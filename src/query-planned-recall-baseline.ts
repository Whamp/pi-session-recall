import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, rm } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallControlKind,
  RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallSearchScope,
} from './enums.js';
import type { RecallSearchResult } from './fuse-recall-ranked-lists.js';
import { isPathInsideRecallEvaluationArea } from './recall-evaluation-file-system.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationDependencies,
  type RecallConversationSearch,
  type RecallConversationSearchResult,
  type RecallSearchCandidateLimits,
} from './recall-conversation-service.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

const SHA256_SCHEMA = Type.String({ pattern: '^[a-f0-9]{64}$' });
const NONEMPTY_STRING_SCHEMA = Type.String({ minLength: 1 });
const POSITIVE_INTEGER_SCHEMA = Type.Integer({ minimum: 1 });

const PRIVATE_EXPECTED_SOURCE_SCHEMA = Type.Object(
  {
    snapshotId: Type.String({ pattern: '^snapshot-[0-9]{3}$' }),
    entryId: NONEMPTY_STRING_SCHEMA,
    requiredText: Type.Array(NONEMPTY_STRING_SCHEMA, { minItems: 1 }),
    expectedSessionOrigin: NONEMPTY_STRING_SCHEMA,
    expectedEvidenceRelation: Type.Enum(RecallEvidenceRelation),
    expectedEvidenceKind: Type.Optional(
      Type.Union([
        Type.Literal('conversation'),
        Type.Literal('turn_context'),
        Type.Literal('tool_call'),
        Type.Literal('tool_result'),
        Type.Literal('bash_execution'),
        Type.Literal('compaction_summary'),
        Type.Literal('branch_summary'),
      ]),
    ),
    expectedBranch: Type.Optional(Type.Union([Type.Literal('active'), Type.Literal('abandoned')])),
  },
  { additionalProperties: false },
);

const PRIVATE_RECALL_BASELINE_CASE_SCHEMA = Type.Object(
  {
    id: Type.String({ pattern: '^case-[0-9]{3}$' }),
    category: Type.Enum(QueryPlannedRecallCaseCategory),
    controlKind: Type.Enum(QueryPlannedRecallControlKind),
    query: NONEMPTY_STRING_SCHEMA,
    querySha256: SHA256_SCHEMA,
    scope: Type.Enum(RecallSearchScope),
    invocationDirectory: Type.Optional(NONEMPTY_STRING_SCHEMA),
    expectedSources: Type.Array(PRIVATE_EXPECTED_SOURCE_SCHEMA, { minItems: 1 }),
    relevantDistractors: Type.Array(
      Type.Object(
        {
          snapshotId: Type.String({ pattern: '^snapshot-[0-9]{3}$' }),
          entryId: NONEMPTY_STRING_SCHEMA,
        },
        { additionalProperties: false },
      ),
    ),
    plannedRetrievalLists: Type.Object(
      {
        lexical: Type.Integer({ minimum: 1, maximum: 3 }),
        semantic: Type.Integer({ minimum: 1, maximum: 3 }),
        hypotheticalAnswer: Type.Integer({ minimum: 0, maximum: 1 }),
      },
      { additionalProperties: false },
    ),
    retrievalWorkMatchedCandidateLimits: Type.Object(
      {
        dense: POSITIVE_INTEGER_SCHEMA,
        lexical: POSITIVE_INTEGER_SCHEMA,
        identifier: POSITIVE_INTEGER_SCHEMA,
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

const PRIVATE_RECALL_BASELINE_MANIFEST_SCHEMA = Type.Object(
  {
    version: Type.Literal(1),
    corpus: Type.Object(
      {
        id: Type.String({ pattern: '^[a-z0-9][a-z0-9-]*$' }),
        snapshotDirectory: Type.Literal('snapshots'),
        snapshots: Type.Array(
          Type.Object(
            {
              id: Type.String({ pattern: '^snapshot-[0-9]{3}$' }),
              fileName: Type.String({ pattern: '^snapshot-[0-9]{3}\\.jsonl$' }),
              sha256: SHA256_SCHEMA,
            },
            { additionalProperties: false },
          ),
          { minItems: 1 },
        ),
      },
      { additionalProperties: false },
    ),
    policy: Type.Object(
      {
        normalCandidateLimits: Type.Object(
          {
            dense: Type.Literal(8),
            lexical: Type.Literal(8),
            identifier: Type.Literal(8),
          },
          { additionalProperties: false },
        ),
        plannedCandidateLimit: Type.Literal(20),
        finalResultLimit: Type.Literal(5),
      },
      { additionalProperties: false },
    ),
    cases: Type.Array(PRIVATE_RECALL_BASELINE_CASE_SCHEMA, { minItems: 1 }),
  },
  { additionalProperties: false },
);

type PrivateQueryPlannedRecallManifest = ReturnType<
  typeof Value.Parse<typeof PRIVATE_RECALL_BASELINE_MANIFEST_SCHEMA>
>;
type PrivateQueryPlannedRecallCase = PrivateQueryPlannedRecallManifest['cases'][number];

/** Checksum-verified private corpus plus paths retained only for local evaluation. */
export interface LoadedPrivateQueryPlannedRecallCorpus {
  manifest: PrivateQueryPlannedRecallManifest;
  manifestPath: string;
  manifestSha256: string;
  snapshotDirectory: string;
  snapshots: Array<
    PrivateQueryPlannedRecallManifest['corpus']['snapshots'][number] & { path: string }
  >;
}

/** Retrieval allowance proving the original-query control matches planned pre-fusion work. */
export interface QueryPlannedRecallRetrievalWork {
  submittedQueryLists: 3;
  plannedQueryLists: number;
  candidatesPerList: 20;
  totalCandidateLimit: number;
  originalQueryCandidateLimits: RecallSearchCandidateLimits;
}

/** Privacy-safe controls fixed before any query planner output is inspected. */
export interface PublishableQueryPlannedRecallControls {
  version: 1;
  corpusId: string;
  privateManifestSha256: string;
  policy: {
    normalCandidateLimits: RecallSearchCandidateLimits;
    plannedCandidateLimit: 20;
    finalResultLimit: 5;
  };
  cases: Array<{
    caseId: string;
    category: QueryPlannedRecallCaseCategory;
    controlKind: QueryPlannedRecallControlKind;
    scope: RecallSearchScope;
    querySha256: string;
    privateCaseSha256: string;
    snapshotSha256: string[];
    expectedSourceCount: number;
    relevantDistractorCount: number;
    expectedSessionOriginSha256: string[];
    expectedEvidenceRelations: RecallEvidenceRelation[];
    expectedEvidenceKinds: Array<SessionConversationChunk['evidenceKind']>;
    expectedBranches: Array<'active' | 'abandoned'>;
    retrievalWork: QueryPlannedRecallRetrievalWork;
  }>;
}

function createSha256(content: string | Buffer): string {
  return createHash('sha256').update(content).digest('hex');
}

function assertUniquePrivateRecallValues(values: readonly string[], label: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`Private query-planned recall manifest invalid: ${label} must be unique`);
  }
}

async function assertPrivateRecallArtifactPermissions(
  path: string,
  directory: boolean,
): Promise<void> {
  const artifact = await lstat(path);
  if (
    artifact.isSymbolicLink() ||
    (directory ? !artifact.isDirectory() : !artifact.isFile()) ||
    (artifact.mode & 0o077) !== 0
  ) {
    throw new Error(
      'Private query-planned recall artifact permissions invalid: directories require 0700 and files require 0600 or stricter',
    );
  }
}

function findPrivateRecallDataDirectory(manifestPath: string): string {
  let current = dirname(manifestPath);
  while (basename(current) !== '.recall-data') {
    const parent = dirname(current);
    if (parent === current) {
      throw new Error(
        'Private query-planned recall manifest invalid: artifacts must be stored below a Git-ignored .recall-data directory',
      );
    }
    current = parent;
  }
  return current;
}

function createQueryPlannedRecallRetrievalWork(
  evaluationCase: PrivateQueryPlannedRecallCase,
  plannedCandidateLimit: 20,
): QueryPlannedRecallRetrievalWork {
  const plannedQueryLists =
    evaluationCase.plannedRetrievalLists.lexical +
    evaluationCase.plannedRetrievalLists.semantic +
    evaluationCase.plannedRetrievalLists.hypotheticalAnswer;
  const totalCandidateLimit = (3 + plannedQueryLists) * plannedCandidateLimit;
  const originalQueryCandidateLimits = evaluationCase.retrievalWorkMatchedCandidateLimits;
  const originalQueryTotal =
    originalQueryCandidateLimits.dense +
    originalQueryCandidateLimits.lexical +
    originalQueryCandidateLimits.identifier;
  if (originalQueryTotal !== totalCandidateLimit) {
    throw new Error(
      `Private query-planned recall manifest invalid: retrieval-work-matched candidate limits for ${evaluationCase.id} total ${originalQueryTotal}, expected ${totalCandidateLimit}`,
    );
  }
  return {
    submittedQueryLists: 3,
    plannedQueryLists,
    candidatesPerList: plannedCandidateLimit,
    totalCandidateLimit,
    originalQueryCandidateLimits: { ...originalQueryCandidateLimits },
  };
}

function assertPrivateRecallManifest(manifest: PrivateQueryPlannedRecallManifest): void {
  assertUniquePrivateRecallValues(
    manifest.corpus.snapshots.map(({ id }) => id),
    'snapshot IDs',
  );
  assertUniquePrivateRecallValues(
    manifest.corpus.snapshots.map(({ fileName }) => fileName),
    'snapshot filenames',
  );
  assertUniquePrivateRecallValues(
    manifest.cases.map(({ id }) => id),
    'case IDs',
  );
  const snapshotIds = new Set(manifest.corpus.snapshots.map(({ id }) => id));
  for (const evaluationCase of manifest.cases) {
    if (createSha256(evaluationCase.query) !== evaluationCase.querySha256) {
      throw new Error(
        `Private query-planned recall query checksum mismatch for ${evaluationCase.id}`,
      );
    }
    if (evaluationCase.scope === RecallSearchScope.PROJECT && !evaluationCase.invocationDirectory) {
      throw new Error(
        `Private query-planned recall manifest invalid: project case ${evaluationCase.id} requires an invocation directory`,
      );
    }
    for (const source of [
      ...evaluationCase.expectedSources,
      ...evaluationCase.relevantDistractors,
    ]) {
      if (!snapshotIds.has(source.snapshotId)) {
        throw new Error(
          `Private query-planned recall manifest invalid: ${evaluationCase.id} references undeclared snapshot ${source.snapshotId}`,
        );
      }
    }
    createQueryPlannedRecallRetrievalWork(evaluationCase, manifest.policy.plannedCandidateLimit);
  }
}

/** Loads private real-session inputs only from a permission-restricted, Git-ignored area. */
export async function loadPrivateQueryPlannedRecallCorpus(
  manifestPath: string,
): Promise<LoadedPrivateQueryPlannedRecallCorpus> {
  const resolvedManifestPath = resolve(manifestPath);
  const privateDataDirectory = findPrivateRecallDataDirectory(resolvedManifestPath);
  const privateDirectory = dirname(resolvedManifestPath);
  await assertPrivateRecallArtifactPermissions(privateDataDirectory, true);
  await assertPrivateRecallArtifactPermissions(privateDirectory, true);
  await assertPrivateRecallArtifactPermissions(resolvedManifestPath, false);
  const manifestContent = await readFile(resolvedManifestPath, 'utf8');
  let manifest: PrivateQueryPlannedRecallManifest;
  try {
    const parsed: unknown = JSON.parse(manifestContent);
    manifest = Value.Parse(PRIVATE_RECALL_BASELINE_MANIFEST_SCHEMA, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Private query-planned recall manifest invalid: ${message}`, { cause: error });
  }
  assertPrivateRecallManifest(manifest);

  const snapshotDirectory = join(privateDirectory, manifest.corpus.snapshotDirectory);
  await assertPrivateRecallArtifactPermissions(snapshotDirectory, true);
  const snapshots: LoadedPrivateQueryPlannedRecallCorpus['snapshots'] = [];
  for (const snapshot of manifest.corpus.snapshots) {
    const path = join(snapshotDirectory, snapshot.fileName);
    await assertPrivateRecallArtifactPermissions(path, false);
    const content = await readFile(path);
    if (createSha256(content) !== snapshot.sha256) {
      throw new Error(`Private query-planned recall snapshot checksum mismatch for ${snapshot.id}`);
    }
    snapshots.push({ ...snapshot, path });
  }
  return {
    manifest,
    manifestPath: resolvedManifestPath,
    manifestSha256: createSha256(manifestContent),
    snapshotDirectory,
    snapshots,
  };
}

/** Projects an allowlisted control manifest containing checksums and counts but no private text or paths. */
export function createPublishableQueryPlannedRecallControls(
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
): PublishableQueryPlannedRecallControls {
  const snapshotsById = new Map(corpus.snapshots.map((snapshot) => [snapshot.id, snapshot]));
  return {
    version: 1,
    corpusId: corpus.manifest.corpus.id,
    privateManifestSha256: corpus.manifestSha256,
    policy: {
      normalCandidateLimits: { ...corpus.manifest.policy.normalCandidateLimits },
      plannedCandidateLimit: corpus.manifest.policy.plannedCandidateLimit,
      finalResultLimit: corpus.manifest.policy.finalResultLimit,
    },
    cases: corpus.manifest.cases.map((evaluationCase) => {
      const sourceSnapshotIds = new Set([
        ...evaluationCase.expectedSources.map(({ snapshotId }) => snapshotId),
        ...evaluationCase.relevantDistractors.map(({ snapshotId }) => snapshotId),
      ]);
      return {
        caseId: evaluationCase.id,
        category: evaluationCase.category,
        controlKind: evaluationCase.controlKind,
        scope: evaluationCase.scope,
        querySha256: evaluationCase.querySha256,
        privateCaseSha256: createSha256(JSON.stringify(evaluationCase)),
        snapshotSha256: Array.from(sourceSnapshotIds)
          .toSorted()
          .map((snapshotId) => {
            const snapshot = snapshotsById.get(snapshotId);
            if (!snapshot) {
              throw new Error(
                `Private query-planned recall manifest invalid: missing snapshot ${snapshotId}`,
              );
            }
            return snapshot.sha256;
          }),
        expectedSourceCount: evaluationCase.expectedSources.length,
        relevantDistractorCount: evaluationCase.relevantDistractors.length,
        expectedSessionOriginSha256: Array.from(
          new Set(
            evaluationCase.expectedSources.map(({ expectedSessionOrigin }) =>
              createSha256(expectedSessionOrigin),
            ),
          ),
        ).toSorted(),
        expectedEvidenceRelations: Array.from(
          new Set(
            evaluationCase.expectedSources.map(
              ({ expectedEvidenceRelation }) => expectedEvidenceRelation,
            ),
          ),
        ).toSorted(),
        expectedEvidenceKinds: Array.from(
          new Set(
            evaluationCase.expectedSources.flatMap(({ expectedEvidenceKind }) =>
              expectedEvidenceKind ? [expectedEvidenceKind] : [],
            ),
          ),
        ).toSorted(),
        expectedBranches: Array.from(
          new Set(
            evaluationCase.expectedSources.flatMap(({ expectedBranch }) =>
              expectedBranch ? [expectedBranch] : [],
            ),
          ),
        ).toSorted(),
        retrievalWork: createQueryPlannedRecallRetrievalWork(
          evaluationCase,
          corpus.manifest.policy.plannedCandidateLimit,
        ),
      };
    }),
  };
}

/** Local providers used to index and search a frozen private baseline corpus. */
export type PrivateQueryPlannedRecallBaselineDependencies = Pick<
  RecallConversationDependencies,
  'embeddings' | 'loadTokenizer' | 'resolveProjectIdentity'
>;

/** Inputs for one private hybrid baseline run isolated from production recall data. */
export interface RunPrivateQueryPlannedRecallBaselineOptions {
  corpus: LoadedPrivateQueryPlannedRecallCorpus;
  baseConfig: RecallConversationConfig;
  workDirectory: string;
  dependencies?: PrivateQueryPlannedRecallBaselineDependencies;
}

/** Privacy-safe measurements for one narrow or retrieval-work-matched hybrid arm. */
export interface QueryPlannedRecallBaselineArmMeasurement {
  outcome: QueryPlannedRecallBaselineOutcome;
  expectedSourceRanks: Array<number | null>;
  highestRelevantDistractorRank: number | null;
  provenancePassed: boolean;
  listLimits: RecallSearchCandidateLimits;
  totalCandidatesExamined: number;
  uniqueCandidatesAdmitted: number;
  finalResultCount: number;
  fusedPoolLimit: number;
  rerankPoolLimit: number;
  rankingMode: 'hybrid';
  rankFusionVersion: number;
  reciprocalRankConstant: number;
}

/** Aggregate and per-case evidence from the frozen private hybrid baseline. */
export interface PrivateQueryPlannedRecallBaselineResult {
  version: 1;
  corpusId: string;
  privateManifestSha256: string;
  indexedSnapshotCount: number;
  indexedDocumentCount: number;
  executedSearchRequests: number;
  cases: Array<{
    caseId: string;
    category: QueryPlannedRecallCaseCategory;
    controlKind: QueryPlannedRecallControlKind;
    normal: QueryPlannedRecallBaselineArmMeasurement;
    retrievalWorkMatched: QueryPlannedRecallBaselineArmMeasurement;
  }>;
  outcomeCounts: {
    normal: Record<QueryPlannedRecallBaselineOutcome, number>;
    retrievalWorkMatched: Record<QueryPlannedRecallBaselineOutcome, number>;
  };
}

function assertPrivateBaselineWorkDirectory(
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
  workDirectory: string,
): string {
  const resolvedWorkDirectory = resolve(workDirectory);
  const privateDirectory = dirname(corpus.manifestPath);
  if (
    resolvedWorkDirectory === privateDirectory ||
    !isPathInsideRecallEvaluationArea(privateDirectory, resolvedWorkDirectory)
  ) {
    throw new Error(
      'Private query-planned recall baseline work directory must stay inside the private evaluation area',
    );
  }
  if (
    isPathInsideRecallEvaluationArea(corpus.snapshotDirectory, resolvedWorkDirectory) ||
    isPathInsideRecallEvaluationArea(resolvedWorkDirectory, corpus.snapshotDirectory)
  ) {
    throw new Error(
      'Private query-planned recall baseline work directory overlaps immutable snapshots',
    );
  }
  if (
    isPathInsideRecallEvaluationArea(corpus.manifestPath, resolvedWorkDirectory) ||
    isPathInsideRecallEvaluationArea(resolvedWorkDirectory, corpus.manifestPath)
  ) {
    throw new Error(
      'Private query-planned recall baseline work directory overlaps the private manifest',
    );
  }
  return resolvedWorkDirectory;
}

function createPrivateBaselineConfig(
  options: RunPrivateQueryPlannedRecallBaselineOptions,
  workDirectory: string,
  candidateLimits: RecallSearchCandidateLimits,
): RecallConversationConfig {
  const fusedPoolLimit =
    candidateLimits.dense + candidateLimits.lexical + candidateLimits.identifier;
  return {
    ...options.baseConfig,
    sessionsDirectory: options.corpus.snapshotDirectory,
    databasePath: join(workDirectory, 'zvec'),
    statePath: join(workDirectory, 'index-state.json'),
    manifestPath: join(workDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(workDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(workDirectory, 'embedding-cache'),
    lockPath: join(workDirectory, 'operation.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(workDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(workDirectory, 'diagnostics.previous.jsonl'),
    searchCandidateLimits: { ...candidateLimits },
    fusedPoolLimit,
    rerankPoolLimit: fusedPoolLimit,
  };
}

function createPrivateBaselineDependencies(
  dependencies?: PrivateQueryPlannedRecallBaselineDependencies,
): RecallConversationDependencies {
  return {
    ...dependencies,
    reranker: {
      async rerankDocuments() {
        throw new Error('Private query-planned hybrid baseline must not invoke the reranker');
      },
    },
  };
}

function getPrivateBaselineGroupMembers(
  result: RecallConversationSearchResult,
): RecallSearchResult[] {
  return [result, ...result.duplicateOccurrences];
}

function matchesPrivateBaselineSourceIdentity(
  candidate: RecallSearchResult,
  source: PrivateQueryPlannedRecallCase['expectedSources'][number],
  snapshotFileName: string,
): boolean {
  return (
    basename(candidate.sessionPath) === snapshotFileName &&
    (candidate.entryId.value === source.entryId ||
      candidate.contributingEntryIds.some(({ value }) => value === source.entryId)) &&
    source.requiredText.every((requiredText) => candidate.content.includes(requiredText)) &&
    (!source.expectedEvidenceKind || candidate.evidenceKind === source.expectedEvidenceKind)
  );
}

function verifiesPrivateBaselineProvenance(
  result: RecallConversationSearchResult,
  candidate: RecallSearchResult,
  source: PrivateQueryPlannedRecallCase['expectedSources'][number],
): boolean {
  return (
    candidate.cwd === source.expectedSessionOrigin &&
    result.evidenceRelation === source.expectedEvidenceRelation &&
    (!source.expectedBranch ||
      (source.expectedBranch === 'active' && candidate.isOnActiveBranch) ||
      (source.expectedBranch === 'abandoned' && !candidate.isOnActiveBranch))
  );
}

function findPrivateBaselineSourceMatch(
  search: RecallConversationSearch,
  source: PrivateQueryPlannedRecallCase['expectedSources'][number],
  snapshotFileName: string,
): { rank: number; provenancePassed: boolean } | null {
  for (const [index, result] of search.results.entries()) {
    const candidate = getPrivateBaselineGroupMembers(result).find((groupMember) =>
      matchesPrivateBaselineSourceIdentity(groupMember, source, snapshotFileName),
    );
    if (candidate) {
      return {
        rank: index + 1,
        provenancePassed: verifiesPrivateBaselineProvenance(result, candidate, source),
      };
    }
  }
  return null;
}

function findPrivateBaselineDistractorRank(
  search: RecallConversationSearch,
  distractor: PrivateQueryPlannedRecallCase['relevantDistractors'][number],
  snapshotFileName: string,
): number | null {
  const index = search.results.findIndex((result) =>
    getPrivateBaselineGroupMembers(result).some(
      (candidate) =>
        basename(candidate.sessionPath) === snapshotFileName &&
        (candidate.entryId.value === distractor.entryId ||
          candidate.contributingEntryIds.some(({ value }) => value === distractor.entryId)),
    ),
  );
  return index < 0 ? null : index + 1;
}

function classifyPrivateBaselineOutcome(
  expectedSourceRanks: readonly (number | null)[],
  finalResultLimit: number,
): QueryPlannedRecallBaselineOutcome {
  if (expectedSourceRanks.some((rank) => rank === null)) {
    return QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS;
  }
  if (expectedSourceRanks.some((rank) => rank !== null && rank > finalResultLimit)) {
    return QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS;
  }
  return QueryPlannedRecallBaselineOutcome.SUCCESS;
}

function measurePrivateBaselineSearch(
  search: RecallConversationSearch,
  evaluationCase: PrivateQueryPlannedRecallCase,
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
  finalResultLimit: number,
): QueryPlannedRecallBaselineArmMeasurement {
  const sourceMatches = evaluationCase.expectedSources.map((source) => {
    const snapshot = snapshotsById.get(source.snapshotId);
    if (!snapshot) {
      throw new Error(
        `Private query-planned recall baseline missing snapshot ${source.snapshotId}`,
      );
    }
    return findPrivateBaselineSourceMatch(search, source, snapshot.fileName);
  });
  const distractorRanks = evaluationCase.relevantDistractors.map((distractor) => {
    const snapshot = snapshotsById.get(distractor.snapshotId);
    if (!snapshot) {
      throw new Error(
        `Private query-planned recall baseline missing snapshot ${distractor.snapshotId}`,
      );
    }
    return findPrivateBaselineDistractorRank(search, distractor, snapshot.fileName);
  });
  const groupMembers = search.results.flatMap(getPrivateBaselineGroupMembers);
  return {
    outcome: classifyPrivateBaselineOutcome(
      sourceMatches.map((match) => match?.rank ?? null),
      finalResultLimit,
    ),
    expectedSourceRanks: sourceMatches.map((match) => match?.rank ?? null),
    highestRelevantDistractorRank:
      distractorRanks
        .filter((rank): rank is number => rank !== null)
        .toSorted((a, b) => a - b)[0] ?? null,
    provenancePassed: sourceMatches.every((match) => match?.provenancePassed === true),
    listLimits: { ...search.searchPolicy.candidateLimits },
    totalCandidatesExamined: groupMembers.reduce(
      (total, candidate) => total + candidate.rankedListEvidence.length,
      0,
    ),
    uniqueCandidatesAdmitted: groupMembers.length,
    finalResultCount: Math.min(finalResultLimit, search.results.length),
    fusedPoolLimit: search.searchPolicy.fusedPoolLimit,
    rerankPoolLimit: search.searchPolicy.rerankPoolLimit,
    rankingMode: 'hybrid',
    rankFusionVersion: search.searchPolicy.rankFusionVersion,
    reciprocalRankConstant: search.searchPolicy.reciprocalRankConstant,
  };
}

function countPrivateBaselineOutcomes(
  measurements: readonly QueryPlannedRecallBaselineArmMeasurement[],
): Record<QueryPlannedRecallBaselineOutcome, number> {
  return {
    [QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS]: measurements.filter(
      ({ outcome }) => outcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS,
    ).length,
    [QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS]: measurements.filter(
      ({ outcome }) => outcome === QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
    ).length,
    [QueryPlannedRecallBaselineOutcome.SUCCESS]: measurements.filter(
      ({ outcome }) => outcome === QueryPlannedRecallBaselineOutcome.SUCCESS,
    ).length,
  };
}

/** Publishable model identity and fixed Git point for one pre-planning baseline run. */
export interface QueryPlannedRecallBaselineEvidenceEnvironment {
  recordedAgainstCommit: string;
  embeddingProfile: {
    requestModel: string;
    servedModelId: string;
    artifact: string;
    dimensions: number;
    quantization: string;
    pooling: string;
  };
}

/** Privacy-safe evidence envelope committed before query-planned recall is implemented. */
export interface PublishableQueryPlannedRecallBaselineEvidence {
  version: 1;
  recordedAgainstCommit: string;
  privateManifestSha256: string;
  controlsSha256: string;
  embeddingProfile: QueryPlannedRecallBaselineEvidenceEnvironment['embeddingProfile'];
  baseline: PrivateQueryPlannedRecallBaselineResult;
}

function createPublishableControlsSha256(controls: PublishableQueryPlannedRecallControls): string {
  return createSha256(JSON.stringify(controls));
}

/** Binds safe controls and aggregate hybrid measurements to the pre-planning implementation. */
export function createPublishableQueryPlannedRecallBaselineEvidence(
  controls: PublishableQueryPlannedRecallControls,
  baseline: PrivateQueryPlannedRecallBaselineResult,
  environment: QueryPlannedRecallBaselineEvidenceEnvironment,
): PublishableQueryPlannedRecallBaselineEvidence {
  if (!/^[a-f0-9]{40}$/u.test(environment.recordedAgainstCommit)) {
    throw new Error(
      'Query-planned recall baseline evidence invalid: recorded commit must be a full Git SHA-1',
    );
  }
  if (
    controls.corpusId !== baseline.corpusId ||
    controls.privateManifestSha256 !== baseline.privateManifestSha256
  ) {
    throw new Error(
      'Query-planned recall baseline evidence invalid: controls and measurements must bind the same private manifest',
    );
  }
  const controlsByCaseId = new Map(controls.cases.map((control) => [control.caseId, control]));
  for (const measurement of baseline.cases) {
    const control = controlsByCaseId.get(measurement.caseId);
    if (
      !control ||
      control.category !== measurement.category ||
      control.controlKind !== measurement.controlKind
    ) {
      throw new Error(
        `Query-planned recall baseline evidence invalid: control mismatch for ${measurement.caseId}`,
      );
    }
  }
  return {
    version: 1,
    recordedAgainstCommit: environment.recordedAgainstCommit,
    privateManifestSha256: baseline.privateManifestSha256,
    controlsSha256: createPublishableControlsSha256(controls),
    embeddingProfile: {
      ...environment.embeddingProfile,
      artifact: basename(environment.embeddingProfile.artifact),
    },
    baseline,
  };
}

function formatPrivateBaselineOutcome(outcome: QueryPlannedRecallBaselineOutcome): string {
  return outcome.replaceAll('_', ' ');
}

/** Formats aggregate and per-case baseline evidence without private query or source material. */
export function formatPublishableQueryPlannedRecallBaselineReport(
  evidence: PublishableQueryPlannedRecallBaselineEvidence,
): string {
  const { baseline } = evidence;
  const normal = baseline.outcomeCounts.normal;
  const matched = baseline.outcomeCounts.retrievalWorkMatched;
  const lines = [
    '# Query-Planned Recall: Pre-Planning Hybrid Baseline',
    '',
    'This report records current hybrid behavior before query-planned recall is implemented. Private queries and unchanged real session snapshots remain in a permission-restricted, Git-ignored local corpus; this report contains only opaque case identities, categories, policy values, counts, ranks, and checksums.',
    '',
    '## Identity',
    '',
    `- Recorded against commit: \`${evidence.recordedAgainstCommit}\``,
    `- Private manifest SHA-256: \`${evidence.privateManifestSha256}\``,
    `- Publishable controls canonical JSON SHA-256: \`${evidence.controlsSha256}\``,
    `- Embedding profile: \`${evidence.embeddingProfile.requestModel}\` / \`${evidence.embeddingProfile.servedModelId}\`, ${evidence.embeddingProfile.dimensions} dimensions, ${evidence.embeddingProfile.quantization}, ${evidence.embeddingProfile.pooling} pooling`,
    `- Frozen cases: ${baseline.cases.length} across ${baseline.indexedSnapshotCount} unchanged snapshots and ${baseline.indexedDocumentCount} indexed documents`,
    `- Executed searches: ${baseline.executedSearchRequests} (normal plus retrieval-work-matched original query for every case)`,
    '',
    '## Aggregate outcomes',
    '',
    `- Normal hybrid: ${normal.success} success, ${normal.final_rank_miss} final rank miss, ${normal.candidate_union_miss} candidate union miss.`,
    `- Retrieval-work-matched original query: ${matched.success} success, ${matched.final_rank_miss} final rank miss, ${matched.candidate_union_miss} candidate union miss.`,
    '',
    'A candidate union miss means the exact expected source was absent after all admitted ranked-list candidates were fused. A final rank miss means the source was admitted but ranked below the fixed final-five cutoff.',
    '',
    '## Cases',
    '',
    '| Case | Category | Role | Normal outcome | Normal rank | Normal work (examined / unique) | Work-matched outcome | Work-matched rank | Work-matched work (examined / unique) |',
    '| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |',
  ];
  for (const evaluationCase of baseline.cases) {
    const normalRank = evaluationCase.normal.expectedSourceRanks
      .map((rank) => rank ?? 'absent')
      .join(', ');
    const matchedRank = evaluationCase.retrievalWorkMatched.expectedSourceRanks
      .map((rank) => rank ?? 'absent')
      .join(', ');
    lines.push(
      `| ${evaluationCase.caseId} | ${evaluationCase.category.replaceAll('_', ' ')} | ${evaluationCase.controlKind.replaceAll('_', ' ')} | ${formatPrivateBaselineOutcome(evaluationCase.normal.outcome)} | ${normalRank} | ${evaluationCase.normal.totalCandidatesExamined} / ${evaluationCase.normal.uniqueCandidatesAdmitted} | ${formatPrivateBaselineOutcome(evaluationCase.retrievalWorkMatched.outcome)} | ${matchedRank} | ${evaluationCase.retrievalWorkMatched.totalCandidatesExamined} / ${evaluationCase.retrievalWorkMatched.uniqueCandidatesAdmitted} |`,
    );
  }
  lines.push(
    '',
    '## Interpretation guardrails',
    '',
    '- No query planner was run or inspected while selecting or measuring these cases.',
    '- The larger arm repeats only the original query and matches the anticipated planned arm’s total pre-fusion candidate allowance.',
    '- A future planned-query result earns candidate-generation credit only when it admits the expected source beyond both controls.',
    '- A source already admitted but below the final-five cutoff is a ranking problem, not a candidate-generation win.',
    '- Existing committed recall-quality evidence remains the production rollout gate; this private baseline does not replace it.',
    '',
  );
  return lines.join('\n');
}

/** Indexes unchanged private snapshots and records normal and work-matched hybrid outcomes. */
export async function runPrivateQueryPlannedRecallBaseline(
  options: RunPrivateQueryPlannedRecallBaselineOptions,
): Promise<PrivateQueryPlannedRecallBaselineResult> {
  const workDirectory = assertPrivateBaselineWorkDirectory(options.corpus, options.workDirectory);
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true, mode: 0o700 });
  const dependencies = createPrivateBaselineDependencies(options.dependencies);
  const indexConfig = createPrivateBaselineConfig(
    options,
    workDirectory,
    options.corpus.manifest.policy.normalCandidateLimits,
  );
  const indexService = createRecallConversationService(indexConfig, dependencies);
  const indexed = await indexService.index({ optimize: true });
  if (indexed.indexSummary.failedSessions.length > 0) {
    throw new Error(
      `Private query-planned recall baseline index failed for ${indexed.indexSummary.failedSessions.length} snapshot(s)`,
    );
  }
  if (indexed.indexSummary.scannedSessions !== options.corpus.snapshots.length) {
    throw new Error(
      `Private query-planned recall baseline scan mismatch: expected ${options.corpus.snapshots.length}, received ${indexed.indexSummary.scannedSessions}`,
    );
  }

  const snapshotsById = new Map(
    options.corpus.snapshots.map((snapshot) => [snapshot.id, snapshot]),
  );
  const cases: PrivateQueryPlannedRecallBaselineResult['cases'] = [];
  let executedSearchRequests = 0;
  for (const evaluationCase of options.corpus.manifest.cases) {
    const runArm = async (
      candidateLimits: RecallSearchCandidateLimits,
    ): Promise<QueryPlannedRecallBaselineArmMeasurement> => {
      const config = createPrivateBaselineConfig(options, workDirectory, candidateLimits);
      const service = createRecallConversationService(config, dependencies);
      const fusedPoolLimit =
        candidateLimits.dense + candidateLimits.lexical + candidateLimits.identifier;
      const search = await service.search(evaluationCase.query, fusedPoolLimit, {
        mode: 'hybrid',
        scope: evaluationCase.scope,
        ...(evaluationCase.invocationDirectory
          ? { invocationDirectory: evaluationCase.invocationDirectory }
          : {}),
      });
      executedSearchRequests += 1;
      return measurePrivateBaselineSearch(
        search,
        evaluationCase,
        snapshotsById,
        options.corpus.manifest.policy.finalResultLimit,
      );
    };
    const normal = await runArm(options.corpus.manifest.policy.normalCandidateLimits);
    const retrievalWorkMatched = await runArm(evaluationCase.retrievalWorkMatchedCandidateLimits);
    cases.push({
      caseId: evaluationCase.id,
      category: evaluationCase.category,
      controlKind: evaluationCase.controlKind,
      normal,
      retrievalWorkMatched,
    });
  }
  return {
    version: 1,
    corpusId: options.corpus.manifest.corpus.id,
    privateManifestSha256: options.corpus.manifestSha256,
    indexedSnapshotCount: options.corpus.snapshots.length,
    indexedDocumentCount: indexed.totalChunks,
    executedSearchRequests,
    cases,
    outcomeCounts: {
      normal: countPrivateBaselineOutcomes(cases.map(({ normal }) => normal)),
      retrievalWorkMatched: countPrivateBaselineOutcomes(
        cases.map(({ retrievalWorkMatched }) => retrievalWorkMatched),
      ),
    },
  };
}

import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, rm } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallControlKind,
  RecallDiagnosticsMode,
} from './enums.js';
import type { QueryPlannedRecallCaseCategory } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-ranked-lists.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import { isPathInsideRecallEvaluationArea } from './recall-evaluation-file-system.js';
import type {
  LoadedPrivateQueryPlannedRecallCorpus,
  PublishableQueryPlannedRecallControls,
  QueryPlannedRecallBaselineArmMeasurement,
} from './query-planned-recall-baseline.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationDependencies,
  type RecallConversationSearch,
  type RecallConversationSearchResult,
  type RecallPlannedRetrievalQuery,
  type RecallSearchCandidateLimits,
} from './recall-conversation-service.js';

const SHA256_SCHEMA = Type.String({ pattern: '^[a-f0-9]{64}$' });
/** Fixed vector width for deterministic token-hash quality evaluation embeddings. */
export const QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS = 256;
const PRIVATE_QUERY_PLANNED_RECALL_PLANS_SCHEMA = Type.Object(
  {
    version: Type.Literal(1),
    corpusId: Type.String({ minLength: 1 }),
    privateManifestSha256: SHA256_SCHEMA,
    cases: Type.Array(
      Type.Object(
        {
          caseId: Type.String({ pattern: '^case-[0-9]{3}$' }),
          queries: Type.Array(
            Type.Object(
              {
                type: Type.Union([Type.Literal('lex'), Type.Literal('vec'), Type.Literal('hyde')]),
                query: Type.String({ minLength: 1 }),
              },
              { additionalProperties: false },
            ),
            { minItems: 1, maxItems: 10 },
          ),
        },
        { additionalProperties: false },
      ),
      { minItems: 1 },
    ),
  },
  { additionalProperties: false },
);

type PrivateQueryPlannedRecallPlansDocument = ReturnType<
  typeof Value.Parse<typeof PRIVATE_QUERY_PLANNED_RECALL_PLANS_SCHEMA>
>;

/** Checksum-bound private fixed plans retained only for local quality evaluation. */
export interface LoadedPrivateQueryPlannedRecallPlans {
  document: PrivateQueryPlannedRecallPlansDocument;
  path: string;
  sha256: string;
}

/** Privacy-safe fixed-plan identity containing query hashes instead of private query text. */
export interface PublishableQueryPlannedRecallPlanIdentity {
  source: 'agent';
  planSha256: string;
  cases: Array<{
    caseId: string;
    plannedQueries: Array<{
      type: RecallPlannedRetrievalQuery['type'];
      querySha256: string;
    }>;
  }>;
}

function createQueryPlannedRecallSha256(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

async function assertPrivateQueryPlannedRecallPlanPermissions(path: string): Promise<void> {
  const artifact = await lstat(path);
  if (artifact.isSymbolicLink() || !artifact.isFile() || (artifact.mode & 0o077) !== 0) {
    throw new Error(
      'Private query-planned recall plan permissions invalid: the plan file requires 0600 or stricter',
    );
  }
}

function countPrivatePlannedQueries(queries: readonly RecallPlannedRetrievalQuery[]): {
  lexical: number;
  semantic: number;
  hypotheticalAnswer: number;
} {
  return {
    lexical: queries.filter(({ type }) => type === 'lex').length,
    semantic: queries.filter(({ type }) => type === 'vec').length,
    hypotheticalAnswer: queries.filter(({ type }) => type === 'hyde').length,
  };
}

function assertPrivateQueryPlannedRecallPlans(
  plans: PrivateQueryPlannedRecallPlansDocument,
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
): void {
  if (
    plans.corpusId !== corpus.manifest.corpus.id ||
    plans.privateManifestSha256 !== corpus.manifestSha256
  ) {
    throw new Error(
      'Private query-planned recall plans invalid: plans must bind the loaded private corpus manifest',
    );
  }
  const plansByCaseId = new Map(
    plans.cases.map((plannedCase) => [plannedCase.caseId, plannedCase]),
  );
  if (plansByCaseId.size !== plans.cases.length) {
    throw new Error('Private query-planned recall plans invalid: case IDs must be unique');
  }
  if (plans.cases.length !== corpus.manifest.cases.length) {
    throw new Error(
      'Private query-planned recall plans invalid: every private corpus case requires one fixed plan',
    );
  }
  for (const evaluationCase of corpus.manifest.cases) {
    const plannedCase = plansByCaseId.get(evaluationCase.id);
    if (!plannedCase) {
      throw new Error(
        `Private query-planned recall plans invalid: missing fixed plan for ${evaluationCase.id}`,
      );
    }
    const actualCounts = countPrivatePlannedQueries(plannedCase.queries);
    const expectedCounts = evaluationCase.plannedRetrievalLists;
    if (
      actualCounts.lexical !== expectedCounts.lexical ||
      actualCounts.semantic !== expectedCounts.semantic ||
      actualCounts.hypotheticalAnswer !== expectedCounts.hypotheticalAnswer
    ) {
      throw new Error(
        `Private query-planned recall plans invalid: query-type counts for ${evaluationCase.id} do not match the frozen retrieval work`,
      );
    }
  }
}

/** Loads fixed private plans after permission, schema, corpus, and retrieval-work checks. */
export async function loadPrivateQueryPlannedRecallPlans(
  plansPath: string,
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
): Promise<LoadedPrivateQueryPlannedRecallPlans> {
  const resolvedPlansPath = resolve(plansPath);
  if (dirname(resolvedPlansPath) !== dirname(corpus.manifestPath)) {
    throw new Error(
      'Private query-planned recall plans invalid: the plan file must stay beside the private manifest',
    );
  }
  await assertPrivateQueryPlannedRecallPlanPermissions(resolvedPlansPath);
  const content = await readFile(resolvedPlansPath, 'utf8');
  let document: PrivateQueryPlannedRecallPlansDocument;
  try {
    const parsed: unknown = JSON.parse(content);
    document = Value.Parse(PRIVATE_QUERY_PLANNED_RECALL_PLANS_SCHEMA, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Private query-planned recall plans invalid: ${message}`, { cause: error });
  }
  assertPrivateQueryPlannedRecallPlans(document, corpus);
  return {
    document,
    path: resolvedPlansPath,
    sha256: createQueryPlannedRecallSha256(content),
  };
}

/** Projects fixed private plans to case IDs, query types, and hashes without query text. */
export function createPublishableQueryPlannedRecallPlanIdentity(
  plans: LoadedPrivateQueryPlannedRecallPlans,
): PublishableQueryPlannedRecallPlanIdentity {
  return {
    source: 'agent',
    planSha256: plans.sha256,
    cases: plans.document.cases.map((plannedCase) => ({
      caseId: plannedCase.caseId,
      plannedQueries: plannedCase.queries.map((query) => ({
        type: query.type,
        querySha256: createQueryPlannedRecallSha256(query.query),
      })),
    })),
  };
}

/** Dependencies allowed to vary without changing deterministic embedding or reranker behavior. */
export type PrivateQueryPlannedRecallEvaluationDependencies = Pick<
  RecallConversationDependencies,
  'loadTokenizer' | 'resolveProjectIdentity'
>;

/** Inputs for one deterministic fixed-plan evaluation isolated from production recall data. */
export interface RunPrivateQueryPlannedRecallEvaluationOptions {
  corpus: LoadedPrivateQueryPlannedRecallCorpus;
  plans: LoadedPrivateQueryPlannedRecallPlans;
  baseConfig: RecallConversationConfig;
  workDirectory: string;
  dependencies?: PrivateQueryPlannedRecallEvaluationDependencies;
}

/** Privacy-safe work performed by each planned ranked list. */
export interface QueryPlannedRecallListWorkMeasurement {
  source: string;
  weight: number;
  candidateLimit: number;
  admittedCandidateCount: number;
}

/** Source admission and neutral ranking measurements for one fixed query plan. */
export interface QueryPlannedRecallArmMeasurement {
  outcome: QueryPlannedRecallBaselineOutcome;
  expectedSourceRanks: Array<number | null>;
  admissionProbeSourceRanks: Array<number | null>;
  candidateAdmissionVerified: boolean;
  provenancePassed: boolean;
  listWork: QueryPlannedRecallListWorkMeasurement[];
  totalCandidatesExamined: number;
  rerankCandidatesExamined: number;
  finalResultCount: number;
  fusedPoolLimit: number;
  rerankPoolLimit: number;
  finalResultLimit: number;
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  fusionPolicy: {
    submittedQueryListWeight: number;
    plannedQueryListWeight: number;
    rankOneBonus: number;
    rankTwoOrThreeBonus: number;
  };
  rerankerPolicy: {
    version: number;
    activeBranchPrior: number;
    fusedRankBlend: Array<{
      firstRank: number;
      lastRank: number | null;
      retrievalWeight: number;
      rerankerWeight: number;
    }>;
  };
  rankingProviderPolicy: 'neutral-fused-order-v1';
  admissionProbeProviderPolicy: 'expected-source-promotion-v1';
}

/** Aggregate deterministic evidence comparing controls with one fixed private plan per case. */
export interface PrivateQueryPlannedRecallEvaluationResult {
  version: 1;
  corpusId: string;
  privateManifestSha256: string;
  planIdentity: PublishableQueryPlannedRecallPlanIdentity;
  providerIdentity: {
    embeddingPolicy: 'deterministic-token-hash-v1';
    embeddingDimensions: number;
    rankingRerankerPolicy: 'neutral-fused-order-v1';
    admissionProbeRerankerPolicy: 'expected-source-promotion-v1';
  };
  indexedSnapshotCount: number;
  indexedDocumentCount: number;
  executedSearchRequests: number;
  cases: Array<{
    caseId: string;
    category: QueryPlannedRecallCaseCategory;
    controlKind: QueryPlannedRecallControlKind;
    normal: QueryPlannedRecallBaselineArmMeasurement;
    retrievalWorkMatched: QueryPlannedRecallBaselineArmMeasurement;
    queryPlanned: QueryPlannedRecallArmMeasurement;
    contribution: {
      newCandidateAdmission: boolean;
      rankingOnlyPromotion: boolean;
      preservedExistingSuccess: boolean;
      noImprovement: boolean;
    };
  }>;
  contributionCounts: {
    newCandidateAdmission: number;
    rankingOnlyPromotion: number;
    preservedExistingSuccess: number;
    noImprovement: number;
  };
}

interface PrivateExpectedSourceMatch {
  rank: number;
  provenancePassed: boolean;
}

interface ControlledEvaluationReranker {
  reranker: LocalRerankerClient;
  readLastCandidateCount(): number;
}

function assertPrivateQueryPlannedEvaluationWorkDirectory(
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
      'Private query-planned recall evaluation work directory must stay inside the private evaluation area',
    );
  }
  if (
    isPathInsideRecallEvaluationArea(corpus.snapshotDirectory, resolvedWorkDirectory) ||
    isPathInsideRecallEvaluationArea(resolvedWorkDirectory, corpus.snapshotDirectory) ||
    isPathInsideRecallEvaluationArea(corpus.manifestPath, resolvedWorkDirectory) ||
    isPathInsideRecallEvaluationArea(resolvedWorkDirectory, corpus.manifestPath)
  ) {
    throw new Error(
      'Private query-planned recall evaluation work directory overlaps immutable private inputs',
    );
  }
  return resolvedWorkDirectory;
}

function createPrivateQueryPlannedEvaluationConfig(
  options: RunPrivateQueryPlannedRecallEvaluationOptions,
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
    embeddingModel: 'deterministic-token-hash-v1',
    embeddingServedModelId: 'deterministic-token-hash-v1',
    embeddingArtifact: 'none',
    embeddingQuantization: 'none',
    embeddingPooling: 'token-hash',
    embeddingDimensions: QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS,
    searchCandidateLimits: { ...candidateLimits },
    fusedPoolLimit,
    rerankPoolLimit: fusedPoolLimit,
  };
}

function createDeterministicTokenHashVector(text: string, dimensions: number): number[] {
  const vector = Array.from({ length: dimensions }, () => 0);
  const tokens = text.toLocaleLowerCase('en-US').match(/[\p{L}\p{N}_./:-]+/gu) ?? [];
  for (const token of tokens) {
    let hash = 2_166_136_261;
    for (const character of token) {
      hash ^= character.codePointAt(0) ?? 0;
      hash = Math.imul(hash, 16_777_619) >>> 0;
    }
    const index = hash % dimensions;
    vector[index] = (vector[index] ?? 0) + 1;
  }
  const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (magnitude === 0) {
    vector[0] = 1;
    return vector;
  }
  return vector.map((value) => value / magnitude);
}

function createDeterministicQueryPlannedEvaluationDependencies(
  options: RunPrivateQueryPlannedRecallEvaluationOptions,
  reranker: LocalRerankerClient,
): RecallConversationDependencies {
  const dimensions = QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS;
  return {
    ...options.dependencies,
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => createDeterministicTokenHashVector(text, dimensions));
      },
    },
    reranker,
  };
}

function createControlledEvaluationReranker(
  requiredTextGroups?: readonly (readonly string[])[],
): ControlledEvaluationReranker {
  let lastCandidateCount = 0;
  return {
    reranker: {
      async rerankDocuments(rerankerQuery, documents) {
        void rerankerQuery;
        lastCandidateCount = documents.length;
        return documents.map((document) =>
          requiredTextGroups?.some((requiredTexts) =>
            requiredTexts.every((requiredText) => document.includes(requiredText)),
          )
            ? 1
            : requiredTextGroups
              ? -1
              : 0,
        );
      },
    },
    readLastCandidateCount() {
      return lastCandidateCount;
    },
  };
}

function getQueryPlannedRecallGroupMembers(
  result: RecallConversationSearchResult,
): RecallSearchResult[] {
  return [result, ...result.duplicateOccurrences];
}

function findPrivateExpectedSourceMatch(
  search: RecallConversationSearch,
  source: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number]['expectedSources'][number],
  snapshotFileName: string,
): PrivateExpectedSourceMatch | null {
  for (const [index, result] of search.results.entries()) {
    const candidate = getQueryPlannedRecallGroupMembers(result).find(
      (groupMember) =>
        basename(groupMember.sessionPath) === snapshotFileName &&
        (groupMember.entryId.value === source.entryId ||
          groupMember.contributingEntryIds.some(({ value }) => value === source.entryId)) &&
        source.requiredText.every((requiredText) => groupMember.content.includes(requiredText)) &&
        (!source.expectedEvidenceKind || groupMember.evidenceKind === source.expectedEvidenceKind),
    );
    if (candidate) {
      return {
        rank: index + 1,
        provenancePassed:
          candidate.cwd === source.expectedSessionOrigin &&
          result.evidenceRelation === source.expectedEvidenceRelation &&
          (!source.expectedBranch ||
            (source.expectedBranch === 'active' && candidate.isOnActiveBranch) ||
            (source.expectedBranch === 'abandoned' && !candidate.isOnActiveBranch)),
      };
    }
  }
  return null;
}

function findPrivateExpectedSourceMatches(
  search: RecallConversationSearch,
  evaluationCase: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number],
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
): Array<PrivateExpectedSourceMatch | null> {
  return evaluationCase.expectedSources.map((source) => {
    const snapshot = snapshotsById.get(source.snapshotId);
    if (!snapshot) {
      throw new Error(
        `Private query-planned recall evaluation missing snapshot ${source.snapshotId}`,
      );
    }
    return findPrivateExpectedSourceMatch(search, source, snapshot.fileName);
  });
}

function classifyPrivateQueryPlannedOutcome(
  expectedSourceRanks: readonly (number | null)[],
  candidateAdmissionVerified: boolean,
): QueryPlannedRecallBaselineOutcome {
  if (!candidateAdmissionVerified) {
    return QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS;
  }
  return expectedSourceRanks.every((rank) => rank !== null)
    ? QueryPlannedRecallBaselineOutcome.SUCCESS
    : QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS;
}

function measurePrivateHybridControl(
  search: RecallConversationSearch,
  evaluationCase: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number],
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
  finalResultLimit: number,
): QueryPlannedRecallBaselineArmMeasurement {
  const matches = findPrivateExpectedSourceMatches(search, evaluationCase, snapshotsById);
  const ranks = matches.map((match) => match?.rank ?? null);
  const groupMembers = search.results.flatMap(getQueryPlannedRecallGroupMembers);
  const outcome = ranks.some((rank) => rank === null)
    ? QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS
    : ranks.some((rank) => rank !== null && rank > finalResultLimit)
      ? QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS
      : QueryPlannedRecallBaselineOutcome.SUCCESS;
  return {
    outcome,
    expectedSourceRanks: ranks,
    highestRelevantDistractorRank: null,
    provenancePassed: matches.every((match) => match?.provenancePassed === true),
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

function measurePrivateQueryPlannedArm(
  rankedSearch: RecallConversationSearch,
  admissionProbeSearch: RecallConversationSearch,
  evaluationCase: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number],
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
  rerankCandidatesExamined: number,
): QueryPlannedRecallArmMeasurement {
  const rankedMatches = findPrivateExpectedSourceMatches(
    rankedSearch,
    evaluationCase,
    snapshotsById,
  );
  const admissionMatches = findPrivateExpectedSourceMatches(
    admissionProbeSearch,
    evaluationCase,
    snapshotsById,
  );
  const expectedSourceRanks = rankedMatches.map((match) => match?.rank ?? null);
  const admissionProbeSourceRanks = admissionMatches.map((match) => match?.rank ?? null);
  const candidateAdmissionVerified = admissionMatches.every((match) => match !== null);
  const queryPlan = rankedSearch.searchPolicy.queryPlan;
  if (!queryPlan) {
    throw new Error('Private query-planned recall evaluation missing public query-plan evidence');
  }
  const listWork = queryPlan.rankedLists.map((list) => ({
    source: list.source,
    weight: list.weight,
    candidateLimit: list.candidateLimit,
    admittedCandidateCount: list.admittedCandidateCount,
  }));
  return {
    outcome: classifyPrivateQueryPlannedOutcome(expectedSourceRanks, candidateAdmissionVerified),
    expectedSourceRanks,
    admissionProbeSourceRanks,
    candidateAdmissionVerified,
    provenancePassed: (rankedMatches.some((match) => match !== null)
      ? rankedMatches
      : admissionMatches
    ).every((match) => match?.provenancePassed === true),
    listWork,
    totalCandidatesExamined: listWork.reduce(
      (total, list) => total + list.admittedCandidateCount,
      0,
    ),
    rerankCandidatesExamined,
    finalResultCount: rankedSearch.results.length,
    fusedPoolLimit: rankedSearch.searchPolicy.fusedPoolLimit,
    rerankPoolLimit: rankedSearch.searchPolicy.rerankPoolLimit,
    finalResultLimit: rankedSearch.searchPolicy.finalResultLimit,
    rankFusionVersion: rankedSearch.searchPolicy.rankFusionVersion,
    reciprocalRankConstant: rankedSearch.searchPolicy.reciprocalRankConstant,
    fusionPolicy: {
      submittedQueryListWeight: queryPlan.fusionPolicy.submittedQueryListWeight,
      plannedQueryListWeight: queryPlan.fusionPolicy.plannedQueryListWeight,
      rankOneBonus: queryPlan.fusionPolicy.rankOneBonus,
      rankTwoOrThreeBonus: queryPlan.fusionPolicy.rankTwoOrThreeBonus,
    },
    rerankerPolicy: {
      version: queryPlan.rerankerProfile.policyVersion,
      activeBranchPrior: rankedSearch.searchPolicy.activeBranchPrior,
      fusedRankBlend: queryPlan.rerankerProfile.fusedRankBlend.map((band) => ({ ...band })),
    },
    rankingProviderPolicy: 'neutral-fused-order-v1',
    admissionProbeProviderPolicy: 'expected-source-promotion-v1',
  };
}

function createQueryPlannedContribution(
  controlKind: QueryPlannedRecallControlKind,
  normal: QueryPlannedRecallBaselineArmMeasurement,
  retrievalWorkMatched: QueryPlannedRecallBaselineArmMeasurement,
  queryPlanned: QueryPlannedRecallArmMeasurement,
): PrivateQueryPlannedRecallEvaluationResult['cases'][number]['contribution'] {
  const newCandidateAdmission =
    queryPlanned.candidateAdmissionVerified &&
    normal.outcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS &&
    retrievalWorkMatched.outcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS;
  const rankingOnlyPromotion =
    queryPlanned.outcome === QueryPlannedRecallBaselineOutcome.SUCCESS &&
    !newCandidateAdmission &&
    [normal.outcome, retrievalWorkMatched.outcome].includes(
      QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
    );
  const preservedExistingSuccess =
    controlKind === QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL &&
    queryPlanned.outcome === QueryPlannedRecallBaselineOutcome.SUCCESS;
  return {
    newCandidateAdmission,
    rankingOnlyPromotion,
    preservedExistingSuccess,
    noImprovement: !newCandidateAdmission && !rankingOnlyPromotion && !preservedExistingSuccess,
  };
}

/** Runs fixed plans and equal-work controls through public service searches with deterministic providers. */
export async function runPrivateQueryPlannedRecallEvaluation(
  options: RunPrivateQueryPlannedRecallEvaluationOptions,
): Promise<PrivateQueryPlannedRecallEvaluationResult> {
  if (
    options.plans.document.corpusId !== options.corpus.manifest.corpus.id ||
    options.plans.document.privateManifestSha256 !== options.corpus.manifestSha256
  ) {
    throw new Error(
      'Private query-planned recall evaluation invalid: plans and corpus are not bound together',
    );
  }
  const workDirectory = assertPrivateQueryPlannedEvaluationWorkDirectory(
    options.corpus,
    options.workDirectory,
  );
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true, mode: 0o700 });
  const normalCandidateLimits = options.corpus.manifest.policy.normalCandidateLimits;
  const indexConfig = createPrivateQueryPlannedEvaluationConfig(
    options,
    workDirectory,
    normalCandidateLimits,
  );
  const neutralReranker = createControlledEvaluationReranker();
  const indexService = createRecallConversationService(
    indexConfig,
    createDeterministicQueryPlannedEvaluationDependencies(options, neutralReranker.reranker),
  );
  const indexed = await indexService.index({ optimize: true });
  if (
    indexed.indexSummary.failedSessions.length > 0 ||
    indexed.indexSummary.scannedSessions !== options.corpus.snapshots.length
  ) {
    throw new Error(
      'Private query-planned recall deterministic index did not cover every snapshot',
    );
  }

  const plansByCaseId = new Map(
    options.plans.document.cases.map((plannedCase) => [plannedCase.caseId, plannedCase]),
  );
  const snapshotsById = new Map(
    options.corpus.snapshots.map((snapshot) => [snapshot.id, snapshot]),
  );
  const cases: PrivateQueryPlannedRecallEvaluationResult['cases'] = [];
  let executedSearchRequests = 0;
  for (const evaluationCase of options.corpus.manifest.cases) {
    const plannedCase = plansByCaseId.get(evaluationCase.id);
    if (!plannedCase) {
      throw new Error(
        `Private query-planned recall evaluation missing fixed plan for ${evaluationCase.id}`,
      );
    }
    const runHybridControl = async (
      candidateLimits: RecallSearchCandidateLimits,
    ): Promise<QueryPlannedRecallBaselineArmMeasurement> => {
      const config = createPrivateQueryPlannedEvaluationConfig(
        options,
        workDirectory,
        candidateLimits,
      );
      const service = createRecallConversationService(
        config,
        createDeterministicQueryPlannedEvaluationDependencies(options, neutralReranker.reranker),
      );
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
      return measurePrivateHybridControl(
        search,
        evaluationCase,
        snapshotsById,
        options.corpus.manifest.policy.finalResultLimit,
      );
    };
    const normal = await runHybridControl(normalCandidateLimits);
    const retrievalWorkMatched = await runHybridControl(
      evaluationCase.retrievalWorkMatchedCandidateLimits,
    );

    const plannedConfig = createPrivateQueryPlannedEvaluationConfig(
      options,
      workDirectory,
      normalCandidateLimits,
    );
    const rankedReranker = createControlledEvaluationReranker();
    const rankedService = createRecallConversationService(
      plannedConfig,
      createDeterministicQueryPlannedEvaluationDependencies(options, rankedReranker.reranker),
    );
    const searchOptions = {
      mode: 'query-planned' as const,
      scope: evaluationCase.scope,
      plan: plannedCase.queries,
      ...(evaluationCase.invocationDirectory
        ? { invocationDirectory: evaluationCase.invocationDirectory }
        : {}),
    };
    const rankedSearch = await rankedService.search(
      evaluationCase.query,
      options.corpus.manifest.policy.finalResultLimit,
      searchOptions,
    );
    executedSearchRequests += 1;

    const admissionProbeReranker = createControlledEvaluationReranker(
      evaluationCase.expectedSources.map(({ requiredText }) => requiredText),
    );
    const admissionProbeService = createRecallConversationService(
      plannedConfig,
      createDeterministicQueryPlannedEvaluationDependencies(
        options,
        admissionProbeReranker.reranker,
      ),
    );
    const admissionProbeSearch = await admissionProbeService.search(
      evaluationCase.query,
      options.corpus.manifest.policy.finalResultLimit,
      searchOptions,
    );
    executedSearchRequests += 1;
    const queryPlanned = measurePrivateQueryPlannedArm(
      rankedSearch,
      admissionProbeSearch,
      evaluationCase,
      snapshotsById,
      rankedReranker.readLastCandidateCount(),
    );
    cases.push({
      caseId: evaluationCase.id,
      category: evaluationCase.category,
      controlKind: evaluationCase.controlKind,
      normal,
      retrievalWorkMatched,
      queryPlanned,
      contribution: createQueryPlannedContribution(
        evaluationCase.controlKind,
        normal,
        retrievalWorkMatched,
        queryPlanned,
      ),
    });
  }

  return {
    version: 1,
    corpusId: options.corpus.manifest.corpus.id,
    privateManifestSha256: options.corpus.manifestSha256,
    planIdentity: createPublishableQueryPlannedRecallPlanIdentity(options.plans),
    providerIdentity: {
      embeddingPolicy: 'deterministic-token-hash-v1',
      embeddingDimensions: QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS,
      rankingRerankerPolicy: 'neutral-fused-order-v1',
      admissionProbeRerankerPolicy: 'expected-source-promotion-v1',
    },
    indexedSnapshotCount: options.corpus.snapshots.length,
    indexedDocumentCount: indexed.totalChunks,
    executedSearchRequests,
    cases,
    contributionCounts: {
      newCandidateAdmission: cases.filter(({ contribution }) => contribution.newCandidateAdmission)
        .length,
      rankingOnlyPromotion: cases.filter(({ contribution }) => contribution.rankingOnlyPromotion)
        .length,
      preservedExistingSuccess: cases.filter(
        ({ contribution }) => contribution.preservedExistingSuccess,
      ).length,
      noImprovement: cases.filter(({ contribution }) => contribution.noImprovement).length,
    },
  };
}

/** Fixed Git point used to bind one deterministic query-planned quality result. */
export interface QueryPlannedRecallEvaluationEnvironment {
  recordedAgainstCommit: string;
}

/** Privacy-safe, deterministic fixed-plan evidence accepted by the automated quality gate. */
export interface PublishableQueryPlannedRecallEvaluationEvidence {
  version: 1;
  recordedAgainstCommit: string;
  privateManifestSha256: string;
  controlsSha256: string;
  evaluation: PrivateQueryPlannedRecallEvaluationResult;
}

function countQueryPlannedRecallContributions(
  evaluation: PrivateQueryPlannedRecallEvaluationResult,
): PrivateQueryPlannedRecallEvaluationResult['contributionCounts'] {
  return {
    newCandidateAdmission: evaluation.cases.filter(
      ({ contribution }) => contribution.newCandidateAdmission,
    ).length,
    rankingOnlyPromotion: evaluation.cases.filter(
      ({ contribution }) => contribution.rankingOnlyPromotion,
    ).length,
    preservedExistingSuccess: evaluation.cases.filter(
      ({ contribution }) => contribution.preservedExistingSuccess,
    ).length,
    noImprovement: evaluation.cases.filter(({ contribution }) => contribution.noImprovement).length,
  };
}

function assertQueryPlannedRecallEvaluationGate(
  controls: PublishableQueryPlannedRecallControls,
  evaluation: PrivateQueryPlannedRecallEvaluationResult,
): void {
  if (
    controls.corpusId !== evaluation.corpusId ||
    controls.privateManifestSha256 !== evaluation.privateManifestSha256
  ) {
    throw new Error(
      'Query-planned recall evaluation gate failed: controls and evaluation must bind the same private manifest',
    );
  }
  if (evaluation.contributionCounts.newCandidateAdmission < 1) {
    throw new Error(
      'Query-planned recall evaluation gate requires at least one new candidate admission beyond both original-query controls',
    );
  }
  const measuredContributionCounts = countQueryPlannedRecallContributions(evaluation);
  if (
    JSON.stringify(measuredContributionCounts) !== JSON.stringify(evaluation.contributionCounts)
  ) {
    throw new Error(
      'Query-planned recall evaluation gate failed: aggregate contribution counts do not match case measurements',
    );
  }
  const controlsByCaseId = new Map(controls.cases.map((control) => [control.caseId, control]));
  const planIdentityByCaseId = new Map(
    evaluation.planIdentity.cases.map((plannedCase) => [plannedCase.caseId, plannedCase]),
  );
  if (
    evaluation.cases.length !== controls.cases.length ||
    evaluation.planIdentity.cases.length !== controls.cases.length
  ) {
    throw new Error(
      'Query-planned recall evaluation gate failed: controls, plans, and measurements must cover the same cases',
    );
  }
  for (const measurement of evaluation.cases) {
    const control = controlsByCaseId.get(measurement.caseId);
    const planIdentity = planIdentityByCaseId.get(measurement.caseId);
    if (
      !control ||
      !planIdentity ||
      control.category !== measurement.category ||
      control.controlKind !== measurement.controlKind
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: identity mismatch for ${measurement.caseId}`,
      );
    }
    const queryPlanned = measurement.queryPlanned;
    const plannedCandidateAllowance = queryPlanned.listWork.reduce(
      (total, list) => total + list.candidateLimit,
      0,
    );
    if (
      queryPlanned.listWork.length !== 3 + planIdentity.plannedQueries.length ||
      plannedCandidateAllowance !== control.retrievalWork.totalCandidateLimit ||
      queryPlanned.listWork.some(
        (list) => list.candidateLimit !== control.retrievalWork.candidatesPerList,
      )
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: retrieval work mismatch for ${measurement.caseId}`,
      );
    }
    if (
      queryPlanned.fusedPoolLimit !== 40 ||
      queryPlanned.rerankPoolLimit !== 40 ||
      queryPlanned.finalResultLimit !== controls.policy.finalResultLimit
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: bounded ranking policy mismatch for ${measurement.caseId}`,
      );
    }
    if (
      measurement.contribution.newCandidateAdmission &&
      (measurement.normal.outcome !== QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS ||
        measurement.retrievalWorkMatched.outcome !==
          QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS ||
        !queryPlanned.candidateAdmissionVerified)
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: unsupported candidate admission credit for ${measurement.caseId}`,
      );
    }
    if (
      measurement.contribution.rankingOnlyPromotion &&
      (measurement.contribution.newCandidateAdmission ||
        queryPlanned.outcome !== QueryPlannedRecallBaselineOutcome.SUCCESS)
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: invalid ranking-only promotion for ${measurement.caseId}`,
      );
    }
    if (queryPlanned.candidateAdmissionVerified && !queryPlanned.provenancePassed) {
      throw new Error(
        `Query-planned recall evaluation gate failed: source provenance regression for ${measurement.caseId}`,
      );
    }
    if (
      measurement.controlKind === QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL &&
      queryPlanned.outcome !== QueryPlannedRecallBaselineOutcome.SUCCESS
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: existing successful case regressed for ${measurement.caseId}`,
      );
    }
  }
}

/** Binds deterministic fixed-plan measurements to frozen controls and enforces the quality gate. */
export function createPublishableQueryPlannedRecallEvaluationEvidence(
  controls: PublishableQueryPlannedRecallControls,
  evaluation: PrivateQueryPlannedRecallEvaluationResult,
  environment: QueryPlannedRecallEvaluationEnvironment,
): PublishableQueryPlannedRecallEvaluationEvidence {
  if (!/^[a-f0-9]{40}$/u.test(environment.recordedAgainstCommit)) {
    throw new Error(
      'Query-planned recall evaluation evidence invalid: recorded commit must be a full Git SHA-1',
    );
  }
  assertQueryPlannedRecallEvaluationGate(controls, evaluation);
  return {
    version: 1,
    recordedAgainstCommit: environment.recordedAgainstCommit,
    privateManifestSha256: evaluation.privateManifestSha256,
    controlsSha256: createQueryPlannedRecallSha256(JSON.stringify(controls)),
    evaluation,
  };
}

function formatQueryPlannedRecallOutcome(outcome: QueryPlannedRecallBaselineOutcome): string {
  return outcome.replaceAll('_', ' ');
}

function formatQueryPlannedRecallContribution(
  contribution: PrivateQueryPlannedRecallEvaluationResult['cases'][number]['contribution'],
): string {
  if (contribution.newCandidateAdmission) {
    return 'new candidate admission';
  }
  if (contribution.rankingOnlyPromotion) {
    return 'ranking-only promotion';
  }
  if (contribution.preservedExistingSuccess) {
    return 'preserved existing success';
  }
  return 'no improvement';
}

/** Formats aggregate fixed-plan quality evidence without private query or source text. */
export function formatPublishableQueryPlannedRecallEvaluationReport(
  evidence: PublishableQueryPlannedRecallEvaluationEvidence,
): string {
  const { evaluation } = evidence;
  const firstCase = evaluation.cases[0];
  if (!firstCase) {
    throw new Error('Query-planned recall evaluation report requires at least one case');
  }
  const lines = [
    '# Query-Planned Recall: Deterministic Source-Admission Quality',
    '',
    'This report compares fixed private agent plans with normal hybrid and retrieval-work-matched original-query controls. It contains only opaque case identities, query hashes, policy values, counts, ranks, and checksums; private query and source text remain local.',
    '',
    '## Identity',
    '',
    `- Recorded against commit: \`${evidence.recordedAgainstCommit}\``,
    `- Private manifest SHA-256: \`${evidence.privateManifestSha256}\``,
    `- Fixed plan source: \`${evaluation.planIdentity.source}\``,
    `- Fixed plan SHA-256: \`${evaluation.planIdentity.planSha256}\``,
    `- Publishable controls canonical JSON SHA-256: \`${evidence.controlsSha256}\``,
    `- Embedding policy: \`${evaluation.providerIdentity.embeddingPolicy}\`, ${evaluation.providerIdentity.embeddingDimensions} dimensions`,
    `- Ranking reranker policy: \`${evaluation.providerIdentity.rankingRerankerPolicy}\``,
    `- Admission probe reranker policy: \`${evaluation.providerIdentity.admissionProbeRerankerPolicy}\``,
    `- Frozen cases: ${evaluation.cases.length} across ${evaluation.indexedSnapshotCount} snapshots and ${evaluation.indexedDocumentCount} indexed documents`,
    `- Executed searches: ${evaluation.executedSearchRequests} (normal hybrid, retrieval-work-matched original query, neutral planned ranking, and planned admission probe per case)`,
    '',
    '## Quality gate',
    '',
    `- New candidate admission beyond both controls: ${evaluation.contributionCounts.newCandidateAdmission}`,
    `- Ranking-only promotion of an already admitted source: ${evaluation.contributionCounts.rankingOnlyPromotion}`,
    `- Preserved existing success: ${evaluation.contributionCounts.preservedExistingSuccess}`,
    `- No improvement: ${evaluation.contributionCounts.noImprovement}`,
    '',
    'A planned search receives source-admission credit only when its admission probe finds the expected source and both original-query candidate unions miss. Promotion of a source admitted by either control is reported separately as ranking-only behavior. Existing-success preservation is an independent guard and can overlap a contribution class.',
    '',
    '## Ranking policy',
    '',
    `- Per-list work: ${firstCase.queryPlanned.listWork.map((list) => `${list.source} ${list.candidateLimit}`).join(', ')}`,
    `- Fused pool / rerank pool / final results: ${firstCase.queryPlanned.fusedPoolLimit} / ${firstCase.queryPlanned.rerankPoolLimit} / ${firstCase.queryPlanned.finalResultLimit}`,
    `- Fusion: RRF v${firstCase.queryPlanned.rankFusionVersion}, k=${firstCase.queryPlanned.reciprocalRankConstant}, submitted weight ${firstCase.queryPlanned.fusionPolicy.submittedQueryListWeight}, planned weight ${firstCase.queryPlanned.fusionPolicy.plannedQueryListWeight}, bonuses ${firstCase.queryPlanned.fusionPolicy.rankOneBonus} / ${firstCase.queryPlanned.fusionPolicy.rankTwoOrThreeBonus}`,
    `- QMD reranker policy: v${firstCase.queryPlanned.rerankerPolicy.version}, active-branch prior ${firstCase.queryPlanned.rerankerPolicy.activeBranchPrior}, blend bands ${firstCase.queryPlanned.rerankerPolicy.fusedRankBlend.map((band) => `${band.firstRank}-${band.lastRank ?? 'end'} ${band.retrievalWeight}/${band.rerankerWeight}`).join(', ')}`,
    '',
    '## Cases',
    '',
    '| Case | Category | Normal hybrid | Retrieval-work-matched original query | Query-planned | Contribution | Planned work (admitted / allowed) |',
    '| --- | --- | --- | --- | --- | --- | ---: |',
  ];
  for (const measurement of evaluation.cases) {
    const admitted = measurement.queryPlanned.totalCandidatesExamined;
    const allowed = measurement.queryPlanned.listWork.reduce(
      (total, list) => total + list.candidateLimit,
      0,
    );
    lines.push(
      `| ${measurement.caseId} | ${measurement.category.replaceAll('_', ' ')} | ${formatQueryPlannedRecallOutcome(measurement.normal.outcome)} | ${formatQueryPlannedRecallOutcome(measurement.retrievalWorkMatched.outcome)} | ${formatQueryPlannedRecallOutcome(measurement.queryPlanned.outcome)} | ${formatQueryPlannedRecallContribution(measurement.contribution)} | ${admitted} / ${allowed} |`,
    );
  }
  lines.push('', '## Fixed plan query identities', '');
  for (const plannedCase of evaluation.planIdentity.cases) {
    lines.push(
      `- ${plannedCase.caseId}: ${plannedCase.plannedQueries.map((query) => `${query.type} \`${query.querySha256}\``).join(', ')}`,
    );
  }
  lines.push('');
  return lines.join('\n');
}

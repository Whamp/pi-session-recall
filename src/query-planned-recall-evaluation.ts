import { createHash } from 'node:crypto';
import { lstat, mkdir, readFile, rm } from 'node:fs/promises';
import { basename, dirname, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { isDeepStrictEqual } from 'node:util';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { assertExactEvaluationCaseCoverage } from './assert-exact-evaluation-case-coverage.js';
import { createCanonicalIdentity } from './create-canonical-identity.js';
import {
  EmbeddedInferenceComputeBackend,
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallControlKind,
  RecallInferenceBackend,
  RecallManualMaintenanceTrigger,
} from './enums.js';
import type { QueryPlannedRecallCaseCategory } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-ranked-lists.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import type {
  RecallIdentifiedQueryPlanningProvider,
  RecallIdentifiedRerankingProvider,
} from './recall-inference-capabilities.js';
import {
  createPrivateRecallEvaluationConfig,
  isPathInsideRecallEvaluationArea,
} from './recall-evaluation-file-system.js';
import { classifyQueryPlannedRecallContribution } from './query-planned-recall-contribution.js';
import { DEFAULT_RECALL_CHUNK_POLICY } from './recall-index-manifest.js';
import { stagePrivateQueryPlannedRecallCorpus } from './query-planned-recall-baseline.js';
import type {
  LoadedPrivateQueryPlannedRecallCorpus,
  StagedPrivateQueryPlannedRecallCorpus,
  PublishableQueryPlannedRecallControls,
  QueryPlannedRecallBaselineArmMeasurement,
} from './query-planned-recall-baseline.js';
import {
  createRecallConversationService,
  type RecallCandidateAdmission,
  type RecallConversationConfig,
  type RecallConversationDependencies,
  type RecallConversationSearch,
  type RecallConversationSearchResult,
  type RecallPlannedRetrievalQuery,
  type RecallQueryPlanningCapabilityVerification,
  type RecallRerankingCapabilityVerification,
  type RecallSearchCandidateLimits,
} from './recall-conversation-service.js';
import type {
  RecallQueryPlanningModelProfile,
  RecallRerankingModelProfile,
} from './recall-model-profiles.js';

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

/** Dependencies allowed to vary without changing deterministic evaluation embeddings. */
export type PrivateQueryPlannedRecallEvaluationDependencies = Pick<
  RecallConversationDependencies,
  'loadTokenizer' | 'resolveProjectIdentity'
>;

interface PrivateQueryPlannedRecallEvaluationBaseOptions {
  corpus: LoadedPrivateQueryPlannedRecallCorpus;
  baseConfig: RecallConversationConfig;
  dependencies?: PrivateQueryPlannedRecallEvaluationDependencies;
}

/** Inputs for one deterministic fixed-plan evaluation isolated from production recall data. */
export interface RunPrivateQueryPlannedRecallEvaluationOptions extends PrivateQueryPlannedRecallEvaluationBaseOptions {
  plans: LoadedPrivateQueryPlannedRecallPlans;
  workDirectory: string;
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
  candidateAdmissionSourceRanks: Array<number | null>;
  candidateAdmissionVerified: boolean;
  sourceProvenance: QueryPlannedSourceProvenance[];
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
  rankingProviderPolicy: 'neutral-fused-order-v1' | 'live-profile-v1';
  candidateAdmissionBoundaryPolicy: 'fused-candidate-pool-v1';
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
    candidateAdmissionBoundaryPolicy: 'fused-candidate-pool-v1';
  };
  indexedSnapshotCount: number;
  indexedSnapshotSha256: string[];
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

/** Per-source provenance selected from ranked output first and candidate admission second. */
export interface QueryPlannedSourceProvenance {
  selectedFrom: 'ranked_result' | 'candidate_admission' | 'missing';
  passed: boolean;
}

/** Selects provenance independently for each expected source without mixing search arms. */
export function selectQueryPlannedSourceProvenance(
  rankedMatches: readonly (Pick<PrivateExpectedSourceMatch, 'provenancePassed'> | null)[],
  admissionMatches: readonly (Pick<PrivateExpectedSourceMatch, 'provenancePassed'> | null)[],
): QueryPlannedSourceProvenance[] {
  if (rankedMatches.length !== admissionMatches.length) {
    throw new Error(
      'Query-planned source provenance invalid: ranked and candidate-admission sources must align',
    );
  }
  return rankedMatches.map((rankedMatch, index) => {
    const admissionMatch = admissionMatches[index];
    return rankedMatch
      ? { selectedFrom: 'ranked_result', passed: rankedMatch.provenancePassed }
      : admissionMatch
        ? { selectedFrom: 'candidate_admission', passed: admissionMatch.provenancePassed }
        : { selectedFrom: 'missing', passed: false };
  });
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
  options: PrivateQueryPlannedRecallEvaluationBaseOptions,
  workDirectory: string,
  candidateLimits: RecallSearchCandidateLimits,
  stagedCorpus: StagedPrivateQueryPlannedRecallCorpus,
  additionalImmutableInputPaths: readonly string[] = [],
): RecallConversationConfig {
  return createPrivateRecallEvaluationConfig({
    baseConfig: options.baseConfig,
    evaluationRootDirectory: dirname(options.corpus.manifestPath),
    workDirectory,
    sessionsDirectory: stagedCorpus.snapshotDirectory,
    immutableInputPaths: [
      stagedCorpus.snapshotDirectory,
      options.corpus.manifestPath,
      ...additionalImmutableInputPaths,
    ],
    candidateLimits,
    embeddingIdentity: {
      model: 'deterministic-token-hash-v1',
      servedModelId: 'deterministic-token-hash-v1',
      artifact: 'none',
      quantization: 'none',
      pooling: 'token-hash',
      dimensions: QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS,
    },
  });
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
  options: PrivateQueryPlannedRecallEvaluationBaseOptions,
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

function createControlledEvaluationReranker(): ControlledEvaluationReranker {
  let lastCandidateCount = 0;
  return {
    reranker: {
      async rerankDocuments(rerankerQuery, documents) {
        void rerankerQuery;
        lastCandidateCount = documents.length;
        return documents.map(() => 0);
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

function matchesPrivateExpectedSource(
  candidate: RecallSearchResult,
  source: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number]['expectedSources'][number],
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

function verifiesPrivateExpectedSourceProvenance(
  candidate: RecallCandidateAdmission,
  source: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number]['expectedSources'][number],
): boolean {
  return (
    candidate.cwd === source.expectedSessionOrigin &&
    candidate.evidenceRelation === source.expectedEvidenceRelation &&
    (!source.expectedBranch ||
      (source.expectedBranch === 'active' && candidate.isOnActiveBranch) ||
      (source.expectedBranch === 'abandoned' && !candidate.isOnActiveBranch))
  );
}

function findPrivateExpectedSourceRankedMatch(
  search: RecallConversationSearch,
  source: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number]['expectedSources'][number],
  snapshotFileName: string,
): PrivateExpectedSourceMatch | null {
  const admissionById = new Map(
    search.candidateAdmission.map((candidate) => [candidate.id, candidate]),
  );
  for (const [index, result] of search.results.entries()) {
    const candidate = getQueryPlannedRecallGroupMembers(result).find((groupMember) =>
      matchesPrivateExpectedSource(groupMember, source, snapshotFileName),
    );
    if (candidate) {
      const admittedCandidate = admissionById.get(candidate.id);
      if (!admittedCandidate) {
        throw new Error(
          `Private query-planned recall evaluation ranked candidate ${candidate.id} was absent from candidate admission`,
        );
      }
      return {
        rank: index + 1,
        provenancePassed: verifiesPrivateExpectedSourceProvenance(admittedCandidate, source),
      };
    }
  }
  return null;
}

function findPrivateExpectedSourceAdmissionMatch(
  search: RecallConversationSearch,
  source: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number]['expectedSources'][number],
  snapshotFileName: string,
): PrivateExpectedSourceMatch | null {
  const index = search.candidateAdmission.findIndex((candidate) =>
    matchesPrivateExpectedSource(candidate, source, snapshotFileName),
  );
  const candidate = search.candidateAdmission[index];
  return index < 0 || !candidate
    ? null
    : {
        rank: index + 1,
        provenancePassed: verifiesPrivateExpectedSourceProvenance(candidate, source),
      };
}

function findPrivateExpectedSourceMatches(
  search: RecallConversationSearch,
  evaluationCase: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number],
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
  boundary: 'ranked_result' | 'candidate_admission',
): Array<PrivateExpectedSourceMatch | null> {
  return evaluationCase.expectedSources.map((source) => {
    const snapshot = snapshotsById.get(source.snapshotId);
    if (!snapshot) {
      throw new Error(
        `Private query-planned recall evaluation missing snapshot ${source.snapshotId}`,
      );
    }
    return boundary === 'ranked_result'
      ? findPrivateExpectedSourceRankedMatch(search, source, snapshot.fileName)
      : findPrivateExpectedSourceAdmissionMatch(search, source, snapshot.fileName);
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
  const rankedMatches = findPrivateExpectedSourceMatches(
    search,
    evaluationCase,
    snapshotsById,
    'ranked_result',
  );
  const admissionMatches = findPrivateExpectedSourceMatches(
    search,
    evaluationCase,
    snapshotsById,
    'candidate_admission',
  );
  const ranks = rankedMatches.map((match) => match?.rank ?? null);
  const outcome = classifyPrivateQueryPlannedOutcome(
    ranks.map((rank) => (rank !== null && rank <= finalResultLimit ? rank : null)),
    admissionMatches.every((match) => match !== null),
  );
  return {
    outcome,
    expectedSourceRanks: ranks,
    highestRelevantDistractorRank: null,
    provenancePassed: rankedMatches.every(
      (rankedMatch, index) => (rankedMatch ?? admissionMatches[index])?.provenancePassed === true,
    ),
    listLimits: { ...search.searchPolicy.candidateLimits },
    totalCandidatesExamined: search.candidateAdmission.reduce(
      (total, candidate) => total + candidate.rankedListEvidence.length,
      0,
    ),
    uniqueCandidatesAdmitted: search.candidateAdmission.length,
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
  evaluationCase: LoadedPrivateQueryPlannedRecallCorpus['manifest']['cases'][number],
  snapshotsById: ReadonlyMap<string, { fileName: string }>,
  rerankCandidatesExamined: number,
  rankingProviderPolicy: QueryPlannedRecallArmMeasurement['rankingProviderPolicy'] = 'neutral-fused-order-v1',
): QueryPlannedRecallArmMeasurement {
  const rankedMatches = findPrivateExpectedSourceMatches(
    rankedSearch,
    evaluationCase,
    snapshotsById,
    'ranked_result',
  );
  const admissionMatches = findPrivateExpectedSourceMatches(
    rankedSearch,
    evaluationCase,
    snapshotsById,
    'candidate_admission',
  );
  const expectedSourceRanks = rankedMatches.map((match) => match?.rank ?? null);
  const candidateAdmissionSourceRanks = admissionMatches.map((match) => match?.rank ?? null);
  const candidateAdmissionVerified = admissionMatches.every((match) => match !== null);
  const sourceProvenance = selectQueryPlannedSourceProvenance(rankedMatches, admissionMatches);
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
    candidateAdmissionSourceRanks,
    candidateAdmissionVerified,
    sourceProvenance,
    provenancePassed: sourceProvenance.every(({ passed }) => passed),
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
    rankingProviderPolicy,
    candidateAdmissionBoundaryPolicy: 'fused-candidate-pool-v1',
  };
}

function createQueryPlannedContribution(
  controlKind: QueryPlannedRecallControlKind,
  normal: QueryPlannedRecallBaselineArmMeasurement,
  retrievalWorkMatched: QueryPlannedRecallBaselineArmMeasurement,
  queryPlanned: QueryPlannedRecallArmMeasurement,
  planSource: 'planner' | 'fallback' = 'planner',
): PrivateQueryPlannedRecallEvaluationResult['cases'][number]['contribution'] {
  if (planSource === 'fallback') {
    return {
      newCandidateAdmission: false,
      rankingOnlyPromotion: false,
      preservedExistingSuccess: false,
      noImprovement: true,
    };
  }
  return classifyQueryPlannedRecallContribution({
    controlKind,
    normalOutcome: normal.outcome,
    retrievalWorkMatchedOutcome: retrievalWorkMatched.outcome,
    queryPlannedOutcome: queryPlanned.outcome,
    candidateAdmissionVerified: queryPlanned.candidateAdmissionVerified,
  });
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
  const normalCandidateLimits = options.corpus.manifest.policy.normalCandidateLimits;
  createPrivateQueryPlannedEvaluationConfig(
    options,
    workDirectory,
    normalCandidateLimits,
    { snapshotDirectory: options.corpus.snapshotDirectory, snapshots: [] },
    [options.plans.path],
  );
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true, mode: 0o700 });
  const stagedCorpus = await stagePrivateQueryPlannedRecallCorpus(options.corpus, workDirectory);
  const indexWorkDirectory = resolve(workDirectory, 'index');
  const immutableInputPaths = [options.plans.path];
  const indexConfig = createPrivateQueryPlannedEvaluationConfig(
    options,
    indexWorkDirectory,
    normalCandidateLimits,
    stagedCorpus,
    immutableInputPaths,
  );
  const neutralReranker = createControlledEvaluationReranker();
  const indexService = createRecallConversationService(
    indexConfig,
    createDeterministicQueryPlannedEvaluationDependencies(options, neutralReranker.reranker),
  );
  const indexed = await indexService.index({
    rebuild: true,
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
    optimize: true,
  });
  if (
    indexed.indexSummary.failedSessions.length > 0 ||
    indexed.indexSummary.scannedSessions !== stagedCorpus.snapshots.length
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
        indexWorkDirectory,
        candidateLimits,
        stagedCorpus,
        immutableInputPaths,
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
      indexWorkDirectory,
      normalCandidateLimits,
      stagedCorpus,
      immutableInputPaths,
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

    const queryPlanned = measurePrivateQueryPlannedArm(
      rankedSearch,
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
      candidateAdmissionBoundaryPolicy: 'fused-candidate-pool-v1',
    },
    indexedSnapshotCount: stagedCorpus.snapshots.length,
    indexedSnapshotSha256: stagedCorpus.snapshots.map(({ sha256 }) => sha256),
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

/** Backend and device class measured by one live query-planned profile run. */
export interface LiveQueryPlannedProfileRunIdentity {
  id: string;
  backend: RecallInferenceBackend;
  deviceClass: 'cpu' | 'accelerated';
  device: string;
  backendVersion?: string;
}

/** Fixed public reranker fixture used to reject score-semantic drift before evaluation. */
export interface LiveRerankerConformanceFixture {
  query: string;
  documents: readonly string[];
  expectedScores: readonly number[];
  maximumAbsoluteDifference: number;
}

/** Canonical non-corpus evaluation inputs that can change measurements or release gates. */
export interface LiveQueryPlannedEvaluationConfigurationIdentity {
  version: 1;
  effectiveConfigurationIdentity: string;
  rerankerConformanceFixtureIdentity: string;
}

/** Software revisions that bound one published live profile measurement. */
export interface LiveQueryPlannedSoftwareIdentity {
  repositoryCommit: string;
  backendVersion: string;
  nodeVersion: string;
  platform: string;
  architecture: string;
}

/** Complete canonical profile, adapter, evaluation, and software execution identity. */
export interface LiveQueryPlannedProfileIdentity {
  embeddingPolicy: 'deterministic-token-hash-v1';
  embeddingDimensions: number;
  evaluationConfiguration: LiveQueryPlannedEvaluationConfigurationIdentity;
  software: LiveQueryPlannedSoftwareIdentity;
  queryPlanning: {
    profileId: string;
    model: string;
    promptPolicy: string;
    grammarVersion: string;
    executionIdentity: RecallIdentifiedQueryPlanningProvider['executionIdentity'];
  };
  reranking: {
    profileId: string;
    model: string;
    scorePolicy: string;
    executionIdentity: RecallIdentifiedRerankingProvider['executionIdentity'];
  };
}

/** Creates the canonical effective evaluation and conformance-fixture identity. */
export function createLiveQueryPlannedEvaluationConfigurationIdentity(
  baseConfig: RecallConversationConfig,
  rerankerConformance: LiveRerankerConformanceFixture,
): LiveQueryPlannedEvaluationConfigurationIdentity {
  return {
    version: 1,
    effectiveConfigurationIdentity: createCanonicalIdentity(
      'live-query-planned-effective-evaluation-config-v1',
      {
        chunkPolicy: baseConfig.chunkPolicy ?? DEFAULT_RECALL_CHUNK_POLICY,
        embeddingBatchSize: baseConfig.embeddingBatchSize,
        embeddingDimensions: QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS,
        embeddingPolicy: 'deterministic-token-hash-v1',
        projectLineages: baseConfig.projectLineages,
      },
    ),
    rerankerConformanceFixtureIdentity: createCanonicalIdentity(
      'live-query-planned-reranker-conformance-fixture-v1',
      rerankerConformance,
    ),
  };
}

/** Creates the one canonical identity published by and checked against a live profile result. */
export function createLiveQueryPlannedProfileIdentity(options: {
  evaluationConfiguration: LiveQueryPlannedEvaluationConfigurationIdentity;
  software: LiveQueryPlannedSoftwareIdentity;
  queryPlanningProfile: RecallQueryPlanningModelProfile;
  queryPlanningExecutionIdentity: RecallIdentifiedQueryPlanningProvider['executionIdentity'];
  rerankingProfile: RecallRerankingModelProfile;
  rerankingExecutionIdentity: RecallIdentifiedRerankingProvider['executionIdentity'];
}): LiveQueryPlannedProfileIdentity {
  return {
    embeddingPolicy: 'deterministic-token-hash-v1',
    embeddingDimensions: QUERY_PLANNED_RECALL_EVALUATION_EMBEDDING_DIMENSIONS,
    evaluationConfiguration: { ...options.evaluationConfiguration },
    software: { ...options.software },
    queryPlanning: {
      profileId: options.queryPlanningProfile.profileId,
      model: options.queryPlanningProfile.model,
      promptPolicy: options.queryPlanningProfile.promptPolicy,
      grammarVersion: options.queryPlanningProfile.grammarVersion,
      executionIdentity: options.queryPlanningExecutionIdentity,
    },
    reranking: {
      profileId: options.rerankingProfile.profileId,
      model: options.rerankingProfile.model,
      scorePolicy: options.rerankingProfile.scorePolicy,
      executionIdentity: options.rerankingExecutionIdentity,
    },
  };
}

/** Live planner and reranker inputs for one privacy-safe private-corpus evaluation. */
export interface RunLiveQueryPlannedProfileEvaluationOptions extends PrivateQueryPlannedRecallEvaluationBaseOptions {
  workDirectory: string;
  profileRun: LiveQueryPlannedProfileRunIdentity;
  evaluationConfiguration: LiveQueryPlannedEvaluationConfigurationIdentity;
  software: LiveQueryPlannedSoftwareIdentity;
  queryPlanningProfile: RecallQueryPlanningModelProfile;
  queryPlanner: RecallIdentifiedQueryPlanningProvider;
  rerankingProfile: RecallRerankingModelProfile;
  reranker: RecallIdentifiedRerankingProvider;
  rerankerConformance: LiveRerankerConformanceFixture;
  reportProgress?(message: string): void;
}

/** Aggregate latency summary in milliseconds for complete live service searches. */
export interface LiveQueryPlannedSearchLatencySummary {
  minimum: number;
  median: number;
  maximum: number;
}

/** Immutable private corpus identity required to validate live profile evidence. */
export interface LiveQueryPlannedEvaluationCorpusIdentity {
  id: string;
  privateManifestSha256: string;
  snapshotSha256: readonly string[];
  cases: readonly {
    caseId: string;
    category: QueryPlannedRecallCaseCategory;
    controlKind: QueryPlannedRecallControlKind;
    expectedSourceCount: number;
  }[];
}

/** Creates the manifest-bound snapshot and case identity expected from every live profile. */
export function createLiveQueryPlannedEvaluationCorpusIdentity(
  corpus: LoadedPrivateQueryPlannedRecallCorpus,
): LiveQueryPlannedEvaluationCorpusIdentity {
  return {
    id: corpus.manifest.corpus.id,
    privateManifestSha256: corpus.manifestSha256,
    snapshotSha256: corpus.snapshots.map(({ sha256 }) => sha256),
    cases: corpus.manifest.cases.map((evaluationCase) => ({
      caseId: evaluationCase.id,
      category: evaluationCase.category,
      controlKind: evaluationCase.controlKind,
      expectedSourceCount: evaluationCase.expectedSources.length,
    })),
  };
}

/** Live planner and reranker quality over the checksum-fixed committed corpus. */
export interface CommittedCorpusQueryPlannedProfileEvidence {
  corpusId: string;
  specificationSha256: string;
  caseCount: number;
  qualityPassed: boolean;
  candidatePoolRecall: number;
  finalRecall: number;
  contextUsefulness: number;
  sourceOccurrencePreservation: number;
  sessionOriginVerification: number;
  evidenceRelationVerification: number;
  contributingEntryVerification: number;
  branchVerification: number;
  policyFailureCaseIds: readonly string[];
  queryLatencyMilliseconds: { median: number; p95: number };
  executedSearchRequests: number;
  plannerRequests: number;
  rerankerRequests: number;
}

/** Publishable profile-bound quality and latency without private query or source text. */
export interface LiveQueryPlannedProfileEvaluationResult {
  version: 1;
  profileRun: LiveQueryPlannedProfileRunIdentity;
  corpus: {
    id: string;
    privateManifestSha256: string;
    snapshotCount: number;
    snapshotSha256: string[];
    indexedDocumentCount: number;
    caseCount: number;
  };
  profileIdentity: LiveQueryPlannedProfileIdentity;
  capabilityConformance: {
    queryPlanning: RecallQueryPlanningCapabilityVerification;
    reranking: RecallRerankingCapabilityVerification;
  };
  latency: {
    coldPlanningMilliseconds: number;
    warmPlanningMilliseconds: number;
    warmPlanningSucceeded: boolean;
    coldRerankingMilliseconds: number;
    warmRerankingMilliseconds: number;
    totalSearchMilliseconds: LiveQueryPlannedSearchLatencySummary;
  };
  cases: Array<{
    caseId: string;
    category: QueryPlannedRecallCaseCategory;
    controlKind: QueryPlannedRecallControlKind;
    planSource: 'planner' | 'fallback';
    plannedQueries: Array<{ type: RecallPlannedRetrievalQuery['type']; querySha256: string }>;
    normal: QueryPlannedRecallBaselineArmMeasurement;
    retrievalWorkMatched: QueryPlannedRecallBaselineArmMeasurement;
    queryPlanned: QueryPlannedRecallArmMeasurement;
    planningMilliseconds: number;
    rerankingMilliseconds: number;
    totalSearchMilliseconds: number;
    contribution: PrivateQueryPlannedRecallEvaluationResult['cases'][number]['contribution'];
  }>;
  committedCorpus?: CommittedCorpusQueryPlannedProfileEvidence;
  quality: {
    newCandidateAdmissionCount: number;
    rankingOnlyPromotionCount: number;
    preservedExistingSuccessCount: number;
    noImprovementCount: number;
    plannerFallbackCount: number;
  };
}

function createLiveQueryPlannedProfileQuality(
  cases: readonly LiveQueryPlannedProfileEvaluationResult['cases'][number][],
): LiveQueryPlannedProfileEvaluationResult['quality'] {
  return {
    newCandidateAdmissionCount: cases.filter(
      ({ contribution }) => contribution.newCandidateAdmission,
    ).length,
    rankingOnlyPromotionCount: cases.filter(({ contribution }) => contribution.rankingOnlyPromotion)
      .length,
    preservedExistingSuccessCount: cases.filter(
      ({ contribution }) => contribution.preservedExistingSuccess,
    ).length,
    noImprovementCount: cases.filter(({ contribution }) => contribution.noImprovement).length,
    plannerFallbackCount: cases.filter(({ planSource }) => planSource === 'fallback').length,
  };
}

/** Rejects live profile evidence that is not corpus-bound, case-complete, and recomputable. */
export function assertLiveQueryPlannedProfileEvaluationResult(
  result: LiveQueryPlannedProfileEvaluationResult,
  expectedCorpus: LiveQueryPlannedEvaluationCorpusIdentity,
): void {
  if (
    result.corpus.id !== expectedCorpus.id ||
    result.corpus.privateManifestSha256 !== expectedCorpus.privateManifestSha256 ||
    result.corpus.snapshotCount !== expectedCorpus.snapshotSha256.length ||
    !isDeepStrictEqual(result.corpus.snapshotSha256, expectedCorpus.snapshotSha256) ||
    result.corpus.caseCount !== expectedCorpus.cases.length ||
    !Number.isInteger(result.corpus.indexedDocumentCount) ||
    result.corpus.indexedDocumentCount < 1
  ) {
    throw new Error('Live query-planned profile evidence invalid: corpus identity mismatch');
  }
  assertExactEvaluationCaseCoverage({
    controls: expectedCorpus.cases.map(({ caseId }) => caseId),
    measurements: result.cases.map(({ caseId }) => caseId),
  });
  const expectedCasesById = new Map(
    expectedCorpus.cases.map((expectedCase) => [expectedCase.caseId, expectedCase]),
  );
  for (const measuredCase of result.cases) {
    const expectedCase = expectedCasesById.get(measuredCase.caseId);
    if (
      !expectedCase ||
      measuredCase.category !== expectedCase.category ||
      measuredCase.controlKind !== expectedCase.controlKind
    ) {
      throw new Error(
        `Live query-planned profile evidence invalid: case identity mismatch for ${measuredCase.caseId}`,
      );
    }
    const queryPlanned = measuredCase.queryPlanned;
    if (
      queryPlanned.expectedSourceRanks.length !== expectedCase.expectedSourceCount ||
      queryPlanned.candidateAdmissionSourceRanks.length !== expectedCase.expectedSourceCount ||
      queryPlanned.sourceProvenance.length !== expectedCase.expectedSourceCount
    ) {
      throw new Error(
        `Live query-planned profile evidence invalid: source evidence length mismatch for ${measuredCase.caseId}`,
      );
    }
    const sourceRanks = [
      ...queryPlanned.expectedSourceRanks,
      ...queryPlanned.candidateAdmissionSourceRanks,
    ];
    if (sourceRanks.some((rank) => rank !== null && (!Number.isInteger(rank) || rank < 1))) {
      throw new Error(
        `Live query-planned profile evidence invalid: source ranks must be positive integers for ${measuredCase.caseId}`,
      );
    }
    const candidateAdmissionVerified = queryPlanned.candidateAdmissionSourceRanks.every(
      (rank) => rank !== null,
    );
    if (queryPlanned.candidateAdmissionVerified !== candidateAdmissionVerified) {
      throw new Error(
        `Live query-planned profile evidence invalid: candidate admission does not match source ranks for ${measuredCase.caseId}`,
      );
    }
    const outcome = classifyPrivateQueryPlannedOutcome(
      queryPlanned.expectedSourceRanks,
      candidateAdmissionVerified,
    );
    if (queryPlanned.outcome !== outcome) {
      throw new Error(
        `Live query-planned profile evidence invalid: outcome does not match measured admission and ranks for ${measuredCase.caseId}`,
      );
    }
    for (const [sourceIndex, provenance] of queryPlanned.sourceProvenance.entries()) {
      const selectedFrom =
        queryPlanned.expectedSourceRanks[sourceIndex] !== null
          ? 'ranked_result'
          : queryPlanned.candidateAdmissionSourceRanks[sourceIndex] !== null
            ? 'candidate_admission'
            : 'missing';
      if (provenance.selectedFrom !== selectedFrom) {
        throw new Error(
          `Live query-planned profile evidence invalid: provenance boundary does not match source ranks for ${measuredCase.caseId}`,
        );
      }
    }
    if (
      queryPlanned.provenancePassed !== queryPlanned.sourceProvenance.every(({ passed }) => passed)
    ) {
      throw new Error(
        `Live query-planned profile evidence invalid: provenance aggregate mismatch for ${measuredCase.caseId}`,
      );
    }
    const contribution = createQueryPlannedContribution(
      measuredCase.controlKind,
      measuredCase.normal,
      measuredCase.retrievalWorkMatched,
      queryPlanned,
      measuredCase.planSource,
    );
    if (!isDeepStrictEqual(measuredCase.contribution, contribution)) {
      throw new Error(
        `Live query-planned profile evidence invalid: contribution does not match recomputed case evidence for ${measuredCase.caseId}`,
      );
    }
    if (Object.values(measuredCase.contribution).filter(Boolean).length !== 1) {
      throw new Error(
        `Live query-planned profile evidence invalid: contribution must be exclusive for ${measuredCase.caseId}`,
      );
    }
  }
  if (!isDeepStrictEqual(result.quality, createLiveQueryPlannedProfileQuality(result.cases))) {
    throw new Error(
      'Live query-planned profile evidence invalid: quality does not match recomputed case evidence',
    );
  }
}

function createRetrievalWorkMatchedCandidateLimits(
  plannedQueryCount: number,
): RecallSearchCandidateLimits {
  const totalCandidateLimit = (3 + plannedQueryCount) * 20;
  const baseLimit = Math.floor(totalCandidateLimit / 3);
  const remainder = totalCandidateLimit % 3;
  return {
    dense: baseLimit + (remainder > 0 ? 1 : 0),
    lexical: baseLimit + (remainder > 1 ? 1 : 0),
    identifier: baseLimit,
  };
}

function summarizeLiveSearchLatencies(
  measurements: readonly number[],
): LiveQueryPlannedSearchLatencySummary {
  if (measurements.length === 0) {
    throw new Error('Live query-planned profile evaluation requires search latency measurements');
  }
  const ordered = [...measurements].sort((left, right) => left - right);
  const middleIndex = Math.floor(ordered.length / 2);
  const lowerMiddle = ordered[Math.max(middleIndex - 1, 0)] ?? 0;
  const upperMiddle = ordered[middleIndex] ?? lowerMiddle;
  return {
    minimum: ordered[0] ?? 0,
    median: ordered.length % 2 === 0 ? (lowerMiddle + upperMiddle) / 2 : upperMiddle,
    maximum: ordered[ordered.length - 1] ?? 0,
  };
}

/** Runs live planner/reranker conformance and public searches while publishing only aggregates. */
export async function runLiveQueryPlannedProfileEvaluation(
  options: RunLiveQueryPlannedProfileEvaluationOptions,
): Promise<LiveQueryPlannedProfileEvaluationResult> {
  const workDirectory = assertPrivateQueryPlannedEvaluationWorkDirectory(
    options.corpus,
    options.workDirectory,
  );
  const normalCandidateLimits = options.corpus.manifest.policy.normalCandidateLimits;
  createPrivateQueryPlannedEvaluationConfig(options, workDirectory, normalCandidateLimits, {
    snapshotDirectory: options.corpus.snapshotDirectory,
    snapshots: [],
  });
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true, mode: 0o700 });
  const stagedCorpus = await stagePrivateQueryPlannedRecallCorpus(options.corpus, workDirectory);
  const indexWorkDirectory = resolve(workDirectory, 'index');
  const normalConfig = createPrivateQueryPlannedEvaluationConfig(
    options,
    indexWorkDirectory,
    normalCandidateLimits,
    stagedCorpus,
  );
  const warnings: string[] = [];
  let planningMilliseconds = 0;
  let rerankingMilliseconds = 0;
  let lastRerankCandidateCount = 0;
  const timedQueryPlanner: RecallIdentifiedQueryPlanningProvider = {
    get executionIdentity() {
      return options.queryPlanner.executionIdentity;
    },
    async planRecallQuery(request, signal) {
      const startedAt = performance.now();
      try {
        return await options.queryPlanner.planRecallQuery(request, signal);
      } finally {
        planningMilliseconds += Math.max(performance.now() - startedAt, 0);
      }
    },
  };
  const timedReranker: RecallIdentifiedRerankingProvider = {
    get executionIdentity() {
      return options.reranker.executionIdentity;
    },
    async rerankDocuments(query, documents, signal) {
      lastRerankCandidateCount = documents.length;
      const startedAt = performance.now();
      try {
        return await options.reranker.rerankDocuments(query, documents, signal);
      } finally {
        rerankingMilliseconds += Math.max(performance.now() - startedAt, 0);
      }
    },
  };
  const liveDependencies: RecallConversationDependencies = {
    ...createDeterministicQueryPlannedEvaluationDependencies(options, timedReranker),
    rerankingProfile: options.rerankingProfile,
    reranker: timedReranker,
    rerankerExecutionIdentity: timedReranker.executionIdentity,
    queryPlanningProfile: options.queryPlanningProfile,
    queryPlanner: timedQueryPlanner,
    notifyWarning(warning) {
      warnings.push(warning);
    },
  };
  options.reportProgress?.(`Verifying live profile ${options.profileRun.id} capabilities`);
  const conformanceService = createRecallConversationService(normalConfig, liveDependencies);
  const queryPlanningConformance = await conformanceService.verifyQueryPlanningCapability();
  const rerankingConformance = await conformanceService.verifyRerankingCapability({
    query: options.rerankerConformance.query,
    documents: options.rerankerConformance.documents,
    expectedScores: options.rerankerConformance.expectedScores,
    maximumAbsoluteDifference: options.rerankerConformance.maximumAbsoluteDifference,
  });

  const warmPlanningStartedAt = performance.now();
  let warmPlanningSucceeded = true;
  try {
    await options.queryPlanner.planRecallQuery({
      query: options.queryPlanningProfile.conformanceCanary.query,
      recallIntent: options.queryPlanningProfile.conformanceCanary.recallIntent,
    });
  } catch {
    warmPlanningSucceeded = false;
  }
  const warmPlanningMilliseconds = Math.max(performance.now() - warmPlanningStartedAt, 0);
  const warmRerankingStartedAt = performance.now();
  await options.reranker.rerankDocuments(
    options.rerankerConformance.query,
    options.rerankerConformance.documents,
  );
  const warmRerankingMilliseconds = Math.max(performance.now() - warmRerankingStartedAt, 0);

  options.reportProgress?.(`Indexing live profile ${options.profileRun.id} private corpus`);
  const indexReranker = createControlledEvaluationReranker();
  const indexService = createRecallConversationService(
    normalConfig,
    createDeterministicQueryPlannedEvaluationDependencies(options, indexReranker.reranker),
  );
  const indexed = await indexService.index({
    rebuild: true,
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
    optimize: true,
  });
  if (
    indexed.indexSummary.failedSessions.length > 0 ||
    indexed.indexSummary.scannedSessions !== stagedCorpus.snapshots.length
  ) {
    throw new Error('Live query-planned profile index did not cover every private snapshot');
  }

  const snapshotsById = new Map(
    options.corpus.snapshots.map((snapshot) => [snapshot.id, snapshot]),
  );
  const cases: LiveQueryPlannedProfileEvaluationResult['cases'] = [];
  const totalSearchMeasurements: number[] = [];
  for (const [caseIndex, evaluationCase] of options.corpus.manifest.cases.entries()) {
    options.reportProgress?.(
      `Evaluating live profile ${options.profileRun.id} case ${caseIndex + 1}/${options.corpus.manifest.cases.length}`,
    );
    const normalService = createRecallConversationService(
      normalConfig,
      createDeterministicQueryPlannedEvaluationDependencies(options, indexReranker.reranker),
    );
    const normalSearch = await normalService.search(
      evaluationCase.query,
      normalCandidateLimits.dense +
        normalCandidateLimits.lexical +
        normalCandidateLimits.identifier,
      {
        mode: 'hybrid',
        scope: evaluationCase.scope,
        ...(evaluationCase.invocationDirectory
          ? { invocationDirectory: evaluationCase.invocationDirectory }
          : {}),
      },
    );
    const normal = measurePrivateHybridControl(
      normalSearch,
      evaluationCase,
      snapshotsById,
      options.corpus.manifest.policy.finalResultLimit,
    );

    const planningBeforeSearch = planningMilliseconds;
    const rerankingBeforeSearch = rerankingMilliseconds;
    const warningsBeforeSearch = warnings.length;
    const searchStartedAt = performance.now();
    const liveService = createRecallConversationService(normalConfig, liveDependencies);
    const liveSearch = await liveService.search(
      evaluationCase.query,
      options.corpus.manifest.policy.finalResultLimit,
      {
        mode: 'query-planned',
        scope: evaluationCase.scope,
        ...(evaluationCase.invocationDirectory
          ? { invocationDirectory: evaluationCase.invocationDirectory }
          : {}),
      },
    );
    const totalSearchMilliseconds = Math.max(performance.now() - searchStartedAt, 0);
    totalSearchMeasurements.push(totalSearchMilliseconds);
    const queryPlan = liveSearch.searchPolicy.queryPlan;
    if (!queryPlan || (queryPlan.source !== 'planner' && queryPlan.source !== 'fallback')) {
      throw new Error(
        'Live query-planned profile evaluation missing planner or fallback policy evidence',
      );
    }

    const matchedCandidateLimits = createRetrievalWorkMatchedCandidateLimits(
      queryPlan.plannedQueries.length,
    );
    const matchedConfig = createPrivateQueryPlannedEvaluationConfig(
      options,
      indexWorkDirectory,
      matchedCandidateLimits,
      stagedCorpus,
    );
    const matchedService = createRecallConversationService(
      matchedConfig,
      createDeterministicQueryPlannedEvaluationDependencies(options, indexReranker.reranker),
    );
    const matchedPoolLimit =
      matchedCandidateLimits.dense +
      matchedCandidateLimits.lexical +
      matchedCandidateLimits.identifier;
    const matchedSearch = await matchedService.search(evaluationCase.query, matchedPoolLimit, {
      mode: 'hybrid',
      scope: evaluationCase.scope,
      ...(evaluationCase.invocationDirectory
        ? { invocationDirectory: evaluationCase.invocationDirectory }
        : {}),
    });
    const retrievalWorkMatched = measurePrivateHybridControl(
      matchedSearch,
      evaluationCase,
      snapshotsById,
      options.corpus.manifest.policy.finalResultLimit,
    );

    const measuredQueryPlanned = measurePrivateQueryPlannedArm(
      liveSearch,
      evaluationCase,
      snapshotsById,
      lastRerankCandidateCount,
      'live-profile-v1',
    );
    const queryPlanned = measuredQueryPlanned;
    const contribution = createQueryPlannedContribution(
      evaluationCase.controlKind,
      normal,
      retrievalWorkMatched,
      queryPlanned,
      queryPlan.source,
    );
    cases.push({
      caseId: evaluationCase.id,
      category: evaluationCase.category,
      controlKind: evaluationCase.controlKind,
      planSource: queryPlan.source,
      plannedQueries: queryPlan.plannedQueries.map((plannedQuery) => ({
        type: plannedQuery.type,
        querySha256: createQueryPlannedRecallSha256(plannedQuery.query),
      })),
      normal,
      retrievalWorkMatched,
      queryPlanned,
      planningMilliseconds: Math.max(planningMilliseconds - planningBeforeSearch, 0),
      rerankingMilliseconds: Math.max(rerankingMilliseconds - rerankingBeforeSearch, 0),
      totalSearchMilliseconds,
      contribution,
    });
    if (queryPlan.source === 'fallback' && warnings.length === warningsBeforeSearch) {
      throw new Error(
        `Live query-planned profile fallback warning missing for ${evaluationCase.id}`,
      );
    }
  }

  return {
    version: 1,
    profileRun: { ...options.profileRun },
    corpus: {
      id: options.corpus.manifest.corpus.id,
      privateManifestSha256: options.corpus.manifestSha256,
      snapshotCount: stagedCorpus.snapshots.length,
      snapshotSha256: stagedCorpus.snapshots.map(({ sha256 }) => sha256),
      indexedDocumentCount: indexed.totalChunks,
      caseCount: cases.length,
    },
    profileIdentity: createLiveQueryPlannedProfileIdentity({
      evaluationConfiguration: options.evaluationConfiguration,
      software: options.software,
      queryPlanningProfile: options.queryPlanningProfile,
      queryPlanningExecutionIdentity: options.queryPlanner.executionIdentity,
      rerankingProfile: options.rerankingProfile,
      rerankingExecutionIdentity: options.reranker.executionIdentity,
    }),
    capabilityConformance: {
      queryPlanning: queryPlanningConformance,
      reranking: rerankingConformance,
    },
    latency: {
      coldPlanningMilliseconds: queryPlanningConformance.measurement.planningMilliseconds,
      warmPlanningMilliseconds,
      warmPlanningSucceeded,
      coldRerankingMilliseconds: rerankingConformance.measurement.rerankingMilliseconds,
      warmRerankingMilliseconds,
      totalSearchMilliseconds: summarizeLiveSearchLatencies(totalSearchMeasurements),
    },
    cases,
    quality: createLiveQueryPlannedProfileQuality(cases),
  };
}

/** Publishable committed-corpus profile result used by the live release gate. */
export interface CommittedCorpusLiveProfileEvidence {
  evidenceKind: 'accepted-hybrid-baseline' | 'live-profile-candidate';
  deviceClass: 'baseline' | 'cpu' | 'accelerated';
  profileId: string;
  evidenceSha256: string;
  qualityPassed: boolean;
  candidatePoolRecall: number;
  finalRecall: number;
}

/** Focused public-service and Pi contract results for required failure semantics. */
export interface LiveQueryPlannedFailureSemanticsEvidence {
  plannerFallbackPublicServicePassed: boolean;
  rerankerFailurePublicServicePassed: boolean;
  piToolContractPassed: boolean;
}

/** Inputs for the identity-bound explicit-mode release gate. */
export interface CreateLiveQueryPlannedProfileAcceptanceOptions {
  recordedAgainstCommit: string;
  defaultSearchMode: 'hybrid';
  committedCorpus: readonly CommittedCorpusLiveProfileEvidence[];
  expectedCorpus: LiveQueryPlannedEvaluationCorpusIdentity;
  expectedProfileRuns: readonly LiveQueryPlannedProfileRunIdentity[];
  profileRuns: readonly LiveQueryPlannedProfileEvaluationResult[];
  requiredSuccessfulBaselineControlCount: number;
  privacyAudit: { checkedValueCount: number; leakCount: 0 };
  failureSemantics: LiveQueryPlannedFailureSemanticsEvidence;
}

/** Publishable evidence approving the measured identities as an explicit post-hybrid fallback. */
export interface PublishableLiveQueryPlannedProfileAcceptance {
  version: 1;
  releaseDecision: 'approved-explicit-fallback';
  recordedAgainstCommit: string;
  approvedSearchMode: 'query-planned';
  defaultSearchMode: 'hybrid';
  committedCorpus: readonly CommittedCorpusLiveProfileEvidence[];
  profileRuns: readonly LiveQueryPlannedProfileEvaluationResult[];
  privacyAudit: { checkedValueCount: number; leakCount: 0 };
  failureSemantics: LiveQueryPlannedFailureSemanticsEvidence;
  aggregateQuality: {
    newCandidateAdmissionCount: number;
    rankingOnlyPromotionCount: number;
    preservedExistingSuccessCount: number;
    noImprovementCount: number;
    plannerFallbackCount: number;
  };
  fallbackCharacterization: {
    liveNewCandidateAdmissionObserved: boolean;
    existingSuccessPreservedAcrossProfiles: boolean;
    existingSuccessRegressionProfileRunIds: readonly string[];
  };
  limitations: readonly string[];
}

function assertLiveProfileExecutionIdentity(run: LiveQueryPlannedProfileEvaluationResult): void {
  const expectedPlannerAdapter =
    run.profileRun.backend === RecallInferenceBackend.EMBEDDED
      ? 'node-llama-cpp-qmd-query-planning-v1'
      : run.profileRun.backend === RecallInferenceBackend.LLAMA_CPP_HTTP
        ? 'llama-cpp-http-query-planning-v1'
        : null;
  const expectedRerankerAdapter =
    run.profileRun.backend === RecallInferenceBackend.EMBEDDED
      ? 'node-llama-cpp-qwen-reranking-logit-recovery-v1'
      : run.profileRun.backend === RecallInferenceBackend.LLAMA_CPP_HTTP
        ? 'llama-cpp-http-reranking-v1'
        : null;
  const planner = run.profileIdentity.queryPlanning;
  const reranker = run.profileIdentity.reranking;
  if (
    !expectedPlannerAdapter ||
    !expectedRerankerAdapter ||
    planner.profileId !== 'qmd-query-expansion-1.7b-q4-k-m-v1' ||
    planner.promptPolicy !== 'qmd-query-expansion-no-think-v1' ||
    planner.grammarVersion !== 'qmd-bounded-query-plan-v2' ||
    planner.executionIdentity.adapterId !== expectedPlannerAdapter ||
    planner.executionIdentity.adapterVersion !== '1' ||
    planner.executionIdentity.backend !== run.profileRun.backend ||
    reranker.profileId !== 'qwen3-reranker-0.6b-q8-0-v1' ||
    reranker.scorePolicy !== 'llama-cpp-qwen3-rank-probability-v1' ||
    reranker.executionIdentity.adapterId !== expectedRerankerAdapter ||
    reranker.executionIdentity.adapterVersion !== '1' ||
    reranker.executionIdentity.backend !== run.profileRun.backend
  ) {
    throw new Error(
      `Live query-planned profile acceptance identity mismatch for ${run.profileRun.id}`,
    );
  }
  const conformancePlannerIdentity = run.capabilityConformance.queryPlanning.executionIdentity;
  const conformanceRerankerIdentity = run.capabilityConformance.reranking.executionIdentity;
  if (
    !isDeepStrictEqual(conformancePlannerIdentity, planner.executionIdentity) ||
    !isDeepStrictEqual(conformanceRerankerIdentity, reranker.executionIdentity)
  ) {
    throw new Error(
      `Live query-planned profile acceptance conformance identity mismatch for ${run.profileRun.id}`,
    );
  }
  if (run.profileIdentity.software.backendVersion !== (run.profileRun.backendVersion ?? '')) {
    throw new Error(
      `Live query-planned profile acceptance software identity mismatch for ${run.profileRun.id}`,
    );
  }
  if (run.profileRun.backend === RecallInferenceBackend.EMBEDDED) {
    const plannerComputeBackend =
      'computeBackend' in planner.executionIdentity
        ? planner.executionIdentity.computeBackend
        : undefined;
    const rerankerComputeBackend =
      'computeBackend' in reranker.executionIdentity
        ? reranker.executionIdentity.computeBackend
        : undefined;
    const plannerDevicePolicy =
      'devicePolicy' in planner.executionIdentity
        ? planner.executionIdentity.devicePolicy
        : undefined;
    const rerankerDevicePolicy =
      'devicePolicy' in reranker.executionIdentity
        ? reranker.executionIdentity.devicePolicy
        : undefined;
    const plannerFallback =
      'fallbackFromComputeBackend' in planner.executionIdentity
        ? planner.executionIdentity.fallbackFromComputeBackend
        : undefined;
    const rerankerFallback =
      'fallbackFromComputeBackend' in reranker.executionIdentity
        ? reranker.executionIdentity.fallbackFromComputeBackend
        : undefined;
    const plannerDeviceNames =
      'deviceNames' in planner.executionIdentity
        ? planner.executionIdentity.deviceNames
        : undefined;
    const rerankerDeviceNames =
      'deviceNames' in reranker.executionIdentity
        ? reranker.executionIdentity.deviceNames
        : undefined;
    const plannerPhysicalDeviceIdentity =
      'physicalDeviceIdentity' in planner.executionIdentity
        ? planner.executionIdentity.physicalDeviceIdentity
        : undefined;
    const rerankerPhysicalDeviceIdentity =
      'physicalDeviceIdentity' in reranker.executionIdentity
        ? reranker.executionIdentity.physicalDeviceIdentity
        : undefined;
    const expectedComputeBackend =
      run.profileRun.device === 'cpu'
        ? EmbeddedInferenceComputeBackend.CPU
        : run.profileRun.device === 'metal'
          ? EmbeddedInferenceComputeBackend.METAL
          : run.profileRun.device === 'cuda'
            ? EmbeddedInferenceComputeBackend.CUDA
            : run.profileRun.device === 'vulkan'
              ? EmbeddedInferenceComputeBackend.VULKAN
              : null;
    const expectedDeviceClass =
      expectedComputeBackend === EmbeddedInferenceComputeBackend.CPU ? 'cpu' : 'accelerated';
    if (
      !expectedComputeBackend ||
      plannerComputeBackend !== expectedComputeBackend ||
      rerankerComputeBackend !== expectedComputeBackend ||
      plannerDevicePolicy !== run.profileRun.device ||
      rerankerDevicePolicy !== run.profileRun.device ||
      expectedDeviceClass !== run.profileRun.deviceClass ||
      plannerFallback !== null ||
      rerankerFallback !== null ||
      !Array.isArray(plannerDeviceNames) ||
      plannerDeviceNames.length === 0 ||
      plannerDeviceNames.some(
        (deviceName) => typeof deviceName !== 'string' || !deviceName.trim(),
      ) ||
      !isDeepStrictEqual(plannerDeviceNames, rerankerDeviceNames) ||
      !Array.isArray(plannerPhysicalDeviceIdentity) ||
      plannerPhysicalDeviceIdentity.length === 0 ||
      !isDeepStrictEqual(plannerPhysicalDeviceIdentity, rerankerPhysicalDeviceIdentity)
    ) {
      throw new Error(
        `Live query-planned profile acceptance resolved physical device identity mismatch for ${run.profileRun.id}`,
      );
    }
  }
  for (const measuredCase of run.cases) {
    const fusedDocumentLimit = measuredCase.queryPlanned.listWork.reduce(
      (total, list) => total + list.candidateLimit,
      0,
    );
    if (
      measuredCase.queryPlanned.rankingProviderPolicy !== 'live-profile-v1' ||
      measuredCase.queryPlanned.rankFusionVersion !== 2 ||
      measuredCase.queryPlanned.reciprocalRankConstant !== 60 ||
      measuredCase.queryPlanned.fusedPoolLimit !== fusedDocumentLimit ||
      measuredCase.queryPlanned.rerankPoolLimit !== 40 ||
      measuredCase.queryPlanned.finalResultLimit !== 5
    ) {
      throw new Error(
        `Live query-planned profile acceptance search policy mismatch for ${run.profileRun.id}/${measuredCase.caseId}`,
      );
    }
  }
}

function assertExactLiveProfileAcceptanceMatrix(
  expectedProfileRuns: readonly LiveQueryPlannedProfileRunIdentity[],
  profileRuns: readonly LiveQueryPlannedProfileEvaluationResult[],
): void {
  const expectedEmbeddedCpuCount = expectedProfileRuns.filter(
    (run) => run.backend === RecallInferenceBackend.EMBEDDED && run.deviceClass === 'cpu',
  ).length;
  const expectedEmbeddedAcceleratedCount = expectedProfileRuns.filter(
    (run) => run.backend === RecallInferenceBackend.EMBEDDED && run.deviceClass === 'accelerated',
  ).length;
  const expectedHttpCount = expectedProfileRuns.filter(
    (run) => run.backend === RecallInferenceBackend.LLAMA_CPP_HTTP,
  ).length;
  if (
    expectedProfileRuns.length !== 3 ||
    expectedEmbeddedCpuCount !== 1 ||
    expectedEmbeddedAcceleratedCount !== 1 ||
    expectedHttpCount !== 1
  ) {
    throw new Error(
      'Live query-planned profile acceptance failed: producer matrix must declare exactly embedded CPU, embedded accelerated, and one HTTP tuple',
    );
  }
  const unmatchedActualRuns = profileRuns.map(({ profileRun }) => profileRun);
  for (const expectedProfileRun of expectedProfileRuns) {
    const matchingIndex = unmatchedActualRuns.findIndex((actualProfileRun) =>
      isDeepStrictEqual(actualProfileRun, expectedProfileRun),
    );
    if (matchingIndex < 0) {
      throw new Error(
        `Live query-planned profile acceptance failed: missing or substituted profile tuple ${expectedProfileRun.id}`,
      );
    }
    unmatchedActualRuns.splice(matchingIndex, 1);
  }
  if (unmatchedActualRuns.length > 0) {
    throw new Error(
      'Live query-planned profile acceptance failed: duplicate or extra profile tuples are not allowed',
    );
  }
}

/** Approves an explicit post-hybrid fallback for the exact measured profile matrix. */
export function createPublishableLiveQueryPlannedProfileAcceptance(
  options: CreateLiveQueryPlannedProfileAcceptanceOptions,
): PublishableLiveQueryPlannedProfileAcceptance {
  if (!/^[a-f0-9]{40}$/u.test(options.recordedAgainstCommit)) {
    throw new Error(
      'Live query-planned profile acceptance invalid: recorded commit must be a full Git SHA-1',
    );
  }
  if (
    !Number.isInteger(options.requiredSuccessfulBaselineControlCount) ||
    options.requiredSuccessfulBaselineControlCount < 0
  ) {
    throw new Error(
      'Live query-planned profile acceptance invalid: successful baseline control count must be a nonnegative integer',
    );
  }
  if (
    !Number.isInteger(options.privacyAudit.checkedValueCount) ||
    options.privacyAudit.checkedValueCount < 1 ||
    options.privacyAudit.leakCount !== 0
  ) {
    throw new Error(
      'Live query-planned profile acceptance failed: privacy audit must check private values with zero leaks',
    );
  }
  if (
    !options.failureSemantics.plannerFallbackPublicServicePassed ||
    !options.failureSemantics.rerankerFailurePublicServicePassed ||
    !options.failureSemantics.piToolContractPassed
  ) {
    throw new Error(
      'Live query-planned profile acceptance failed: planner fallback, reranker failure, and Pi tool semantics must pass',
    );
  }
  const acceptedHybridBaseline = options.committedCorpus.some(
    (evidence) =>
      evidence.evidenceKind === 'accepted-hybrid-baseline' &&
      evidence.profileId.length > 0 &&
      evidence.qualityPassed &&
      evidence.candidatePoolRecall === 1 &&
      evidence.finalRecall === 1 &&
      /^[a-f0-9]{64}$/u.test(evidence.evidenceSha256),
  );
  const measuredEmbeddingGemmaDeviceClasses = new Set(
    options.committedCorpus
      .filter(
        (evidence) =>
          evidence.evidenceKind === 'live-profile-candidate' &&
          evidence.profileId === 'embeddinggemma-300m-q8-0-v1' &&
          /^[a-f0-9]{64}$/u.test(evidence.evidenceSha256),
      )
      .map(({ deviceClass }) => deviceClass),
  );
  if (
    !acceptedHybridBaseline ||
    !measuredEmbeddingGemmaDeviceClasses.has('cpu') ||
    !measuredEmbeddingGemmaDeviceClasses.has('accelerated')
  ) {
    throw new Error(
      'Live query-planned profile acceptance failed: accepted current hybrid baseline plus measured CPU and accelerated EmbeddingGemma candidates are required',
    );
  }
  assertExactLiveProfileAcceptanceMatrix(options.expectedProfileRuns, options.profileRuns);
  const runIds = options.profileRuns.map(({ profileRun }) => profileRun.id);
  if (new Set(runIds).size !== runIds.length) {
    throw new Error('Live query-planned profile acceptance failed: profile run IDs must be unique');
  }
  const privateManifestSha256 = options.profileRuns[0]?.corpus.privateManifestSha256;
  if (!privateManifestSha256) {
    throw new Error(
      'Live query-planned profile acceptance failed: at least one profile run is required',
    );
  }
  const existingSuccessRegressionProfileRunIds: string[] = [];
  const profileQualities: LiveQueryPlannedProfileEvaluationResult['quality'][] = [];
  const indexedDocumentCount = options.profileRuns[0]?.corpus.indexedDocumentCount;
  let committedCorpusId: string | undefined;
  let committedCorpusSpecificationSha256: string | undefined;
  for (const run of options.profileRuns) {
    assertLiveQueryPlannedProfileEvaluationResult(run, options.expectedCorpus);
    const quality = createLiveQueryPlannedProfileQuality(run.cases);
    profileQualities.push(quality);
    if (
      run.corpus.privateManifestSha256 !== privateManifestSha256 ||
      run.corpus.indexedDocumentCount !== indexedDocumentCount
    ) {
      throw new Error(
        'Live query-planned profile acceptance failed: every run must bind the same private manifest and indexed corpus',
      );
    }
    if (run.profileIdentity.software.repositoryCommit !== options.recordedAgainstCommit) {
      throw new Error(
        `Live query-planned profile acceptance failed: software commit mismatch for ${run.profileRun.id}`,
      );
    }
    assertLiveProfileExecutionIdentity(run);
    const committedCorpus = run.committedCorpus;
    if (
      !committedCorpus ||
      !committedCorpus.qualityPassed ||
      committedCorpus.caseCount < 1 ||
      committedCorpus.candidatePoolRecall !== 1 ||
      committedCorpus.finalRecall !== 1 ||
      committedCorpus.contextUsefulness !== 1 ||
      committedCorpus.sourceOccurrencePreservation !== 1 ||
      committedCorpus.sessionOriginVerification !== 1 ||
      committedCorpus.evidenceRelationVerification !== 1 ||
      committedCorpus.contributingEntryVerification !== 1 ||
      committedCorpus.branchVerification !== 1 ||
      committedCorpus.policyFailureCaseIds.length !== 0 ||
      committedCorpus.executedSearchRequests < committedCorpus.caseCount ||
      committedCorpus.plannerRequests < 1 ||
      committedCorpus.rerankerRequests < 1 ||
      !/^[a-f0-9]{64}$/u.test(committedCorpus.specificationSha256)
    ) {
      throw new Error(
        `Live query-planned profile acceptance failed: committed-corpus planner/reranker quality invalid for ${run.profileRun.id}`,
      );
    }
    committedCorpusId ??= committedCorpus.corpusId;
    committedCorpusSpecificationSha256 ??= committedCorpus.specificationSha256;
    if (
      committedCorpus.corpusId !== committedCorpusId ||
      committedCorpus.specificationSha256 !== committedCorpusSpecificationSha256
    ) {
      throw new Error(
        'Live query-planned profile acceptance failed: every profile must measure the same committed corpus',
      );
    }
    if (quality.preservedExistingSuccessCount < options.requiredSuccessfulBaselineControlCount) {
      existingSuccessRegressionProfileRunIds.push(run.profileRun.id);
    }
  }
  const aggregateQuality = {
    newCandidateAdmissionCount: profileQualities.reduce(
      (total, quality) => total + quality.newCandidateAdmissionCount,
      0,
    ),
    rankingOnlyPromotionCount: profileQualities.reduce(
      (total, quality) => total + quality.rankingOnlyPromotionCount,
      0,
    ),
    preservedExistingSuccessCount: profileQualities.reduce(
      (total, quality) => total + quality.preservedExistingSuccessCount,
      0,
    ),
    noImprovementCount: profileQualities.reduce(
      (total, quality) => total + quality.noImprovementCount,
      0,
    ),
    plannerFallbackCount: profileQualities.reduce(
      (total, quality) => total + quality.plannerFallbackCount,
      0,
    ),
  };
  const fallbackCharacterization = {
    liveNewCandidateAdmissionObserved: aggregateQuality.newCandidateAdmissionCount >= 1,
    existingSuccessPreservedAcrossProfiles: existingSuccessRegressionProfileRunIds.length === 0,
    existingSuccessRegressionProfileRunIds,
  };
  return {
    version: 1,
    releaseDecision: 'approved-explicit-fallback',
    recordedAgainstCommit: options.recordedAgainstCommit,
    approvedSearchMode: 'query-planned',
    defaultSearchMode: options.defaultSearchMode,
    committedCorpus: options.committedCorpus.map((evidence) => ({ ...evidence })),
    profileRuns: options.profileRuns.map((run) => ({ ...run })),
    privacyAudit: { ...options.privacyAudit },
    failureSemantics: { ...options.failureSemantics },
    aggregateQuality,
    fallbackCharacterization,
    limitations: [
      'Approval applies only as an explicit fallback after hybrid recall misses, with the accepted committed hybrid baseline and recorded planner, reranker, adapter, grammar, score, and search-policy identities.',
      'Live candidate admissions and preservation of queries already answered by hybrid are reported as fallback characterization, not release gates.',
      'Committed-corpus query-planned correctness must pass; its latency is recorded as explicit-fallback characterization rather than compared with the hybrid latency gate.',
      'EmbeddingGemma live candidates remain separate and are not approved when their committed-corpus quality gate fails.',
      'The committed corpus is synthetic-but-session-shaped; the private corpus is bounded and does not establish broad superiority.',
      'Private queries, plans, source text, session paths, and model artifacts remain outside Git.',
      'Hybrid remains the default search mode.',
    ],
  };
}

function formatQualityRate(rate: number): string {
  return `${(rate * 100).toFixed(1)}%`;
}

/** Formats profile-bound live acceptance evidence without private query or source text. */
export function formatPublishableLiveQueryPlannedProfileAcceptanceReport(
  evidence: PublishableLiveQueryPlannedProfileAcceptance,
): string {
  const firstRun = evidence.profileRuns[0];
  if (!firstRun) {
    throw new Error('Live query-planned profile acceptance report requires a profile run');
  }
  const decision =
    '**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.';
  const fusedDocumentLimits = [
    ...new Set(
      evidence.profileRuns.flatMap((run) =>
        run.cases.map((measuredCase) => measuredCase.queryPlanned.fusedPoolLimit),
      ),
    ),
  ].toSorted((left, right) => left - right);
  const firstQueryPlannedMeasurement = firstRun.cases[0]?.queryPlanned;
  const lines = [
    '# Query-Planned Recall: Live Profile Acceptance',
    '',
    decision,
    '',
    'This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.',
    '',
    '## Bounds and identity',
    '',
    `- Recorded against commit: \`${evidence.recordedAgainstCommit}\``,
    `- Private manifest SHA-256: \`${firstRun.corpus.privateManifestSha256}\``,
    `- Private corpus: ${firstRun.corpus.caseCount} cases, ${firstRun.corpus.snapshotCount} snapshots, ${firstRun.corpus.indexedDocumentCount} indexed documents`,
    `- Retrieval embedding policy for every live matrix row: \`${firstRun.profileIdentity.embeddingPolicy}\`, ${firstRun.profileIdentity.embeddingDimensions} dimensions`,
    `- Planner profile: \`${firstRun.profileIdentity.queryPlanning.profileId}\` / \`${firstRun.profileIdentity.queryPlanning.model}\``,
    `- Prompt / grammar: \`${firstRun.profileIdentity.queryPlanning.promptPolicy}\` / \`${firstRun.profileIdentity.queryPlanning.grammarVersion}\``,
    `- Reranker profile / score policy: \`${firstRun.profileIdentity.reranking.profileId}\` / \`${firstRun.profileIdentity.reranking.scorePolicy}\``,
    `- Search policy: RRF v${firstQueryPlannedMeasurement?.rankFusionVersion ?? 'unknown'}, k=${firstQueryPlannedMeasurement?.reciprocalRankConstant ?? 'unknown'}, fused-document limits ${fusedDocumentLimits.join(', ') || 'unknown'}; duplicate-group rerank/final limits ${firstQueryPlannedMeasurement?.rerankPoolLimit ?? 'unknown'}/${firstQueryPlannedMeasurement?.finalResultLimit ?? 'unknown'}`,
    '',
    '## Committed-corpus EmbeddingGemma evidence',
    '',
    '| Evidence | Device class | Profile | Candidate / final recall | Quality gate | Evidence SHA-256 |',
    '| --- | --- | --- | ---: | --- | --- |',
  ];
  for (const committed of evidence.committedCorpus) {
    lines.push(
      `| ${committed.evidenceKind} | ${committed.deviceClass} | \`${committed.profileId}\` | ${committed.candidatePoolRecall.toFixed(3)} / ${committed.finalRecall.toFixed(3)} | ${committed.qualityPassed ? 'pass' : 'fail'} | \`${committed.evidenceSha256}\` |`,
    );
  }
  lines.push(
    '',
    '## What the live matrix measures',
    '',
    `Every matrix row uses \`${firstRun.profileIdentity.embeddingPolicy}\` test embeddings. The run name, backend, and device describe only where the query planner and reranker execute.`,
    '',
    'The private-corpus total search time includes retrieval with fixed test embeddings plus live planning and live reranking. The same live planner/reranker profiles also run over the checksum-fixed committed corpus with deterministic embeddings; this does not measure end-to-end production inference with EmbeddingGemma embeddings.',
    '',
    '## Live planner and reranker matrix',
    '',
    '| Planner/reranker run | Planner/reranker backend | Planner/reranker device | Planner adapter | Reranker adapter | Cold planning | Warm planning | Cold reranking | Warm reranking | Query-planned search min / median / max |',
    '| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |',
  );
  for (const run of evidence.profileRuns) {
    lines.push(
      `| ${run.profileRun.id} | ${run.profileRun.backend}${run.profileRun.backendVersion ? ` (${run.profileRun.backendVersion})` : ''} | ${run.profileRun.deviceClass} / ${run.profileRun.device} | \`${run.profileIdentity.queryPlanning.executionIdentity.adapterId}\` | \`${run.profileIdentity.reranking.executionIdentity.adapterId}\` | ${run.latency.coldPlanningMilliseconds.toFixed(1)} ms | ${run.latency.warmPlanningMilliseconds.toFixed(1)} ms (${run.latency.warmPlanningSucceeded ? 'pass' : 'planner failure'}) | ${run.latency.coldRerankingMilliseconds.toFixed(1)} ms | ${run.latency.warmRerankingMilliseconds.toFixed(1)} ms | ${run.latency.totalSearchMilliseconds.minimum.toFixed(1)} / ${run.latency.totalSearchMilliseconds.median.toFixed(1)} / ${run.latency.totalSearchMilliseconds.maximum.toFixed(1)} ms |`,
    );
  }
  lines.push(
    '',
    '## Live planner/reranker quality on the committed corpus',
    '',
    'Correctness uses the frozen committed-corpus retrieval and provenance gates. Median and p95 latency are recorded as explicit-fallback characterization; the hybrid 2-second latency gate does not apply to this mode.',
    '',
    '| Profile run | Cases | Candidate pool recall | Final recall | Context | Source occurrences | Session origins | Evidence relations | Contributing entries | Branches | Planner / reranker calls | Median / p95 |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
  );
  for (const run of evidence.profileRuns) {
    const committed = run.committedCorpus;
    if (!committed) {
      throw new Error(
        `Live query-planned profile report invalid: committed-corpus evidence missing for ${run.profileRun.id}`,
      );
    }
    lines.push(
      `| ${run.profileRun.id} | ${committed.caseCount} | ${formatQualityRate(committed.candidatePoolRecall)} | ${formatQualityRate(committed.finalRecall)} | ${formatQualityRate(committed.contextUsefulness)} | ${formatQualityRate(committed.sourceOccurrencePreservation)} | ${formatQualityRate(committed.sessionOriginVerification)} | ${formatQualityRate(committed.evidenceRelationVerification)} | ${formatQualityRate(committed.contributingEntryVerification)} | ${formatQualityRate(committed.branchVerification)} | ${committed.plannerRequests} / ${committed.rerankerRequests} | ${committed.queryLatencyMilliseconds.median.toFixed(1)} / ${committed.queryLatencyMilliseconds.p95.toFixed(1)} ms |`,
    );
  }
  lines.push(
    '',
    '## Fallback characterization',
    '',
    `- New candidate admissions beyond normal and retrieval-work-matched original-query controls: ${evidence.aggregateQuality.newCandidateAdmissionCount}`,
    `- Ranking-only promotions: ${evidence.aggregateQuality.rankingOnlyPromotionCount}`,
    `- Preserved existing successes across profile runs: ${evidence.aggregateQuality.preservedExistingSuccessCount}`,
    `- No improvement: ${evidence.aggregateQuality.noImprovementCount}`,
    `- Planner fallbacks: ${evidence.aggregateQuality.plannerFallbackCount}`,
    '- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.',
    `- Live new-candidate admission observed: ${evidence.fallbackCharacterization.liveNewCandidateAdmissionObserved ? 'yes' : 'no'}`,
    `- Existing successes preserved across profiles: ${evidence.fallbackCharacterization.existingSuccessPreservedAcrossProfiles ? 'yes' : 'no'}`,
    `- Existing-success regression profiles: ${evidence.fallbackCharacterization.existingSuccessRegressionProfileRunIds.length > 0 ? evidence.fallbackCharacterization.existingSuccessRegressionProfileRunIds.join(', ') : 'none'}`,
    '',
    '## Candidate work by opaque case',
    '',
    '| Planner/reranker run / case | Plan source | Normal | Equal-work control | Query-planned | Candidate work (admitted / allowed) | Planning / reranking / total |',
    '| --- | --- | --- | --- | --- | ---: | ---: |',
  );
  for (const run of evidence.profileRuns) {
    for (const measuredCase of run.cases) {
      const admittedCandidates = measuredCase.queryPlanned.listWork.reduce(
        (total, list) => total + list.admittedCandidateCount,
        0,
      );
      const allowedCandidates = measuredCase.queryPlanned.listWork.reduce(
        (total, list) => total + list.candidateLimit,
        0,
      );
      lines.push(
        `| ${run.profileRun.id} / ${measuredCase.caseId} | ${measuredCase.planSource} | ${formatQueryPlannedRecallOutcome(measuredCase.normal.outcome)} | ${formatQueryPlannedRecallOutcome(measuredCase.retrievalWorkMatched.outcome)} | ${formatQueryPlannedRecallOutcome(measuredCase.queryPlanned.outcome)} | ${admittedCandidates} / ${allowedCandidates} | ${measuredCase.planningMilliseconds.toFixed(1)} / ${measuredCase.rerankingMilliseconds.toFixed(1)} / ${measuredCase.totalSearchMilliseconds.toFixed(1)} ms |`,
      );
    }
  }
  lines.push(
    '',
    '## Failure, tool, and privacy semantics',
    '',
    `- Privacy audit: ${evidence.privacyAudit.checkedValueCount} private values checked, ${evidence.privacyAudit.leakCount} leaks`,
    `- Planner fallback through public service: ${evidence.failureSemantics.plannerFallbackPublicServicePassed ? 'pass' : 'fail'}`,
    `- Reranker failure through public service: ${evidence.failureSemantics.rerankerFailurePublicServicePassed ? 'pass' : 'fail'}`,
    `- Pi tool contract and policy evidence: ${evidence.failureSemantics.piToolContractPassed ? 'pass' : 'fail'}`,
    '',
    '## Limitations',
    '',
  );
  for (const limitation of evidence.limitations) {
    lines.push(`- ${limitation}`);
  }
  return `${lines.join('\n')}\n`;
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
  if (
    evaluation.indexedSnapshotCount !== controls.snapshotSha256.length ||
    !isDeepStrictEqual(evaluation.indexedSnapshotSha256, controls.snapshotSha256)
  ) {
    throw new Error(
      'Query-planned recall evaluation gate failed: indexed snapshots must exactly match manifest hashes',
    );
  }
  assertExactEvaluationCaseCoverage({
    controls: controls.cases.map(({ caseId }) => caseId),
    planIdentities: evaluation.planIdentity.cases.map(({ caseId }) => caseId),
    evaluationMeasurements: evaluation.cases.map(({ caseId }) => caseId),
  });
  for (const measurement of evaluation.cases) {
    if (Object.values(measurement.contribution).filter(Boolean).length !== 1) {
      throw new Error(
        `Query-planned recall evaluation gate failed: contribution classification must be exclusive for ${measurement.caseId}`,
      );
    }
    const expectedContribution = createQueryPlannedContribution(
      measurement.controlKind,
      measurement.normal,
      measurement.retrievalWorkMatched,
      measurement.queryPlanned,
    );
    if (!isDeepStrictEqual(measurement.contribution, expectedContribution)) {
      throw new Error(
        `Query-planned recall evaluation gate failed: contribution classification mismatch for ${measurement.caseId}`,
      );
    }
  }
  const aggregateContributionCount = Object.values(evaluation.contributionCounts).reduce(
    (total, count) => total + count,
    0,
  );
  if (aggregateContributionCount !== evaluation.cases.length) {
    throw new Error(
      'Query-planned recall evaluation gate failed: contribution counts must equal measured case count',
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
    const sourceMeasurementCount = control.expectedSourceCount;
    if (
      queryPlanned.expectedSourceRanks.length !== sourceMeasurementCount ||
      queryPlanned.candidateAdmissionSourceRanks.length !== sourceMeasurementCount ||
      queryPlanned.sourceProvenance.length !== sourceMeasurementCount
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: source measurement count mismatch for ${measurement.caseId}`,
      );
    }
    const candidateAdmissionVerified = queryPlanned.candidateAdmissionSourceRanks.every(
      (rank) => rank !== null,
    );
    const expectedOutcome = classifyPrivateQueryPlannedOutcome(
      queryPlanned.expectedSourceRanks,
      candidateAdmissionVerified,
    );
    const sourceProvenancePassed = queryPlanned.sourceProvenance.every(({ passed }) => passed);
    const sourceSelectionPassed = queryPlanned.sourceProvenance.every(({ selectedFrom }, index) => {
      const rankedRank = queryPlanned.expectedSourceRanks[index] ?? null;
      const admissionRank = queryPlanned.candidateAdmissionSourceRanks[index] ?? null;
      if (rankedRank !== null) {
        return selectedFrom === 'ranked_result' && admissionRank !== null;
      }
      if (admissionRank !== null) {
        return selectedFrom === 'candidate_admission';
      }
      return selectedFrom === 'missing';
    });
    if (
      queryPlanned.candidateAdmissionVerified !== candidateAdmissionVerified ||
      queryPlanned.outcome !== expectedOutcome ||
      queryPlanned.provenancePassed !== sourceProvenancePassed ||
      !sourceSelectionPassed
    ) {
      throw new Error(
        `Query-planned recall evaluation gate failed: source evidence mismatch for ${measurement.caseId}`,
      );
    }
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
      queryPlanned.fusedPoolLimit !== plannedCandidateAllowance ||
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
  const fusedDocumentLimits = [
    ...new Set(evaluation.cases.map((measurement) => measurement.queryPlanned.fusedPoolLimit)),
  ].toSorted((left, right) => left - right);
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
    `- Candidate-admission boundary policy: \`${evaluation.providerIdentity.candidateAdmissionBoundaryPolicy}\``,
    `- Frozen cases: ${evaluation.cases.length} across ${evaluation.indexedSnapshotCount} snapshots and ${evaluation.indexedDocumentCount} indexed documents`,
    `- Executed searches: ${evaluation.executedSearchRequests} (normal hybrid, retrieval-work-matched original query, and planned ranking per case)`,
    '',
    '## Quality gate',
    '',
    `- New candidate admission beyond both controls: ${evaluation.contributionCounts.newCandidateAdmission}`,
    `- Ranking-only promotion of an already admitted source: ${evaluation.contributionCounts.rankingOnlyPromotion}`,
    `- Preserved existing success: ${evaluation.contributionCounts.preservedExistingSuccess}`,
    `- No improvement: ${evaluation.contributionCounts.noImprovement}`,
    '',
    'A planned search receives source-admission credit only when its fused candidate pool contains every expected source before duplicate grouping, rerank limits, and final truncation, while both original-query candidate unions miss. Promotion of a source admitted by either control is reported separately as ranking-only behavior. Existing-success preservation requires both the measured normal hybrid arm and query-planned arm to succeed.',
    '',
    '## Ranking policy',
    '',
    `- Per-list work: ${firstCase.queryPlanned.listWork.map((list) => `${list.source} ${list.candidateLimit}`).join(', ')}`,
    `- Fused-document limits before duplicate grouping: ${fusedDocumentLimits.join(', ')}`,
    `- Duplicate-group rerank limit / final results: ${firstCase.queryPlanned.rerankPoolLimit} / ${firstCase.queryPlanned.finalResultLimit}`,
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

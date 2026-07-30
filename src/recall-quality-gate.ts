import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-ranked-lists.js';
import { RECALL_ACTIVE_BRANCH_PRIOR } from './rank-recall-search-results.js';
import { RECALL_GENERATION_FORMAT_VERSION } from './recall-generation-manifest.js';
import { RECALL_GENERATION_STORE_FORMAT_VERSION } from './recall-generation-stores.js';
import { RECALL_GENERATION_VALIDATION_RECEIPT_VERSION } from './recall-generation-validation-receipt.js';
import { parseQualityCaseId } from './recall-quality-corpus.js';
import { INCREMENTAL_RECALL_ELIGIBILITY_POLICY_VERSION } from './reduce-recall-eligibility.js';
import {
  createLineageDigest,
  normalizeRecallProjectLineages,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  PROJECT_IDENTITY_POLICY_VERSION,
  PROJECT_LINEAGE_POLICY_VERSION,
  PROJECT_SCOPE_POLICY_VERSION,
} from './resolve-project-identity.js';
import { selectRecallQualityPolicy } from './select-recall-quality-policy.js';

/** Highest final-result count accepted from evidence and exposed by the recall tool. */
export const MAX_RECALL_FINAL_RESULT_COUNT = 10;

/** Packaged bounded-evaluation evidence consulted before production indexing. */
export const RECALL_QUALITY_RESULTS_PATH = fileURLToPath(
  new URL('../docs/evaluation/recall-quality-results.json', import.meta.url),
);

/** Exact measured chunk, candidate, and final-result policy from a clean passing gate. */
export interface RecallQualityApprovedPolicy {
  chunkPolicy: {
    id: string;
    maxTokens: number;
    overlapTokens: number;
  };
  candidateCount: number;
  finalCount: number;
}

/** Automated backfill decision reconstructed from committed bounded-evaluation evidence. */
export interface RecallQualityGateDecision {
  automatedGatePassed: boolean;
  selectedPolicy: RecallQualityApprovedPolicy | null;
  blockers: string[];
}

function areRecallQualityPoliciesEqual(
  left: RecallQualityApprovedPolicy,
  right: RecallQualityApprovedPolicy,
): boolean {
  return (
    left.chunkPolicy.id === right.chunkPolicy.id &&
    left.chunkPolicy.maxTokens === right.chunkPolicy.maxTokens &&
    left.chunkPolicy.overlapTokens === right.chunkPolicy.overlapTokens &&
    left.candidateCount === right.candidateCount &&
    left.finalCount === right.finalCount
  );
}

const CHUNK_POLICY_SCHEMA = Type.Object({
  id: Type.String({ minLength: 1 }),
  maxTokens: Type.Integer({ minimum: 1, maximum: 1_024 }),
  overlapTokens: Type.Integer({ minimum: 0, maximum: 128 }),
});
const RATE_SCHEMA = Type.Number({ minimum: 0, maximum: 1 });
const NONNEGATIVE_NUMBER_SCHEMA = Type.Number({ minimum: 0 });
const NONNEGATIVE_INTEGER_SCHEMA = Type.Integer({ minimum: 0 });
const POSITIVE_INTEGER_SCHEMA = Type.Integer({ minimum: 1 });
const LATENCY_SCHEMA = Type.Object({
  median: NONNEGATIVE_NUMBER_SCHEMA,
  p95: NONNEGATIVE_NUMBER_SCHEMA,
});
const CASE_FINAL_SCHEMA = Type.Object({
  finalCount: POSITIVE_INTEGER_SCHEMA,
  finalRecalled: Type.Boolean(),
  contextUseful: Type.Boolean(),
  sourceOccurrencesPreserved: Type.Boolean(),
  preservedSourceOccurrences: NONNEGATIVE_INTEGER_SCHEMA,
  sessionOriginsVerified: Type.Boolean(),
  evidenceRelationsVerified: Type.Boolean(),
  contributingEntriesVerified: Type.Boolean(),
  branchesVerified: Type.Boolean(),
  finalDuplicateSlots: NONNEGATIVE_INTEGER_SCHEMA,
  finalResultSlots: NONNEGATIVE_INTEGER_SCHEMA,
});
const CHANNEL_LIMIT_SCHEMA = Type.Object({
  channel: Type.Union([Type.Literal('dense'), Type.Literal('lexical'), Type.Literal('identifier')]),
  projectSourceAdmitted: Type.Boolean(),
  globalSourceDisplaced: Type.Boolean(),
  pollutingCandidateCount: NONNEGATIVE_INTEGER_SCHEMA,
  passed: Type.Boolean(),
});
const CASE_MEASUREMENT_SCHEMA = Type.Object({
  caseId: Type.String({ minLength: 1 }),
  category: Type.Union([
    Type.Literal('semantic_paraphrase'),
    Type.Literal('exact_identifier'),
    Type.Literal('tool_evidence'),
    Type.Literal('context_dependent_reply'),
    Type.Literal('branch'),
    Type.Literal('summary'),
    Type.Literal('duplicate_content'),
    Type.Literal('project_scope'),
  ]),
  scope: Type.Enum(RecallSearchScope),
  searchScopeVerified: Type.Boolean(),
  invocationProjectIdentityVerified: Type.Boolean(),
  excludedSessionFilesAbsent: Type.Boolean(),
  preLimitChannelsVerified: Type.Boolean(),
  preLimitChannelMeasurements: Type.Array(CHANNEL_LIMIT_SCHEMA),
  candidatePoolRecalled: Type.Boolean(),
  rawCandidateCount: NONNEGATIVE_INTEGER_SCHEMA,
  groupedCandidateCount: NONNEGATIVE_INTEGER_SCHEMA,
  candidatePoolDuplicateSlots: NONNEGATIVE_INTEGER_SCHEMA,
  queryLatencyMilliseconds: NONNEGATIVE_NUMBER_SCHEMA,
  finalCounts: Type.Array(CASE_FINAL_SCHEMA, { minItems: 1 }),
});
const FINAL_MEASUREMENT_SCHEMA = Type.Object({
  finalCount: POSITIVE_INTEGER_SCHEMA,
  finalRecall: RATE_SCHEMA,
  contextUsefulness: RATE_SCHEMA,
  sourceOccurrencePreservation: RATE_SCHEMA,
  sessionOriginVerification: RATE_SCHEMA,
  evidenceRelationVerification: RATE_SCHEMA,
  contributingEntryVerification: RATE_SCHEMA,
  branchVerification: RATE_SCHEMA,
  finalDuplicateRate: RATE_SCHEMA,
  missedCaseIds: Type.Array(Type.String()),
  contextFailureCaseIds: Type.Array(Type.String()),
  sourceOccurrenceFailureCaseIds: Type.Array(Type.String()),
  sessionOriginFailureCaseIds: Type.Array(Type.String()),
  evidenceRelationFailureCaseIds: Type.Array(Type.String()),
  contributingEntryFailureCaseIds: Type.Array(Type.String()),
  branchFailureCaseIds: Type.Array(Type.String()),
  finalDuplicateSlots: NONNEGATIVE_INTEGER_SCHEMA,
  finalResultSlots: NONNEGATIVE_INTEGER_SCHEMA,
});
const MEASUREMENT_SCHEMA = Type.Object({
  caseCount: POSITIVE_INTEGER_SCHEMA,
  candidatePoolRecall: RATE_SCHEMA,
  candidatePoolDuplicateRate: RATE_SCHEMA,
  queryLatencyMilliseconds: LATENCY_SCHEMA,
  queryLatencyByScope: Type.Object({
    project: Type.Union([Type.Null(), LATENCY_SCHEMA]),
    global: Type.Union([Type.Null(), LATENCY_SCHEMA]),
  }),
  policyFailureCaseIds: Type.Array(Type.String()),
  missedCandidatePoolCaseIds: Type.Array(Type.String()),
  caseMeasurements: Type.Array(CASE_MEASUREMENT_SCHEMA, { minItems: 1 }),
  finalCounts: Type.Array(FINAL_MEASUREMENT_SCHEMA, { minItems: 1 }),
});
const CONFIGURATION_SCHEMA = Type.Object({
  chunkPolicy: CHUNK_POLICY_SCHEMA,
  candidateCount: Type.Integer({ minimum: 1, maximum: 200 }),
  totalChunks: POSITIVE_INTEGER_SCHEMA,
  indexLatencyMilliseconds: NONNEGATIVE_NUMBER_SCHEMA,
  measurement: MEASUREMENT_SCHEMA,
});
const SELECTED_POLICY_SCHEMA = Type.Object({
  chunkPolicy: CHUNK_POLICY_SCHEMA,
  candidateCount: Type.Integer({ minimum: 1, maximum: 200 }),
  finalCount: Type.Integer({ minimum: 1, maximum: MAX_RECALL_FINAL_RESULT_COUNT }),
  gatePassed: Type.Boolean(),
});
const EVALUATION_IDENTITY_SCHEMA = Type.Object({
  defaultScope: Type.Enum(RecallSearchScope),
  projectScopePolicyVersion: POSITIVE_INTEGER_SCHEMA,
  projectIdentityPolicyVersion: POSITIVE_INTEGER_SCHEMA,
  projectIdentityMetadataSchemaVersion: POSITIVE_INTEGER_SCHEMA,
  lineagePolicyVersion: POSITIVE_INTEGER_SCHEMA,
  lineageDigest: Type.String({ pattern: '^[a-f0-9]{64}$' }),
  rankingMode: Type.String(),
  rankFusionVersion: POSITIVE_INTEGER_SCHEMA,
  reciprocalRankConstant: Type.Number({ exclusiveMinimum: 0 }),
  activeBranchPrior: NONNEGATIVE_NUMBER_SCHEMA,
  candidateLimits: Type.Object({
    dense: POSITIVE_INTEGER_SCHEMA,
    lexical: POSITIVE_INTEGER_SCHEMA,
    identifier: POSITIVE_INTEGER_SCHEMA,
  }),
  fusedPoolLimit: POSITIVE_INTEGER_SCHEMA,
  rerankPoolLimit: POSITIVE_INTEGER_SCHEMA,
  finalResultCount: Type.Integer({ minimum: 1, maximum: MAX_RECALL_FINAL_RESULT_COUNT }),
});
const INDEX_SUMMARY_SCHEMA = Type.Object({
  scannedSessions: NONNEGATIVE_INTEGER_SCHEMA,
  indexedSessions: NONNEGATIVE_INTEGER_SCHEMA,
  removedSessions: NONNEGATIVE_INTEGER_SCHEMA,
  cacheHits: NONNEGATIVE_INTEGER_SCHEMA,
  newlyEmbeddedChunks: NONNEGATIVE_INTEGER_SCHEMA,
  embeddingRequestCount: NONNEGATIVE_INTEGER_SCHEMA,
  deletedChunks: NONNEGATIVE_INTEGER_SCHEMA,
  failedSessions: Type.Array(Type.Unknown()),
});
const RECALL_QUALITY_GATE_EVIDENCE_SCHEMA = Type.Object({
  version: POSITIVE_INTEGER_SCHEMA,
  environment: Type.Object({ gitDirty: Type.Boolean() }),
  specification: Type.Optional(
    Type.Object({
      version: POSITIVE_INTEGER_SCHEMA,
      corpus: Type.Object({ sessionFiles: Type.Array(Type.Unknown(), { minItems: 1 }) }),
      projectIdentityFixtures: Type.Array(
        Type.Object({ identitySource: Type.Enum(RecallProjectIdentitySource) }),
      ),
      projectLineages: Type.Record(
        Type.String({ minLength: 1 }),
        Type.Array(Type.String({ minLength: 1 })),
      ),
      bounds: Type.Object({
        maximumSessionFiles: POSITIVE_INTEGER_SCHEMA,
        maximumEvaluationCases: POSITIVE_INTEGER_SCHEMA,
        maximumChunkPolicies: POSITIVE_INTEGER_SCHEMA,
        maximumCandidateCounts: POSITIVE_INTEGER_SCHEMA,
        maximumFinalCounts: POSITIVE_INTEGER_SCHEMA,
        maximumSearchRequests: POSITIVE_INTEGER_SCHEMA,
        maximumChunkEmbeddingRequests: POSITIVE_INTEGER_SCHEMA,
      }),
      chunkPolicies: Type.Array(CHUNK_POLICY_SCHEMA, { minItems: 1 }),
      candidateCounts: Type.Array(POSITIVE_INTEGER_SCHEMA, { minItems: 1 }),
      finalCounts: Type.Array(POSITIVE_INTEGER_SCHEMA, { minItems: 1 }),
      warmupQueriesPerCombination: NONNEGATIVE_INTEGER_SCHEMA,
      qualityGate: Type.Object({
        minimumCandidatePoolRecall: RATE_SCHEMA,
        minimumFinalRecall: RATE_SCHEMA,
        minimumContextUsefulness: RATE_SCHEMA,
        minimumSourceOccurrencePreservation: RATE_SCHEMA,
        maximumFinalDuplicateRate: RATE_SCHEMA,
        maximumQueryP95Milliseconds: Type.Number({ exclusiveMinimum: 0 }),
      }),
      cases: Type.Array(
        Type.Object({
          id: Type.String({ minLength: 1 }),
          scope: Type.Enum(RecallSearchScope),
          preLimitChannelProof: Type.Optional(Type.Unknown()),
        }),
        { minItems: 1 },
      ),
    }),
  ),
  result: Type.Object({
    version: POSITIVE_INTEGER_SCHEMA,
    storageIdentity: Type.Optional(
      Type.Object({
        generationFormatVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
        generationStoreFormatVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
        validationReceiptVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
        incrementalEligibilityPolicyVersion: NONNEGATIVE_INTEGER_SCHEMA,
        conversationSchemaVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
        zvecSchemaVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
        indexManifestVersion: Type.Optional(POSITIVE_INTEGER_SCHEMA),
      }),
    ),
    evaluationIdentity: Type.Optional(EVALUATION_IDENTITY_SCHEMA),
    indexRuns: Type.Optional(
      Type.Array(
        Type.Object({
          chunkPolicy: CHUNK_POLICY_SCHEMA,
          generationId: Type.Optional(Type.String({ pattern: '^[A-Za-z0-9_-]+$' })),
          manifestFingerprint: Type.Optional(Type.String({ pattern: '^[a-f0-9]{64}$' })),
          startingSnapshotFingerprint: Type.Optional(Type.String({ pattern: '^[a-f0-9]{64}$' })),
          storeCounts: Type.Optional(
            Type.Object({
              lexicalSource: POSITIVE_INTEGER_SCHEMA,
              dense: NONNEGATIVE_INTEGER_SCHEMA,
              sessionProjection: POSITIVE_INTEGER_SCHEMA,
            }),
          ),
          totalChunks: POSITIVE_INTEGER_SCHEMA,
          indexLatencyMilliseconds: NONNEGATIVE_NUMBER_SCHEMA,
          indexSummary: INDEX_SUMMARY_SCHEMA,
        }),
      ),
    ),
    configurations: Type.Optional(Type.Array(CONFIGURATION_SCHEMA, { minItems: 1 })),
    selection: Type.Object({
      passed: Type.Boolean(),
      selected: Type.Union([Type.Null(), SELECTED_POLICY_SCHEMA]),
      blockers: Type.Array(Type.String()),
      combinations: Type.Optional(Type.Array(SELECTED_POLICY_SCHEMA, { minItems: 1 })),
    }),
    boundedWork: Type.Optional(
      Type.Object({
        sessionFiles: POSITIVE_INTEGER_SCHEMA,
        evaluationCases: POSITIVE_INTEGER_SCHEMA,
        indexRuns: POSITIVE_INTEGER_SCHEMA,
        executedSearchRequests: POSITIVE_INTEGER_SCHEMA,
        rerankerRequests: NONNEGATIVE_INTEGER_SCHEMA,
        chunkEmbeddingRequests: NONNEGATIVE_INTEGER_SCHEMA,
        maximumCandidatesPerSearch: POSITIVE_INTEGER_SCHEMA,
        repositoryIdentityResolutions: NONNEGATIVE_INTEGER_SCHEMA,
      }),
    ),
  }),
});

/** Reads committed quality evidence and returns a policy only for a consistent clean pass. */
export async function readRecallQualityGateDecision(
  resultsPath: string,
): Promise<RecallQualityGateDecision> {
  let evidence: ReturnType<typeof Value.Parse<typeof RECALL_QUALITY_GATE_EVIDENCE_SCHEMA>>;
  try {
    const parsed: unknown = JSON.parse(await readFile(resultsPath, 'utf8'));
    evidence = Value.Parse(RECALL_QUALITY_GATE_EVIDENCE_SCHEMA, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall quality gate evidence invalid at ${resultsPath}: ${message}`, {
      cause: error,
    });
  }

  const { selection } = evidence.result;
  if (selection.passed !== (selection.selected !== null && selection.selected.gatePassed)) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: pass and selected-policy decisions disagree`,
    );
  }
  if (evidence.version !== 2 || evidence.result.version !== 6) {
    const staleReason =
      evidence.result.version < 5
        ? 'predates project-scoped measurement'
        : 'predates target-generation certification';
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        `Recall quality evidence version ${evidence.result.version} ${staleReason}; rerun npm run evaluate:recall`,
      ],
    };
  }
  const specification = evidence.specification;
  const storageIdentity = evidence.result.storageIdentity;
  const evaluationIdentity = evidence.result.evaluationIdentity;
  const indexRuns = evidence.result.indexRuns;
  const configurations = evidence.result.configurations;
  const boundedWork = evidence.result.boundedWork;
  if (
    !specification ||
    !storageIdentity ||
    !evaluationIdentity ||
    !indexRuns ||
    !configurations ||
    !boundedWork
  ) {
    throw new Error(
      `Recall quality gate evidence invalid at ${resultsPath}: project-scoped evidence requires complete specification, measurements, index runs, and bounded work`,
    );
  }
  if (
    storageIdentity.generationFormatVersion !== RECALL_GENERATION_FORMAT_VERSION ||
    storageIdentity.generationStoreFormatVersion !== RECALL_GENERATION_STORE_FORMAT_VERSION ||
    storageIdentity.validationReceiptVersion !== RECALL_GENERATION_VALIDATION_RECEIPT_VERSION ||
    storageIdentity.incrementalEligibilityPolicyVersion !==
      INCREMENTAL_RECALL_ELIGIBILITY_POLICY_VERSION
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        'Recall quality target generation storage identity does not match current generation, store, validation receipt, or eligibility contracts; rerun npm run evaluate:recall',
      ],
    };
  }
  const expectedLineageDigest = createLineageDigest(
    normalizeRecallProjectLineages(specification.projectLineages),
  );
  const repositoryFixtureCount = specification.projectIdentityFixtures.filter(
    ({ identitySource }) => identitySource !== RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
  ).length;
  const expectedConfigurationCount =
    specification.chunkPolicies.length * specification.candidateCounts.length;
  const expectedMaximumCandidates = Math.max(...specification.candidateCounts) * 3;
  const representedScopeCount = new Set(specification.cases.map(({ scope }) => scope)).size;
  const preLimitProofCount = specification.cases.filter(
    ({ preLimitChannelProof }) => preLimitChannelProof !== undefined,
  ).length;
  const expectedSearchRequests =
    expectedConfigurationCount *
    (specification.cases.length +
      preLimitProofCount +
      specification.warmupQueriesPerCombination * representedScopeCount);
  const expectedCaseIds = new Set(specification.cases.map(({ id }) => id));
  const expectedConfigurationKeys = new Set(
    specification.chunkPolicies.flatMap((chunkPolicy) =>
      specification.candidateCounts.map((candidateCount) => `${chunkPolicy.id}\0${candidateCount}`),
    ),
  );
  const actualConfigurationKeys = new Set(
    configurations.map(({ chunkPolicy, candidateCount }) => `${chunkPolicy.id}\0${candidateCount}`),
  );
  const embeddedRequestCount = indexRuns.reduce(
    (total, { indexSummary }) => total + indexSummary.embeddingRequestCount,
    0,
  );
  const boundedWorkValid =
    boundedWork.sessionFiles === specification.corpus.sessionFiles.length &&
    boundedWork.sessionFiles <= specification.bounds.maximumSessionFiles &&
    boundedWork.evaluationCases === specification.cases.length &&
    boundedWork.evaluationCases <= specification.bounds.maximumEvaluationCases &&
    boundedWork.indexRuns === indexRuns.length &&
    boundedWork.indexRuns === specification.chunkPolicies.length &&
    boundedWork.indexRuns <= specification.bounds.maximumChunkPolicies &&
    configurations.length === expectedConfigurationCount &&
    specification.candidateCounts.length <= specification.bounds.maximumCandidateCounts &&
    specification.finalCounts.length <= specification.bounds.maximumFinalCounts &&
    boundedWork.executedSearchRequests === expectedSearchRequests &&
    boundedWork.executedSearchRequests <= specification.bounds.maximumSearchRequests &&
    boundedWork.rerankerRequests === 0 &&
    boundedWork.chunkEmbeddingRequests === embeddedRequestCount &&
    boundedWork.chunkEmbeddingRequests <= specification.bounds.maximumChunkEmbeddingRequests &&
    boundedWork.maximumCandidatesPerSearch === expectedMaximumCandidates &&
    boundedWork.repositoryIdentityResolutions === repositoryFixtureCount &&
    indexRuns.every(
      ({
        generationId,
        manifestFingerprint,
        startingSnapshotFingerprint,
        storeCounts,
        totalChunks,
        indexSummary,
      }) =>
        generationId !== undefined &&
        manifestFingerprint !== undefined &&
        startingSnapshotFingerprint !== undefined &&
        storeCounts !== undefined &&
        storeCounts.lexicalSource >= totalChunks &&
        storeCounts.dense === indexSummary.newlyEmbeddedChunks &&
        storeCounts.sessionProjection >= boundedWork.sessionFiles &&
        indexSummary.scannedSessions === boundedWork.sessionFiles &&
        indexSummary.indexedSessions === boundedWork.sessionFiles &&
        indexSummary.cacheHits === 0 &&
        indexSummary.failedSessions.length === 0,
    ) &&
    actualConfigurationKeys.size === expectedConfigurationKeys.size &&
    [...expectedConfigurationKeys].every((key) => actualConfigurationKeys.has(key)) &&
    indexRuns.every(({ chunkPolicy }) =>
      specification.chunkPolicies.some(
        (expected) =>
          expected.id === chunkPolicy.id &&
          expected.maxTokens === chunkPolicy.maxTokens &&
          expected.overlapTokens === chunkPolicy.overlapTokens,
      ),
    ) &&
    configurations.every(({ measurement }) => {
      const measuredCaseIds = new Set(measurement.caseMeasurements.map(({ caseId }) => caseId));
      return (
        measurement.caseCount === boundedWork.evaluationCases &&
        measurement.caseMeasurements.length === boundedWork.evaluationCases &&
        measuredCaseIds.size === expectedCaseIds.size &&
        [...expectedCaseIds].every((caseId) => measuredCaseIds.has(caseId)) &&
        measurement.finalCounts.length === specification.finalCounts.length &&
        specification.finalCounts.every((finalCount) =>
          measurement.finalCounts.some((measured) => measured.finalCount === finalCount),
        ) &&
        measurement.caseMeasurements.every(
          (caseMeasurement) =>
            caseMeasurement.finalCounts.length === specification.finalCounts.length &&
            specification.finalCounts.every((finalCount) =>
              caseMeasurement.finalCounts.some((measured) => measured.finalCount === finalCount),
            ),
        )
      );
    });
  if (!boundedWorkValid) {
    throw new Error(
      `Recall quality gate evidence invalid at ${resultsPath}: bounded-work counters or complete measurements disagree with the fixed corpus and limits`,
    );
  }
  const typedConfigurations = configurations.map((configuration) => ({
    ...configuration,
    measurement: {
      ...configuration.measurement,
      policyFailureCaseIds: configuration.measurement.policyFailureCaseIds.map(parseQualityCaseId),
      missedCandidatePoolCaseIds:
        configuration.measurement.missedCandidatePoolCaseIds.map(parseQualityCaseId),
      caseMeasurements: configuration.measurement.caseMeasurements.map((measurement) => ({
        ...measurement,
        caseId: parseQualityCaseId(measurement.caseId),
      })),
      finalCounts: configuration.measurement.finalCounts.map((measurement) => ({
        ...measurement,
        missedCaseIds: measurement.missedCaseIds.map(parseQualityCaseId),
        contextFailureCaseIds: measurement.contextFailureCaseIds.map(parseQualityCaseId),
        sourceOccurrenceFailureCaseIds:
          measurement.sourceOccurrenceFailureCaseIds.map(parseQualityCaseId),
        sessionOriginFailureCaseIds:
          measurement.sessionOriginFailureCaseIds.map(parseQualityCaseId),
        evidenceRelationFailureCaseIds:
          measurement.evidenceRelationFailureCaseIds.map(parseQualityCaseId),
        contributingEntryFailureCaseIds:
          measurement.contributingEntryFailureCaseIds.map(parseQualityCaseId),
        branchFailureCaseIds: measurement.branchFailureCaseIds.map(parseQualityCaseId),
      })),
    },
  }));
  const measuredSelection = selectRecallQualityPolicy(
    typedConfigurations,
    specification.qualityGate,
  );
  const selectedMatchesMeasurement =
    selection.selected !== null &&
    measuredSelection.selected !== null &&
    areRecallQualityPoliciesEqual(selection.selected, measuredSelection.selected) &&
    selection.selected.gatePassed === measuredSelection.selected.gatePassed;
  const combinationsMatchMeasurements =
    selection.combinations?.length === measuredSelection.combinations.length &&
    measuredSelection.combinations.every((measured) =>
      selection.combinations?.some(
        (recorded) =>
          areRecallQualityPoliciesEqual(recorded, measured) &&
          recorded.gatePassed === measured.gatePassed,
      ),
    );
  if (
    selection.passed !== measuredSelection.passed ||
    (selection.passed && !selectedMatchesMeasurement) ||
    !combinationsMatchMeasurements
  ) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: selection was not reproduced from complete measurements`,
    );
  }
  if (
    specification.version !== 3 ||
    evaluationIdentity.defaultScope !== RecallSearchScope.PROJECT ||
    evaluationIdentity.projectScopePolicyVersion !== PROJECT_SCOPE_POLICY_VERSION ||
    evaluationIdentity.projectIdentityPolicyVersion !== PROJECT_IDENTITY_POLICY_VERSION ||
    evaluationIdentity.projectIdentityMetadataSchemaVersion !==
      PROJECT_IDENTITY_METADATA_SCHEMA_VERSION ||
    evaluationIdentity.lineagePolicyVersion !== PROJECT_LINEAGE_POLICY_VERSION ||
    evaluationIdentity.lineageDigest !== expectedLineageDigest
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        'Recall quality project identity does not match the current scope, repository, or lineage policy; rerun npm run evaluate:recall',
      ],
    };
  }
  if (
    evaluationIdentity.rankingMode !== 'hybrid' ||
    evaluationIdentity.rankFusionVersion !== RECALL_RANK_FUSION_VERSION ||
    evaluationIdentity.reciprocalRankConstant !== RECALL_RRF_RANK_CONSTANT ||
    evaluationIdentity.activeBranchPrior !== RECALL_ACTIVE_BRANCH_PRIOR ||
    evaluationIdentity.fusedPoolLimit !== 24 ||
    evaluationIdentity.rerankPoolLimit !== 24
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        ...selection.blockers,
        'Recall quality ranking identity does not match the current hybrid policy; rerun npm run evaluate:recall',
      ],
    };
  }
  if (!selection.passed) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [...selection.blockers],
    };
  }
  if (evidence.environment.gitDirty) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: ['Recall quality evidence was generated from a dirty worktree'],
    };
  }
  const selected = selection.selected;
  if (!selected) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: passing selection is missing`,
    );
  }
  if (
    selected.chunkPolicy.id !== '512-64' ||
    selected.chunkPolicy.maxTokens !== 512 ||
    selected.chunkPolicy.overlapTokens !== 64 ||
    selected.candidateCount !== 8 ||
    selected.finalCount !== 5 ||
    evaluationIdentity.candidateLimits.dense !== 8 ||
    evaluationIdentity.candidateLimits.lexical !== 8 ||
    evaluationIdentity.candidateLimits.identifier !== 8 ||
    evaluationIdentity.fusedPoolLimit !== 24 ||
    evaluationIdentity.rerankPoolLimit !== 24 ||
    evaluationIdentity.finalResultCount !== 5
  ) {
    return {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: [
        'Recall quality selected policy differs from the approved 512/64, 8 candidates/channel, 5-result policy; obtain human approval before changing production',
      ],
    };
  }
  const wasMeasuredPassing = selection.combinations?.some(
    (combination) => combination.gatePassed && areRecallQualityPoliciesEqual(combination, selected),
  );
  if (!wasMeasuredPassing) {
    throw new Error(
      `Recall quality gate evidence inconsistent at ${resultsPath}: selected policy was not a passing measured combination`,
    );
  }
  return {
    automatedGatePassed: true,
    selectedPolicy: {
      chunkPolicy: { ...selected.chunkPolicy },
      candidateCount: selected.candidateCount,
      finalCount: selected.finalCount,
    },
    blockers: [],
  };
}

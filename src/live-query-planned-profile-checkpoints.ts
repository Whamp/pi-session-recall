import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { Type } from 'typebox';
import { Value } from 'typebox/value';
import {
  EmbeddedInferenceComputeBackend,
  EmbeddedInferenceDevicePolicy,
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallControlKind,
  RecallInferenceBackend,
} from './enums.js';
import { writeAtomicRecallEvaluationFile } from './recall-evaluation-file-system.js';
import type { LiveQueryPlannedProfileEvaluationResult } from './query-planned-recall-evaluation.js';

const NULLABLE_NUMBER_SCHEMA = Type.Union([Type.Number(), Type.Null()]);
const EXECUTION_BACKEND_SCHEMA = Type.Enum(RecallInferenceBackend);
const PROFILE_RUN_SCHEMA = Type.Object({
  id: Type.String(),
  backend: EXECUTION_BACKEND_SCHEMA,
  deviceClass: Type.Union([Type.Literal('cpu'), Type.Literal('accelerated')]),
  device: Type.String(),
  backendVersion: Type.Optional(Type.String()),
});
const RESOLVED_COMPUTE_BACKEND_SCHEMA = Type.Enum(EmbeddedInferenceComputeBackend);
const EMBEDDED_DEVICE_POLICY_SCHEMA = Type.Enum(EmbeddedInferenceDevicePolicy);
const NULLABLE_ACCELERATED_COMPUTE_BACKEND_SCHEMA = Type.Union([
  Type.Literal(EmbeddedInferenceComputeBackend.METAL),
  Type.Literal(EmbeddedInferenceComputeBackend.CUDA),
  Type.Literal(EmbeddedInferenceComputeBackend.VULKAN),
  Type.Null(),
]);
const EMBEDDED_PHYSICAL_DEVICE_EXECUTION_IDENTITY_PROPERTIES = {
  computeBackend: RESOLVED_COMPUTE_BACKEND_SCHEMA,
  deviceNames: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
  devicePolicy: EMBEDDED_DEVICE_POLICY_SCHEMA,
  fallbackFromComputeBackend: NULLABLE_ACCELERATED_COMPUTE_BACKEND_SCHEMA,
  physicalDeviceIdentity: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
  probedComputeBackends: Type.Array(
    Type.Union([
      Type.Literal(EmbeddedInferenceComputeBackend.METAL),
      Type.Literal(EmbeddedInferenceComputeBackend.CUDA),
      Type.Literal(EmbeddedInferenceComputeBackend.VULKAN),
    ]),
  ),
};
const QUERY_PLANNING_EXECUTION_IDENTITY_PROPERTIES = {
  adapterId: Type.String(),
  adapterVersion: Type.String(),
  adapterConfigurationIdentity: Type.String(),
  cacheIdentity: Type.String(),
  modelProfileId: Type.String(),
  modelProfileIdentity: Type.String(),
  promptPolicy: Type.String(),
  grammarVersion: Type.String(),
  requestTimeoutMilliseconds: Type.Number(),
};
const QUERY_PLANNING_EXECUTION_IDENTITY_SCHEMA = Type.Union([
  Type.Object({
    ...QUERY_PLANNING_EXECUTION_IDENTITY_PROPERTIES,
    backend: Type.Literal(RecallInferenceBackend.EMBEDDED),
    ...EMBEDDED_PHYSICAL_DEVICE_EXECUTION_IDENTITY_PROPERTIES,
    contextSize: Type.Integer({ minimum: 1 }),
    threads: Type.Union([Type.Integer({ minimum: 1 }), Type.Null()]),
    nodeLlamaCppVersion: Type.String({ minLength: 1 }),
  }),
  Type.Object({
    ...QUERY_PLANNING_EXECUTION_IDENTITY_PROPERTIES,
    backend: Type.Literal(RecallInferenceBackend.LLAMA_CPP_HTTP),
  }),
]);
const RERANKING_EXECUTION_IDENTITY_PROPERTIES = {
  adapterId: Type.String(),
  adapterVersion: Type.String(),
  adapterConfigurationIdentity: Type.String(),
  cacheIdentity: Type.String(),
  modelProfileId: Type.String(),
  modelProfileIdentity: Type.String(),
  requestTimeoutMilliseconds: Type.Number(),
};
const RERANKING_EXECUTION_IDENTITY_SCHEMA = Type.Union([
  Type.Object({
    ...RERANKING_EXECUTION_IDENTITY_PROPERTIES,
    backend: Type.Literal(RecallInferenceBackend.EMBEDDED),
    ...EMBEDDED_PHYSICAL_DEVICE_EXECUTION_IDENTITY_PROPERTIES,
    contextSize: Type.Integer({ minimum: 1 }),
    threads: Type.Union([Type.Integer({ minimum: 1 }), Type.Null()]),
    nodeLlamaCppVersion: Type.String({ minLength: 1 }),
    parallelism: Type.Integer({ minimum: 1 }),
  }),
  Type.Object({
    ...RERANKING_EXECUTION_IDENTITY_PROPERTIES,
    backend: Type.Literal(RecallInferenceBackend.LLAMA_CPP_HTTP),
  }),
]);
const CANDIDATE_LIMITS_SCHEMA = Type.Object({
  dense: Type.Number(),
  lexical: Type.Number(),
  identifier: Type.Number(),
});
const BASELINE_ARM_SCHEMA = Type.Object({
  outcome: Type.Enum(QueryPlannedRecallBaselineOutcome),
  expectedSourceRanks: Type.Array(NULLABLE_NUMBER_SCHEMA),
  highestRelevantDistractorRank: NULLABLE_NUMBER_SCHEMA,
  provenancePassed: Type.Boolean(),
  listLimits: CANDIDATE_LIMITS_SCHEMA,
  totalCandidatesExamined: Type.Number(),
  uniqueCandidatesAdmitted: Type.Number(),
  finalResultCount: Type.Number(),
  fusedPoolLimit: Type.Number(),
  rerankPoolLimit: Type.Number(),
  rankingMode: Type.Literal('hybrid'),
  rankFusionVersion: Type.Number(),
  reciprocalRankConstant: Type.Number(),
});
const QUERY_PLANNED_ARM_SCHEMA = Type.Object({
  outcome: Type.Enum(QueryPlannedRecallBaselineOutcome),
  expectedSourceRanks: Type.Array(NULLABLE_NUMBER_SCHEMA),
  admissionProbeSourceRanks: Type.Array(NULLABLE_NUMBER_SCHEMA),
  candidateAdmissionVerified: Type.Boolean(),
  provenancePassed: Type.Boolean(),
  listWork: Type.Array(
    Type.Object({
      source: Type.String(),
      weight: Type.Number(),
      candidateLimit: Type.Number(),
      admittedCandidateCount: Type.Number(),
    }),
  ),
  totalCandidatesExamined: Type.Number(),
  rerankCandidatesExamined: Type.Number(),
  finalResultCount: Type.Number(),
  fusedPoolLimit: Type.Number(),
  rerankPoolLimit: Type.Number(),
  finalResultLimit: Type.Number(),
  rankFusionVersion: Type.Number(),
  reciprocalRankConstant: Type.Number(),
  fusionPolicy: Type.Object({
    submittedQueryListWeight: Type.Number(),
    plannedQueryListWeight: Type.Number(),
    rankOneBonus: Type.Number(),
    rankTwoOrThreeBonus: Type.Number(),
  }),
  rerankerPolicy: Type.Object({
    version: Type.Number(),
    activeBranchPrior: Type.Number(),
    fusedRankBlend: Type.Array(
      Type.Object({
        firstRank: Type.Number(),
        lastRank: NULLABLE_NUMBER_SCHEMA,
        retrievalWeight: Type.Number(),
        rerankerWeight: Type.Number(),
      }),
    ),
  }),
  rankingProviderPolicy: Type.Union([
    Type.Literal('neutral-fused-order-v1'),
    Type.Literal('live-profile-v1'),
  ]),
  admissionProbeProviderPolicy: Type.Literal('expected-source-promotion-v1'),
});
const CONTRIBUTION_SCHEMA = Type.Object({
  newCandidateAdmission: Type.Boolean(),
  rankingOnlyPromotion: Type.Boolean(),
  preservedExistingSuccess: Type.Boolean(),
  noImprovement: Type.Boolean(),
});
const LIVE_PROFILE_RESULT_SCHEMA = Type.Object({
  version: Type.Literal(1),
  profileRun: PROFILE_RUN_SCHEMA,
  corpus: Type.Object({
    id: Type.String(),
    privateManifestSha256: Type.String(),
    snapshotCount: Type.Number(),
    indexedDocumentCount: Type.Number(),
    caseCount: Type.Number(),
  }),
  profileIdentity: Type.Object({
    embeddingPolicy: Type.Literal('deterministic-token-hash-v1'),
    embeddingDimensions: Type.Number(),
    evaluationConfiguration: Type.Object({
      version: Type.Literal(1),
      effectiveConfigurationIdentity: Type.String(),
      rerankerConformanceFixtureIdentity: Type.String(),
    }),
    software: Type.Object({
      repositoryCommit: Type.String(),
      backendVersion: Type.String(),
      nodeVersion: Type.String(),
      platform: Type.String(),
      architecture: Type.String(),
    }),
    queryPlanning: Type.Object({
      profileId: Type.String(),
      model: Type.String(),
      promptPolicy: Type.String(),
      grammarVersion: Type.String(),
      executionIdentity: QUERY_PLANNING_EXECUTION_IDENTITY_SCHEMA,
    }),
    reranking: Type.Object({
      profileId: Type.String(),
      model: Type.String(),
      scorePolicy: Type.String(),
      executionIdentity: RERANKING_EXECUTION_IDENTITY_SCHEMA,
    }),
  }),
  capabilityConformance: Type.Object({
    queryPlanning: Type.Object({
      profileId: Type.String(),
      model: Type.String(),
      promptPolicy: Type.String(),
      grammarVersion: Type.String(),
      executionIdentity: QUERY_PLANNING_EXECUTION_IDENTITY_SCHEMA,
      measurement: Type.Object({
        plannedQueryCount: Type.Number(),
        lexQueryCount: Type.Number(),
        vecQueryCount: Type.Number(),
        hydeQueryCount: Type.Number(),
        planningMilliseconds: Type.Number(),
      }),
    }),
    reranking: Type.Object({
      profileId: Type.String(),
      model: Type.String(),
      scorePolicy: Type.String(),
      executionIdentity: RERANKING_EXECUTION_IDENTITY_SCHEMA,
      measurement: Type.Object({
        queryCount: Type.Literal(1),
        documentCount: Type.Number(),
        rerankingMilliseconds: Type.Number(),
      }),
    }),
  }),
  latency: Type.Object({
    coldPlanningMilliseconds: Type.Number(),
    warmPlanningMilliseconds: Type.Number(),
    warmPlanningSucceeded: Type.Boolean(),
    coldRerankingMilliseconds: Type.Number(),
    warmRerankingMilliseconds: Type.Number(),
    totalSearchMilliseconds: Type.Object({
      minimum: Type.Number(),
      median: Type.Number(),
      maximum: Type.Number(),
    }),
  }),
  cases: Type.Array(
    Type.Object({
      caseId: Type.String(),
      category: Type.Enum(QueryPlannedRecallCaseCategory),
      controlKind: Type.Enum(QueryPlannedRecallControlKind),
      planSource: Type.Union([Type.Literal('planner'), Type.Literal('fallback')]),
      plannedQueries: Type.Array(
        Type.Object({
          type: Type.Union([Type.Literal('lex'), Type.Literal('vec'), Type.Literal('hyde')]),
          querySha256: Type.String(),
        }),
      ),
      normal: BASELINE_ARM_SCHEMA,
      retrievalWorkMatched: BASELINE_ARM_SCHEMA,
      queryPlanned: QUERY_PLANNED_ARM_SCHEMA,
      planningMilliseconds: Type.Number(),
      rerankingMilliseconds: Type.Number(),
      totalSearchMilliseconds: Type.Number(),
      contribution: CONTRIBUTION_SCHEMA,
    }),
  ),
  quality: Type.Object({
    newCandidateAdmissionCount: Type.Number(),
    rankingOnlyPromotionCount: Type.Number(),
    preservedExistingSuccessCount: Type.Number(),
    plannerFallbackCount: Type.Number(),
  }),
});
const CHECKPOINT_IDENTITY_SCHEMA = Type.Object({
  version: Type.Literal(2),
  recordedAgainstCommit: Type.String(),
  privateManifestSha256: Type.String(),
  profileRun: PROFILE_RUN_SCHEMA,
  profileIdentity: LIVE_PROFILE_RESULT_SCHEMA.properties.profileIdentity,
});
const LIVE_PROFILE_CHECKPOINT_SCHEMA = Type.Object({
  checkpointIdentity: CHECKPOINT_IDENTITY_SCHEMA,
  result: LIVE_PROFILE_RESULT_SCHEMA,
});

/** Exact code, corpus, canonical execution, evaluation, software, and physical device identity for resume. */
export interface LiveProfileEvaluationCheckpointIdentity {
  version: 2;
  recordedAgainstCommit: string;
  privateManifestSha256: string;
  profileRun: LiveQueryPlannedProfileEvaluationResult['profileRun'];
  profileIdentity: LiveQueryPlannedProfileEvaluationResult['profileIdentity'];
}

/** One expensive live profile operation guarded by an identity-bound private checkpoint. */
export interface CheckpointedLiveProfileEvaluation {
  profileRun: LiveQueryPlannedProfileEvaluationResult['profileRun'];
  resolveCheckpointIdentity(): Promise<LiveProfileEvaluationCheckpointIdentity>;
  evaluateProfile(): Promise<LiveQueryPlannedProfileEvaluationResult>;
  disposeProfile?(): Promise<void>;
}

/** Inputs for a sequential live profile matrix that can resume completed profile evidence. */
export interface RunCheckpointedLiveProfileEvaluationMatrixOptions {
  checkpointDirectory: string;
  profiles: readonly CheckpointedLiveProfileEvaluation[];
  reportProgress(message: string): void;
}

function createLiveProfileCheckpointPath(checkpointDirectory: string, profileId: string): string {
  const profileKey = createHash('sha256').update(profileId).digest('hex');
  return join(checkpointDirectory, `${profileKey}.json`);
}

function checkpointResultMatchesIdentity(
  result: LiveQueryPlannedProfileEvaluationResult,
  identity: LiveProfileEvaluationCheckpointIdentity,
): boolean {
  return (
    isDeepStrictEqual(result.profileRun, identity.profileRun) &&
    result.corpus.privateManifestSha256 === identity.privateManifestSha256 &&
    identity.profileIdentity.software.repositoryCommit === identity.recordedAgainstCommit &&
    isDeepStrictEqual(result.profileIdentity, identity.profileIdentity) &&
    isDeepStrictEqual(
      result.capabilityConformance.queryPlanning.executionIdentity,
      identity.profileIdentity.queryPlanning.executionIdentity,
    ) &&
    isDeepStrictEqual(
      result.capabilityConformance.reranking.executionIdentity,
      identity.profileIdentity.reranking.executionIdentity,
    )
  );
}

async function readMatchingLiveProfileCheckpoint(
  checkpointPath: string,
  expectedIdentity: LiveProfileEvaluationCheckpointIdentity,
): Promise<LiveQueryPlannedProfileEvaluationResult | null> {
  let content: string;
  try {
    content = await readFile(checkpointPath, 'utf8');
  } catch (error) {
    if (error instanceof Error && 'code' in error && error.code === 'ENOENT') {
      return null;
    }
    throw error;
  }
  try {
    const parsed: unknown = JSON.parse(content);
    const checkpoint = Value.Parse(LIVE_PROFILE_CHECKPOINT_SCHEMA, parsed);
    const result: LiveQueryPlannedProfileEvaluationResult = checkpoint.result;
    if (
      !isDeepStrictEqual(checkpoint.checkpointIdentity, expectedIdentity) ||
      !checkpointResultMatchesIdentity(result, expectedIdentity)
    ) {
      return null;
    }
    return result;
  } catch {
    return null;
  }
}

/** Runs or resumes each live profile in order and atomically saves privacy-safe evidence. */
export async function runCheckpointedLiveProfileEvaluationMatrix(
  options: RunCheckpointedLiveProfileEvaluationMatrixOptions,
): Promise<LiveQueryPlannedProfileEvaluationResult[]> {
  const results: LiveQueryPlannedProfileEvaluationResult[] = [];
  for (const [profileIndex, profile] of options.profiles.entries()) {
    const position = `${profileIndex + 1}/${options.profiles.length}`;
    const profileId = profile.profileRun.id;
    try {
      const checkpointIdentity = await profile.resolveCheckpointIdentity();
      if (!isDeepStrictEqual(checkpointIdentity.profileRun, profile.profileRun)) {
        throw new Error(`Live profile resolved checkpoint tuple mismatch for ${profileId}`);
      }
      const checkpointPath = createLiveProfileCheckpointPath(
        options.checkpointDirectory,
        profileId,
      );
      const checkpointResult = await readMatchingLiveProfileCheckpoint(
        checkpointPath,
        checkpointIdentity,
      );
      if (checkpointResult) {
        options.reportProgress(`Resumed live profile ${profileId} (${position})`);
        results.push(checkpointResult);
        continue;
      }

      options.reportProgress(`Starting live profile ${profileId} (${position})`);
      const result = await profile.evaluateProfile();
      if (!checkpointResultMatchesIdentity(result, checkpointIdentity)) {
        throw new Error(`Live profile checkpoint identity mismatch for ${profileId}`);
      }
      await writeAtomicRecallEvaluationFile(
        checkpointPath,
        `${JSON.stringify({ checkpointIdentity, result }, null, 2)}\n`,
      );
      options.reportProgress(`Completed live profile ${profileId} (${position})`);
      results.push(result);
    } finally {
      await profile.disposeProfile?.();
    }
  }
  return results;
}

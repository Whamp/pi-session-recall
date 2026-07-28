import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { isDeepStrictEqual } from 'node:util';
import { Type } from 'typebox';
import { Value } from 'typebox/value';
import {
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
const QUERY_PLANNING_EXECUTION_IDENTITY_SCHEMA = Type.Object({
  adapterId: Type.String(),
  adapterConfigurationIdentity: Type.String(),
  backend: EXECUTION_BACKEND_SCHEMA,
  cacheIdentity: Type.String(),
  modelProfileId: Type.String(),
  promptPolicy: Type.String(),
  grammarVersion: Type.String(),
  requestTimeoutMilliseconds: Type.Number(),
});
const RERANKING_EXECUTION_IDENTITY_SCHEMA = Type.Object({
  adapterId: Type.String(),
  backend: EXECUTION_BACKEND_SCHEMA,
  cacheIdentity: Type.String(),
  modelProfileId: Type.String(),
});
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
  version: Type.Literal(1),
  recordedAgainstCommit: Type.String(),
  privateManifestSha256: Type.String(),
  profileRun: PROFILE_RUN_SCHEMA,
  queryPlanningProfileId: Type.String(),
  queryPlanningCacheIdentity: Type.String(),
  rerankingProfileId: Type.String(),
  rerankingCacheIdentity: Type.String(),
  adapterConfigurationIdentity: Type.String(),
});
const LIVE_PROFILE_CHECKPOINT_SCHEMA = Type.Object({
  checkpointIdentity: CHECKPOINT_IDENTITY_SCHEMA,
  result: LIVE_PROFILE_RESULT_SCHEMA,
});

/** Exact code, corpus, profile, backend, and device identity required to resume one live profile. */
export interface LiveProfileEvaluationCheckpointIdentity {
  version: 1;
  recordedAgainstCommit: string;
  privateManifestSha256: string;
  profileRun: LiveQueryPlannedProfileEvaluationResult['profileRun'];
  queryPlanningProfileId: string;
  queryPlanningCacheIdentity: string;
  rerankingProfileId: string;
  rerankingCacheIdentity: string;
  adapterConfigurationIdentity: string;
}

/** One expensive live profile operation guarded by an identity-bound private checkpoint. */
export interface CheckpointedLiveProfileEvaluation {
  checkpointIdentity: LiveProfileEvaluationCheckpointIdentity;
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
    result.profileIdentity.queryPlanning.profileId === identity.queryPlanningProfileId &&
    result.profileIdentity.queryPlanning.executionIdentity.cacheIdentity ===
      identity.queryPlanningCacheIdentity &&
    result.capabilityConformance.queryPlanning.executionIdentity.cacheIdentity ===
      identity.queryPlanningCacheIdentity &&
    result.profileIdentity.reranking.profileId === identity.rerankingProfileId &&
    result.profileIdentity.reranking.executionIdentity.cacheIdentity ===
      identity.rerankingCacheIdentity &&
    result.capabilityConformance.reranking.executionIdentity.cacheIdentity ===
      identity.rerankingCacheIdentity
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
    const profileId = profile.checkpointIdentity.profileRun.id;
    try {
      const checkpointPath = createLiveProfileCheckpointPath(
        options.checkpointDirectory,
        profileId,
      );
      const checkpointResult = await readMatchingLiveProfileCheckpoint(
        checkpointPath,
        profile.checkpointIdentity,
      );
      if (checkpointResult) {
        options.reportProgress(`Resumed live profile ${profileId} (${position})`);
        results.push(checkpointResult);
        continue;
      }

      options.reportProgress(`Starting live profile ${profileId} (${position})`);
      const result = await profile.evaluateProfile();
      if (!checkpointResultMatchesIdentity(result, profile.checkpointIdentity)) {
        throw new Error(`Live profile checkpoint identity mismatch for ${profileId}`);
      }
      await writeAtomicRecallEvaluationFile(
        checkpointPath,
        `${JSON.stringify({ checkpointIdentity: profile.checkpointIdentity, result }, null, 2)}\n`,
      );
      options.reportProgress(`Completed live profile ${profileId} (${position})`);
      results.push(result);
    } finally {
      await profile.disposeProfile?.();
    }
  }
  return results;
}

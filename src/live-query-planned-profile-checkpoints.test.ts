import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import {
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallControlKind,
  RecallInferenceBackend,
} from './enums.js';
import {
  runCheckpointedLiveProfileEvaluationMatrix,
  type LiveProfileEvaluationCheckpointIdentity,
} from './live-query-planned-profile-checkpoints.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  createRecallRerankingExecutionIdentity,
} from './recall-inference-capabilities.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import type { LiveQueryPlannedProfileEvaluationResult } from './query-planned-recall-evaluation.js';

function createCheckpointIdentity(
  recordedAgainstCommit: string,
  result: LiveQueryPlannedProfileEvaluationResult,
): LiveProfileEvaluationCheckpointIdentity {
  return {
    version: 2,
    recordedAgainstCommit,
    privateManifestSha256: result.corpus.privateManifestSha256,
    profileRun: result.profileRun,
    profileIdentity: result.profileIdentity,
  };
}

function createProfileResult(
  recordedAgainstCommit = 'commit-one',
  physicalDeviceIdentity = 'cpu',
): LiveQueryPlannedProfileEvaluationResult {
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const queryPlanningExecutionIdentity = {
    ...createRecallQueryPlanningExecutionIdentity(
      queryPlanningProfile,
      'embedded-qmd-query-planner-v1',
      `planner-configuration-${physicalDeviceIdentity}`,
      RecallInferenceBackend.EMBEDDED,
      300_000,
    ),
    computeBackend: 'cpu',
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    physicalDeviceIdentity: [physicalDeviceIdentity],
  };
  const rerankingExecutionIdentity = {
    ...createRecallRerankingExecutionIdentity(
      rerankingProfile,
      'embedded-qwen-reranker-v1',
      `reranker-configuration-${physicalDeviceIdentity}`,
      RecallInferenceBackend.EMBEDDED,
      300_000,
    ),
    computeBackend: 'cpu',
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    physicalDeviceIdentity: [physicalDeviceIdentity],
  };
  return {
    version: 1,
    profileRun: {
      id: 'embedded-cpu',
      backend: RecallInferenceBackend.EMBEDDED,
      deviceClass: 'cpu',
      device: 'cpu',
      backendVersion: 'node-llama-cpp@3.18.1 / llama.cpp b8390',
    },
    corpus: {
      id: 'private-corpus',
      privateManifestSha256: 'private-manifest-sha256',
      snapshotCount: 1,
      indexedDocumentCount: 2,
      caseCount: 1,
    },
    profileIdentity: {
      embeddingPolicy: 'deterministic-token-hash-v1',
      embeddingDimensions: 64,
      evaluationConfiguration: {
        version: 1,
        effectiveConfigurationIdentity: 'effective-evaluation-configuration-identity',
        rerankerConformanceFixtureIdentity: 'reranker-conformance-fixture-identity',
      },
      software: {
        repositoryCommit: recordedAgainstCommit,
        backendVersion: 'node-llama-cpp@3.18.1 / llama.cpp b8390',
        nodeVersion: 'v24.0.0',
        platform: 'linux',
        architecture: 'x64',
      },
      queryPlanning: {
        profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
        model: 'qmd-query-expansion-1.7B-q4_k_m',
        promptPolicy: 'qmd-query-expansion-v1',
        grammarVersion: 'qmd-query-plan-v1',
        executionIdentity: queryPlanningExecutionIdentity,
      },
      reranking: {
        profileId: 'qwen3-reranker-0.6b-q8-0-v1',
        model: 'qwen3-reranker-0.6b-q8_0',
        scorePolicy: 'qwen3-logit-sigmoid-v1',
        executionIdentity: rerankingExecutionIdentity,
      },
    },
    capabilityConformance: {
      queryPlanning: {
        profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
        model: 'qmd-query-expansion-1.7B-q4_k_m',
        promptPolicy: 'qmd-query-expansion-v1',
        grammarVersion: 'qmd-query-plan-v1',
        executionIdentity: queryPlanningExecutionIdentity,
        measurement: {
          plannedQueryCount: 2,
          lexQueryCount: 1,
          vecQueryCount: 1,
          hydeQueryCount: 0,
          planningMilliseconds: 10,
        },
      },
      reranking: {
        profileId: 'qwen3-reranker-0.6b-q8-0-v1',
        model: 'qwen3-reranker-0.6b-q8_0',
        scorePolicy: 'qwen3-logit-sigmoid-v1',
        executionIdentity: rerankingExecutionIdentity,
        measurement: {
          queryCount: 1,
          documentCount: 2,
          rerankingMilliseconds: 5,
        },
      },
    },
    latency: {
      coldPlanningMilliseconds: 10,
      warmPlanningMilliseconds: 4,
      warmPlanningSucceeded: true,
      coldRerankingMilliseconds: 5,
      warmRerankingMilliseconds: 2,
      totalSearchMilliseconds: { minimum: 20, median: 20, maximum: 20 },
    },
    cases: [
      {
        caseId: 'case-1',
        category: QueryPlannedRecallCaseCategory.VOCABULARY_DRIFT,
        controlKind: QueryPlannedRecallControlKind.DIFFICULT_CASE,
        planSource: 'planner',
        plannedQueries: [{ type: 'lex', querySha256: 'query-sha256' }],
        normal: {
          outcome: QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS,
          expectedSourceRanks: [null],
          highestRelevantDistractorRank: null,
          provenancePassed: false,
          listLimits: { dense: 8, lexical: 8, identifier: 8 },
          totalCandidatesExamined: 24,
          uniqueCandidatesAdmitted: 20,
          finalResultCount: 5,
          fusedPoolLimit: 40,
          rerankPoolLimit: 40,
          rankingMode: 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
        },
        retrievalWorkMatched: {
          outcome: QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS,
          expectedSourceRanks: [null],
          highestRelevantDistractorRank: null,
          provenancePassed: false,
          listLimits: { dense: 14, lexical: 13, identifier: 13 },
          totalCandidatesExamined: 40,
          uniqueCandidatesAdmitted: 30,
          finalResultCount: 5,
          fusedPoolLimit: 40,
          rerankPoolLimit: 40,
          rankingMode: 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
        },
        queryPlanned: {
          outcome: QueryPlannedRecallBaselineOutcome.SUCCESS,
          expectedSourceRanks: [1],
          admissionProbeSourceRanks: [1],
          candidateAdmissionVerified: true,
          provenancePassed: true,
          listWork: [
            { source: 'planned-lex-0', weight: 1, candidateLimit: 20, admittedCandidateCount: 1 },
          ],
          totalCandidatesExamined: 20,
          rerankCandidatesExamined: 1,
          finalResultCount: 1,
          fusedPoolLimit: 40,
          rerankPoolLimit: 40,
          finalResultLimit: 5,
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
          fusionPolicy: {
            submittedQueryListWeight: 2,
            plannedQueryListWeight: 1,
            rankOneBonus: 0.05,
            rankTwoOrThreeBonus: 0.02,
          },
          rerankerPolicy: {
            version: 1,
            activeBranchPrior: 0.01,
            fusedRankBlend: [
              { firstRank: 1, lastRank: null, retrievalWeight: 0.3, rerankerWeight: 0.7 },
            ],
          },
          rankingProviderPolicy: 'live-profile-v1',
          admissionProbeProviderPolicy: 'expected-source-promotion-v1',
        },
        planningMilliseconds: 10,
        rerankingMilliseconds: 5,
        totalSearchMilliseconds: 20,
        contribution: {
          newCandidateAdmission: true,
          rankingOnlyPromotion: false,
          preservedExistingSuccess: false,
          noImprovement: false,
        },
      },
    ],
    quality: {
      newCandidateAdmissionCount: 1,
      rankingOnlyPromotionCount: 0,
      preservedExistingSuccessCount: 0,
      plannerFallbackCount: 0,
    },
  };
}

void test('live profile matrix resumes only an exact identity-bound checkpoint', async () => {
  const checkpointDirectory = await mkdtemp(join(tmpdir(), 'recall-live-profile-checkpoints-'));
  const progress: string[] = [];
  let runCount = 0;
  const result = createProfileResult();
  const firstIdentity = createCheckpointIdentity('commit-one', result);

  try {
    const firstRun = await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          profileRun: firstIdentity.profileRun,
          async resolveCheckpointIdentity() {
            return firstIdentity;
          },
          async evaluateProfile() {
            runCount += 1;
            return result;
          },
        },
      ],
      reportProgress(message) {
        progress.push(message);
      },
    });
    assert.deepEqual(firstRun, [result]);
    assert.equal(runCount, 1);
    assert.deepEqual(progress, [
      'Starting live profile embedded-cpu (1/1)',
      'Completed live profile embedded-cpu (1/1)',
    ]);

    progress.length = 0;
    const resumed = await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          profileRun: firstIdentity.profileRun,
          async resolveCheckpointIdentity() {
            return firstIdentity;
          },
          async evaluateProfile() {
            throw new Error('Exact checkpoint should have resumed');
          },
        },
      ],
      reportProgress(message) {
        progress.push(message);
      },
    });
    assert.deepEqual(resumed, [result]);
    assert.equal(runCount, 1);
    assert.deepEqual(progress, ['Resumed live profile embedded-cpu (1/1)']);

    progress.length = 0;
    const changedResult = createProfileResult('commit-two');
    const changedIdentity = createCheckpointIdentity('commit-two', changedResult);
    await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          profileRun: changedIdentity.profileRun,
          async resolveCheckpointIdentity() {
            return changedIdentity;
          },
          async evaluateProfile() {
            runCount += 1;
            return changedResult;
          },
        },
      ],
      reportProgress(message) {
        progress.push(message);
      },
    });
    assert.equal(runCount, 2);
    assert.deepEqual(progress, [
      'Starting live profile embedded-cpu (1/1)',
      'Completed live profile embedded-cpu (1/1)',
    ]);

    progress.length = 0;
    const replacementDeviceResult = createProfileResult('commit-two', 'replacement cpu');
    const replacementDeviceIdentity = createCheckpointIdentity(
      'commit-two',
      replacementDeviceResult,
    );
    await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          profileRun: replacementDeviceIdentity.profileRun,
          async resolveCheckpointIdentity() {
            return replacementDeviceIdentity;
          },
          async evaluateProfile() {
            runCount += 1;
            return replacementDeviceResult;
          },
        },
      ],
      reportProgress(message) {
        progress.push(message);
      },
    });
    assert.equal(runCount, 3);
    assert.deepEqual(progress, [
      'Starting live profile embedded-cpu (1/1)',
      'Completed live profile embedded-cpu (1/1)',
    ]);
  } finally {
    await rm(checkpointDirectory, { recursive: true, force: true });
  }
});

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
import type { LiveQueryPlannedProfileEvaluationResult } from './query-planned-recall-evaluation.js';

function createCheckpointIdentity(
  recordedAgainstCommit: string,
): LiveProfileEvaluationCheckpointIdentity {
  return {
    version: 1,
    recordedAgainstCommit,
    privateManifestSha256: 'private-manifest-sha256',
    profileRun: {
      id: 'embedded-cpu',
      backend: RecallInferenceBackend.EMBEDDED,
      deviceClass: 'cpu',
      device: 'cpu',
      backendVersion: 'node-llama-cpp@3.18.1 / llama.cpp b8390',
    },
    queryPlanningProfileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
    queryPlanningCacheIdentity: 'planner-cache-identity',
    rerankingProfileId: 'qwen3-reranker-0.6b-q8-0-v1',
    rerankingCacheIdentity: 'reranker-cache-identity',
    adapterConfigurationIdentity: 'embedded-cpu-configuration',
  };
}

function createProfileResult(): LiveQueryPlannedProfileEvaluationResult {
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
      queryPlanning: {
        profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
        model: 'qmd-query-expansion-1.7B-q4_k_m',
        promptPolicy: 'qmd-query-expansion-v1',
        grammarVersion: 'qmd-query-plan-v1',
        executionIdentity: {
          adapterId: 'embedded-qmd-query-planner-v1',
          backend: RecallInferenceBackend.EMBEDDED,
          cacheIdentity: 'planner-cache-identity',
          modelProfileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
          promptPolicy: 'qmd-query-expansion-v1',
          grammarVersion: 'qmd-query-plan-v1',
          requestTimeoutMilliseconds: 300_000,
        },
      },
      reranking: {
        profileId: 'qwen3-reranker-0.6b-q8-0-v1',
        model: 'qwen3-reranker-0.6b-q8_0',
        scorePolicy: 'qwen3-logit-sigmoid-v1',
        executionIdentity: {
          adapterId: 'embedded-qwen-reranker-v1',
          backend: RecallInferenceBackend.EMBEDDED,
          cacheIdentity: 'reranker-cache-identity',
          modelProfileId: 'qwen3-reranker-0.6b-q8-0-v1',
        },
      },
    },
    capabilityConformance: {
      queryPlanning: {
        profileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
        model: 'qmd-query-expansion-1.7B-q4_k_m',
        promptPolicy: 'qmd-query-expansion-v1',
        grammarVersion: 'qmd-query-plan-v1',
        executionIdentity: {
          adapterId: 'embedded-qmd-query-planner-v1',
          backend: RecallInferenceBackend.EMBEDDED,
          cacheIdentity: 'planner-cache-identity',
          modelProfileId: 'qmd-query-expansion-1.7b-q4-k-m-v1',
          promptPolicy: 'qmd-query-expansion-v1',
          grammarVersion: 'qmd-query-plan-v1',
          requestTimeoutMilliseconds: 300_000,
        },
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
        executionIdentity: {
          adapterId: 'embedded-qwen-reranker-v1',
          backend: RecallInferenceBackend.EMBEDDED,
          cacheIdentity: 'reranker-cache-identity',
          modelProfileId: 'qwen3-reranker-0.6b-q8-0-v1',
        },
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
  const firstIdentity = createCheckpointIdentity('commit-one');
  const result = createProfileResult();

  try {
    const firstRun = await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          checkpointIdentity: firstIdentity,
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
          checkpointIdentity: firstIdentity,
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
    const changedIdentity = createCheckpointIdentity('commit-two');
    await runCheckpointedLiveProfileEvaluationMatrix({
      checkpointDirectory,
      profiles: [
        {
          checkpointIdentity: changedIdentity,
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
    assert.equal(runCount, 2);
    assert.deepEqual(progress, [
      'Starting live profile embedded-cpu (1/1)',
      'Completed live profile embedded-cpu (1/1)',
    ]);
  } finally {
    await rm(checkpointDirectory, { recursive: true, force: true });
  }
});

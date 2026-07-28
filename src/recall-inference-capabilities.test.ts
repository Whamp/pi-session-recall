import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallInferenceBackend } from './enums.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  createRecallRerankingExecutionIdentity,
} from './recall-inference-capabilities.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

void test('capability identities include adapter policy without changing model profile identity', () => {
  const plannerProfile = createRecommendedQmdQueryPlanningModelProfile();

  assert.deepEqual(
    createRecallRerankingExecutionIdentity(
      'reranker-profile',
      'custom-reranker-v1',
      RecallInferenceBackend.CUSTOM,
    ),
    {
      adapterId: 'custom-reranker-v1',
      backend: RecallInferenceBackend.CUSTOM,
      cacheIdentity: 'reranker-profile:custom-reranker-v1',
      modelProfileId: 'reranker-profile',
    },
  );
  assert.match(
    createRecallQueryPlanningExecutionIdentity(
      plannerProfile,
      'custom-planner-v1',
      'custom-planner-configuration-v1',
      RecallInferenceBackend.CUSTOM,
      1_000,
    ).cacheIdentity,
    /custom-planner-v1.*qmd-query-expansion-no-think-v1/u,
  );
});

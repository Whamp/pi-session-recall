import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallInferenceBackend } from './enums.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  createRecallRerankingExecutionIdentity,
  normalizeRecallPhysicalDeviceIdentity,
} from './recall-inference-capabilities.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

void test('capability identities include adapter policy without changing model profile identity', () => {
  const plannerProfile = createRecommendedQmdQueryPlanningModelProfile();

  const rerankerIdentity = createRecallRerankingExecutionIdentity(
    createRecommendedQwenRerankingModelProfile(),
    'custom-reranker-v1',
    'custom-reranker-configuration-v1',
    RecallInferenceBackend.CUSTOM,
    1_000,
  );
  assert.equal(rerankerIdentity.adapterId, 'custom-reranker-v1');
  assert.equal(rerankerIdentity.backend, RecallInferenceBackend.CUSTOM);
  assert.match(rerankerIdentity.cacheIdentity, /recall-reranking-execution-v1/u);
  const changedRerankerProfileIdentity = createRecallRerankingExecutionIdentity(
    {
      ...createRecommendedQwenRerankingModelProfile(),
      scoreRange: { minimum: 0.1, maximum: 1 },
    },
    'custom-reranker-v1',
    'custom-reranker-configuration-v1',
    RecallInferenceBackend.CUSTOM,
    1_000,
  );
  assert.notEqual(
    rerankerIdentity.modelProfileIdentity,
    changedRerankerProfileIdentity.modelProfileIdentity,
  );
  const plannerIdentity = createRecallQueryPlanningExecutionIdentity(
    plannerProfile,
    'custom-planner-v1',
    'custom-planner-configuration-v1',
    RecallInferenceBackend.CUSTOM,
    1_000,
  );
  assert.match(plannerIdentity.cacheIdentity, /^recall-query-planning-execution-v1:[a-f0-9]{64}$/u);
  assert.equal(plannerIdentity.adapterVersion, 'custom-planner-v1');
  assert.match(plannerIdentity.modelProfileIdentity, /^recall-query-planning-model-profile-v1:/u);
});

void test('physical device identity normalizes case, whitespace, duplicates, and order', () => {
  assert.deepEqual(normalizeRecallPhysicalDeviceIdentity(['  NVIDIA   RTX 4090 ', 'cpu', 'CPU']), [
    'cpu',
    'nvidia rtx 4090',
  ]);
});

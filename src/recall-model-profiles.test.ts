import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

void test('recommended model profiles retain distinct immutable semantic identities', () => {
  assert.deepEqual(
    [
      createRecommendedEmbeddingGemmaModelProfile().profileId,
      createRecommendedQwenRerankingModelProfile().profileId,
      createRecommendedQmdQueryPlanningModelProfile().profileId,
    ],
    [
      'embeddinggemma-300m-q8-0-v1',
      'qwen3-reranker-0.6b-q8-0-v1',
      'qmd-query-expansion-1.7b-q4-k-m-v1',
    ],
  );
});

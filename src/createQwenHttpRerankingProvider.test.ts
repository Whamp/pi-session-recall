import assert from 'node:assert/strict';
import test from 'node:test';

import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import { createRecommendedQwenRerankingModelProfile } from './recall-model-profiles.js';

void test('Qwen HTTP reranking provider rejects a non-HTTP endpoint', () => {
  assert.throws(
    () =>
      createQwenHttpRerankingProvider(createRecommendedQwenRerankingModelProfile(), {
        baseUrl: 'file:///models',
      }),
    /reranker base URL invalid protocol/u,
  );
});

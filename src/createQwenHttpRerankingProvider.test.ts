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
    /Qwen HTTP reranking base URL invalid protocol/u,
  );
});

void test('Qwen HTTP reranking identity normalizes endpoints and binds timeout changes', () => {
  const profile = createRecommendedQwenRerankingModelProfile();
  const first = createQwenHttpRerankingProvider(profile, {
    baseUrl: 'HTTP://RERANKER.TEST:80/v1/',
    requestTimeoutMilliseconds: 1_000,
  });
  const equivalent = createQwenHttpRerankingProvider(profile, {
    baseUrl: 'http://reranker.test/v1',
    requestTimeoutMilliseconds: 1_000,
  });
  const changedTimeout = createQwenHttpRerankingProvider(profile, {
    baseUrl: 'http://reranker.test/v1',
    requestTimeoutMilliseconds: 2_000,
  });

  assert.equal(first.executionIdentity.adapterVersion, '1');
  assert.notEqual(first.executionIdentity.adapterVersion, first.executionIdentity.adapterId);
  assert.deepEqual(first.executionIdentity, equivalent.executionIdentity);
  assert.notEqual(
    first.executionIdentity.cacheIdentity,
    changedTimeout.executionIdentity.cacheIdentity,
  );
});

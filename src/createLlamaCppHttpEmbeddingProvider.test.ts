import assert from 'node:assert/strict';
import test from 'node:test';

import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

void test('llama.cpp HTTP embedding provider rejects a non-HTTP endpoint', () => {
  assert.throws(
    () =>
      createLlamaCppHttpEmbeddingProvider(createRecommendedEmbeddingGemmaModelProfile(), {
        baseUrl: 'file:///models',
      }),
    /llama\.cpp HTTP embedding base URL invalid protocol/u,
  );
});

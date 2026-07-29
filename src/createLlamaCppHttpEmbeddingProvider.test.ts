import assert from 'node:assert/strict';
import test from 'node:test';

import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

void test('llama.cpp HTTP embedding provider rejects a non-HTTP endpoint', async () => {
  const provider = createLlamaCppHttpEmbeddingProvider(
    createRecommendedEmbeddingGemmaModelProfile(),
    { baseUrl: 'file:///models' },
  );

  await assert.rejects(() => provider.embedQuery('source provenance'), /fetch failed/u);
});

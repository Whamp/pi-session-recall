import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallInferenceBackend, RecallInferenceCapability } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecommendedEmbeddingGemmaHttpInferenceCandidate } from './recommended-embeddinggemma-inference-candidate.js';

void test('recommended HTTP embedding candidate preserves the EmbeddingGemma semantic identity', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: {
      PI_RECALL_DATA_DIRECTORY: '/recall/data',
      PI_RECALL_EMBEDDING_BASE_URL: 'http://embedding.test/v1',
      PI_RECALL_EMBEDDING_BATCH_SIZE: '8',
    },
  });
  const candidate = createRecommendedEmbeddingGemmaHttpInferenceCandidate(config);

  assert.equal(candidate.capability, RecallInferenceCapability.EMBEDDING);
  assert.equal(candidate.backend, RecallInferenceBackend.LLAMA_CPP_HTTP);
  assert.equal(candidate.profileId, 'embeddinggemma-300m-q8-0-v1');
  assert.equal(candidate.endpoint, 'http://embedding.test/v1');
});

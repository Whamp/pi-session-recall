import assert from 'node:assert/strict';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecommendedEmbeddingGemmaConversationRuntime } from './recommended-embeddinggemma-conversation-service.js';

void test('recommended EmbeddingGemma runtime stays lazy until a model operation is requested', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: { PI_RECALL_DATA_DIRECTORY: '/recall/data' },
  });
  let nativeLoadCount = 0;
  const runtime = createRecommendedEmbeddingGemmaConversationRuntime(config, {
    async loadNodeLlamaCpp() {
      nativeLoadCount += 1;
      throw new Error('native runtime should remain lazy');
    },
  });

  assert.equal(runtime.executionIdentity.computeBackend, 'pending');
  assert.equal(nativeLoadCount, 0);
  await runtime.dispose();
  assert.equal(nativeLoadCount, 0);
});

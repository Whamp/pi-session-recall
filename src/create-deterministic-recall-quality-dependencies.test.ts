import assert from 'node:assert/strict';
import test from 'node:test';

import { createDeterministicRecallQualityDependencies } from './create-deterministic-recall-quality-dependencies.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';

void test('deterministic quality dependencies are stable and network-free', async () => {
  const dependencies = createDeterministicRecallQualityDependencies();
  const texts = [
    RECALL_EMBEDDING_CANARY_TEXT,
    'How do queued deliveries survive a worker crash?',
    'Use an append-only SQLite outbox with remote acknowledgement.',
  ];
  const first = await dependencies.embeddingProvider?.embedDocuments(texts);
  const second = await dependencies.embeddingProvider?.embedDocuments(texts);
  const tokenizer = await dependencies.loadTokenizer?.();

  assert.deepEqual(second, first);
  assert.deepEqual(await dependencies.embeddingProvider?.embedQuery(texts[1] ?? ''), first?.[1]);
  assert.equal(
    first?.every((vector) => vector.length === 64),
    true,
  );
  assert.deepEqual(tokenizer?.encodeConversationText('three fixed tokens').ids, [0, 1, 2]);
  assert.ok((first?.[1]?.[0] ?? 0) > 0.9);
  assert.ok((first?.[2]?.[0] ?? 0) > 0.9);
});

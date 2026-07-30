import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service creates three independent real disposable zvec stores', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-stores-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const generationId = 'generation_stores_owner';
  await createRecallBackgroundIndexWorkerFixtureService(config).createEmptyRecallGeneration({
    generationId,
  });
  const generationDirectory = join(config.generationRootDirectory, generationId);
  const lexical = ZVecOpen(join(generationDirectory, 'lexical-source'), { readOnly: true });
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  const projections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(lexical.schema.vectors().length, 0);
    assert.equal(dense.schema.vectors().length, 1);
    assert.equal(projections.schema.vectors().length, 0);
    assert.notEqual(lexical.schema.name, dense.schema.name);
  } finally {
    lexical.closeSync();
    dense.closeSync();
    projections.closeSync();
  }
});

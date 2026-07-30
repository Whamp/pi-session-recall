import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service exposes a generation only with its matching successful validation receipt', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-receipt-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const created = await service.createEmptyRecallGeneration({
    generationId: 'generation_receipt_owner',
  });
  const receipt: unknown = JSON.parse(await readFile(created.validationReceiptPath, 'utf8'));
  assert.ok(isUnknownRecord(receipt));
  assert.equal(receipt.generationId, created.generationId);
  assert.equal(receipt.manifestFingerprint, created.manifestFingerprint);
  assert.equal(receipt.successful, true);
  assert.deepEqual(await service.openValidatedRecallGeneration(created.generationId), created);
});

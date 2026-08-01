import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service persists one immutable manifest before exposing a generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-manifest-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const created = await createRecallBackgroundIndexWorkerFixtureService(
    config,
  ).createEmptyRecallGeneration({ generationId: 'generation_manifest_owner' });
  const manifest: unknown = JSON.parse(await readFile(created.manifestPath, 'utf8'));
  assert.ok(isUnknownRecord(manifest));
  assert.equal(manifest.generationId, created.generationId);
  assert.equal(manifest.generationFormatVersion, 1);
  assert.ok(isUnknownRecord(manifest.stores));
  assert.ok(isUnknownRecord(manifest.stores.dense));
  assert.ok(Array.isArray(manifest.stores.dense.vectorFields));
  const denseVectorField: unknown = manifest.stores.dense.vectorFields[0];
  assert.ok(isUnknownRecord(denseVectorField));
  assert.ok(isUnknownRecord(denseVectorField.index));
  assert.equal(denseVectorField.index.metric, 'inner-product');
  assert.equal(created.manifestFingerprint.length, 64);
});

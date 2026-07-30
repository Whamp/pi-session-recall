import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service builds one fixed snapshot into complete disposable generation stores', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const dataDirectory = join(root, 'data');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'fixed.jsonl');
  await writeFile(
    sessionPath,
    `${JSON.stringify({ type: 'session', version: 3, id: 'fixed', timestamp: '2026-08-15T00:00:00.000Z', cwd: '/fixture' })}\n${JSON.stringify({ type: 'message', id: 'entry', parentId: null, timestamp: '2026-08-15T00:00:01.000Z', message: { role: 'assistant', content: 'fixed snapshot ownership evidence' } })}\n`,
  );
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);

  const built = await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_fixed_snapshot_owner',
    physicalSessionPaths: [sessionPath],
  });

  assert.equal(built.generationId, 'generation_fixed_snapshot_owner');
  assert.ok(built.storeCounts.lexicalSource > 0);
  assert.ok(built.storeCounts.dense > 0);
  assert.ok(built.storeCounts.sessionProjection > 0);
  assert.ok(built.startingSnapshotFingerprint.length > 0);
});

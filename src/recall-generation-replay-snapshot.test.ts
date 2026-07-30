import assert from 'node:assert/strict';
import { access, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured activation persists one fixed replay snapshot before serving the generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-replay-snapshot-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_replay_snapshot_owner';
  await service.createEmptyRecallGeneration({ generationId });
  await service.activateValidatedRecallGeneration(generationId);
  const status = await service.readOperatorStatus();
  assert.equal(status.replay.state, 'pending');
  assert.ok(status.replay.snapshotFileName);
  await access(join(config.generationRootDirectory, generationId, status.replay.snapshotFileName));
});

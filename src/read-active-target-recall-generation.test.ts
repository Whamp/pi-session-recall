import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured service reads only the pointer-and-registry-selected target generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-active-target-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_active_target_owner';
  await service.createEmptyRecallGeneration({ generationId });
  await service.activateValidatedRecallGeneration(generationId);
  const status = await service.readOperatorStatus();
  assert.equal(status.activeGeneration?.generationId, generationId);
  assert.equal(status.readiness, 'ready');
  assert.deepEqual(await service.searchRecallGenerationLexical(generationId, 'absent', 5), []);
});

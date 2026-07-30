import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured rollback accepts a healthy retained target generation through its bounded check', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-rollback-health-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  await service.createEmptyRecallGeneration({ generationId: 'generation_rollback_health_a' });
  await service.activateValidatedRecallGeneration('generation_rollback_health_a');
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await service.createEmptyRecallGeneration({ generationId: 'generation_rollback_health_b' });
  await service.activateValidatedRecallGeneration('generation_rollback_health_b');
  assert.equal(await service.completeRecallGenerationReplay(), true);
  const rollback = await service.rollback();
  assert.equal(rollback.activeGenerationId, 'generation_rollback_health_a');
  assert.equal(rollback.rollbackGenerationId, 'generation_rollback_health_b');
});

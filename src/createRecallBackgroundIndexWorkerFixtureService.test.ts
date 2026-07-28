import assert from 'node:assert/strict';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';

void test('background worker test utility creates an isolated fixture service', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: { PI_RECALL_DATA_DIRECTORY: '/tmp/recall-worker-fixture' },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);

  assert.deepEqual(await service.readIndexGenerationStatus(), { active: null, staging: null });
});

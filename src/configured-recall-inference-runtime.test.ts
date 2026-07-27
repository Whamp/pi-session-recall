import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveRecallInferenceConfigurationPath } from './configured-recall-inference-runtime.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('configured runtime resolves inference state beside the index manifest', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: { PI_RECALL_DATA_DIRECTORY: '/recall/data' },
  });

  assert.equal(
    resolveRecallInferenceConfigurationPath(config),
    '/recall/data/inference-configuration.json',
  );
});

import assert from 'node:assert/strict';
import test from 'node:test';

import { applyRecallQualityPolicyToConversationConfig } from './applyRecallQualityPolicyToConversationConfig.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('selected quality policy updates chunk and candidate limits only', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: {},
  });
  const updated = applyRecallQualityPolicyToConversationConfig(config, {
    automatedGatePassed: true,
    selectedPolicy: {
      chunkPolicy: { id: '256-32', maxTokens: 256, overlapTokens: 32 },
      candidateCount: 6,
      finalCount: 4,
    },
    blockers: [],
  });

  assert.deepEqual(updated.chunkPolicy, { maxTokens: 256, overlapTokens: 32 });
  assert.deepEqual(updated.searchCandidateLimits, { dense: 6, lexical: 6, identifier: 6 });
  assert.equal(updated.manifestPath, config.manifestPath);
});

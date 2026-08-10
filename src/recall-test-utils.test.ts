import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createTestRankedRecallDenseSearchResult,
  createTestSessionConversationChunk,
} from './recall-test-utils.js';

void test('recall test fixtures derive source geometry and preserve explicit overrides', () => {
  const chunk = createTestSessionConversationChunk({
    id: 'fixture',
    content: 'fixture text',
    role: 'user',
  });
  const result = createTestRankedRecallDenseSearchResult({
    id: 'result',
    content: 'result text',
    rankingScore: 0.75,
  });

  assert.equal(chunk.characterEnd, 12);
  assert.equal(chunk.role, 'user');
  assert.equal(result.rankingScore, 0.75);
  assert.equal(result.content, 'result text');
});

import assert from 'node:assert/strict';
import test from 'node:test';

import { assertRecallChunkPolicy } from './recall-chunk-policy.js';

void test('recall chunk policy accepts production bounds and rejects equal overlap', () => {
  assert.doesNotThrow(() => assertRecallChunkPolicy({ maxTokens: 1_024, overlapTokens: 128 }));
  assert.throws(
    () => assertRecallChunkPolicy({ maxTokens: 128, overlapTokens: 128 }),
    /Recall chunk policy invalid/,
  );
});

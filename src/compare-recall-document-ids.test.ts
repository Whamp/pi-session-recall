import assert from 'node:assert/strict';
import test from 'node:test';

import { compareRecallDocumentIds } from './compare-recall-document-ids.js';

void test('recall document IDs use deterministic bytewise ordering', () => {
  assert.equal(compareRecallDocumentIds('chunk-a', 'chunk-b'), -1);
  assert.equal(compareRecallDocumentIds('chunk-b', 'chunk-a'), 1);
  assert.equal(compareRecallDocumentIds('chunk-a', 'chunk-a'), 0);
});

import assert from 'node:assert/strict';
import test from 'node:test';

import {
  convertNormalizedRecallInnerProductToCosineDistance,
  createStoredRecallEmbedding,
} from './recall-stored-embedding.js';

void test('stored recall embeddings retain the configured native prefix and L2-normalize it', () => {
  assert.deepEqual(createStoredRecallEmbedding([3, 4, 0, 8], 4, 2), [
    Math.fround(0.6),
    Math.fround(0.8),
  ]);
});

void test('stored recall embeddings reject invalid native and stored dimensions', () => {
  assert.throws(
    () => createStoredRecallEmbedding([1, 2], 3, 2),
    /Recall native embedding dimension mismatch: expected 3, received 2/,
  );
  assert.throws(
    () => createStoredRecallEmbedding([1, 2], 2, 3),
    /Recall stored embedding dimensions invalid: expected an integer from 1 to 2, received 3/,
  );
});

void test('normalized inner-product scores preserve cosine-distance semantics', () => {
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(1), 0);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(0.25), 0.75);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(-1), 2);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(1.000_001), 0);
});

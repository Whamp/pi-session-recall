import assert from 'node:assert/strict';
import test from 'node:test';

import fc from 'fast-check';

import {
  assertRepeatableStoredRecallEmbeddings,
  convertNormalizedRecallInnerProductToCosineDistance,
  createStoredRecallEmbedding,
} from './recall-stored-embedding.js';

void test('stored recall embeddings retain the first N components at unit L2 norm', () => {
  fc.assert(
    fc.property(
      fc.array(fc.integer({ min: -1_000, max: 1_000 }), {
        minLength: 1,
        maxLength: 32,
      }),
      fc.integer({ min: 1, max: 32 }),
      (nativeVector, requestedStoredDimensions) => {
        fc.pre(nativeVector.some((value) => value !== 0));
        const storedDimensions = Math.min(requestedStoredDimensions, nativeVector.length);
        fc.pre(nativeVector.slice(0, storedDimensions).some((value) => value !== 0));
        const stored = createStoredRecallEmbedding(nativeVector, {
          nativeDimensions: nativeVector.length,
          storedDimensions,
          source: 'property-fixture',
        });
        assert.equal(stored.length, storedDimensions);
        assert.ok(Math.abs(Math.hypot(...stored) - 1) <= 1e-12);
        const firstNativeValue = nativeVector[0];
        const firstStoredValue = stored[0];
        assert.ok(firstNativeValue !== undefined);
        assert.ok(firstStoredValue !== undefined);
        for (let index = 0; index < stored.length; index += 1) {
          const nativeValue = nativeVector[index];
          const storedValue = stored[index];
          assert.ok(nativeValue !== undefined);
          assert.ok(storedValue !== undefined);
          assert.ok(
            Math.abs(storedValue * firstNativeValue - nativeValue * firstStoredValue) <= 1e-10,
          );
        }
      },
    ),
  );
});

void test('normalized recall inner product converts to bounded cosine distance', () => {
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(1), 0);
  assert.ok(Math.abs(convertNormalizedRecallInnerProductToCosineDistance(0.8) - 0.2) <= 1e-12);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(0), 1);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(-1), 2);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(1 + Number.EPSILON), 0);
  assert.equal(convertNormalizedRecallInnerProductToCosineDistance(-1 - Number.EPSILON), 2);
  assert.throws(
    () => convertNormalizedRecallInnerProductToCosineDistance(Number.NaN),
    /Recall normalized inner product invalid: score must be finite/u,
  );

  fc.assert(
    fc.property(
      fc.double({ min: -1.000_001, max: 1.000_001, noNaN: true, noDefaultInfinity: true }),
      fc.double({ min: -1.000_001, max: 1.000_001, noNaN: true, noDefaultInfinity: true }),
      (left, right) => {
        const leftDistance = convertNormalizedRecallInnerProductToCosineDistance(left);
        const rightDistance = convertNormalizedRecallInnerProductToCosineDistance(right);
        assert.ok(leftDistance >= 0 && leftDistance <= 2);
        assert.ok(rightDistance >= 0 && rightDistance <= 2);
        if (left <= right) {
          assert.ok(leftDistance >= rightDistance);
        }
      },
    ),
  );
});

void test('stored recall embedding validation rejects malformed and non-repeatable provider output', () => {
  assert.throws(
    () =>
      createStoredRecallEmbedding([1, 2], {
        nativeDimensions: 3,
        storedDimensions: 2,
        source: 'query canary',
      }),
    /Recall stored embedding invalid for query canary: expected native width 3, received 2/u,
  );
  assert.throws(
    () =>
      createStoredRecallEmbedding([1, Number.NaN], {
        nativeDimensions: 2,
        storedDimensions: 2,
        source: 'document occurrence_1',
      }),
    /Recall stored embedding invalid for document occurrence_1: dimension 1 is not finite/u,
  );
  assert.throws(
    () =>
      createStoredRecallEmbedding([1, 2, Number.POSITIVE_INFINITY], {
        nativeDimensions: 3,
        storedDimensions: 2,
        source: 'document occurrence_2',
      }),
    /Recall stored embedding invalid for document occurrence_2: dimension 2 is not finite/u,
  );
  assert.throws(
    () =>
      createStoredRecallEmbedding([0, 0, 1], {
        nativeDimensions: 3,
        storedDimensions: 2,
        source: 'query canary',
      }),
    /Recall stored embedding invalid for query canary: retained prefix norm must be positive/u,
  );
  assert.throws(
    () =>
      assertRepeatableStoredRecallEmbeddings([1, 0], [0, 1], {
        minimumCosineSimilarity: 0.9995,
        source: 'query canary',
      }),
    /Recall stored embedding repeatability mismatch for query canary/u,
  );
});

import assert from 'node:assert/strict';
import test from 'node:test';

import fc from 'fast-check';

import { openRecallZvecValidationStore } from './open-recall-zvec-validation-store.js';

void test('zvec validation open retries every transient lock failure and returns the opened store', async () => {
  await fc.assert(
    fc.asyncProperty(fc.integer({ min: 0, max: 4 }), async (transientFailureCount) => {
      const openedStore = { kind: 'opened-validation-store' } as const;
      let openAttemptCount = 0;

      const result = await openRecallZvecValidationStore(() => {
        openAttemptCount += 1;
        if (openAttemptCount <= transientFailureCount) {
          throw new Error("Can't lock read-only collection: /tmp/generated-zvec/LOCK");
        }
        return openedStore;
      });

      assert.equal(result, openedStore);
      assert.equal(openAttemptCount, transientFailureCount + 1);
    }),
    { numRuns: 20 },
  );
});

void test('zvec validation open preserves unrelated failures without retrying', async () => {
  await fc.assert(
    fc.asyncProperty(fc.string(), async (failureDetail) => {
      const permanentFailure = new Error(`Recall validation failed permanently: ${failureDetail}`);
      let openAttemptCount = 0;

      await assert.rejects(
        openRecallZvecValidationStore(() => {
          openAttemptCount += 1;
          throw permanentFailure;
        }),
        (error: unknown) => error === permanentFailure,
      );
      assert.equal(openAttemptCount, 1);
    }),
    { numRuns: 20 },
  );
});

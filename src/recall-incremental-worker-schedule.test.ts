import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallEligibilityThreshold } from './enums.js';
import {
  persistRecallIncrementalWorkerSchedule,
  readRecallIncrementalWorkerSchedule,
  type RecallLargeTransferDeferral,
} from './recall-incremental-worker-schedule.js';

void test('incremental worker schedule persists the earliest future wake and large-transfer deferral', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-worker-schedule-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const schedulePath = join(directory, 'control', 'worker-schedule.json');
  const largeTransferDeferral: RecallLargeTransferDeferral = {
    physicalSessionId: 'physical-1',
    sourceModifiedAtEpochMilliseconds: 1_000,
    sourceMarkerIds: ['marker-1'],
    threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
    readyAtEpochMilliseconds: 5_000,
  };

  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 1_000,
      schedule: {
        version: 1,
        nextWakeAtEpochMilliseconds: 5_000,
        largeTransferDeferrals: [largeTransferDeferral],
      },
    }),
    true,
  );
  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 1_000,
      schedule: {
        version: 1,
        nextWakeAtEpochMilliseconds: 8_000,
        largeTransferDeferrals: [largeTransferDeferral],
      },
    }),
    true,
  );

  assert.deepEqual(await readRecallIncrementalWorkerSchedule(schedulePath), {
    version: 1,
    nextWakeAtEpochMilliseconds: 5_000,
    largeTransferDeferrals: [largeTransferDeferral],
  });
  assert.match(await readFile(schedulePath, 'utf8'), /^\{"version":1,/u);
  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 1_000,
      schedule: {
        version: 1,
        nextWakeAtEpochMilliseconds: null,
        largeTransferDeferrals: [],
      },
    }),
    false,
  );
  assert.equal(
    (await readRecallIncrementalWorkerSchedule(schedulePath))?.nextWakeAtEpochMilliseconds,
    null,
  );
});

void test('incremental worker schedule re-arms an equal future wake after a sleeper crash gap', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-worker-schedule-rearm-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const schedulePath = join(directory, 'worker-schedule.json');
  const schedule = {
    version: 1 as const,
    nextWakeAtEpochMilliseconds: 5_000,
    largeTransferDeferrals: [],
  };

  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 1_000,
      schedule,
    }),
    true,
  );
  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 1_000,
      schedule,
    }),
    true,
  );
});

void test('incremental worker schedule replaces an elapsed wake with the next deadline', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-worker-schedule-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const schedulePath = join(directory, 'worker-schedule.json');
  await persistRecallIncrementalWorkerSchedule({
    schedulePath,
    nowEpochMilliseconds: 1_000,
    schedule: {
      version: 1,
      nextWakeAtEpochMilliseconds: 2_000,
      largeTransferDeferrals: [],
    },
  });

  assert.equal(
    await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: 2_000,
      schedule: {
        version: 1,
        nextWakeAtEpochMilliseconds: 5_000,
        largeTransferDeferrals: [],
      },
    }),
    true,
  );
  assert.equal(
    (await readRecallIncrementalWorkerSchedule(schedulePath))?.nextWakeAtEpochMilliseconds,
    5_000,
  );
});

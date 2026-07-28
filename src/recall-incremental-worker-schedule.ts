import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdir, open, readFile, rename, rm } from 'node:fs/promises';
import { dirname } from 'node:path';

import { RecallEligibilityThreshold } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Current strict version for the generation-independent incremental worker schedule. */
export const RECALL_INCREMENTAL_WORKER_SCHEDULE_VERSION = 1;

/** One prepared large transfer retained until its measured quiet deadline. */
export interface RecallLargeTransferDeferral {
  physicalSessionId: string;
  sourceModifiedAtEpochMilliseconds: number;
  sourceMarkerIds: string[];
  threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER;
  readyAtEpochMilliseconds: number;
}

/** Durable generation-independent wake deadline and prepared-transfer deferrals. */
export interface RecallIncrementalWorkerSchedule {
  version: 1;
  nextWakeAtEpochMilliseconds: number | null;
  largeTransferDeferrals: RecallLargeTransferDeferral[];
}

/** Atomic schedule replacement input using one observed wall-clock instant. */
export interface PersistRecallIncrementalWorkerScheduleOptions {
  schedulePath: string;
  nowEpochMilliseconds: number;
  schedule: RecallIncrementalWorkerSchedule;
}

/** Production paths needed to signal one detached worker at a persisted deadline. */
export interface SignalRecallIncrementalWorkerWakeOptions {
  readyAtEpochMilliseconds: number;
  workerOwnershipLockPath: string;
  workerExecutablePath: string;
  nowEpochMilliseconds?: () => number;
}

function parseNonnegativeInteger(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw new Error(`Recall incremental worker schedule ${field} invalid`);
  }
  return Number(value);
}

function parseLargeTransferDeferral(value: unknown): RecallLargeTransferDeferral {
  if (
    !isUnknownRecord(value) ||
    typeof value.physicalSessionId !== 'string' ||
    value.physicalSessionId.length === 0 ||
    !Array.isArray(value.sourceMarkerIds) ||
    value.threshold !== RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER
  ) {
    throw new Error('Recall incremental worker schedule large transfer deferral invalid');
  }
  const sourceMarkerIds: string[] = [];
  for (const markerIdValue of value.sourceMarkerIds) {
    const markerId: unknown = markerIdValue;
    if (typeof markerId !== 'string' || markerId.length === 0) {
      throw new Error('Recall incremental worker schedule marker ID invalid');
    }
    sourceMarkerIds.push(markerId);
  }
  return {
    physicalSessionId: value.physicalSessionId,
    sourceModifiedAtEpochMilliseconds: parseNonnegativeInteger(
      value.sourceModifiedAtEpochMilliseconds,
      'source modified time',
    ),
    sourceMarkerIds: [...new Set(sourceMarkerIds)].toSorted((left, right) =>
      left.localeCompare(right),
    ),
    threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
    readyAtEpochMilliseconds: parseNonnegativeInteger(
      value.readyAtEpochMilliseconds,
      'large transfer deadline',
    ),
  };
}

function parseRecallIncrementalWorkerSchedule(value: unknown): RecallIncrementalWorkerSchedule {
  if (
    !isUnknownRecord(value) ||
    value.version !== RECALL_INCREMENTAL_WORKER_SCHEDULE_VERSION ||
    !Array.isArray(value.largeTransferDeferrals)
  ) {
    throw new Error('Recall incremental worker schedule invalid');
  }
  const nextWakeAtEpochMilliseconds =
    value.nextWakeAtEpochMilliseconds === null
      ? null
      : parseNonnegativeInteger(value.nextWakeAtEpochMilliseconds, 'wake deadline');
  const largeTransferDeferrals = value.largeTransferDeferrals.map(parseLargeTransferDeferral);
  const physicalSessionIds = largeTransferDeferrals.map(
    ({ physicalSessionId }) => physicalSessionId,
  );
  if (new Set(physicalSessionIds).size !== physicalSessionIds.length) {
    throw new Error('Recall incremental worker schedule contains duplicate physical sessions');
  }
  return {
    version: RECALL_INCREMENTAL_WORKER_SCHEDULE_VERSION,
    nextWakeAtEpochMilliseconds,
    largeTransferDeferrals: largeTransferDeferrals.toSorted((left, right) =>
      left.physicalSessionId.localeCompare(right.physicalSessionId),
    ),
  };
}

/** Reads the durable worker schedule, returning null only before the first deferral. */
export async function readRecallIncrementalWorkerSchedule(
  schedulePath: string,
): Promise<RecallIncrementalWorkerSchedule | null> {
  try {
    return parseRecallIncrementalWorkerSchedule(JSON.parse(await readFile(schedulePath, 'utf8')));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function writeRecallIncrementalWorkerSchedule(
  schedulePath: string,
  schedule: RecallIncrementalWorkerSchedule,
): Promise<void> {
  const parsed = parseRecallIncrementalWorkerSchedule(schedule);
  const directoryPath = dirname(schedulePath);
  await mkdir(directoryPath, { recursive: true });
  const temporaryPath = `${schedulePath}.${randomUUID()}.tmp`;
  const file = await open(temporaryPath, 'wx', 0o600);
  try {
    await file.writeFile(`${JSON.stringify(parsed)}\n`);
    await file.sync();
  } finally {
    await file.close();
  }
  try {
    await rename(temporaryPath, schedulePath);
    await syncRecallDirectory(directoryPath);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

/** Atomically persists the earliest future wake and returns whether a new signal is needed. */
export async function persistRecallIncrementalWorkerSchedule(
  options: PersistRecallIncrementalWorkerScheduleOptions,
): Promise<boolean> {
  const current = await readRecallIncrementalWorkerSchedule(options.schedulePath);
  const currentFutureWake =
    current?.nextWakeAtEpochMilliseconds !== null &&
    current?.nextWakeAtEpochMilliseconds !== undefined &&
    current.nextWakeAtEpochMilliseconds > options.nowEpochMilliseconds
      ? current.nextWakeAtEpochMilliseconds
      : null;
  const proposedWake = options.schedule.nextWakeAtEpochMilliseconds;
  const nextWakeAtEpochMilliseconds =
    proposedWake === null
      ? null
      : currentFutureWake === null
        ? proposedWake
        : Math.min(currentFutureWake, proposedWake);
  await writeRecallIncrementalWorkerSchedule(options.schedulePath, {
    ...options.schedule,
    nextWakeAtEpochMilliseconds,
  });
  return nextWakeAtEpochMilliseconds !== null && nextWakeAtEpochMilliseconds !== currentFutureWake;
}

/** Starts one detached sleep followed by a blocking kernel-lock worker wake. */
export function signalRecallIncrementalWorkerWake(
  options: SignalRecallIncrementalWorkerWakeOptions,
): void {
  const nowEpochMilliseconds = options.nowEpochMilliseconds?.() ?? Date.now();
  const delaySeconds = Math.max(
    0,
    Math.ceil((options.readyAtEpochMilliseconds - nowEpochMilliseconds) / 1_000),
  );
  const child = spawn(
    '/bin/sh',
    [
      '-c',
      'sleep "$1"; exec /usr/bin/flock "$2" "$3" --import tsx "$4"',
      'recall-worker-wake',
      String(delaySeconds),
      options.workerOwnershipLockPath,
      process.execPath,
      options.workerExecutablePath,
    ],
    { detached: true, stdio: 'ignore' },
  );
  child.once('error', (error) => {
    process.emitWarning(
      `Recall scheduled worker signal failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
    );
  });
  child.unref();
}

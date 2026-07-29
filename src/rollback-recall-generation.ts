import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { access, copyFile, mkdir, open, readdir, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import { resolveRecallGenerationDirectory } from './recall-generation-state.js';
import {
  recallGenerationTransitionRequiresReplaySignal,
  rollbackRecallGenerationTransition,
} from './recall-generation-transitions.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Explicit pointer rollback inputs, including generation-independent retained marker storage. */
export interface RollbackRecallGenerationOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  generationRootDirectory: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  retainedMarkerDirectory: string;
  lockPath: string;
  workerSignal: RecallDetachedWorkerSignal;
  rollbackRetentionMilliseconds?: number;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
}

/** Explicit rollback outcome and number of markers restored for deterministic replay. */
export interface RollbackRecallGenerationResult {
  activeGenerationId: string;
  rollbackGenerationId: string;
  restoredMarkerCount: number;
}

const RETAINED_MARKER_FILE_PATTERN = /^[A-Za-z0-9_-]+\.json$/u;

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function restoreRetainedRecallMarkers(
  retainedMarkerDirectory: string,
  markerSpoolDirectory: string,
  onMarkerRestored: (restoredMarkerCount: number) => void,
): Promise<number> {
  let markerNames: string[];
  try {
    markerNames = (await readdir(retainedMarkerDirectory))
      .filter((name) => RETAINED_MARKER_FILE_PATTERN.test(name))
      .toSorted();
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return 0;
    }
    throw error;
  }
  await mkdir(markerSpoolDirectory, { recursive: true });
  let restoredMarkerCount = 0;
  for (const markerName of markerNames) {
    const destinationPath = join(markerSpoolDirectory, markerName);
    if (await pathExists(destinationPath)) {
      restoredMarkerCount += 1;
      onMarkerRestored(restoredMarkerCount);
      continue;
    }
    const temporaryPath = `${destinationPath}.${randomUUID()}.rollback.tmp`;
    try {
      await copyFile(
        join(retainedMarkerDirectory, markerName),
        temporaryPath,
        constants.COPYFILE_EXCL,
      );
      const temporaryFile = await open(temporaryPath, 'r');
      try {
        await temporaryFile.sync();
      } finally {
        await temporaryFile.close();
      }
      await rename(temporaryPath, destinationPath);
      restoredMarkerCount += 1;
      onMarkerRestored(restoredMarkerCount);
    } catch (error) {
      await rm(temporaryPath, { force: true });
      throw error;
    }
  }
  if (restoredMarkerCount > 0) {
    await syncRecallDirectory(markerSpoolDirectory);
  }
  return restoredMarkerCount;
}

/** Atomically restores the bounded rollback pointer and republishes retained markers for replay. */
export async function rollbackRecallGeneration(
  options: RollbackRecallGenerationOptions,
): Promise<RollbackRecallGenerationResult> {
  let restoredMarkerCountBeforeFailure = 0;
  let rollback: {
    result: RollbackRecallGenerationResult;
    replayRequired: boolean;
  };
  try {
    rollback = await coordinateRecallWriteWindow(
      {
        lockPath: options.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      (writeWindow) =>
        rollbackRecallGenerationTransition({
          activeGenerationPointerPath: options.activeGenerationPointerPath,
          generationRegistryPath: options.generationRegistryPath,
          backlogSummaryPath: options.backlogSummaryPath,
          ...(options.rollbackRetentionMilliseconds === undefined
            ? {}
            : { rollbackRetentionMilliseconds: options.rollbackRetentionMilliseconds }),
          ...(options.nowEpochMilliseconds
            ? { nowEpochMilliseconds: options.nowEpochMilliseconds }
            : {}),
          async validateRollbackGeneration(generationId) {
            await resolveRecallGenerationDirectory(options.generationRootDirectory, generationId);
          },
          restoreRetainedMarkers: () =>
            restoreRetainedRecallMarkers(
              options.retainedMarkerDirectory,
              options.markerSpoolDirectory,
              (restoredCount) => {
                restoredMarkerCountBeforeFailure = restoredCount;
              },
            ),
          retainRecoveryRequired() {
            writeWindow.retainRecoveryRequired();
          },
        }),
    );
  } catch (error) {
    if (restoredMarkerCountBeforeFailure > 0) {
      options.workerSignal.signalDetachedWorker();
      throw error;
    }
    if (await recallGenerationTransitionRequiresReplaySignal(options.generationRegistryPath)) {
      options.workerSignal.signalDetachedWorker();
    }
    throw error;
  }
  if (rollback.replayRequired) {
    options.workerSignal.signalDetachedWorker();
  }
  return rollback.result;
}

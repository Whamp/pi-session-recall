import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { access, copyFile, mkdir, open, readdir, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';

import { checkRecallGenerationRollbackHealth } from './check-recall-generation-rollback-health.js';
import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import { RECALL_INDEX_MANIFEST_VERSION, RecallTargetGenerationRollbackStage } from './enums.js';
import {
  listPendingRecallMarkerIds,
  listQuarantinedRecallMarkerIds,
} from './recall-generation-replay-markers.js';
import {
  RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION,
  writeRecallGenerationReplaySnapshot,
} from './recall-generation-replay-snapshot.js';
import {
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
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
  markerQuarantineDirectory: string;
  retainedMarkerDirectory: string;
  lockPath: string;
  workerSignal: RecallDetachedWorkerSignal;
  rollbackRetentionMilliseconds?: number;
  rollbackFault?: (stage: RecallTargetGenerationRollbackStage) => void | Promise<void>;
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
  const inspectedRegistry = await readRecallGenerationRegistry(options.generationRegistryPath);
  const inspectedRollbackEntry = inspectedRegistry?.generations.find(
    ({ generationId }) => generationId === inspectedRegistry.rollbackGenerationId,
  );
  if (
    inspectedRollbackEntry === undefined ||
    inspectedRollbackEntry.indexManifestVersion !== RECALL_INDEX_MANIFEST_VERSION
  ) {
    throw new Error('Recall generation rollback unavailable: no retained target generation');
  }
  await checkRecallGenerationRollbackHealth({
    generationRootDirectory: options.generationRootDirectory,
    generationId: inspectedRollbackEntry.generationId,
    expectedManifestFingerprint: inspectedRollbackEntry.indexManifestFingerprint,
  });
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
            if (generationId !== inspectedRollbackEntry.generationId) {
              throw new Error(
                `Recall rollback health target changed before cutover: expected ${inspectedRollbackEntry.generationId}, received ${generationId}`,
              );
            }
          },
          async prepareRollbackReplay(generationId) {
            const generationDirectory = await resolveRecallGenerationDirectory(
              options.generationRootDirectory,
              generationId,
            );
            const restoredMarkerCount = await restoreRetainedRecallMarkers(
              options.retainedMarkerDirectory,
              options.markerSpoolDirectory,
              (restoredCount) => {
                restoredMarkerCountBeforeFailure = restoredCount;
              },
            );
            const [pendingMarkerIds, quarantinedMarkerIds] = await Promise.all([
              listPendingRecallMarkerIds(options.markerSpoolDirectory),
              listQuarantinedRecallMarkerIds(options.markerQuarantineDirectory),
            ]);
            const replaySnapshotFileName = `generation-replay-snapshot-${randomUUID()}.json`;
            await writeRecallGenerationReplaySnapshot(
              join(generationDirectory, replaySnapshotFileName),
              {
                snapshotVersion: RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION,
                generationId,
                pendingMarkerIds,
                quarantinedMarkerIds,
                capturedAtEpochMilliseconds: options.nowEpochMilliseconds?.() ?? Date.now(),
              },
            );
            return {
              restoredMarkerCount,
              replayMarkerIds: pendingMarkerIds,
              replaySnapshotFileName,
            };
          },
          async afterRollbackRegistry() {
            await options.rollbackFault?.(RecallTargetGenerationRollbackStage.AFTER_REGISTRY);
          },
          async afterRollbackPointer() {
            await options.rollbackFault?.(RecallTargetGenerationRollbackStage.AFTER_POINTER);
          },
          async afterRollbackBacklog() {
            await options.rollbackFault?.(RecallTargetGenerationRollbackStage.AFTER_BACKLOG);
          },
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

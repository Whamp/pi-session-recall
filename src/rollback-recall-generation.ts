import { randomUUID } from 'node:crypto';
import { constants } from 'node:fs';
import { access, copyFile, mkdir, open, readdir, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
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

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;
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
  return coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: false,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      const [pointer, registry] = await Promise.all([
        readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
        readRecallGenerationRegistry(options.generationRegistryPath),
      ]);
      if (!pointer || !registry || !registry.rollbackGenerationId) {
        throw new Error('Recall generation rollback unavailable: no retained rollback generation');
      }
      if (
        registry.activeGenerationId !== pointer.activeGenerationId ||
        registry.activePointerChecksum !== pointer.checksum
      ) {
        throw new Error('Recall generation rollback found pointer and registry disagreement');
      }
      if (registry.buildingGenerationId !== null) {
        throw new Error('Recall generation rollback unavailable while a replacement builds');
      }
      const rollbackEntry = registry.generations.find(
        ({ generationId }) => generationId === registry.rollbackGenerationId,
      );
      const activeEntry = registry.generations.find(
        ({ generationId }) => generationId === registry.activeGenerationId,
      );
      if (
        !rollbackEntry ||
        !activeEntry ||
        rollbackEntry.state !== RecallGenerationCutoverState.ROLLBACK
      ) {
        throw new Error('Recall generation rollback registry roles invalid');
      }
      await resolveRecallGenerationDirectory(
        options.generationRootDirectory,
        rollbackEntry.generationId,
      );
      const rolledBackAt = options.nowEpochMilliseconds?.() ?? Date.now();
      if (
        rollbackEntry.retireAfterEpochMilliseconds !== undefined &&
        rollbackEntry.retireAfterEpochMilliseconds !== null &&
        rollbackEntry.retireAfterEpochMilliseconds <= rolledBackAt
      ) {
        throw new Error('Recall generation rollback unavailable: retention period expired');
      }
      const targetState =
        rollbackEntry.indexManifestVersion === 5
          ? RecallGenerationCutoverState.LEGACY_READ_ONLY
          : RecallGenerationCutoverState.REPLAY_PENDING;
      const activeReplacement: RecallGenerationRegistryEntry = {
        ...rollbackEntry,
        state: targetState,
        stateChangedAtEpochMilliseconds: rolledBackAt,
        retireAfterEpochMilliseconds: null,
      };
      const rollbackReplacement: RecallGenerationRegistryEntry = {
        ...activeEntry,
        state: RecallGenerationCutoverState.ROLLBACK,
        stateChangedAtEpochMilliseconds: rolledBackAt,
        retireAfterEpochMilliseconds:
          rolledBackAt +
          (options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS),
      };
      const targetPointer = createRecallActiveGenerationPointer(rollbackEntry.generationId);
      const nextRegistry = {
        ...registry,
        activeGenerationId: rollbackEntry.generationId,
        rollbackGenerationId: activeEntry.generationId,
        activePointerChecksum: targetPointer.checksum,
        generations: registry.generations.map((entry) => {
          if (entry.generationId === activeReplacement.generationId) {
            return activeReplacement;
          }
          if (entry.generationId === rollbackReplacement.generationId) {
            return rollbackReplacement;
          }
          return entry;
        }),
      };
      const restoredMarkerCount = await restoreRetainedRecallMarkers(
        options.retainedMarkerDirectory,
        options.markerSpoolDirectory,
      );
      try {
        // Registry-first ordering leaves the old pointer searchable until recovery completes.
        await writeRecallGenerationRegistry(options.generationRegistryPath, nextRegistry);
        await writeRecallActiveGenerationPointer(
          options.activeGenerationPointerPath,
          targetPointer,
        );
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: restoredMarkerCount,
          oldestEligibleMarkerAgeMilliseconds: null,
          activeGenerationId: rollbackEntry.generationId,
          buildingGenerationId: null,
          generationState: targetState,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: null,
          lastFailureCategory: null,
          observedAtEpochMilliseconds: rolledBackAt,
        });
      } catch (error) {
        writeWindow.retainRecoveryRequired();
        throw error;
      }
      return {
        activeGenerationId: rollbackEntry.generationId,
        rollbackGenerationId: activeEntry.generationId,
        restoredMarkerCount,
      };
    },
  );
}

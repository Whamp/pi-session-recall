import type { Dirent } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { completeRecallGenerationReplayTransition } from './recall-generation-transitions.js';

/** Registry, spool, and scalar-warning inputs for proving replacement replay complete. */
export interface CompleteRecallGenerationReplayOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  lockPath: string;
  nowEpochMilliseconds?: () => number;
}

async function hasPendingRecallMarkers(markerSpoolDirectory: string): Promise<boolean> {
  try {
    return (await readdir(markerSpoolDirectory, { withFileTypes: true })).some(
      (entry) => entry.isFile() && !entry.name.startsWith('.') && entry.name.endsWith('.json'),
    );
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function hasQuarantinedRecallMarkers(markerQuarantineDirectory: string): Promise<boolean> {
  let categoryEntries: Dirent[];
  try {
    categoryEntries = await readdir(markerQuarantineDirectory, { withFileTypes: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
  for (const categoryEntry of categoryEntries) {
    if (!categoryEntry.isDirectory()) {
      continue;
    }
    const quarantinedEntries = await readdir(join(markerQuarantineDirectory, categoryEntry.name), {
      withFileTypes: true,
    });
    if (quarantinedEntries.some((entry) => entry.isFile())) {
      return true;
    }
  }
  return false;
}

/** Clears replay state only after the lock-held caller proves pointer agreement and empty spool. */
export async function completeRecallGenerationReplayWithinWriteWindow(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  return completeRecallGenerationReplayTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    backlogSummaryPath: options.backlogSummaryPath,
    ...(options.nowEpochMilliseconds ? { nowEpochMilliseconds: options.nowEpochMilliseconds } : {}),
    async proveReplayWorkComplete() {
      return (
        !(await hasPendingRecallMarkers(options.markerSpoolDirectory)) &&
        !(await hasQuarantinedRecallMarkers(options.markerQuarantineDirectory))
      );
    },
  });
}

/** Completes replay under the operation lock so rebuild state cannot be overwritten. */
export async function completeRecallGenerationReplay(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  return coordinateRecallWriteWindow({ lockPath: options.lockPath, allowRecovery: false }, () =>
    completeRecallGenerationReplayWithinWriteWindow(options),
  );
}

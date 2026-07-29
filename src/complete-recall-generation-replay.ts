import type { Dirent } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import { readNodeErrorCode } from './read-node-error-code.js';

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
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error(
      'Recall generation replay completion requires initialized pointer and registry',
    );
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation replay completion found pointer and registry disagreement');
  }
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  if (!activeEntry) {
    throw new Error('Recall generation replay completion active registry entry missing');
  }
  const completedAt = options.nowEpochMilliseconds?.() ?? Date.now();
  if (activeEntry.state === RecallGenerationCutoverState.ACTIVE) {
    await writeRecallBacklogSummary(options.backlogSummaryPath, {
      version: RECALL_BACKLOG_SUMMARY_VERSION,
      pendingEligibleSessionCount: 0,
      oldestEligibleMarkerAgeMilliseconds: null,
      activeGenerationId: pointer.activeGenerationId,
      buildingGenerationId: null,
      generationState: RecallGenerationCutoverState.ACTIVE,
      activeGenerationAgeMilliseconds: 0,
      rebuildAgeMilliseconds: null,
      lastFailureCategory: null,
      observedAtEpochMilliseconds: completedAt,
    });
    return true;
  }
  if (activeEntry.state !== RecallGenerationCutoverState.REPLAY_PENDING) {
    return false;
  }
  if (
    (await hasPendingRecallMarkers(options.markerSpoolDirectory)) ||
    (await hasQuarantinedRecallMarkers(options.markerQuarantineDirectory))
  ) {
    return false;
  }
  const activeReplacement: RecallGenerationRegistryEntry = {
    ...activeEntry,
    state: RecallGenerationCutoverState.ACTIVE,
    stateChangedAtEpochMilliseconds: completedAt,
  };
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...registry,
    generations: registry.generations.map((entry) =>
      entry.generationId === activeReplacement.generationId ? activeReplacement : entry,
    ),
  });
  await writeRecallBacklogSummary(options.backlogSummaryPath, {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 0,
    oldestEligibleMarkerAgeMilliseconds: null,
    activeGenerationId: pointer.activeGenerationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.ACTIVE,
    activeGenerationAgeMilliseconds: 0,
    rebuildAgeMilliseconds: null,
    lastFailureCategory: null,
    observedAtEpochMilliseconds: completedAt,
  });
  return true;
}

/** Completes replay under the operation lock so rebuild state cannot be overwritten. */
export async function completeRecallGenerationReplay(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  return coordinateRecallWriteWindow({ lockPath: options.lockPath, allowRecovery: false }, () =>
    completeRecallGenerationReplayWithinWriteWindow(options),
  );
}

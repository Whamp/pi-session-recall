import { join } from 'node:path';

import { ZVecOpen } from '@zvec/zvec';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { RecallSessionProjectionKind } from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
import { listQuarantinedRecallMarkerIds } from './recall-generation-replay-markers.js';
import {
  readRecallGenerationReplaySnapshot,
  RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
} from './recall-generation-replay-snapshot.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { completeRecallGenerationReplayTransition } from './recall-generation-transitions.js';
import { decodeRecallSessionProjection } from './recall-session-projection.js';

/** Registry, spool, and fixed-snapshot inputs for proving replacement replay complete. */
export interface CompleteRecallGenerationReplayOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  lockPath: string;
  /** Target generation root containing the immutable fixed replay snapshot. */
  generationRootDirectory: string;
  nowEpochMilliseconds?: () => number;
}

function readTargetPhysicalProjectionCoveredMarkerIds(
  generationDirectory: string,
  generationId: string,
): Set<string> {
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  const collection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  try {
    if (collection.stats.docCount === 0) {
      return new Set();
    }
    const records = collection.querySync({
      topk: collection.stats.docCount,
      outputFields: ['projectionKind', 'projectionJson'],
      includeVector: false,
    });
    const coveredMarkerIds = new Set<string>();
    for (const record of records) {
      if (
        record.fields.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION ||
        typeof record.fields.projectionJson !== 'string'
      ) {
        continue;
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(record.fields.projectionJson);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Recall generation replay physical projection JSON invalid for ${record.id}: ${message}`,
          { cause: error },
        );
      }
      if (
        typeof parsed !== 'object' ||
        parsed === null ||
        !('ingestionProjectionPayload' in parsed)
      ) {
        throw new Error(`Recall generation replay ingestion projection missing for ${record.id}`);
      }
      const projection = decodeRecallSessionProjection(parsed.ingestionProjectionPayload, {
        expectedGenerationId: generationId,
      });
      if (projection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
        throw new Error(
          `Recall generation replay physical projection kind mismatch for ${record.id}`,
        );
      }
      for (const markerId of projection.markerCheckpoint.coveredMarkerIds) {
        coveredMarkerIds.add(markerId);
      }
    }
    return coveredMarkerIds;
  } finally {
    collection.closeSync();
  }
}

async function proveFixedRecallGenerationReplayComplete(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (pointer === null) {
    throw new Error('Recall generation replay fixed-snapshot proof requires an active pointer');
  }
  const generationDirectory = await resolveRecallGenerationDirectory(
    options.generationRootDirectory,
    pointer.activeGenerationId,
  );
  const activeEntry = registry?.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  const snapshot = await readRecallGenerationReplaySnapshot(
    join(
      generationDirectory,
      activeEntry?.replaySnapshotFileName ?? RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
    ),
  );
  if (snapshot.generationId !== pointer.activeGenerationId) {
    throw new Error(
      `Recall generation replay snapshot identity mismatch: expected ${pointer.activeGenerationId}, received ${snapshot.generationId}`,
    );
  }
  const coveredMarkerIds = readTargetPhysicalProjectionCoveredMarkerIds(
    generationDirectory,
    pointer.activeGenerationId,
  );
  const quarantinedMarkerIds = new Set(
    await listQuarantinedRecallMarkerIds(options.markerQuarantineDirectory),
  );
  return (
    snapshot.pendingMarkerIds.every((markerId) => coveredMarkerIds.has(markerId)) &&
    snapshot.quarantinedMarkerIds.every((markerId) => !quarantinedMarkerIds.has(markerId))
  );
}

/** Clears replay state only after the lock-held caller proves the selected replay contract. */
export async function completeRecallGenerationReplayWithinWriteWindow(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  return completeRecallGenerationReplayTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    backlogSummaryPath: options.backlogSummaryPath,
    ...(options.nowEpochMilliseconds ? { nowEpochMilliseconds: options.nowEpochMilliseconds } : {}),
    proveReplayWorkComplete: () => proveFixedRecallGenerationReplayComplete(options),
  });
}

/** Completes replay under the operation lock so generation state cannot be overwritten. */
export async function completeRecallGenerationReplay(
  options: CompleteRecallGenerationReplayOptions,
): Promise<boolean> {
  return coordinateRecallWriteWindow({ lockPath: options.lockPath, allowRecovery: false }, () =>
    completeRecallGenerationReplayWithinWriteWindow(options),
  );
}

import { rm } from 'node:fs/promises';
import { join } from 'node:path';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { RecallMarkerCheckpoint } from './recall-session-projection.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

const RECALL_MARKER_ID_PATTERN = /^[A-Za-z0-9_-]+$/u;

/** Deletes only planned marker IDs proven by an exact target-generation projection checkpoint. */
export async function acknowledgeCoveredRecallMarkers(
  workPlan: RecallMarkerReplayWorkPlan,
  durableProjectionCheckpoint: RecallMarkerCheckpoint,
): Promise<number> {
  if (durableProjectionCheckpoint.generationId !== workPlan.targetGenerationId) {
    return 0;
  }
  const coveredMarkerIds = new Set(durableProjectionCheckpoint.coveredMarkerIds);
  let acknowledgedMarkerCount = 0;
  for (const markerId of workPlan.sourceMarkerIds) {
    if (!RECALL_MARKER_ID_PATTERN.test(markerId)) {
      throw new Error('Recall marker acknowledgement marker ID invalid');
    }
    if (!coveredMarkerIds.has(markerId)) {
      continue;
    }
    try {
      await rm(join(workPlan.markerSpoolDirectory, `${markerId}.json`));
      acknowledgedMarkerCount += 1;
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ENOENT') {
        throw error;
      }
    }
  }
  if (acknowledgedMarkerCount > 0) {
    await syncRecallDirectory(workPlan.markerSpoolDirectory);
  }
  return acknowledgedMarkerCount;
}

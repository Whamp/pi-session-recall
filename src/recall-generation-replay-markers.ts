import type { Dirent } from 'node:fs';
import { readdir } from 'node:fs/promises';
import { join } from 'node:path';

import { readNodeErrorCode } from './read-node-error-code.js';

const PENDING_MARKER_FILE_PATTERN = /^([A-Za-z0-9_-]+)\.json$/u;
const QUARANTINED_MARKER_FILE_PATTERN = /^([A-Za-z0-9_-]+)\.json(?:\..+)?$/u;

/** Lists exact pending marker IDs in deterministic order without reading marker contents. */
export async function listPendingRecallMarkerIds(markerSpoolDirectory: string): Promise<string[]> {
  try {
    return (await readdir(markerSpoolDirectory, { withFileTypes: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => PENDING_MARKER_FILE_PATTERN.exec(entry.name)?.[1])
      .filter((markerId) => markerId !== undefined)
      .toSorted();
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

/** Lists exact quarantined marker IDs across failure categories in deterministic order. */
export async function listQuarantinedRecallMarkerIds(
  markerQuarantineDirectory: string,
): Promise<string[]> {
  let categoryEntries: Dirent[];
  try {
    categoryEntries = await readdir(markerQuarantineDirectory, { withFileTypes: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
  const markerIds = (
    await Promise.all(
      categoryEntries
        .filter((entry) => entry.isDirectory())
        .map(async (category) =>
          (await readdir(join(markerQuarantineDirectory, category.name), { withFileTypes: true }))
            .filter((entry) => entry.isFile())
            .map((entry) => QUARANTINED_MARKER_FILE_PATTERN.exec(entry.name)?.[1])
            .filter((markerId) => markerId !== undefined),
        ),
    )
  ).flat();
  return [...new Set(markerIds)].toSorted();
}

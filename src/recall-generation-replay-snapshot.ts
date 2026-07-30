import { open, readFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { syncRecallDirectory } from './sync-recall-directory.js';

/** Current immutable generation replay snapshot format. */
export const RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION = 1;

/** Fixed filename retained for the first activation replay snapshot contract. */
export const RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME = 'generation-replay-snapshot.json';

/** Exact pending and quarantined marker IDs fixed at one generation cutover boundary. */
export interface RecallGenerationReplaySnapshot {
  snapshotVersion: 1;
  generationId: string;
  pendingMarkerIds: string[];
  quarantinedMarkerIds: string[];
  capturedAtEpochMilliseconds: number;
}

const markerIdSchema = Type.String({ pattern: '^[A-Za-z0-9_-]+$' });
const recallGenerationReplaySnapshotSchema = Type.Object(
  {
    snapshotVersion: Type.Literal(RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION),
    generationId: markerIdSchema,
    pendingMarkerIds: Type.Array(markerIdSchema, { uniqueItems: true }),
    quarantinedMarkerIds: Type.Array(markerIdSchema, { uniqueItems: true }),
    capturedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

function parseRecallGenerationReplaySnapshot(value: unknown): RecallGenerationReplaySnapshot {
  const snapshot = Value.Parse(recallGenerationReplaySnapshotSchema, value);
  if (
    snapshot.pendingMarkerIds.some(
      (markerId, index) => markerId !== snapshot.pendingMarkerIds.toSorted()[index],
    ) ||
    snapshot.quarantinedMarkerIds.some(
      (markerId, index) => markerId !== snapshot.quarantinedMarkerIds.toSorted()[index],
    )
  ) {
    throw new Error('Recall generation replay snapshot marker IDs must be sorted');
  }
  return snapshot;
}

/** Writes one immutable fixed replay snapshot and refuses replacement. */
export async function writeRecallGenerationReplaySnapshot(
  snapshotPath: string,
  snapshot: RecallGenerationReplaySnapshot,
): Promise<void> {
  const validated = parseRecallGenerationReplaySnapshot(snapshot);
  const snapshotFile = await open(snapshotPath, 'wx', 0o600);
  try {
    await snapshotFile.writeFile(`${JSON.stringify(validated, null, 2)}\n`, 'utf8');
    await snapshotFile.sync();
  } finally {
    await snapshotFile.close();
  }
  await syncRecallDirectory(dirname(snapshotPath));
}

/** Reads one fixed replay snapshot through its strict generation-local contract. */
export async function readRecallGenerationReplaySnapshot(
  snapshotPath: string,
): Promise<RecallGenerationReplaySnapshot> {
  let source: string;
  try {
    source = await readFile(snapshotPath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall generation replay snapshot unreadable at ${snapshotPath}: ${message}`, {
      cause: error,
    });
  }
  try {
    const parsed: unknown = JSON.parse(source);
    return parseRecallGenerationReplaySnapshot(parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall generation replay snapshot invalid at ${snapshotPath}: ${message}`, {
      cause: error,
    });
  }
}

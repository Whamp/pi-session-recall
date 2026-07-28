import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import {
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

const generationId = 'generation_projection_test';
const physicalSessionId = 'physical-session-1';
const physicalProjectionId = createPhysicalSessionProjectionId(physicalSessionId);

function createPhysicalProjection(): PhysicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: physicalProjectionId,
    generationId,
    physicalSessionId,
    sourcePath: '/isolated/session.jsonl',
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 100,
    appendCursorLines: 2,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: 'entry-1',
    logicalSessionIds: ['logical-session-1'],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: ['marker_1'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 1 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createLogicalProjection(): LogicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(physicalSessionId, 'logical-session-1'),
    generationId,
    physicalSessionId,
    physicalProjectionId,
    logicalSessionId: 'logical-session-1',
    effectiveLeafEntryId: 'entry-1',
    activeContextBoundary: { firstEntryId: 'entry-1', lastEntryId: 'entry-1' },
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    entryDescriptors: [
      {
        entryId: 'entry-1',
        parentEntryId: null,
        entryType: 'message',
        sourceLine: 2,
        startByte: 50,
        endByte: 100,
        firstKeptEntryId: null,
        hasRetainedTail: false,
        toolCalls: [],
        toolResult: null,
      },
    ],
    eligibleContributorEntryIds: ['entry-1'],
    eligibleSpans: [
      {
        startByte: 50,
        endByte: 100,
        startEntryId: 'entry-1',
        endEntryId: 'entry-1',
        contributorEntryIds: ['entry-1'],
      },
    ],
    labels: [],
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: ['marker_1'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 1 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

void test('scalar-only zvec projection store strictly round-trips physical and logical checkpoints', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-projections-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const databasePath = join(directory, 'collection');
  const projections = [createPhysicalProjection(), createLogicalProjection()];

  const writer = openZvecSessionProjectionStore({ databasePath, generationId });
  await writer.upsertProjections(projections);
  writer.close();

  const raw = ZVecOpen(databasePath, { readOnly: true });
  assert.equal(raw.schema.vectors().length, 0);
  assert.deepEqual(
    raw.schema
      .fields()
      .map(({ name }) => name)
      .toSorted(),
    [
      'generationId',
      'logicalSessionId',
      'physicalSessionProjectionId',
      'projectionJson',
      'projectionKind',
      'schemaVersion',
    ],
  );
  raw.closeSync();

  const reader = openZvecSessionProjectionStore({
    databasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  assert.deepEqual(
    reader.fetchProjections(projections.map(({ projectionId }) => projectionId)),
    new Map(projections.map((projection) => [projection.projectionId, projection])),
  );
  reader.close();
});

void test('projection store checked delete removes only requested scalar records', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-projection-delete-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const databasePath = join(directory, 'collection');
  const physical = createPhysicalProjection();
  const logical = createLogicalProjection();
  const store = openZvecSessionProjectionStore({ databasePath, generationId });
  await store.upsertProjections([physical, logical]);
  await store.deleteProjections([logical.projectionId]);
  assert.deepEqual(
    store.fetchProjections([physical.projectionId, logical.projectionId]),
    new Map([[physical.projectionId, physical]]),
  );
  store.close();
});

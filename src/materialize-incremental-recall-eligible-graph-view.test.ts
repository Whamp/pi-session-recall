import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { materializeIncrementalRecallEligibleGraphView } from './materialize-incremental-recall-eligible-graph-view.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import {
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { readRecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';

void test('cursor-zero graph materialization reuses append records without rereading source body', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-eligible-graph-view-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sourcePath = join(directory, 'session.jsonl');
  const physicalSessionId = 'physical-graph-view';
  const generationId = 'generation-graph-view';
  await writeFile(
    sourcePath,
    [
      {
        type: 'session',
        version: 3,
        id: physicalSessionId,
        timestamp: '2026-07-28T00:00:00Z',
        cwd: '/isolated/project',
      },
      {
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-28T00:00:01Z',
        message: { role: 'user', content: 'cursor zero bounded evidence' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n',
  );
  const metadata = await stat(sourcePath, { bigint: true });
  const physicalProjection: PhysicalSessionProjection = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId,
    physicalSessionId,
    sourcePath,
    sourceDevice: metadata.dev.toString(),
    sourceInode: metadata.ino.toString(),
    appendCursorBytes: 0,
    appendCursorLines: 0,
    boundaryFingerprint: createHash('sha256').update('').digest('hex'),
    lastEntryId: null,
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sourcePath,
    runtimeInstanceId: 'runtime-graph-view',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'entry-1',
    },
  } as const;
  const marker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  const appendDelta = await readRecallSessionAppendDelta(sourcePath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [marker],
    quiescenceObserved: false,
  });
  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  const logicalProjection = projected.logicalProjections[0];
  assert.ok(logicalProjection);
  const graphView = await materializeIncrementalRecallEligibleGraphView({
    physicalProjection: projected.physicalProjection,
    logicalProjection,
    newlyEligibleSpans: projected.newlyEligibleSpans,
    appendDelta,
    readRange() {
      throw new Error('cursor-zero graph view must not reread append source bytes');
    },
  });

  assert.equal(graphView.graph.header.id, physicalSessionId);
  assert.deepEqual(graphView.graph.entries[0]?.record.message, {
    role: 'user',
    content: 'cursor zero bounded evidence',
  });
  assert.equal(graphView.graph.entries[0]?.lineIndex, 2);
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { appendFile, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import fc from 'fast-check';

import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  projectRecallSessionAppend,
  type ProjectRecallSessionAppendInput,
} from './project-recall-session-append.js';
import {
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { readRecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import type { RecallWorkMarker, RecallWorkMarkerTriggerPayload } from './recall-work-marker.js';

function jsonl(records: readonly object[]): Buffer {
  return Buffer.from(`${records.map((record) => JSON.stringify(record)).join('\n')}\n`);
}

async function emptyPhysicalProjection(sessionPath: string): Promise<PhysicalSessionProjection> {
  const metadata = await stat(sessionPath, { bigint: true });
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId('physical'),
    generationId: 'generation',
    physicalSessionId: 'physical',
    sourcePath: sessionPath,
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
    markerCheckpoint: { generationId: 'generation', coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function marker(
  markerId: string,
  trigger: RecallWorkMarkerTriggerPayload,
  runtimeSequence = 1,
): RecallWorkMarker {
  return {
    version: 1,
    markerId,
    physicalSessionId: 'physical',
    physicalSessionPath: '/isolated/session.jsonl',
    runtimeInstanceId: 'runtime',
    runtimeSequence,
    createdAtEpochMilliseconds: runtimeSequence,
    trigger,
  };
}

void test('projector creates multiple logical sessions from one current canonical physical append', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-append-'));
  const sessionPath = join(directory, 'session.jsonl');
  const source = jsonl([
    {
      type: 'session',
      version: 3,
      id: 'logical-1',
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/one',
    },
    {
      type: 'message',
      id: 'e1',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'one' },
    },
    {
      type: 'session',
      version: 3,
      id: 'logical-2',
      timestamp: '2026-01-02T00:00:00Z',
      cwd: '/two',
    },
    {
      type: 'message',
      id: 'e2',
      parentId: null,
      timestamp: '2026-01-02T00:00:01Z',
      message: { role: 'user', content: 'two' },
    },
  ]);
  await writeFile(sessionPath, source);
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }

  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [],
    quiescenceObserved: false,
  });

  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(projected.physicalProjection.logicalSessionIds, ['logical-1@1', 'logical-2@3']);
  assert.equal(projected.physicalProjection.appendCursorBytes, source.length);
  assert.deepEqual(
    projected.logicalProjections.map((projection) => ({
      id: projection.logicalSessionId,
      leaf: projection.effectiveLeafEntryId,
      entries: projection.entryDescriptors.map(({ entryId }) => entryId),
      eligible: projection.eligibleContributorEntryIds,
    })),
    [
      { id: 'logical-1@1', leaf: 'e1', entries: ['e1'], eligible: [] },
      { id: 'logical-2@3', leaf: 'e2', entries: ['e2'], eligible: [] },
    ],
  );
});

void test('projector gives repeated raw header IDs distinct occurrence identities', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-repeated-header-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    jsonl([
      {
        type: 'session',
        version: 3,
        id: 'reused-logical',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/one',
      },
      {
        type: 'message',
        id: 'first-entry',
        parentId: null,
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: 'one' },
      },
      {
        type: 'session',
        version: 3,
        id: 'reused-logical',
        timestamp: '2026-01-02T00:00:00Z',
        cwd: '/two',
      },
      {
        type: 'message',
        id: 'second-entry',
        parentId: null,
        timestamp: '2026-01-02T00:00:01Z',
        message: { role: 'user', content: 'two' },
      },
    ]),
  );
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }

  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [],
    quiescenceObserved: false,
  });

  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(projected.physicalProjection.logicalSessionIds, [
    'reused-logical@1',
    'reused-logical@3',
  ]);
  assert.deepEqual(
    projected.logicalProjections.map(({ logicalSessionId, rawSessionId }) => ({
      logicalSessionId,
      rawSessionId,
    })),
    [
      { logicalSessionId: 'reused-logical@1', rawSessionId: 'reused-logical' },
      { logicalSessionId: 'reused-logical@3', rawSessionId: 'reused-logical' },
    ],
  );
});

void test('projector keeps the first repeated-header occurrence identity stable across later appends and replay', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-stable-occurrence-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'session.jsonl');
  const firstOccurrence = [
    {
      type: 'session',
      version: 3,
      id: 'reused-logical',
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/one',
    },
    {
      type: 'message',
      id: 'first-entry',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'one' },
    },
  ];
  await writeFile(sessionPath, jsonl(firstOccurrence));
  const initialPhysicalProjection = await emptyPhysicalProjection(sessionPath);
  const firstDelta = await readRecallSessionAppendDelta(sessionPath, initialPhysicalProjection);
  assert.equal(firstDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (firstDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const firstProjection = projectRecallSessionAppend({
    physicalProjection: initialPhysicalProjection,
    logicalProjections: [],
    appendDelta: firstDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(firstProjection.status, RecallAppendProjectionStatus.PROJECTED);
  if (firstProjection.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  const firstLogicalProjection = firstProjection.logicalProjections[0];
  assert.equal(firstLogicalProjection?.logicalSessionId, 'reused-logical@1');
  assert.equal(firstLogicalProjection?.rawSessionId, 'reused-logical');

  await appendFile(
    sessionPath,
    jsonl([
      {
        type: 'session',
        version: 3,
        id: 'reused-logical',
        timestamp: '2026-01-02T00:00:00Z',
        cwd: '/two',
      },
      {
        type: 'message',
        id: 'second-entry',
        parentId: null,
        timestamp: '2026-01-02T00:00:01Z',
        message: { role: 'user', content: 'two' },
      },
    ]),
  );
  const secondDelta = await readRecallSessionAppendDelta(
    sessionPath,
    firstProjection.physicalProjection,
  );
  assert.equal(secondDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (secondDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const secondProjection = projectRecallSessionAppend({
    physicalProjection: firstProjection.physicalProjection,
    logicalProjections: firstProjection.logicalProjections,
    appendDelta: secondDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(secondProjection.status, RecallAppendProjectionStatus.PROJECTED);
  if (secondProjection.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(
    secondProjection.logicalProjections.map(({ logicalSessionId, projectionId }) => ({
      logicalSessionId,
      projectionId,
    })),
    [
      {
        logicalSessionId: 'reused-logical@1',
        projectionId: firstLogicalProjection?.projectionId,
      },
      {
        logicalSessionId: 'reused-logical@3',
        projectionId: secondProjection.logicalProjections[1]?.projectionId,
      },
    ],
  );

  const uniqueBranchMarker = marker('unique-repeated-raw-branch', {
    kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
    logicalSessionId: 'reused-logical',
    oldLeafEntryId: 'first-entry',
    newLeafEntryId: null,
  });
  const routed = projectRecallSessionAppend({
    physicalProjection: secondProjection.physicalProjection,
    logicalProjections: secondProjection.logicalProjections,
    appendDelta: { ...secondDelta, records: [] },
    markers: [uniqueBranchMarker],
    quiescenceObserved: false,
  });
  assert.equal(routed.status, RecallAppendProjectionStatus.PROJECTED);
  if (routed.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(
    routed.logicalProjections.map((projection) => ({
      logicalSessionId: projection.logicalSessionId,
      eligibleEntryIds: projection.eligibleContributorEntryIds,
      coveredMarkerIds: projection.markerCheckpoint.coveredMarkerIds,
    })),
    [
      {
        logicalSessionId: 'reused-logical@1',
        eligibleEntryIds: ['first-entry'],
        coveredMarkerIds: [uniqueBranchMarker.markerId],
      },
      {
        logicalSessionId: 'reused-logical@3',
        eligibleEntryIds: [],
        coveredMarkerIds: [],
      },
    ],
  );

  const replay = projectRecallSessionAppend({
    physicalProjection: routed.physicalProjection,
    logicalProjections: routed.logicalProjections,
    appendDelta: { ...secondDelta, records: [] },
    markers: [uniqueBranchMarker],
    quiescenceObserved: false,
  });
  assert.equal(replay.status, RecallAppendProjectionStatus.PROJECTED);
  if (replay.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(replay.newlyEligibleSpans, []);
  assert.deepEqual(replay.physicalProjection, routed.physicalProjection);
  assert.deepEqual(replay.logicalProjections, routed.logicalProjections);
});

void test('projector scopes colliding context exits to their logical sessions with durable provenance', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-reused-session-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'session.jsonl');
  const records = [
    {
      type: 'session',
      version: 3,
      id: 'logical-1',
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/one',
    },
    {
      type: 'message',
      id: 'shared-root',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'first root' },
    },
    {
      type: 'message',
      id: 'shared-old',
      parentId: 'shared-root',
      timestamp: '2026-01-01T00:00:02Z',
      message: { role: 'assistant', content: 'first old branch' },
    },
    {
      type: 'message',
      id: 'shared-new',
      parentId: 'shared-root',
      timestamp: '2026-01-01T00:00:03Z',
      message: { role: 'assistant', content: 'first new branch' },
    },
    {
      type: 'branch_summary',
      id: 'shared-summary',
      parentId: 'shared-new',
      timestamp: '2026-01-01T00:00:04Z',
      fromId: 'shared-old',
      summary: 'first branch summary',
    },
    {
      type: 'compaction',
      id: 'shared-compaction',
      parentId: 'shared-summary',
      timestamp: '2026-01-01T00:00:05Z',
      summary: 'first compaction',
      firstKeptEntryId: 'shared-new',
      tokensBefore: 100,
    },
    {
      type: 'message',
      id: 'shared-tail',
      parentId: 'shared-compaction',
      timestamp: '2026-01-01T00:00:06Z',
      message: { role: 'user', content: 'first active tail' },
    },
    {
      type: 'session',
      version: 3,
      id: 'logical-2',
      timestamp: '2026-01-02T00:00:00Z',
      cwd: '/two',
    },
    {
      type: 'message',
      id: 'shared-root',
      parentId: null,
      timestamp: '2026-01-02T00:00:01Z',
      message: { role: 'user', content: 'second root' },
    },
    {
      type: 'message',
      id: 'shared-old',
      parentId: 'shared-root',
      timestamp: '2026-01-02T00:00:02Z',
      message: { role: 'assistant', content: 'second old branch' },
    },
    {
      type: 'message',
      id: 'shared-new',
      parentId: 'shared-root',
      timestamp: '2026-01-02T00:00:03Z',
      message: { role: 'assistant', content: 'second new branch' },
    },
    {
      type: 'branch_summary',
      id: 'shared-summary',
      parentId: 'shared-new',
      timestamp: '2026-01-02T00:00:04Z',
      fromId: 'shared-old',
      summary: 'second branch summary',
    },
    {
      type: 'compaction',
      id: 'shared-compaction',
      parentId: 'shared-summary',
      timestamp: '2026-01-02T00:00:05Z',
      summary: 'second compaction',
      firstKeptEntryId: 'shared-new',
      tokensBefore: 100,
    },
    {
      type: 'message',
      id: 'shared-tail',
      parentId: 'shared-compaction',
      timestamp: '2026-01-02T00:00:06Z',
      message: { role: 'user', content: 'second active tail' },
    },
  ];
  await writeFile(sessionPath, jsonl(records));
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const branchExitTrigger = {
    kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
    logicalSessionId: 'logical-1',
    oldLeafEntryId: 'shared-old',
    newLeafEntryId: 'shared-new',
    summaryEntryId: 'shared-summary',
  } as const;
  const compactionTrigger = {
    kind: RecallWorkMarkerTrigger.COMPACTION,
    logicalSessionId: 'logical-2',
    compactionEntryId: 'shared-compaction',
  } as const;
  const branchMarker = marker('branch-logical-1', branchExitTrigger);
  const compactionMarker = marker('compaction-logical-2', compactionTrigger, 2);

  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [branchMarker, compactionMarker],
    quiescenceObserved: false,
  });

  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(
    projected.logicalProjections.map((projection) => ({
      logicalSessionId: projection.logicalSessionId,
      eligibleEntryIds: projection.eligibleContributorEntryIds,
      coveredMarkerIds: projection.markerCheckpoint.coveredMarkerIds,
    })),
    [
      {
        logicalSessionId: 'logical-1@1',
        eligibleEntryIds: ['shared-old', 'shared-summary'],
        coveredMarkerIds: [branchMarker.markerId],
      },
      {
        logicalSessionId: 'logical-2@8',
        eligibleEntryIds: ['shared-root', 'shared-compaction'],
        coveredMarkerIds: [compactionMarker.markerId],
      },
    ],
  );
  assert.deepEqual(projected.physicalProjection.markerCheckpoint.coveredMarkerIds, [
    branchMarker.markerId,
    compactionMarker.markerId,
  ]);
  const expectedSourceRecords = [3, 5, 9, 13].map((sourceLine) => {
    const record = appendDelta.records.find((candidate) => candidate.sourceLine === sourceLine);
    assert.ok(record);
    return {
      startByte: record.startByte,
      endByte: record.endByte,
    };
  });
  assert.deepEqual(
    projected.newlyEligibleSpans.map(({ startByte, endByte }) => ({ startByte, endByte })),
    expectedSourceRecords,
  );

  const replayed = projectRecallSessionAppend({
    physicalProjection: projected.physicalProjection,
    logicalProjections: projected.logicalProjections,
    appendDelta: { ...appendDelta, records: [] },
    markers: [branchMarker, compactionMarker],
    quiescenceObserved: false,
  });
  assert.equal(replayed.status, RecallAppendProjectionStatus.PROJECTED);
  if (replayed.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(replayed.newlyEligibleSpans, []);
  assert.deepEqual(replayed.physicalProjection, projected.physicalProjection);
  assert.deepEqual(replayed.logicalProjections, projected.logicalProjections);
});

void test('projector reconciles context-exit markers that resolve to zero or multiple logical occurrences', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-ambiguous-marker-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'session.jsonl');
  const repeatedGraph = (timestampDate: string) => [
    {
      type: 'session',
      version: 3,
      id: 'reused-logical',
      timestamp: `${timestampDate}T00:00:00Z`,
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'shared-root',
      parentId: null,
      timestamp: `${timestampDate}T00:00:01Z`,
      message: { role: 'user', content: 'root' },
    },
    {
      type: 'message',
      id: 'shared-old',
      parentId: 'shared-root',
      timestamp: `${timestampDate}T00:00:02Z`,
      message: { role: 'assistant', content: 'old' },
    },
    {
      type: 'message',
      id: 'shared-new',
      parentId: 'shared-root',
      timestamp: `${timestampDate}T00:00:03Z`,
      message: { role: 'assistant', content: 'new' },
    },
    {
      type: 'branch_summary',
      id: 'shared-summary',
      parentId: 'shared-new',
      timestamp: `${timestampDate}T00:00:04Z`,
      fromId: 'shared-old',
      summary: 'summary',
    },
    {
      type: 'compaction',
      id: 'shared-compaction',
      parentId: 'shared-summary',
      timestamp: `${timestampDate}T00:00:05Z`,
      summary: 'compaction',
      firstKeptEntryId: 'shared-new',
      tokensBefore: 100,
    },
  ];
  await writeFile(
    sessionPath,
    jsonl([...repeatedGraph('2026-01-01'), ...repeatedGraph('2026-01-02')]),
  );
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const baseline = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(baseline.status, RecallAppendProjectionStatus.PROJECTED);
  if (baseline.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }

  const ambiguousMarkers = [
    marker('ambiguous-compaction', {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'reused-logical',
      compactionEntryId: 'shared-compaction',
    }),
    marker('ambiguous-branch', {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'reused-logical',
      oldLeafEntryId: 'shared-old',
      newLeafEntryId: 'shared-new',
      summaryEntryId: 'shared-summary',
    }),
    marker('missing-compaction', {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'reused-logical',
      compactionEntryId: 'missing-compaction',
    }),
  ];
  for (const contextExitMarker of ambiguousMarkers) {
    const replayInput: ProjectRecallSessionAppendInput = {
      physicalProjection: baseline.physicalProjection,
      logicalProjections: baseline.logicalProjections,
      appendDelta: { ...appendDelta, records: [] },
      markers: [contextExitMarker],
      quiescenceObserved: false,
    };
    const expected = {
      status: RecallAppendProjectionStatus.REQUIRES_RECONCILIATION,
      repairReason: RecallProjectionRepairReason.MALFORMED_GRAPH,
    } as const;
    assert.deepEqual(projectRecallSessionAppend(replayInput), expected);
    assert.deepEqual(projectRecallSessionAppend(replayInput), expected);
    assert.deepEqual(baseline.physicalProjection.markerCheckpoint.coveredMarkerIds, []);
    assert.ok(
      baseline.logicalProjections.every(
        (projection) => projection.markerCheckpoint.coveredMarkerIds.length === 0,
      ),
    );
  }
});

void test('projector applies label metadata and explicit leaf records without making content eligible', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-label-leaf-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    jsonl([
      {
        type: 'session',
        version: 3,
        id: 'logical',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'e1',
        parentId: null,
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: 'one' },
      },
      {
        type: 'message',
        id: 'e2',
        parentId: 'e1',
        timestamp: '2026-01-01T00:00:02Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'two' }] },
      },
      {
        type: 'session_info',
        id: 'label',
        parentId: 'e2',
        timestamp: '2026-01-01T00:00:03Z',
        name: 'Focused work',
      },
      { type: 'leaf', targetId: 'e1' },
    ]),
  );
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.equal(projected.logicalProjections[0]?.effectiveLeafEntryId, 'e1');
  assert.deepEqual(projected.logicalProjections[0]?.labels, ['Focused work']);
  assert.deepEqual(projected.newlyEligibleSpans, []);
});

void test('projector emits immediate compaction and branch-exit spans while excluding the new active tail', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-boundaries-'));
  const sessionPath = join(directory, 'session.jsonl');
  const source = jsonl([
    {
      type: 'session',
      version: 3,
      id: 'logical',
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'root',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'root' },
    },
    {
      type: 'message',
      id: 'old',
      parentId: 'root',
      timestamp: '2026-01-01T00:00:02Z',
      message: { role: 'assistant', content: [{ type: 'text', text: 'old' }] },
    },
    {
      type: 'message',
      id: 'new',
      parentId: 'root',
      timestamp: '2026-01-01T00:00:03Z',
      message: { role: 'assistant', content: [{ type: 'text', text: 'new' }] },
    },
    {
      type: 'branch_summary',
      id: 'summary',
      parentId: 'new',
      timestamp: '2026-01-01T00:00:04Z',
      fromId: 'root',
      summary: 'old branch summary',
    },
    {
      type: 'compaction',
      id: 'compact',
      parentId: 'summary',
      timestamp: '2026-01-01T00:00:05Z',
      summary: 'compact summary',
      firstKeptEntryId: 'new',
      tokensBefore: 100,
    },
    {
      type: 'message',
      id: 'tail',
      parentId: 'compact',
      timestamp: '2026-01-01T00:00:06Z',
      message: { role: 'user', content: 'active tail' },
    },
  ]);
  await writeFile(sessionPath, source);
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }

  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: [],
    appendDelta,
    markers: [
      marker('branch', {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        logicalSessionId: 'logical',
        oldLeafEntryId: 'old',
        newLeafEntryId: 'new',
        summaryEntryId: 'summary',
      }),
      marker(
        'compact',
        {
          kind: RecallWorkMarkerTrigger.COMPACTION,
          logicalSessionId: 'logical',
          compactionEntryId: 'compact',
        },
        2,
      ),
    ],
    quiescenceObserved: false,
  });

  assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.deepEqual(
    projected.newlyEligibleSpans.map(({ contributorEntryIds }) => contributorEntryIds),
    [['root'], ['old'], ['summary'], ['compact']],
  );
  assert.equal(
    projected.logicalProjections[0]?.eligibleContributorEntryIds.includes('tail'),
    false,
  );
});

void test('applying a valid append in arbitrary record partitions equals applying it once', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-partitions-'));
  await fc.assert(
    fc.asyncProperty(fc.integer({ min: 1, max: 12 }), async (entryCount) => {
      const header = {
        type: 'session',
        version: 3,
        id: 'logical-property',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/project',
      };
      const entries = Array.from({ length: entryCount }, (_, index) => ({
        type: 'message',
        id: `e${index}`,
        parentId: index === 0 ? null : `e${index - 1}`,
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: `entry ${index}` },
      }));
      const oncePath = join(directory, 'once.jsonl');
      await writeFile(oncePath, '');
      const onceInitial = await emptyPhysicalProjection(oncePath);
      await writeFile(oncePath, jsonl([header, ...entries]));
      const onceDelta = await readRecallSessionAppendDelta(oncePath, onceInitial);
      assert.equal(onceDelta.status, RecallAppendDeltaStatus.APPENDED);
      if (onceDelta.status !== RecallAppendDeltaStatus.APPENDED) {
        return;
      }
      const once = projectRecallSessionAppend({
        physicalProjection: onceInitial,
        logicalProjections: [],
        appendDelta: onceDelta,
        markers: [],
        quiescenceObserved: false,
      });
      assert.equal(once.status, RecallAppendProjectionStatus.PROJECTED);
      if (once.status !== RecallAppendProjectionStatus.PROJECTED) {
        return;
      }

      const partitionedPath = join(directory, 'partitioned.jsonl');
      await writeFile(partitionedPath, '');
      let partitionedPhysical = await emptyPhysicalProjection(partitionedPath);
      let partitionedLogical: LogicalSessionProjection[] = [];
      for (const nextRecord of [header, ...entries]) {
        await appendFile(partitionedPath, jsonl([nextRecord]));
        const delta = await readRecallSessionAppendDelta(partitionedPath, partitionedPhysical);
        assert.equal(delta.status, RecallAppendDeltaStatus.APPENDED);
        if (delta.status !== RecallAppendDeltaStatus.APPENDED) {
          return;
        }
        const projected = projectRecallSessionAppend({
          physicalProjection: partitionedPhysical,
          logicalProjections: partitionedLogical,
          appendDelta: delta,
          markers: [],
          quiescenceObserved: false,
        });
        assert.equal(projected.status, RecallAppendProjectionStatus.PROJECTED);
        if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
          return;
        }
        partitionedPhysical = projected.physicalProjection;
        partitionedLogical = projected.logicalProjections;
      }
      assert.deepEqual(partitionedLogical, once.logicalProjections);
      assert.equal(
        partitionedPhysical.appendCursorBytes,
        once.physicalProjection.appendCursorBytes,
      );
      assert.equal(
        partitionedPhysical.appendCursorLines,
        once.physicalProjection.appendCursorLines,
      );
      assert.equal(
        partitionedPhysical.boundaryFingerprint,
        once.physicalProjection.boundaryFingerprint,
      );
    }),
    { numRuns: 30 },
  );
});

void test('projector rejects malformed links without imposing a whole-session payload ceiling', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-project-reconcile-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    jsonl([
      {
        type: 'session',
        version: 3,
        id: 'logical',
        timestamp: '2026-01-01T00:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'orphan',
        parentId: 'missing',
        timestamp: '2026-01-01T00:00:01Z',
        message: { role: 'user', content: 'orphan' },
      },
    ]),
  );
  const physicalProjection = await emptyPhysicalProjection(sessionPath);
  const appendDelta = await readRecallSessionAppendDelta(sessionPath, physicalProjection);
  assert.equal(appendDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  assert.deepEqual(
    projectRecallSessionAppend({
      physicalProjection,
      logicalProjections: [],
      appendDelta,
      markers: [],
      quiescenceObserved: false,
    }),
    {
      status: RecallAppendProjectionStatus.REQUIRES_RECONCILIATION,
      repairReason: RecallProjectionRepairReason.MALFORMED_GRAPH,
    },
  );

  const validDelta = {
    ...appendDelta,
    records: appendDelta.records.filter(({ value }) => value.type === 'session'),
  };
  assert.equal(
    projectRecallSessionAppend({
      physicalProjection,
      logicalProjections: [],
      appendDelta: validDelta,
      markers: [],
      quiescenceObserved: false,
    }).status,
    RecallAppendProjectionStatus.PROJECTED,
  );
});

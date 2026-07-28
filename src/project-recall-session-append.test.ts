import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { appendFile, mkdtemp, stat, writeFile } from 'node:fs/promises';
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
import { projectRecallSessionAppend } from './project-recall-session-append.js';
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
  assert.deepEqual(projected.physicalProjection.logicalSessionIds, ['logical-1', 'logical-2']);
  assert.equal(projected.physicalProjection.appendCursorBytes, source.length);
  assert.deepEqual(
    projected.logicalProjections.map((projection) => ({
      id: projection.logicalSessionId,
      leaf: projection.effectiveLeafEntryId,
      entries: projection.entryDescriptors.map(({ entryId }) => entryId),
      eligible: projection.eligibleContributorEntryIds,
    })),
    [
      { id: 'logical-1', leaf: 'e1', entries: ['e1'], eligible: [] },
      { id: 'logical-2', leaf: 'e2', entries: ['e2'], eligible: [] },
    ],
  );
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
        oldLeafEntryId: 'old',
        newLeafEntryId: 'new',
        summaryEntryId: 'summary',
      }),
      marker(
        'compact',
        { kind: RecallWorkMarkerTrigger.COMPACTION, compactionEntryId: 'compact' },
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

void test('projector rejects malformed links and payload overflow with explicit reconciliation', async () => {
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
  assert.deepEqual(
    projectRecallSessionAppend({
      physicalProjection,
      logicalProjections: [],
      appendDelta: validDelta,
      markers: [],
      quiescenceObserved: false,
      maxProjectionPayloadBytes: 1,
    }),
    {
      status: RecallAppendProjectionStatus.REQUIRES_RECONCILIATION,
      repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW,
    },
  );
});

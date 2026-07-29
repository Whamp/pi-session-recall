import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RecallProjectionEncodingStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  decodeRecallSessionProjection,
  encodeRecallSessionProjection,
  mergeRecallMarkerCheckpoint,
  RECALL_SESSION_PROJECTION_MAX_BYTES,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallSessionProjection,
} from './recall-session-projection.js';

const generationId = 'generation_2026_07_24';

function createPhysicalProjection(): PhysicalSessionProjection {
  const physicalSessionId = 'physical-session-1';
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId,
    physicalSessionId,
    sourcePath: '/isolated/sessions/2026-07-24/session.jsonl',
    sourceDevice: '2049',
    sourceInode: '123456',
    appendCursorBytes: 4_096,
    appendCursorLines: 17,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: 'entry-last-1',
    logicalSessionIds: ['logical-session-1'],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: ['marker_abc'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 7 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createLogicalProjection(): LogicalSessionProjection {
  const physicalSessionId = 'physical-session-1';
  const logicalSessionId = 'logical-session-1';
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(physicalSessionId, logicalSessionId),
    generationId,
    physicalSessionId,
    physicalProjectionId: createPhysicalSessionProjectionId(physicalSessionId),
    logicalSessionId,
    effectiveLeafEntryId: 'entry-leaf-1',
    activeContextBoundary: {
      firstEntryId: 'entry-active-first',
      lastEntryId: 'entry-active-last',
    },
    compactionBoundary: {
      compactionEntryId: 'entry-compaction-1',
      firstRetainedEntryId: 'entry-active-first',
    },
    runtimeLeafObservations: [{ runtimeInstanceId: 'runtime-1', leafEntryId: 'entry-leaf-1' }],
    preservedBranchExits: [
      {
        oldLeafEntryId: 'entry-old-leaf',
        newLeafEntryId: 'entry-new-leaf',
        summaryEntryId: 'entry-branch-summary',
      },
      {
        oldLeafEntryId: null,
        newLeafEntryId: null,
        summaryEntryId: null,
      },
    ],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 128,
      sourceFingerprint: 'a'.repeat(64),
      cwd: '/isolated/project',
      parentSessionPath: null,
    },
    entryDescriptors: [
      {
        entryId: 'entry-span-first',
        parentEntryId: null,
        entryType: 'message',
        timestamp: '2026-01-01T00:00:00Z',
        messageRole: 'user',
        branchSummaryFromEntryId: null,
        sourceLine: 2,
        startByte: 128,
        endByte: 512,
        sourceFingerprint: 'b'.repeat(64),
        firstKeptEntryId: null,
        hasRetainedTail: false,
        toolCalls: [],
        toolResult: null,
      },
    ],
    eligibleContributorEntryIds: ['entry-span-first'],
    eligibleSpans: [
      {
        startByte: 128,
        endByte: 512,
        startEntryId: 'entry-span-first',
        endEntryId: 'entry-span-first',
        contributorEntryIds: ['entry-span-first'],
      },
    ],
    labels: ['feature/incremental-recall'],
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: ['marker_def'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 8 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

void test('physical and logical session projections round-trip through scalar-only payloads', () => {
  for (const projection of [createPhysicalProjection(), createLogicalProjection()]) {
    const encoded = encodeRecallSessionProjection(projection);
    assert.equal(encoded.status, RecallProjectionEncodingStatus.ENCODED);
    if (encoded.status !== RecallProjectionEncodingStatus.ENCODED) {
      continue;
    }
    assert.ok(
      Object.values(encoded.payload).every(
        (value) =>
          typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean',
      ),
    );
    assert.deepEqual(
      decodeRecallSessionProjection(encoded.payload, { expectedGenerationId: generationId }),
      projection,
    );
  }
});

void test('projection IDs are deterministic, kind-separated, and zvec-safe', () => {
  const physical = createPhysicalSessionProjectionId('same-session');
  const logical = createLogicalSessionProjectionId('same-session', 'same-session');
  assert.equal(createPhysicalSessionProjectionId('same-session'), physical);
  assert.equal(createLogicalSessionProjectionId('same-session', 'same-session'), logical);
  assert.equal(physical, 'physical_MjLfurftPHJkzSoZZSdU6_OyaZR0ytETpVpD4vLtUHU');
  assert.equal(logical, 'logical_SK1RG8upIyXNXxxRGtuOUIqbtPLtSSJifAronzFRQ4U');
  assert.notEqual(physical, logical);
  assert.notEqual(
    createLogicalSessionProjectionId('same-session', 'logical-a'),
    createLogicalSessionProjectionId('same-session', 'logical-b'),
  );
  for (const projectionId of [physical, logical]) {
    assert.match(projectionId, /^[A-Za-z0-9_-]+$/u);
    assert.equal(projectionId.includes(':'), false);
  }
});

void test('projection codec rejects forward versions, extra fields, generation mismatch, and malformed checkpoints', () => {
  const projection = createLogicalProjection();
  const encoded = encodeRecallSessionProjection(projection);
  assert.equal(encoded.status, RecallProjectionEncodingStatus.ENCODED);
  if (encoded.status !== RecallProjectionEncodingStatus.ENCODED) {
    return;
  }
  const decodeInner = (inner: unknown, expectedGenerationId = generationId): unknown =>
    decodeRecallSessionProjection(
      { ...encoded.payload, projectionJson: JSON.stringify(inner) },
      { expectedGenerationId },
    );

  assert.throws(
    () => decodeInner({ ...projection, schemaVersion: 4 }),
    /projection|schema|invalid/iu,
  );
  assert.throws(() => decodeInner({ ...projection, unexpected: true }), /projection|invalid/iu);
  assert.throws(
    () =>
      decodeRecallSessionProjection(encoded.payload, { expectedGenerationId: 'generation-new' }),
    /generation mismatch/iu,
  );
  assert.throws(
    () =>
      decodeInner({
        ...projection,
        markerCheckpoint: { ...projection.markerCheckpoint, generationId: 'generation-other' },
      }),
    /generation mismatch/iu,
  );
  assert.throws(
    () =>
      decodeInner({
        ...projection,
        markerCheckpoint: {
          ...projection.markerCheckpoint,
          runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 0 }],
        },
      }),
    /projection|checkpoint|invalid/iu,
  );
  assert.throws(
    () =>
      decodeInner({
        ...projection,
        projectionId: createLogicalSessionProjectionId('physical-session-1', 'different-logical'),
      }),
    /projection ID mismatch/iu,
  );
  assert.throws(
    () =>
      decodeRecallSessionProjection({ schemaVersion: 2 }, { expectedGenerationId: generationId }),
    /projection|invalid/iu,
  );
});

void test('projection codec enforces repair-state consistency and does not accept legacy index state', () => {
  const projection = createPhysicalProjection();
  assert.throws(
    () =>
      encodeRecallSessionProjection({
        ...projection,
        repairState: RecallProjectionRepairState.REQUIRES_RECONCILIATION,
        repairReason: null,
      }),
    /repair state|repair reason/iu,
  );
  assert.throws(
    () =>
      encodeRecallSessionProjection({
        ...projection,
        sourceAvailability: RecallSourceAvailability.SOURCE_MISSING,
        sourceMissingObservedAtEpochMilliseconds: 1_753_315_200_000,
        sourceMissingObservationCount: 0,
        sourceMissingSweepId: null,
        deletionCheckpoint: null,
      }),
    /source state/iu,
  );
  assert.throws(
    () =>
      decodeRecallSessionProjection(
        {
          schemaVersion: 2,
          sessions: {},
          documentOwners: {},
        },
        { expectedGenerationId: generationId },
      ),
    /projection|invalid/iu,
  );
});

void test('projection serialization excludes conversation payloads and arbitrary repair messages', () => {
  const sentinel = 'PRIVATE_CONVERSATION_SENTINEL_49';
  const projection = createLogicalProjection();
  const encoded = encodeRecallSessionProjection(projection);
  assert.equal(encoded.status, RecallProjectionEncodingStatus.ENCODED);
  assert.equal(JSON.stringify(encoded).includes(sentinel), false);

  const unsafeProjection: RecallSessionProjection & { conversationText: string } = {
    ...projection,
    conversationText: sentinel,
  };
  assert.throws(() => encodeRecallSessionProjection(unsafeProjection), /projection|invalid/iu);
  const unsafeRepairProjection: RecallSessionProjection & { repairMessage: string } = {
    ...projection,
    repairMessage: sentinel,
  };
  assert.throws(
    () => encodeRecallSessionProjection(unsafeRepairProjection),
    /projection|invalid/iu,
  );
});

void test('mergeRecallMarkerCheckpoint unions covered IDs and keeps max runtime sequences', () => {
  const merged = mergeRecallMarkerCheckpoint({
    generationId: 'generation_next',
    current: {
      generationId: 'generation_prior',
      coveredMarkerIds: ['marker_b', 'marker_a'],
      runtimeSequences: [
        { runtimeInstanceId: 'runtime-2', sequence: 3 },
        { runtimeInstanceId: 'runtime-1', sequence: 7 },
      ],
    },
    coveredMarkerIds: ['marker_a', 'marker_c'],
    runtimeSequences: [
      { runtimeInstanceId: 'runtime-1', sequence: 4 },
      { runtimeInstanceId: 'runtime-1', sequence: 9 },
      { runtimeInstanceId: 'runtime-3', sequence: 1 },
    ],
  });
  assert.deepEqual(merged, {
    generationId: 'generation_next',
    coveredMarkerIds: ['marker_a', 'marker_b', 'marker_c'],
    runtimeSequences: [
      { runtimeInstanceId: 'runtime-1', sequence: 9 },
      { runtimeInstanceId: 'runtime-2', sequence: 3 },
      { runtimeInstanceId: 'runtime-3', sequence: 1 },
    ],
  });
});

void test('projection serialization accepts the exact production bound and rejects one byte more', () => {
  const projection = createLogicalProjection();
  const baseline = encodeRecallSessionProjection(
    { ...projection, labels: [''] },
    { maxPayloadBytes: Number.MAX_SAFE_INTEGER },
  );
  assert.equal(baseline.status, RecallProjectionEncodingStatus.ENCODED);
  if (baseline.status !== RecallProjectionEncodingStatus.ENCODED) {
    return;
  }
  const paddingLength = RECALL_SESSION_PROJECTION_MAX_BYTES - baseline.byteLength;
  assert.ok(paddingLength > 0);

  const exact = encodeRecallSessionProjection({
    ...projection,
    labels: ['x'.repeat(paddingLength)],
  });
  assert.equal(exact.status, RecallProjectionEncodingStatus.ENCODED);
  assert.equal(exact.byteLength, RECALL_SESSION_PROJECTION_MAX_BYTES);

  const overflow = encodeRecallSessionProjection({
    ...projection,
    labels: ['x'.repeat(paddingLength + 1)],
  });
  assert.deepEqual(overflow, {
    status: RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW,
    byteLength: RECALL_SESSION_PROJECTION_MAX_BYTES + 1,
  });
});

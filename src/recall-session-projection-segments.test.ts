import assert from 'node:assert/strict';
import test from 'node:test';

import fc from 'fast-check';

import { RecallProjectionRepairState, RecallSessionProjectionKind } from './enums.js';
import {
  decodeRecallSessionProjectionSegments,
  encodeRecallSessionProjectionSegments,
} from './recall-session-projection-segments.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES,
  type LogicalSessionProjection,
  type RecallProjectedEntryDescriptor,
} from './recall-session-projection.js';

function createEntryDescriptor(
  entryId: string,
  sourceLine: number,
): RecallProjectedEntryDescriptor {
  return {
    entryId,
    parentEntryId: sourceLine === 2 ? null : `entry-${sourceLine - 1}`,
    entryType: 'message',
    timestamp: `2026-08-01T00:00:${String(sourceLine).padStart(2, '0')}.000Z`,
    messageRole: sourceLine % 2 === 0 ? 'user' : 'assistant',
    branchSummaryFromEntryId: null,
    sourceLine,
    startByte: sourceLine * 100,
    endByte: sourceLine * 100 + 99,
    sourceFingerprint: String(sourceLine).padStart(64, '0'),
    firstKeptEntryId: null,
    hasRetainedTail: false,
    toolCalls: [],
    toolResult: null,
  };
}

function createLogicalProjection(
  labels: string[],
  entryDescriptors: RecallProjectedEntryDescriptor[],
): LogicalSessionProjection {
  return {
    schemaVersion: 3,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId('physical_fixture', 'logical-session@1'),
    generationId: 'generation_fixture',
    physicalSessionId: 'physical_fixture',
    physicalProjectionId: createPhysicalSessionProjectionId('physical_fixture'),
    logicalSessionId: 'logical-session@1',
    rawSessionId: 'logical-session',
    effectiveLeafEntryId: entryDescriptors.at(-1)?.entryId ?? null,
    activeContextBoundary: null,
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 99,
      sourceFingerprint: 'a'.repeat(64),
      cwd: '/tmp/project',
      parentSessionPath: null,
    },
    entryDescriptors,
    eligibleContributorEntryIds: entryDescriptors.map(({ entryId }) => entryId),
    eligibleSpans: entryDescriptors.map((descriptor) => ({
      startByte: descriptor.startByte,
      endByte: descriptor.endByte,
      startEntryId: descriptor.entryId,
      endEntryId: descriptor.entryId,
      contributorEntryIds: [descriptor.entryId],
    })),
    labels,
    markerCheckpoint: {
      generationId: 'generation_fixture',
      coveredMarkerIds: [],
      runtimeSequences: [],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

void test('logical session projection round-trips through bounded segments without duplicating entry descriptors', () => {
  const entryDescriptors = Array.from({ length: 256 }, (_, index) =>
    createEntryDescriptor(`entry-${index + 2}`, index + 2),
  );
  const projection = createLogicalProjection(['x'.repeat(9_000_000)], entryDescriptors);

  const encoded = encodeRecallSessionProjectionSegments(projection);

  assert.ok(encoded.manifest.payloadByteLength > RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES);
  assert.ok(encoded.segments.length > 1);
  assert.ok(
    encoded.segments.every(
      (segment) =>
        Buffer.byteLength(JSON.stringify(segment), 'utf8') <=
        RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES,
    ),
  );
  assert.ok(
    encoded.segments.every(
      (segment) =>
        !Buffer.from(segment.payloadBase64, 'base64').includes(Buffer.from('entryDescriptors')),
    ),
  );
  assert.deepEqual(
    decodeRecallSessionProjectionSegments(encoded.manifest, encoded.segments, {
      logicalEntryDescriptors: entryDescriptors,
    }),
    projection,
  );
});

void test('projection segment codec preserves arbitrary Unicode across byte boundaries', () => {
  fc.assert(
    fc.property(
      fc.array(fc.string(), { minLength: 0, maxLength: 8 }),
      fc.integer({ min: 160, max: 512 }),
      (labels, maxSegmentRecordBytes) => {
        const projection = createLogicalProjection(labels, [createEntryDescriptor('entry-2', 2)]);
        const encoded = encodeRecallSessionProjectionSegments(projection, {
          maxSegmentRecordBytes,
        });
        assert.ok(
          encoded.segments.every(
            (segment) =>
              Buffer.byteLength(JSON.stringify(segment), 'utf8') <= maxSegmentRecordBytes,
          ),
        );
        assert.deepEqual(
          decodeRecallSessionProjectionSegments(encoded.manifest, encoded.segments, {
            logicalEntryDescriptors: projection.entryDescriptors,
          }),
          projection,
        );
      },
    ),
    { numRuns: 100 },
  );
});

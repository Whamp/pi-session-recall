import { createHash } from 'node:crypto';

import { RecallSessionProjectionKind } from './enums.js';
import {
  parseRecallSessionProjection,
  RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES,
  type RecallProjectedEntryDescriptor,
  type RecallSessionProjection,
} from './recall-session-projection.js';

const RECALL_SESSION_PROJECTION_SEGMENT_SCHEMA_VERSION = 1;
const RECALL_SESSION_PROJECTION_SEGMENT_ENCODING = 'base64-json-segments-v1';

/** Constant-size identity and integrity metadata for one segmented session projection. */
export interface RecallSessionProjectionSegmentManifest {
  schemaVersion: 1;
  encoding: 'base64-json-segments-v1';
  projectionKind: RecallSessionProjectionKind;
  projectionId: string;
  segmentCount: number;
  payloadByteLength: number;
  payloadSha256: string;
  logicalEntryDescriptorCount: number | null;
  logicalEntryDescriptorSha256: string | null;
}

/** One bounded byte segment of serialized mutable session projection state. */
export interface RecallSessionProjectionSegment {
  schemaVersion: 1;
  segmentIndex: number;
  payloadBase64: string;
}

/** Bounded projection segments plus the constant-size manifest needed to restore them. */
export interface EncodedRecallSessionProjectionSegments {
  manifest: RecallSessionProjectionSegmentManifest;
  segments: RecallSessionProjectionSegment[];
}

/** Optional segment sizing override used by bounded persistence tests. */
export interface EncodeRecallSessionProjectionSegmentsOptions {
  maxSegmentRecordBytes?: number;
}

/** Immutable entry descriptors loaded from entry anchors when restoring a logical projection. */
export interface DecodeRecallSessionProjectionSegmentsOptions {
  logicalEntryDescriptors?: readonly RecallProjectedEntryDescriptor[];
}

function createPayloadSha256(payload: Buffer): string {
  return createHash('sha256').update(payload).digest('hex');
}

function createSerializedProjectionState(projection: RecallSessionProjection): Buffer {
  if (projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION) {
    return Buffer.from(JSON.stringify(projection), 'utf8');
  }
  const { entryDescriptors: omittedEntryDescriptors, ...logicalState } = projection;
  void omittedEntryDescriptors;
  return Buffer.from(JSON.stringify(logicalState), 'utf8');
}

function createSegmentRecord(
  segmentIndex: number,
  payload: Buffer,
): RecallSessionProjectionSegment {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SEGMENT_SCHEMA_VERSION,
    segmentIndex,
    payloadBase64: payload.toString('base64'),
  };
}

function maxSegmentPayloadBytes(maxSegmentRecordBytes: number, segmentIndex: number): number {
  const emptyRecordBytes = Buffer.byteLength(
    JSON.stringify(createSegmentRecord(segmentIndex, Buffer.alloc(0))),
    'utf8',
  );
  const availableBase64Bytes = maxSegmentRecordBytes - emptyRecordBytes;
  if (availableBase64Bytes < 4) {
    throw new Error(
      `Recall session projection segment bound too small: ${maxSegmentRecordBytes} bytes`,
    );
  }
  let payloadBytes = Math.floor(availableBase64Bytes / 4) * 3;
  while (
    payloadBytes > 0 &&
    Buffer.byteLength(
      JSON.stringify(createSegmentRecord(segmentIndex, Buffer.alloc(payloadBytes))),
      'utf8',
    ) > maxSegmentRecordBytes
  ) {
    payloadBytes -= 1;
  }
  if (payloadBytes === 0) {
    throw new Error(
      `Recall session projection segment bound has no payload capacity: ${maxSegmentRecordBytes} bytes`,
    );
  }
  return payloadBytes;
}

/** Serializes unbounded session state into independently bounded records without duplicating entry descriptors. */
export function encodeRecallSessionProjectionSegments(
  projection: RecallSessionProjection,
  options: EncodeRecallSessionProjectionSegmentsOptions = {},
): EncodedRecallSessionProjectionSegments {
  parseRecallSessionProjection(projection);
  const maxSegmentRecordBytes =
    options.maxSegmentRecordBytes ?? RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES;
  const payload = createSerializedProjectionState(projection);
  const segments: RecallSessionProjectionSegment[] = [];
  let offset = 0;
  while (offset < payload.length || segments.length === 0) {
    const segmentIndex = segments.length;
    const payloadBytes = maxSegmentPayloadBytes(maxSegmentRecordBytes, segmentIndex);
    const nextOffset = Math.min(payload.length, offset + payloadBytes);
    const segment = createSegmentRecord(segmentIndex, payload.subarray(offset, nextOffset));
    const recordBytes = Buffer.byteLength(JSON.stringify(segment), 'utf8');
    if (recordBytes > maxSegmentRecordBytes) {
      throw new Error(
        `Recall session projection segment exceeds bounded record: ${recordBytes} bytes`,
      );
    }
    segments.push(segment);
    offset = nextOffset;
  }
  return {
    manifest: {
      schemaVersion: RECALL_SESSION_PROJECTION_SEGMENT_SCHEMA_VERSION,
      encoding: RECALL_SESSION_PROJECTION_SEGMENT_ENCODING,
      projectionKind: projection.projectionKind,
      projectionId: projection.projectionId,
      segmentCount: segments.length,
      payloadByteLength: payload.length,
      payloadSha256: createPayloadSha256(payload),
      logicalEntryDescriptorCount:
        projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
          ? projection.entryDescriptors.length
          : null,
      logicalEntryDescriptorSha256:
        projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
          ? createPayloadSha256(Buffer.from(JSON.stringify(projection.entryDescriptors), 'utf8'))
          : null,
    },
    segments,
  };
}

function assertSegmentManifest(manifest: RecallSessionProjectionSegmentManifest): void {
  if (
    manifest.schemaVersion !== RECALL_SESSION_PROJECTION_SEGMENT_SCHEMA_VERSION ||
    manifest.encoding !== RECALL_SESSION_PROJECTION_SEGMENT_ENCODING ||
    !Number.isSafeInteger(manifest.segmentCount) ||
    manifest.segmentCount < 1 ||
    !Number.isSafeInteger(manifest.payloadByteLength) ||
    manifest.payloadByteLength < 0 ||
    !/^[a-f0-9]{64}$/u.test(manifest.payloadSha256) ||
    (manifest.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
      ? !Number.isSafeInteger(manifest.logicalEntryDescriptorCount) ||
        Number(manifest.logicalEntryDescriptorCount) < 0 ||
        typeof manifest.logicalEntryDescriptorSha256 !== 'string' ||
        !/^[a-f0-9]{64}$/u.test(manifest.logicalEntryDescriptorSha256)
      : manifest.logicalEntryDescriptorCount !== null ||
        manifest.logicalEntryDescriptorSha256 !== null)
  ) {
    throw new Error(`Recall session projection segment manifest invalid: ${manifest.projectionId}`);
  }
}

/** Restores one complete projection after its immutable logical entry descriptors are loaded from anchors. */
export function decodeRecallSessionProjectionSegments(
  manifest: RecallSessionProjectionSegmentManifest,
  segments: readonly RecallSessionProjectionSegment[],
  options: DecodeRecallSessionProjectionSegmentsOptions = {},
): RecallSessionProjection {
  assertSegmentManifest(manifest);
  if (segments.length !== manifest.segmentCount) {
    throw new Error(
      `Recall session projection segment count mismatch for ${manifest.projectionId}: expected ${manifest.segmentCount}, received ${segments.length}`,
    );
  }
  const orderedSegments = [...segments].toSorted(
    (left, right) => left.segmentIndex - right.segmentIndex,
  );
  const payloadParts = orderedSegments.map((segment, expectedIndex) => {
    const payload = Buffer.from(segment.payloadBase64, 'base64');
    if (
      segment.schemaVersion !== RECALL_SESSION_PROJECTION_SEGMENT_SCHEMA_VERSION ||
      segment.segmentIndex !== expectedIndex ||
      payload.toString('base64') !== segment.payloadBase64
    ) {
      throw new Error(
        `Recall session projection segment invalid for ${manifest.projectionId}: ${expectedIndex}`,
      );
    }
    return payload;
  });
  const payload = Buffer.concat(payloadParts);
  if (
    payload.length !== manifest.payloadByteLength ||
    createPayloadSha256(payload) !== manifest.payloadSha256
  ) {
    throw new Error(
      `Recall session projection segment checksum mismatch: ${manifest.projectionId}`,
    );
  }
  let parsedState: unknown;
  try {
    parsedState = JSON.parse(payload.toString('utf8'));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session projection segmented JSON invalid: ${message}`, {
      cause: error,
    });
  }
  if (typeof parsedState !== 'object' || parsedState === null) {
    throw new Error(`Recall session projection segmented state invalid: ${manifest.projectionId}`);
  }
  const logicalEntryDescriptors = [...(options.logicalEntryDescriptors ?? [])];
  if (
    manifest.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION &&
    (logicalEntryDescriptors.length !== manifest.logicalEntryDescriptorCount ||
      createPayloadSha256(Buffer.from(JSON.stringify(logicalEntryDescriptors), 'utf8')) !==
        manifest.logicalEntryDescriptorSha256)
  ) {
    throw new Error(
      `Recall session projection entry descriptor checksum mismatch: ${manifest.projectionId}`,
    );
  }
  const candidate =
    manifest.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION
      ? {
          ...parsedState,
          entryDescriptors: logicalEntryDescriptors,
        }
      : parsedState;
  const projection = parseRecallSessionProjection(candidate);
  if (
    projection.projectionKind !== manifest.projectionKind ||
    projection.projectionId !== manifest.projectionId
  ) {
    throw new Error(
      `Recall session projection segment identity mismatch: ${manifest.projectionId}`,
    );
  }
  return projection;
}

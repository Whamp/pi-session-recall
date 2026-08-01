import { createHash } from 'node:crypto';

import type { ZVecCollection } from '@zvec/zvec';

import {
  RecallGenerationSessionProjectionRecordKind,
  RecallSessionProjectionKind,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  decodeRecallSessionProjectionSegments,
  encodeRecallSessionProjectionSegments,
  type RecallSessionProjectionSegment,
  type RecallSessionProjectionSegmentManifest,
} from './recall-session-projection-segments.js';
import {
  parseRecallProjectedEntryDescriptor,
  RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES,
  type RecallProjectedEntryDescriptor,
  type RecallSessionProjection,
} from './recall-session-projection.js';
import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';

const RECALL_GENERATION_SESSION_PROJECTION_RECORD_SCHEMA_VERSION = 1;

/** Caller-owned target metadata stored beside one segmented ingestion projection manifest. */
export type RecallGenerationSessionProjectionMetadata = Readonly<Record<string, unknown>>;

/** Inputs for one constant-size projection head and its deterministic bounded segment rows. */
export interface CreateRecallGenerationSessionProjectionRecordsOptions {
  generationId: string;
  physicalSourceIdentity: string;
  logicalSessionOccurrenceId: string;
  projectionRowId: string;
  projection: RecallSessionProjection;
  metadata: RecallGenerationSessionProjectionMetadata;
}

/** One scalar-only session projection store record. */
export interface RecallGenerationSessionProjectionRow {
  id: string;
  fields: Record<string, unknown>;
}

/** Complete scalar row set for one physical or logical session projection. */
export interface RecallGenerationSessionProjectionRecords {
  headRow: RecallGenerationSessionProjectionRow;
  segmentRows: RecallGenerationSessionProjectionRow[];
}

/** Restored ingestion projection and caller-owned metadata from one projection head. */
export interface ReadRecallGenerationSessionProjectionRecord {
  metadata: Record<string, unknown>;
  projection: RecallSessionProjection;
}

/** Options for restoring one segmented projection from a generation-local collection. */
export interface ReadRecallGenerationSessionProjectionRecordOptions {
  collection: ZVecCollection;
  lexicalSourceCollection?: ZVecCollection;
  generationId: string;
  projectionRowId: string;
}

/** Derives one stable segment row ID from its projection head and zero-based position. */
export function createRecallSessionProjectionSegmentRecordId(
  projectionRowId: string,
  segmentIndex: number,
): string {
  if (!Number.isSafeInteger(segmentIndex) || segmentIndex < 0) {
    throw new Error(`Recall session projection segment index invalid: ${segmentIndex}`);
  }
  const digest = createHash('sha256')
    .update(JSON.stringify(['recall_session_projection_segment_v1', projectionRowId, segmentIndex]))
    .digest('base64url');
  return `projection_segment_${digest}`;
}

function createProjectionRecordFields(
  options: CreateRecallGenerationSessionProjectionRecordsOptions,
  projectionRecordId: string,
  projectionKind: RecallGenerationSessionProjectionRecordKind,
  projectionJson: string,
): RecallGenerationSessionProjectionRow['fields'] {
  return {
    schemaVersion: RECALL_GENERATION_SESSION_PROJECTION_RECORD_SCHEMA_VERSION,
    generationId: options.generationId,
    projectionRecordId,
    projectionKind,
    physicalSourceIdentity: options.physicalSourceIdentity,
    logicalSessionOccurrenceId: options.logicalSessionOccurrenceId,
    projectionJson,
  };
}

function assertBoundedProjectionRecord(projectionRecordId: string, projectionJson: string): void {
  const payloadBytes = Buffer.byteLength(projectionJson, 'utf8');
  if (payloadBytes > RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES) {
    throw new Error(
      `Recall generation session projection record exceeds bounded payload: ${projectionRecordId} (${payloadBytes} bytes)`,
    );
  }
}

/** Creates bounded generation rows without storing immutable logical entry descriptors twice. */
export function createRecallGenerationSessionProjectionRecords(
  options: CreateRecallGenerationSessionProjectionRecordsOptions,
): RecallGenerationSessionProjectionRecords {
  if (
    'ingestionProjectionPayload' in options.metadata ||
    'ingestionProjectionSegments' in options.metadata ||
    'ingestionProjectionInlineSegment' in options.metadata
  ) {
    throw new Error(
      `Recall generation session projection metadata contains reserved ingestion state: ${options.projectionRowId}`,
    );
  }
  const encoded = encodeRecallSessionProjectionSegments(options.projection);
  const headMetadata = {
    ...options.metadata,
    schemaVersion: RECALL_GENERATION_SESSION_PROJECTION_RECORD_SCHEMA_VERSION,
    ingestionProjectionSegments: encoded.manifest,
  };
  const inlineHeadJson =
    encoded.segments.length === 1
      ? JSON.stringify({
          ...headMetadata,
          ingestionProjectionInlineSegment: encoded.segments[0],
        })
      : null;
  const storesInlineSegment =
    inlineHeadJson !== null &&
    Buffer.byteLength(inlineHeadJson, 'utf8') <= RECALL_SESSION_PROJECTION_RECORD_MAX_BYTES;
  const headJson = storesInlineSegment ? inlineHeadJson : JSON.stringify(headMetadata);
  assertBoundedProjectionRecord(options.projectionRowId, headJson);
  const headRow: RecallGenerationSessionProjectionRow = {
    id: options.projectionRowId,
    fields: createProjectionRecordFields(
      options,
      options.projectionRowId,
      options.projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION
        ? RecallGenerationSessionProjectionRecordKind.PHYSICAL_SESSION
        : RecallGenerationSessionProjectionRecordKind.LOGICAL_SESSION,
      headJson,
    ),
  };
  const segmentRows = (storesInlineSegment ? [] : encoded.segments).map((segment) => {
    const projectionRecordId = createRecallSessionProjectionSegmentRecordId(
      options.projectionRowId,
      segment.segmentIndex,
    );
    const projectionJson = JSON.stringify(segment);
    assertBoundedProjectionRecord(projectionRecordId, projectionJson);
    return {
      id: projectionRecordId,
      fields: createProjectionRecordFields(
        options,
        projectionRecordId,
        RecallGenerationSessionProjectionRecordKind.PROJECTION_SEGMENT,
        projectionJson,
      ),
    };
  });
  return { headRow, segmentRows };
}

function parseProjectionRecordJson(
  projectionRecordId: string,
  value: unknown,
): Record<string, unknown> {
  if (typeof value !== 'string') {
    throw new Error(`Recall generation session projection JSON missing: ${projectionRecordId}`);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall generation session projection JSON invalid for ${projectionRecordId}: ${message}`,
      { cause: error },
    );
  }
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Recall generation session projection record invalid: ${projectionRecordId}`);
  }
  return parsed;
}

function readProjectionSegmentManifest(
  projectionRowId: string,
  metadata: Record<string, unknown>,
): RecallSessionProjectionSegmentManifest {
  const value = metadata.ingestionProjectionSegments;
  if (
    !isUnknownRecord(value) ||
    value.schemaVersion !== 1 ||
    value.encoding !== 'base64-json-segments-v1' ||
    (value.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION &&
      value.projectionKind !== RecallSessionProjectionKind.LOGICAL_SESSION) ||
    typeof value.projectionId !== 'string' ||
    !Number.isSafeInteger(value.segmentCount) ||
    !Number.isSafeInteger(value.payloadByteLength) ||
    typeof value.payloadSha256 !== 'string' ||
    !(
      value.logicalEntryDescriptorCount === null ||
      Number.isSafeInteger(value.logicalEntryDescriptorCount)
    ) ||
    !(
      value.logicalEntryDescriptorSha256 === null ||
      typeof value.logicalEntryDescriptorSha256 === 'string'
    )
  ) {
    throw new Error(`Recall generation session projection segments missing: ${projectionRowId}`);
  }
  return {
    schemaVersion: value.schemaVersion,
    encoding: value.encoding,
    projectionKind: value.projectionKind,
    projectionId: value.projectionId,
    segmentCount: Number(value.segmentCount),
    payloadByteLength: Number(value.payloadByteLength),
    payloadSha256: value.payloadSha256,
    logicalEntryDescriptorCount:
      value.logicalEntryDescriptorCount === null ? null : Number(value.logicalEntryDescriptorCount),
    logicalEntryDescriptorSha256: value.logicalEntryDescriptorSha256,
  };
}

function readProjectionSegment(
  projectionRecordId: string,
  projectionJson: unknown,
): RecallSessionProjectionSegment {
  const value =
    typeof projectionJson === 'string'
      ? parseProjectionRecordJson(projectionRecordId, projectionJson)
      : projectionJson;
  if (
    !isUnknownRecord(value) ||
    value.schemaVersion !== 1 ||
    !Number.isSafeInteger(value.segmentIndex) ||
    typeof value.payloadBase64 !== 'string'
  ) {
    throw new Error(`Recall generation session projection segment invalid: ${projectionRecordId}`);
  }
  return {
    schemaVersion: value.schemaVersion,
    segmentIndex: Number(value.segmentIndex),
    payloadBase64: value.payloadBase64,
  };
}

function readLogicalEntryDescriptorsFromAnchors(
  collection: ZVecCollection,
  logicalSessionOccurrenceId: string,
  expectedDescriptorCount: number,
): RecallProjectedEntryDescriptor[] {
  if (!/^[A-Za-z0-9_-]+$/u.test(logicalSessionOccurrenceId)) {
    throw new Error(
      `Recall generation logical session occurrence ID invalid: ${logicalSessionOccurrenceId}`,
    );
  }
  const descriptors: RecallProjectedEntryDescriptor[] = [];
  visitExactZvecDocuments(
    collection,
    {
      filter: `recordKind = 'entry-anchor' AND logicalSessionOccurrenceId = '${logicalSessionOccurrenceId}'`,
      uniquePartitionField: 'entryAnchorId',
      outputFields: ['recordJson'],
    },
    (document) => {
      const parsed = parseProjectionRecordJson(document.id, document.fields.recordJson);
      if (
        parsed.recordKind !== 'entry-anchor' ||
        parsed.logicalSessionOccurrenceId !== logicalSessionOccurrenceId
      ) {
        throw new Error(`Recall generation logical session entry anchor invalid: ${document.id}`);
      }
      descriptors.push(parseRecallProjectedEntryDescriptor(parsed.descriptor));
    },
  );
  const orderedDescriptors = descriptors.toSorted(
    (left, right) => left.sourceLine - right.sourceLine,
  );
  if (orderedDescriptors.length < expectedDescriptorCount) {
    throw new Error(
      `Recall generation logical session entry anchors incomplete: ${logicalSessionOccurrenceId}`,
    );
  }
  return orderedDescriptors.slice(0, expectedDescriptorCount);
}

function readExternalProjectionSegments(
  options: ReadRecallGenerationSessionProjectionRecordOptions,
  manifest: RecallSessionProjectionSegmentManifest,
): RecallSessionProjectionSegment[] {
  const segmentIds = Array.from({ length: manifest.segmentCount }, (_, segmentIndex) =>
    createRecallSessionProjectionSegmentRecordId(options.projectionRowId, segmentIndex),
  );
  const fetchedSegments = options.collection.fetchSync({
    ids: segmentIds,
    outputFields: [
      'schemaVersion',
      'generationId',
      'projectionRecordId',
      'projectionKind',
      'physicalSourceIdentity',
      'logicalSessionOccurrenceId',
      'projectionJson',
    ],
    includeVector: false,
  });
  return segmentIds.map((segmentId) => {
    const segment = fetchedSegments[segmentId];
    if (
      segment === undefined ||
      segment.fields.schemaVersion !== RECALL_GENERATION_SESSION_PROJECTION_RECORD_SCHEMA_VERSION ||
      segment.fields.generationId !== options.generationId ||
      segment.fields.projectionRecordId !== segmentId ||
      segment.fields.projectionKind !==
        RecallGenerationSessionProjectionRecordKind.PROJECTION_SEGMENT
    ) {
      throw new Error(`Recall generation session projection segment mismatch: ${segmentId}`);
    }
    return readProjectionSegment(segmentId, segment.fields.projectionJson);
  });
}

/** Restores one projection from bounded records and immutable logical entry anchors. */
export function readRecallGenerationSessionProjectionRecord(
  options: ReadRecallGenerationSessionProjectionRecordOptions,
): ReadRecallGenerationSessionProjectionRecord {
  const head = options.collection.fetchSync({
    ids: [options.projectionRowId],
    outputFields: [
      'schemaVersion',
      'generationId',
      'projectionRecordId',
      'projectionKind',
      'physicalSourceIdentity',
      'logicalSessionOccurrenceId',
      'projectionJson',
    ],
    includeVector: false,
  })[options.projectionRowId];
  if (
    head === undefined ||
    head.fields.schemaVersion !== RECALL_GENERATION_SESSION_PROJECTION_RECORD_SCHEMA_VERSION ||
    head.fields.generationId !== options.generationId ||
    head.fields.projectionRecordId !== options.projectionRowId
  ) {
    throw new Error(
      `Recall generation session projection head mismatch: ${options.projectionRowId}`,
    );
  }
  const metadata = parseProjectionRecordJson(options.projectionRowId, head.fields.projectionJson);
  const manifest = readProjectionSegmentManifest(options.projectionRowId, metadata);
  const inlineSegment = metadata.ingestionProjectionInlineSegment;
  const segments =
    inlineSegment === undefined
      ? readExternalProjectionSegments(options, manifest)
      : [readProjectionSegment(options.projectionRowId, inlineSegment)];
  if (inlineSegment !== undefined && manifest.segmentCount !== 1) {
    throw new Error(
      `Recall generation inline session projection segment count mismatch: ${options.projectionRowId}`,
    );
  }
  const logicalEntryDescriptors =
    manifest.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION &&
    options.lexicalSourceCollection !== undefined
      ? readLogicalEntryDescriptorsFromAnchors(
          options.lexicalSourceCollection,
          String(head.fields.logicalSessionOccurrenceId),
          Number(manifest.logicalEntryDescriptorCount),
        )
      : undefined;
  if (
    manifest.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION &&
    logicalEntryDescriptors === undefined
  ) {
    throw new Error(
      `Recall generation logical session projection entry anchors unavailable: ${options.projectionRowId}`,
    );
  }
  const projection =
    logicalEntryDescriptors === undefined
      ? decodeRecallSessionProjectionSegments(manifest, segments)
      : decodeRecallSessionProjectionSegments(manifest, segments, {
          logicalEntryDescriptors,
        });
  const {
    ingestionProjectionSegments: omittedSegments,
    ingestionProjectionInlineSegment: omittedInlineSegment,
    ...callerMetadata
  } = metadata;
  void omittedSegments;
  void omittedInlineSegment;
  return { metadata: callerMetadata, projection };
}

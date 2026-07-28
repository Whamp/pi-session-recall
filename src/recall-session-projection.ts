import { createHash } from 'node:crypto';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  RecallProjectionEncodingStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';

/** Current strict schema version for physical and logical session projections. */
export const RECALL_SESSION_PROJECTION_SCHEMA_VERSION = 1;

/** Maximum encoded scalar projection candidate size accepted without reconciliation. */
export const RECALL_SESSION_PROJECTION_MAX_BYTES = 1_048_576;

/** Highest durably processed marker sequence for one Pi runtime instance. */
export interface RecallMarkerRuntimeCheckpoint {
  runtimeInstanceId: string;
  sequence: number;
}

/** Generation-scoped durable marker coverage committed with a session projection. */
export interface RecallMarkerCheckpoint {
  generationId: string;
  coveredMarkerIds: string[];
  runtimeSequences: RecallMarkerRuntimeCheckpoint[];
}

/** Contiguous entry boundary for the currently active logical-session context. */
export interface RecallActiveContextBoundary {
  firstEntryId: string;
  lastEntryId: string;
}

/** Latest durable compaction boundary and its first retained entry. */
export interface RecallCompactionBoundary {
  compactionEntryId: string;
  firstRetainedEntryId: string;
}

/** One preserved old-leaf transition that cannot be reconstructed from JSONL alone. */
export interface RecallPreservedBranchExit {
  oldLeafEntryId: string;
  newLeafEntryId: string;
  summaryEntryId: string | null;
}

/** One immutable recall-eligible byte span and its bounding source entries. */
export interface RecallEligibleSourceSpan {
  startByte: number;
  endByte: number;
  startEntryId: string;
  endEntryId: string;
}

interface RecallSessionProjectionBase {
  schemaVersion: 1;
  projectionKind: RecallSessionProjectionKind;
  projectionId: string;
  generationId: string;
  physicalSessionId: string;
  markerCheckpoint: RecallMarkerCheckpoint;
  repairState: RecallProjectionRepairState;
  repairReason: RecallProjectionRepairReason | null;
}

/** Mutable append, source-availability, and logical-membership state for one physical file. */
export interface PhysicalSessionProjection extends RecallSessionProjectionBase {
  projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION;
  sourcePath: string;
  appendCursorBytes: number;
  boundaryFingerprint: string;
  lastEntryId: string | null;
  logicalSessionIds: string[];
  sourceAvailability: RecallSourceAvailability;
  sourceMissingObservedAtEpochMilliseconds: number | null;
  sourceMissingObservationCount: number;
}

/** Mutable branch, context, compaction, eligibility, and label state for one logical session. */
export interface LogicalSessionProjection extends RecallSessionProjectionBase {
  projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION;
  physicalProjectionId: string;
  logicalSessionId: string;
  effectiveLeafEntryId: string | null;
  activeContextBoundary: RecallActiveContextBoundary | null;
  compactionBoundary: RecallCompactionBoundary | null;
  preservedBranchExits: RecallPreservedBranchExit[];
  eligibleSpans: RecallEligibleSourceSpan[];
  labels: string[];
}

/** Strict physical or logical mutable session projection. */
export type RecallSessionProjection = PhysicalSessionProjection | LogicalSessionProjection;

/** Scalar-only zvec candidate containing one strict projection as canonical JSON. */
export interface EncodedRecallSessionProjectionPayload {
  schemaVersion: 1;
  projectionKind: RecallSessionProjectionKind;
  projectionId: string;
  generationId: string;
  projectionJson: string;
}

/** Options controlling projection encoding bounds for production and boundary tests. */
export interface RecallSessionProjectionEncodingOptions {
  maxPayloadBytes?: number;
}

/** Expected generation applied while decoding one projection collection record. */
export interface RecallSessionProjectionDecodingOptions {
  expectedGenerationId: string;
}

/** Successfully encoded scalar-only projection candidate. */
export interface EncodedRecallSessionProjection {
  status: RecallProjectionEncodingStatus.ENCODED;
  payload: EncodedRecallSessionProjectionPayload;
  byteLength: number;
}

/** Oversized projection classification that requires explicit reconciliation without truncation. */
export interface OverflowRecallSessionProjection {
  status: RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION;
  repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW;
  byteLength: number;
}

/** Bounded projection encoding result. */
export type RecallSessionProjectionEncodingResult =
  | EncodedRecallSessionProjection
  | OverflowRecallSessionProjection;

const nonemptyStringSchema = Type.String({ minLength: 1 });
const nullableIdentifierSchema = Type.Union([nonemptyStringSchema, Type.Null()]);
const repairReasonSchema = Type.Union([Type.Enum(RecallProjectionRepairReason), Type.Null()]);
const markerRuntimeCheckpointSchema = Type.Object(
  {
    runtimeInstanceId: nonemptyStringSchema,
    sequence: Type.Integer({ minimum: 1 }),
  },
  { additionalProperties: false },
);
const markerCheckpointSchema = Type.Object(
  {
    generationId: nonemptyStringSchema,
    coveredMarkerIds: Type.Array(nonemptyStringSchema),
    runtimeSequences: Type.Array(markerRuntimeCheckpointSchema),
  },
  { additionalProperties: false },
);
const projectionBaseSchema = {
  schemaVersion: Type.Literal(RECALL_SESSION_PROJECTION_SCHEMA_VERSION),
  projectionId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
  generationId: nonemptyStringSchema,
  physicalSessionId: nonemptyStringSchema,
  markerCheckpoint: markerCheckpointSchema,
  repairState: Type.Enum(RecallProjectionRepairState),
  repairReason: repairReasonSchema,
};
const physicalSessionProjectionSchema = Type.Object(
  {
    ...projectionBaseSchema,
    projectionKind: Type.Literal(RecallSessionProjectionKind.PHYSICAL_SESSION),
    sourcePath: nonemptyStringSchema,
    appendCursorBytes: Type.Integer({ minimum: 0 }),
    boundaryFingerprint: Type.String({ pattern: '^[a-f0-9]{64}$' }),
    lastEntryId: nullableIdentifierSchema,
    logicalSessionIds: Type.Array(nonemptyStringSchema),
    sourceAvailability: Type.Enum(RecallSourceAvailability),
    sourceMissingObservedAtEpochMilliseconds: Type.Union([
      Type.Integer({ minimum: 0 }),
      Type.Null(),
    ]),
    sourceMissingObservationCount: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);
const logicalSessionProjectionSchema = Type.Object(
  {
    ...projectionBaseSchema,
    projectionKind: Type.Literal(RecallSessionProjectionKind.LOGICAL_SESSION),
    physicalProjectionId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    logicalSessionId: nonemptyStringSchema,
    effectiveLeafEntryId: nullableIdentifierSchema,
    activeContextBoundary: Type.Union([
      Type.Object(
        { firstEntryId: nonemptyStringSchema, lastEntryId: nonemptyStringSchema },
        { additionalProperties: false },
      ),
      Type.Null(),
    ]),
    compactionBoundary: Type.Union([
      Type.Object(
        {
          compactionEntryId: nonemptyStringSchema,
          firstRetainedEntryId: nonemptyStringSchema,
        },
        { additionalProperties: false },
      ),
      Type.Null(),
    ]),
    preservedBranchExits: Type.Array(
      Type.Object(
        {
          oldLeafEntryId: nonemptyStringSchema,
          newLeafEntryId: nonemptyStringSchema,
          summaryEntryId: nullableIdentifierSchema,
        },
        { additionalProperties: false },
      ),
    ),
    eligibleSpans: Type.Array(
      Type.Object(
        {
          startByte: Type.Integer({ minimum: 0 }),
          endByte: Type.Integer({ minimum: 0 }),
          startEntryId: nonemptyStringSchema,
          endEntryId: nonemptyStringSchema,
        },
        { additionalProperties: false },
      ),
    ),
    labels: Type.Array(Type.String()),
  },
  { additionalProperties: false },
);
const recallSessionProjectionSchema = Type.Union([
  physicalSessionProjectionSchema,
  logicalSessionProjectionSchema,
]);
const encodedRecallSessionProjectionPayloadSchema = Type.Object(
  {
    schemaVersion: Type.Literal(RECALL_SESSION_PROJECTION_SCHEMA_VERSION),
    projectionKind: Type.Enum(RecallSessionProjectionKind),
    projectionId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    generationId: nonemptyStringSchema,
    projectionJson: nonemptyStringSchema,
  },
  { additionalProperties: false },
);

function createProjectionDigest(domain: string, parts: readonly string[]): string {
  return createHash('sha256')
    .update(JSON.stringify([domain, ...parts]))
    .digest('base64url');
}

/** Derives the deterministic zvec-safe ID for one physical session projection. */
export function createPhysicalSessionProjectionId(physicalSessionId: string): string {
  return `physical_${createProjectionDigest('physical_session_projection_v1', [physicalSessionId])}`;
}

/** Derives the deterministic zvec-safe ID for one logical session projection. */
export function createLogicalSessionProjectionId(
  physicalSessionId: string,
  logicalSessionId: string,
): string {
  return `logical_${createProjectionDigest('logical_session_projection_v1', [
    physicalSessionId,
    logicalSessionId,
  ])}`;
}

function assertUniqueProjectionValues(values: readonly string[], fieldName: string): void {
  if (new Set(values).size !== values.length) {
    throw new Error(`Recall session projection invalid ${fieldName}: duplicate values`);
  }
}

function assertRecallMarkerCheckpoint(
  checkpoint: RecallMarkerCheckpoint,
  generationId: string,
): void {
  if (checkpoint.generationId !== generationId) {
    throw new Error(
      `Recall session projection generation mismatch: projection ${generationId}, marker checkpoint ${checkpoint.generationId}`,
    );
  }
  assertUniqueProjectionValues(checkpoint.coveredMarkerIds, 'covered marker IDs');
  assertUniqueProjectionValues(
    checkpoint.runtimeSequences.map(({ runtimeInstanceId }) => runtimeInstanceId),
    'runtime checkpoint instance IDs',
  );
}

function assertRecallProjectionRepairState(projection: RecallSessionProjection): void {
  const isReady = projection.repairState === RecallProjectionRepairState.READY;
  if (isReady !== (projection.repairReason === null)) {
    throw new Error(
      'Recall session projection repair state invalid: ready requires no repair reason and reconciliation requires one repair reason',
    );
  }
}

function assertRecallPhysicalSourceState(projection: PhysicalSessionProjection): void {
  if (projection.sourceAvailability === RecallSourceAvailability.PRESENT) {
    if (
      projection.sourceMissingObservedAtEpochMilliseconds !== null ||
      projection.sourceMissingObservationCount !== 0
    ) {
      throw new Error(
        'Recall physical session projection source state invalid: present sources cannot retain missing-source observations',
      );
    }
    return;
  }
  if (
    projection.sourceMissingObservedAtEpochMilliseconds === null ||
    projection.sourceMissingObservationCount < 1
  ) {
    throw new Error(
      'Recall physical session projection source state invalid: missing sources require an observation time and count',
    );
  }
  if (
    projection.sourceAvailability === RecallSourceAvailability.DELETION_CONFIRMED &&
    projection.sourceMissingObservationCount < 2
  ) {
    throw new Error(
      'Recall physical session projection source state invalid: confirmed deletion requires two missing-source observations',
    );
  }
}

function assertRecallSessionProjectionInvariants(projection: RecallSessionProjection): void {
  assertRecallProjectionRepairState(projection);
  assertRecallMarkerCheckpoint(projection.markerCheckpoint, projection.generationId);
  const expectedProjectionId =
    projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION
      ? createPhysicalSessionProjectionId(projection.physicalSessionId)
      : createLogicalSessionProjectionId(projection.physicalSessionId, projection.logicalSessionId);
  if (projection.projectionId !== expectedProjectionId) {
    throw new Error(
      `Recall session projection ID mismatch: expected ${expectedProjectionId}, received ${projection.projectionId}`,
    );
  }
  if (projection.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION) {
    assertRecallPhysicalSourceState(projection);
    assertUniqueProjectionValues(projection.logicalSessionIds, 'logical session IDs');
    return;
  }
  const expectedPhysicalProjectionId = createPhysicalSessionProjectionId(
    projection.physicalSessionId,
  );
  if (projection.physicalProjectionId !== expectedPhysicalProjectionId) {
    throw new Error(
      `Recall logical session projection physical ID mismatch: expected ${expectedPhysicalProjectionId}, received ${projection.physicalProjectionId}`,
    );
  }
  for (const span of projection.eligibleSpans) {
    if (span.endByte <= span.startByte) {
      throw new Error(
        `Recall logical session projection eligible span invalid: ${span.startByte}-${span.endByte}`,
      );
    }
  }
}

function parseRecallSessionProjection(value: unknown): RecallSessionProjection {
  let projection: RecallSessionProjection;
  try {
    projection = Value.Parse(recallSessionProjectionSchema, value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session projection invalid: ${message}`, { cause: error });
  }
  assertRecallSessionProjectionInvariants(projection);
  return projection;
}

/** Encodes one projection into scalar zvec fields or classifies overflow without truncation. */
export function encodeRecallSessionProjection(
  candidate: RecallSessionProjection,
  options: RecallSessionProjectionEncodingOptions = {},
): RecallSessionProjectionEncodingResult {
  const maxPayloadBytes = options.maxPayloadBytes ?? RECALL_SESSION_PROJECTION_MAX_BYTES;
  if (!Number.isSafeInteger(maxPayloadBytes) || maxPayloadBytes < 1) {
    throw new Error(`Recall session projection maximum payload bytes invalid: ${maxPayloadBytes}`);
  }
  const projection = parseRecallSessionProjection(candidate);
  const payload: EncodedRecallSessionProjectionPayload = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: projection.projectionKind,
    projectionId: projection.projectionId,
    generationId: projection.generationId,
    projectionJson: JSON.stringify(projection),
  };
  const byteLength = Buffer.byteLength(JSON.stringify(payload));
  if (byteLength > maxPayloadBytes) {
    return {
      status: RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION,
      repairReason: RecallProjectionRepairReason.PROJECTION_OVERFLOW,
      byteLength,
    };
  }
  return { status: RecallProjectionEncodingStatus.ENCODED, payload, byteLength };
}

/** Strictly decodes one scalar projection record and rejects cross-generation evidence. */
export function decodeRecallSessionProjection(
  candidate: unknown,
  options: RecallSessionProjectionDecodingOptions,
): RecallSessionProjection {
  let payload: EncodedRecallSessionProjectionPayload;
  try {
    payload = Value.Parse(encodedRecallSessionProjectionPayloadSchema, candidate);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session projection payload invalid: ${message}`, { cause: error });
  }
  if (payload.generationId !== options.expectedGenerationId) {
    throw new Error(
      `Recall session projection generation mismatch: expected ${options.expectedGenerationId}, received ${payload.generationId}`,
    );
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(payload.projectionJson);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall session projection JSON invalid: ${message}`, { cause: error });
  }
  const projection = parseRecallSessionProjection(parsed);
  if (projection.generationId !== payload.generationId) {
    throw new Error(
      `Recall session projection generation mismatch: envelope ${payload.generationId}, record ${projection.generationId}`,
    );
  }
  if (
    projection.schemaVersion !== payload.schemaVersion ||
    projection.projectionKind !== payload.projectionKind ||
    projection.projectionId !== payload.projectionId
  ) {
    throw new Error('Recall session projection payload identity mismatch');
  }
  return projection;
}

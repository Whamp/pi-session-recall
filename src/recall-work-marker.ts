import { createHash } from 'node:crypto';
import { realpath } from 'node:fs/promises';
import { extname, isAbsolute } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallWorkMarkerTrigger } from './enums.js';
import {
  isCanonicalPathWithinBoundary,
  resolveCanonicalPathBoundary,
} from './trusted-path-boundary.js';

/** Current strict wire version for immutable recall work markers. */
export const RECALL_WORK_MARKER_VERSION = 1;

/** Session-root trust boundary applied while encoding or decoding a recall work marker. */
export interface RecallWorkMarkerCodecOptions {
  trustedSessionRoots: readonly string[];
}

/** Marker payload for append activity that introduces no eligibility boundary by itself. */
export interface RecallActivityMarkerTrigger {
  kind: RecallWorkMarkerTrigger.ACTIVITY;
}

/** Marker payload identifying the durable compaction entry that creates an eligibility boundary. */
export interface RecallCompactionMarkerTrigger {
  kind: RecallWorkMarkerTrigger.COMPACTION;
  compactionEntryId: string;
}

/** Marker payload preserving one old-leaf to new-leaf context exit. */
export interface RecallBranchExitMarkerTrigger {
  kind: RecallWorkMarkerTrigger.BRANCH_EXIT;
  oldLeafEntryId: string;
  newLeafEntryId: string;
  summaryEntryId?: string;
}

/** Marker payload for a clean runtime departure from one physical session file. */
export interface RecallDepartureMarkerTrigger {
  kind: RecallWorkMarkerTrigger.DEPARTURE;
}

/** Marker payload for a runtime arriving at one physical session file. */
export interface RecallArrivalMarkerTrigger {
  kind: RecallWorkMarkerTrigger.ARRIVAL;
}

/** Strict trigger-specific payload carried by a recall work marker. */
export type RecallWorkMarkerTriggerPayload =
  | RecallActivityMarkerTrigger
  | RecallCompactionMarkerTrigger
  | RecallBranchExitMarkerTrigger
  | RecallDepartureMarkerTrigger
  | RecallArrivalMarkerTrigger;

/** Identity-bearing marker fields used to derive one deterministic marker ID. */
export interface RecallWorkMarkerIdentity {
  version: 1;
  physicalSessionId: string;
  physicalSessionPath: string;
  runtimeInstanceId: string;
  runtimeSequence: number;
  createdAtEpochMilliseconds: number;
  trigger: RecallWorkMarkerTriggerPayload;
}

/** One strict immutable wake-up marker published outside zvec. */
export interface RecallWorkMarker extends RecallWorkMarkerIdentity {
  markerId: string;
}

const nonemptyIdentifierSchema = Type.String({ minLength: 1 });
const markerTriggerSchema = Type.Union([
  Type.Object(
    { kind: Type.Literal(RecallWorkMarkerTrigger.ACTIVITY) },
    { additionalProperties: false },
  ),
  Type.Object(
    {
      kind: Type.Literal(RecallWorkMarkerTrigger.COMPACTION),
      compactionEntryId: nonemptyIdentifierSchema,
    },
    { additionalProperties: false },
  ),
  Type.Object(
    {
      kind: Type.Literal(RecallWorkMarkerTrigger.BRANCH_EXIT),
      oldLeafEntryId: nonemptyIdentifierSchema,
      newLeafEntryId: nonemptyIdentifierSchema,
      summaryEntryId: Type.Optional(nonemptyIdentifierSchema),
    },
    { additionalProperties: false },
  ),
  Type.Object(
    { kind: Type.Literal(RecallWorkMarkerTrigger.DEPARTURE) },
    { additionalProperties: false },
  ),
  Type.Object(
    { kind: Type.Literal(RecallWorkMarkerTrigger.ARRIVAL) },
    { additionalProperties: false },
  ),
]);

const recallWorkMarkerSchema = Type.Object(
  {
    version: Type.Literal(RECALL_WORK_MARKER_VERSION),
    markerId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    physicalSessionId: nonemptyIdentifierSchema,
    physicalSessionPath: Type.String({ minLength: 1 }),
    runtimeInstanceId: nonemptyIdentifierSchema,
    runtimeSequence: Type.Integer({ minimum: 1 }),
    createdAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
    trigger: markerTriggerSchema,
  },
  { additionalProperties: false },
);

function canonicalizeRecallMarkerTrigger(trigger: RecallWorkMarkerTriggerPayload): object {
  switch (trigger.kind) {
    case RecallWorkMarkerTrigger.ACTIVITY:
      return { kind: RecallWorkMarkerTrigger.ACTIVITY };
    case RecallWorkMarkerTrigger.COMPACTION:
      return {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        compactionEntryId: trigger.compactionEntryId,
      };
    case RecallWorkMarkerTrigger.BRANCH_EXIT:
      return trigger.summaryEntryId === undefined
        ? {
            kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
            oldLeafEntryId: trigger.oldLeafEntryId,
            newLeafEntryId: trigger.newLeafEntryId,
          }
        : {
            kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
            oldLeafEntryId: trigger.oldLeafEntryId,
            newLeafEntryId: trigger.newLeafEntryId,
            summaryEntryId: trigger.summaryEntryId,
          };
    case RecallWorkMarkerTrigger.DEPARTURE:
      return { kind: RecallWorkMarkerTrigger.DEPARTURE };
    case RecallWorkMarkerTrigger.ARRIVAL:
      return { kind: RecallWorkMarkerTrigger.ARRIVAL };
    default:
      throw new Error('Recall work marker trigger unsupported');
  }
}

/** Derives a deterministic zvec-safe marker ID from all immutable marker fields. */
export function createRecallWorkMarkerId(marker: RecallWorkMarkerIdentity): string {
  const canonical = JSON.stringify({
    version: marker.version,
    physicalSessionId: marker.physicalSessionId,
    physicalSessionPath: marker.physicalSessionPath,
    runtimeInstanceId: marker.runtimeInstanceId,
    runtimeSequence: marker.runtimeSequence,
    createdAtEpochMilliseconds: marker.createdAtEpochMilliseconds,
    trigger: canonicalizeRecallMarkerTrigger(marker.trigger),
  });
  return `marker_${createHash('sha256').update(canonical).digest('base64url')}`;
}

async function assertTrustedPhysicalSessionPath(
  physicalSessionPath: string,
  trustedSessionRoots: readonly string[],
): Promise<void> {
  if (!isAbsolute(physicalSessionPath)) {
    throw new Error(
      `Recall work marker physical session path must be absolute: ${physicalSessionPath}`,
    );
  }
  if (extname(physicalSessionPath) !== '.jsonl') {
    throw new Error(
      `Recall work marker physical session path must name a JSONL file: ${physicalSessionPath}`,
    );
  }
  if (trustedSessionRoots.length === 0) {
    throw new Error('Recall work marker requires at least one trusted session root');
  }
  const canonicalPath = await resolveCanonicalPathBoundary(physicalSessionPath);
  for (const trustedSessionRoot of trustedSessionRoots) {
    if (!isAbsolute(trustedSessionRoot)) {
      throw new Error(
        `Recall work marker trusted session root must be absolute: ${trustedSessionRoot}`,
      );
    }
    const canonicalRoot = await realpath(trustedSessionRoot);
    if (isCanonicalPathWithinBoundary(canonicalPath, canonicalRoot)) {
      return;
    }
  }
  throw new Error(
    `Recall work marker physical session path is outside every trusted session root: ${physicalSessionPath}`,
  );
}

async function validateRecallWorkMarker(
  value: unknown,
  options: RecallWorkMarkerCodecOptions,
): Promise<RecallWorkMarker> {
  let marker: RecallWorkMarker;
  try {
    marker = Value.Parse(recallWorkMarkerSchema, value);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall work marker invalid: ${message}`, { cause: error });
  }
  await assertTrustedPhysicalSessionPath(marker.physicalSessionPath, options.trustedSessionRoots);
  const expectedMarkerId = createRecallWorkMarkerId(marker);
  if (marker.markerId !== expectedMarkerId) {
    throw new Error(
      `Recall work marker ID mismatch: expected ${expectedMarkerId}, received ${marker.markerId}`,
    );
  }
  return marker;
}

/** Strictly validates and encodes one marker without publishing or writing it. */
export async function encodeRecallWorkMarker(
  marker: RecallWorkMarker,
  options: RecallWorkMarkerCodecOptions,
): Promise<string> {
  return `${JSON.stringify(await validateRecallWorkMarker(marker, options))}\n`;
}

/** Strictly decodes one marker and enforces its canonical trusted-session-root path. */
export async function decodeRecallWorkMarker(
  source: string,
  options: RecallWorkMarkerCodecOptions,
): Promise<RecallWorkMarker> {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall work marker unreadable: ${message}`, { cause: error });
  }
  return validateRecallWorkMarker(value, options);
}

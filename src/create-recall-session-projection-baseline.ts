import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';

import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  SessionImportFormat,
} from './enums.js';
import { importSessionJsonl } from './import-session-jsonl.js';
import type { ParsedRecallSessionRecord } from './parse-recall-session-record.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  createLogicalSessionOccurrenceId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallSessionProjection,
} from './recall-session-projection.js';
import {
  frameCompleteRecallAppendRecords,
  RECALL_APPEND_BOUNDARY_WINDOW_BYTES,
  readRecallSessionAppendDelta,
  type RecallSessionAppendDelta,
} from './read-recall-session-append-delta.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

/** Scalar identity required to start projecting one physical source from byte zero. */
export interface CreateInitialRecallPhysicalProjectionOptions {
  physicalSessionId: string;
  physicalSessionPath: string;
  generationId: string;
}

/** Creates the shared cursor-zero physical projection used by rebuild and incremental ingestion. */
export async function createInitialRecallPhysicalProjection(
  options: CreateInitialRecallPhysicalProjectionOptions,
): Promise<PhysicalSessionProjection> {
  const metadata = await stat(options.physicalSessionPath, { bigint: true });
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(options.physicalSessionId),
    generationId: options.generationId,
    physicalSessionId: options.physicalSessionId,
    sourcePath: options.physicalSessionPath,
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
    markerCheckpoint: {
      generationId: options.generationId,
      coveredMarkerIds: [],
      runtimeSequences: [],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

/** Inputs for reproducing one fully indexed source as an append-ready projection baseline. */
export interface CreateRecallSessionProjectionBaselineOptions {
  physicalSessionPath: string;
  generationId: string;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy?: RecallChunkPolicy;
  expectedSourceByteSize?: number;
}

async function synthesizePiV1AppendDelta(filePath: string): Promise<RecallSessionAppendDelta> {
  const [allBytes, metadata, jsonlImport] = await Promise.all([
    readFile(filePath),
    stat(filePath, { bigint: true }),
    importSessionJsonl(filePath),
  ]);
  const framed = frameCompleteRecallAppendRecords(allBytes, filePath, 0, 0);
  const physicalByLine = new Map(framed.records.map((record) => [record.sourceLine, record]));
  const synthesizedRecords: ParsedRecallSessionRecord[] = [];
  for (const session of jsonlImport.sessions) {
    for (const canonicalRecord of session.records) {
      const physical = physicalByLine.get(canonicalRecord.sourceLine);
      if (physical === undefined) {
        throw new Error(
          `Recall v1 baseline synthesis missing physical framing at ${filePath}:${canonicalRecord.sourceLine}`,
        );
      }
      synthesizedRecords.push({
        sourceLine: physical.sourceLine,
        startByte: physical.startByte,
        endByte: physical.endByte,
        sourceFingerprint: physical.sourceFingerprint,
        value: canonicalRecord.value,
      });
    }
  }
  const committedBoundaryStart = Math.max(0, framed.appendCursorBytes - RECALL_APPEND_BOUNDARY_WINDOW_BYTES);
  const boundaryFingerprint = createHash('sha256')
    .update(allBytes.subarray(committedBoundaryStart, framed.appendCursorBytes))
    .digest('hex');
  return {
    status: RecallAppendDeltaStatus.APPENDED,
    records: synthesizedRecords,
    appendCursorBytes: framed.appendCursorBytes,
    appendCursorLines: framed.appendCursorLines,
    boundaryFingerprint,
    partialFinalRecordBytes: framed.partialFinalRecordBytes,
    sourceDevice: metadata.dev.toString(),
    sourceInode: metadata.ino.toString(),
  };
}

function eligibleSpanForDescriptor(
  descriptor: LogicalSessionProjection['entryDescriptors'][number],
): RecallEligibleSourceSpan {
  return {
    startByte: descriptor.startByte,
    endByte: descriptor.endByte,
    startEntryId: descriptor.entryId,
    endEntryId: descriptor.entryId,
    contributorEntryIds: [descriptor.entryId],
  };
}

/** Rebuilds scalar projections for one source and marks exactly its indexed contributors eligible. */
export async function createRecallSessionProjectionBaseline(
  options: CreateRecallSessionProjectionBaselineOptions,
): Promise<RecallSessionProjection[]> {
  const imported = await readSessionConversationImport(options.physicalSessionPath, {
    tokenizer: options.tokenizer,
    ...(options.chunkPolicy ?? {}),
  });
  const firstLogicalSession = imported.logicalSessions[0];
  if (firstLogicalSession === undefined) {
    throw new Error(
      `Recall rebuild source contains no logical session: ${options.physicalSessionPath}`,
    );
  }
  const initialPhysicalProjection = await createInitialRecallPhysicalProjection({
    physicalSessionId: firstLogicalSession.sessionId,
    physicalSessionPath: options.physicalSessionPath,
    generationId: options.generationId,
  });
  const appendDeltaResult = await readRecallSessionAppendDelta(
    options.physicalSessionPath,
    initialPhysicalProjection,
  );
  let effectiveAppendDelta: RecallSessionAppendDelta;
  if (appendDeltaResult.status === RecallAppendDeltaStatus.APPENDED) {
    effectiveAppendDelta = appendDeltaResult;
  } else if (
    appendDeltaResult.repairReason === RecallProjectionRepairReason.UNSUPPORTED_LAYOUT &&
    imported.format === SessionImportFormat.PI_V1_LINEAR
  ) {
    effectiveAppendDelta = await synthesizePiV1AppendDelta(options.physicalSessionPath);
  } else {
    throw new Error(
      `Recall rebuild source projection requires reconciliation: ${options.physicalSessionPath}`,
    );
  }
  const projected = projectRecallSessionAppend({
    physicalProjection: initialPhysicalProjection,
    logicalProjections: [],
    appendDelta: effectiveAppendDelta,
    markers: [],
    quiescenceObserved: false,
  });
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    throw new Error(`Recall rebuild source projection failed: ${options.physicalSessionPath}`);
  }
  if (
    options.expectedSourceByteSize !== undefined &&
    projected.physicalProjection.appendCursorBytes !== options.expectedSourceByteSize
  ) {
    throw new Error(
      `Recall rebuild source changed while projections were created: ${options.physicalSessionPath}`,
    );
  }
  const eligibleContributorIdsByLogicalSession = new Map<string, Set<string>>();
  for (const chunk of imported.chunks) {
    const matchingLogicalSessions = imported.logicalSessions.filter(
      ({ sourceLineStart, sourceLineEnd }) =>
        chunk.sourceLineStart >= sourceLineStart && chunk.sourceLineStart <= sourceLineEnd,
    );
    const matchingLogicalSession = matchingLogicalSessions[0];
    if (matchingLogicalSessions.length !== 1 || matchingLogicalSession === undefined) {
      throw new Error(
        `Recall rebuild chunk occurrence is ambiguous: ${options.physicalSessionPath}`,
      );
    }
    const logicalSessionId = createLogicalSessionOccurrenceId(
      matchingLogicalSession.sessionId,
      matchingLogicalSession.sourceLineStart,
    );
    const contributorIds =
      eligibleContributorIdsByLogicalSession.get(logicalSessionId) ?? new Set();
    for (const contributorEntryId of chunk.contributingEntryIds) {
      contributorIds.add(contributorEntryId.value);
    }
    eligibleContributorIdsByLogicalSession.set(logicalSessionId, contributorIds);
  }
  const logicalProjections = projected.logicalProjections.map((projection) => {
    const eligibleContributorIds =
      eligibleContributorIdsByLogicalSession.get(projection.logicalSessionId) ?? new Set<string>();
    const eligibleDescriptors = projection.entryDescriptors.filter(({ entryId }) =>
      eligibleContributorIds.has(entryId),
    );
    if (eligibleDescriptors.length !== eligibleContributorIds.size) {
      throw new Error(
        `Recall rebuild projection contributors missing from ${projection.logicalSessionId}`,
      );
    }
    return {
      ...projection,
      eligibleContributorEntryIds: [...eligibleContributorIds].toSorted(),
      eligibleSpans: eligibleDescriptors.map(eligibleSpanForDescriptor),
    };
  });
  return [projected.physicalProjection, ...logicalProjections];
}

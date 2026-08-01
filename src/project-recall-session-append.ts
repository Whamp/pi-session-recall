import {
  RecallAppendProjectionStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  readProjectedActiveContextPath,
  validateProjectedRecallSessionEntryLinks,
  type ParsedRecallSessionRecord,
} from './parse-recall-session-record.js';
import {
  createLogicalSessionOccurrenceId,
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  mergeRecallMarkerCheckpoint,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallProjectedEntryDescriptor,
  type RecallProjectedToolCall,
} from './recall-session-projection.js';
import type { RecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import { reduceRecallEligibility } from './reduce-recall-eligibility.js';
import {
  isBoundedRecallDepartureMarkerTrigger,
  type RecallWorkMarker,
} from './recall-work-marker.js';

/** Inputs required to apply one validated physical append and its ordered lifecycle facts. */
export interface ProjectRecallSessionAppendInput {
  physicalProjection: PhysicalSessionProjection;
  logicalProjections: readonly LogicalSessionProjection[];
  appendDelta: RecallSessionAppendDelta;
  markers: readonly RecallWorkMarker[];
  quiescenceObserved: boolean;
}

/** Successful scalar projections plus source spans whose contributors just crossed eligibility. */
export interface ProjectedRecallSessionAppend {
  status: RecallAppendProjectionStatus.PROJECTED;
  physicalProjection: PhysicalSessionProjection;
  logicalProjections: LogicalSessionProjection[];
  newlyEligibleSpans: RecallEligibleSourceSpan[];
}

/** Append projection failure delegated to explicit full-session reconciliation. */
export interface RecallSessionAppendProjectionReconciliation {
  status: RecallAppendProjectionStatus.REQUIRES_RECONCILIATION;
  repairReason: RecallProjectionRepairReason;
}

/** Projection result that never emits conversation documents or embedding input. */
export type ProjectRecallSessionAppendResult =
  | ProjectedRecallSessionAppend
  | RecallSessionAppendProjectionReconciliation;

function reconciliation(
  repairReason: RecallProjectionRepairReason,
): RecallSessionAppendProjectionReconciliation {
  return { status: RecallAppendProjectionStatus.REQUIRES_RECONCILIATION, repairReason };
}

function createLogicalProjection(
  physicalProjection: PhysicalSessionProjection,
  logicalSessionId: string,
  rawSessionId: string,
  headerRecord: ParsedRecallSessionRecord,
): LogicalSessionProjection {
  return {
    schemaVersion: physicalProjection.schemaVersion,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(
      physicalProjection.physicalSessionId,
      logicalSessionId,
    ),
    generationId: physicalProjection.generationId,
    physicalSessionId: physicalProjection.physicalSessionId,
    physicalProjectionId: createPhysicalSessionProjectionId(physicalProjection.physicalSessionId),
    logicalSessionId,
    rawSessionId,
    effectiveLeafEntryId: null,
    activeContextBoundary: null,
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: headerRecord.sourceLine,
      startByte: headerRecord.startByte,
      endByte: headerRecord.endByte,
      sourceFingerprint: headerRecord.sourceFingerprint,
      cwd: typeof headerRecord.value.cwd === 'string' ? headerRecord.value.cwd : '',
      parentSessionPath:
        typeof headerRecord.value.parentSession === 'string'
          ? headerRecord.value.parentSession
          : null,
    },
    entryDescriptors: [],
    eligibleContributorEntryIds: [],
    eligibleSpans: [],
    labels: [],
    markerCheckpoint: {
      generationId: physicalProjection.generationId,
      coveredMarkerIds: [],
      runtimeSequences: [],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function readRequiredEntryString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null;
}

function readProjectedToolCalls(record: Record<string, unknown>): RecallProjectedToolCall[] | null {
  if (record.type !== 'message' || !isUnknownRecord(record.message)) {
    return [];
  }
  const message = record.message;
  if (message.role !== 'assistant' || !Array.isArray(message.content)) {
    return [];
  }
  const toolCalls: RecallProjectedToolCall[] = [];
  for (const block of message.content) {
    if (!isUnknownRecord(block) || block.type !== 'toolCall') {
      continue;
    }
    if (block.id === '' && block.name === '') {
      continue;
    }
    const toolCallId = readRequiredEntryString(block.id);
    const toolName = readRequiredEntryString(block.name);
    if (toolCallId === null || toolName === null) {
      return null;
    }
    toolCalls.push({ toolCallId, toolName });
  }
  return toolCalls;
}

function readProjectedToolResult(record: Record<string, unknown>):
  | {
      toolCallId: string;
      toolName: string;
    }
  | null
  | false {
  if (record.type !== 'message' || !isUnknownRecord(record.message)) {
    return null;
  }
  const message = record.message;
  if (message.role !== 'toolResult') {
    return null;
  }
  if (message.toolCallId === '' && message.toolName === '') {
    return null;
  }
  const toolCallId = readRequiredEntryString(message.toolCallId);
  const toolName = readRequiredEntryString(message.toolName);
  return toolCallId === null || toolName === null ? false : { toolCallId, toolName };
}

function createProjectedEntryDescriptor(
  value: Record<string, unknown>,
  sourceLine: number,
  startByte: number,
  endByte: number,
  sourceFingerprint: string,
): RecallProjectedEntryDescriptor | null {
  const entryId = readRequiredEntryString(value.id);
  const timestamp = readRequiredEntryString(value.timestamp);
  if (
    entryId === null ||
    timestamp === null ||
    (value.parentId !== null && typeof value.parentId !== 'string')
  ) {
    return null;
  }
  const toolCalls = readProjectedToolCalls(value);
  const toolResult = readProjectedToolResult(value);
  const messageRole =
    value.type === 'message' &&
    isUnknownRecord(value.message) &&
    typeof value.message.role === 'string'
      ? value.message.role
      : null;
  if (toolCalls === null || toolResult === false) {
    return null;
  }
  return {
    entryId,
    parentEntryId: value.parentId,
    entryType: String(value.type),
    timestamp,
    messageRole,
    branchSummaryFromEntryId:
      value.type === 'branch_summary' && typeof value.fromId === 'string' ? value.fromId : null,
    sourceLine,
    startByte,
    endByte,
    sourceFingerprint,
    firstKeptEntryId:
      value.type === 'compaction' && typeof value.firstKeptEntryId === 'string'
        ? value.firstKeptEntryId
        : null,
    hasRetainedTail: value.type === 'compaction' && Array.isArray(value.retainedTail),
    toolCalls,
    toolResult,
  };
}

function updateLogicalActiveContext(
  projection: LogicalSessionProjection,
): LogicalSessionProjection | null {
  const entriesById = new Map(
    projection.entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  const activePath = readProjectedActiveContextPath(projection.effectiveLeafEntryId, entriesById);
  if (activePath === null) {
    return null;
  }
  if (activePath.length === 0) {
    return { ...projection, activeContextBoundary: null };
  }
  const first = activePath[0];
  const last = activePath.at(-1);
  if (first === undefined || last === undefined) {
    return null;
  }
  return {
    ...projection,
    activeContextBoundary: { firstEntryId: first.entryId, lastEntryId: last.entryId },
  };
}

function projectAppendRecords(
  input: ProjectRecallSessionAppendInput,
): LogicalSessionProjection[] | null {
  const projectionsById = new Map<string, LogicalSessionProjection>(
    input.logicalProjections.map((projection) => [projection.logicalSessionId, projection]),
  );
  let currentLogicalSessionId = input.logicalProjections.at(-1)?.logicalSessionId ?? null;
  for (const record of input.appendDelta.records) {
    const value = record.value;
    if (value.type === 'session') {
      const rawSessionId = readRequiredEntryString(value.id);
      if (rawSessionId === null || (value.version !== 2 && value.version !== 3)) {
        return null;
      }
      const logicalSessionId = createLogicalSessionOccurrenceId(rawSessionId, record.sourceLine);
      if (projectionsById.has(logicalSessionId)) {
        return null;
      }
      projectionsById.set(
        logicalSessionId,
        createLogicalProjection(input.physicalProjection, logicalSessionId, rawSessionId, record),
      );
      currentLogicalSessionId = logicalSessionId;
      continue;
    }
    if (currentLogicalSessionId === null) {
      return null;
    }
    const current = projectionsById.get(currentLogicalSessionId);
    if (current === undefined) {
      return null;
    }
    if (value.type === 'leaf') {
      if (value.targetId !== null && typeof value.targetId !== 'string') {
        return null;
      }
      projectionsById.set(currentLogicalSessionId, {
        ...current,
        effectiveLeafEntryId: value.targetId,
      });
      continue;
    }
    const descriptor = createProjectedEntryDescriptor(
      value,
      record.sourceLine,
      record.startByte,
      record.endByte,
      record.sourceFingerprint,
    );
    if (descriptor === null) {
      return null;
    }
    if (
      value.type === 'branch_summary' &&
      value.fromId !== 'root' &&
      (typeof value.fromId !== 'string' ||
        !current.entryDescriptors.some(({ entryId }) => entryId === value.fromId))
    ) {
      return null;
    }
    const labels =
      (value.type === 'session_info' || value.type === 'label') &&
      typeof (value.name ?? value.label) === 'string'
        ? [String(value.name ?? value.label).trim()].filter(Boolean)
        : current.labels;
    projectionsById.set(currentLogicalSessionId, {
      ...current,
      effectiveLeafEntryId: descriptor.entryId,
      entryDescriptors: [...current.entryDescriptors, descriptor],
      labels,
    });
  }
  const existingLogicalSessionIds = input.logicalProjections.map(
    ({ logicalSessionId }) => logicalSessionId,
  );
  const orderedIds = [
    ...existingLogicalSessionIds,
    ...[...projectionsById.keys()].filter(
      (logicalSessionId) => !existingLogicalSessionIds.includes(logicalSessionId),
    ),
  ];
  const projected: LogicalSessionProjection[] = [];
  for (const logicalSessionId of orderedIds) {
    const candidate = projectionsById.get(logicalSessionId);
    if (
      candidate === undefined ||
      !validateProjectedRecallSessionEntryLinks(candidate.entryDescriptors)
    ) {
      return null;
    }
    const withActiveContext = updateLogicalActiveContext(candidate);
    if (withActiveContext === null) {
      return null;
    }
    projected.push(withActiveContext);
  }
  return projected;
}

function contextExitMarkerMatchesLogicalProjection(
  marker: RecallWorkMarker,
  projection: LogicalSessionProjection,
): boolean {
  const rawSessionId = projection.rawSessionId ?? projection.logicalSessionId;
  switch (marker.trigger.kind) {
    case RecallWorkMarkerTrigger.COMPACTION: {
      if (marker.trigger.logicalSessionId !== rawSessionId) {
        return false;
      }
      const compactionEntryId = marker.trigger.compactionEntryId;
      return projection.entryDescriptors.some(({ entryId }) => entryId === compactionEntryId);
    }
    case RecallWorkMarkerTrigger.BRANCH_EXIT: {
      if (marker.trigger.logicalSessionId !== rawSessionId) {
        return false;
      }
      const entryIds = new Set(projection.entryDescriptors.map(({ entryId }) => entryId));
      const markerEntryIds = [
        marker.trigger.oldLeafEntryId,
        marker.trigger.newLeafEntryId,
        marker.trigger.summaryEntryId,
      ].filter((entryId): entryId is string => typeof entryId === 'string');
      return markerEntryIds.length > 0 && markerEntryIds.every((entryId) => entryIds.has(entryId));
    }
    case RecallWorkMarkerTrigger.DEPARTURE: {
      if (!isBoundedRecallDepartureMarkerTrigger(marker.trigger)) {
        return false;
      }
      const { leafEntryId, logicalSessionId } = marker.trigger;
      return (
        logicalSessionId === rawSessionId &&
        projection.entryDescriptors.some(({ entryId }) => entryId === leafEntryId)
      );
    }
    default:
      return false;
  }
}

function resolveContextExitMarkerOccurrences(
  markers: readonly RecallWorkMarker[],
  projections: readonly LogicalSessionProjection[],
): Map<RecallWorkMarker, string> | null {
  const logicalSessionIdByMarker = new Map<RecallWorkMarker, string>();
  for (const marker of markers) {
    if (
      marker.trigger.kind !== RecallWorkMarkerTrigger.COMPACTION &&
      marker.trigger.kind !== RecallWorkMarkerTrigger.BRANCH_EXIT &&
      (marker.trigger.kind !== RecallWorkMarkerTrigger.DEPARTURE ||
        !isBoundedRecallDepartureMarkerTrigger(marker.trigger))
    ) {
      continue;
    }
    const matches = projections.filter((projection) =>
      contextExitMarkerMatchesLogicalProjection(marker, projection),
    );
    if (matches.length !== 1) {
      return null;
    }
    const match = matches[0];
    if (match === undefined) {
      return null;
    }
    logicalSessionIdByMarker.set(marker, match.logicalSessionId);
  }
  return logicalSessionIdByMarker;
}

function markerAppliesToLogicalProjection(
  marker: RecallWorkMarker,
  projection: LogicalSessionProjection,
  contextExitMarkerOccurrences: ReadonlyMap<RecallWorkMarker, string>,
): boolean {
  switch (marker.trigger.kind) {
    case RecallWorkMarkerTrigger.COMPACTION:
    case RecallWorkMarkerTrigger.BRANCH_EXIT:
      return contextExitMarkerOccurrences.get(marker) === projection.logicalSessionId;
    case RecallWorkMarkerTrigger.DEPARTURE:
      return isBoundedRecallDepartureMarkerTrigger(marker.trigger)
        ? contextExitMarkerOccurrences.get(marker) === projection.logicalSessionId
        : true;
    case RecallWorkMarkerTrigger.ACTIVITY:
    case RecallWorkMarkerTrigger.ARRIVAL:
      return true;
    default:
      return false;
  }
}

function updatePhysicalMarkerCheckpoint(
  projection: PhysicalSessionProjection,
  markers: readonly RecallWorkMarker[],
): PhysicalSessionProjection['markerCheckpoint'] {
  return mergeRecallMarkerCheckpoint({
    generationId: projection.generationId,
    current: projection.markerCheckpoint,
    coveredMarkerIds: markers.map(({ markerId }) => markerId),
    runtimeSequences: markers.map(({ runtimeInstanceId, runtimeSequence }) => ({
      runtimeInstanceId,
      sequence: runtimeSequence,
    })),
  });
}

/** Applies a complete append delta to scalar projections and emits only newly eligible source spans. */
export function projectRecallSessionAppend(
  input: ProjectRecallSessionAppendInput,
): ProjectRecallSessionAppendResult {
  const appendedLogicalProjections = projectAppendRecords(input);
  if (appendedLogicalProjections === null) {
    return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
  }
  const contextExitMarkerOccurrences = resolveContextExitMarkerOccurrences(
    input.markers,
    appendedLogicalProjections,
  );
  if (contextExitMarkerOccurrences === null) {
    return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
  }
  const logicalProjections: LogicalSessionProjection[] = [];
  const newlyEligibleSpans: RecallEligibleSourceSpan[] = [];
  for (const projection of appendedLogicalProjections) {
    const reduced = reduceRecallEligibility({
      physicalProjection: input.physicalProjection,
      logicalProjection: projection,
      markers: input.markers.filter((marker) =>
        markerAppliesToLogicalProjection(marker, projection, contextExitMarkerOccurrences),
      ),
      quiescenceObserved: input.quiescenceObserved,
    });
    if (
      reduced.logicalProjection === null ||
      reduced.logicalProjection.repairState === RecallProjectionRepairState.REQUIRES_RECONCILIATION
    ) {
      return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
    }
    logicalProjections.push(reduced.logicalProjection);
    newlyEligibleSpans.push(...reduced.newlyEligibleSpans);
  }
  const lastEntryId = input.appendDelta.records.findLast(
    ({ value }) => value.type !== 'session' && value.type !== 'leaf',
  )?.value.id;
  const physicalProjection: PhysicalSessionProjection = {
    ...input.physicalProjection,
    sourceDevice: input.appendDelta.sourceDevice,
    sourceInode: input.appendDelta.sourceInode,
    appendCursorBytes: input.appendDelta.appendCursorBytes,
    appendCursorLines: input.appendDelta.appendCursorLines,
    boundaryFingerprint: input.appendDelta.boundaryFingerprint,
    lastEntryId:
      typeof lastEntryId === 'string' ? lastEntryId : input.physicalProjection.lastEntryId,
    logicalSessionIds: logicalProjections.map(({ logicalSessionId }) => logicalSessionId),
    markerCheckpoint: updatePhysicalMarkerCheckpoint(input.physicalProjection, input.markers),
  };
  return {
    status: RecallAppendProjectionStatus.PROJECTED,
    physicalProjection,
    logicalProjections,
    newlyEligibleSpans,
  };
}

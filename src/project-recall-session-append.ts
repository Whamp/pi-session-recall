import {
  RecallAppendProjectionStatus,
  RecallProjectionEncodingStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  readProjectedRecallSessionEntryPath,
  validateProjectedRecallSessionEntryLinks,
  type ParsedRecallSessionRecord,
} from './parse-recall-session-record.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  encodeRecallSessionProjection,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallMarkerRuntimeCheckpoint,
  type RecallProjectedEntryDescriptor,
  type RecallProjectedToolCall,
} from './recall-session-projection.js';
import type { RecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import { reduceRecallEligibility } from './reduce-recall-eligibility.js';
import type { RecallWorkMarker } from './recall-work-marker.js';

/** Inputs required to apply one validated physical append and its ordered lifecycle facts. */
export interface ProjectRecallSessionAppendInput {
  physicalProjection: PhysicalSessionProjection;
  logicalProjections: readonly LogicalSessionProjection[];
  appendDelta: RecallSessionAppendDelta;
  markers: readonly RecallWorkMarker[];
  quiescenceObserved: boolean;
  maxProjectionPayloadBytes?: number;
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
  if (projection.effectiveLeafEntryId === null) {
    return { ...projection, activeContextBoundary: null };
  }
  const entriesById = new Map(
    projection.entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  const path = readProjectedRecallSessionEntryPath(projection.effectiveLeafEntryId, entriesById);
  if (path === null) {
    return null;
  }
  const latestCompactionIndex = path.findLastIndex(({ entryType }) => entryType === 'compaction');
  let activePath = path;
  if (latestCompactionIndex >= 0) {
    const compaction = path[latestCompactionIndex];
    if (compaction === undefined) {
      return null;
    }
    if (compaction.hasRetainedTail) {
      activePath = path.slice(latestCompactionIndex);
    } else {
      const firstKeptIndex = path.findIndex(
        ({ entryId }) => entryId === compaction.firstKeptEntryId,
      );
      if (firstKeptIndex < 0) {
        return null;
      }
      activePath = path.slice(firstKeptIndex);
    }
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
  const projectionsById = new Map(
    input.logicalProjections.map((projection) => [projection.logicalSessionId, projection]),
  );
  let currentLogicalSessionId = input.physicalProjection.logicalSessionIds.at(-1) ?? null;
  for (const record of input.appendDelta.records) {
    const value = record.value;
    if (value.type === 'session') {
      const logicalSessionId = readRequiredEntryString(value.id);
      if (
        logicalSessionId === null ||
        (value.version !== 2 && value.version !== 3) ||
        projectionsById.has(logicalSessionId)
      ) {
        return null;
      }
      projectionsById.set(
        logicalSessionId,
        createLogicalProjection(input.physicalProjection, logicalSessionId, record),
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
  const orderedIds = [
    ...input.physicalProjection.logicalSessionIds,
    ...[...projectionsById.keys()].filter(
      (logicalSessionId) => !input.physicalProjection.logicalSessionIds.includes(logicalSessionId),
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

function markerAppliesToLogicalProjection(
  marker: RecallWorkMarker,
  projection: LogicalSessionProjection,
): boolean {
  const entryIds = new Set(projection.entryDescriptors.map(({ entryId }) => entryId));
  switch (marker.trigger.kind) {
    case RecallWorkMarkerTrigger.COMPACTION:
      return entryIds.has(marker.trigger.compactionEntryId);
    case RecallWorkMarkerTrigger.BRANCH_EXIT:
      return (
        (marker.trigger.oldLeafEntryId !== null && entryIds.has(marker.trigger.oldLeafEntryId)) ||
        (marker.trigger.newLeafEntryId !== null && entryIds.has(marker.trigger.newLeafEntryId)) ||
        (marker.trigger.summaryEntryId !== undefined && entryIds.has(marker.trigger.summaryEntryId))
      );
    case RecallWorkMarkerTrigger.ACTIVITY:
    case RecallWorkMarkerTrigger.ARRIVAL:
    case RecallWorkMarkerTrigger.DEPARTURE:
      return true;
    default:
      return false;
  }
}

function updatePhysicalMarkerCheckpoint(
  projection: PhysicalSessionProjection,
  markers: readonly RecallWorkMarker[],
): PhysicalSessionProjection['markerCheckpoint'] {
  const coveredMarkerIds = new Set(projection.markerCheckpoint.coveredMarkerIds);
  const runtimeSequences = new Map(
    projection.markerCheckpoint.runtimeSequences.map(({ runtimeInstanceId, sequence }) => [
      runtimeInstanceId,
      sequence,
    ]),
  );
  for (const marker of markers) {
    coveredMarkerIds.add(marker.markerId);
    runtimeSequences.set(
      marker.runtimeInstanceId,
      Math.max(runtimeSequences.get(marker.runtimeInstanceId) ?? 0, marker.runtimeSequence),
    );
  }
  const orderedRuntimeSequences: RecallMarkerRuntimeCheckpoint[] = [...runtimeSequences.entries()]
    .map(([runtimeInstanceId, sequence]) => ({ runtimeInstanceId, sequence }))
    .toSorted((left, right) => left.runtimeInstanceId.localeCompare(right.runtimeInstanceId));
  return {
    generationId: projection.generationId,
    coveredMarkerIds: [...coveredMarkerIds].toSorted(),
    runtimeSequences: orderedRuntimeSequences,
  };
}

function projectionPayloadOverflows(
  physicalProjection: PhysicalSessionProjection,
  logicalProjections: readonly LogicalSessionProjection[],
  maxPayloadBytes?: number,
): boolean {
  const options = maxPayloadBytes === undefined ? {} : { maxPayloadBytes };
  return [physicalProjection, ...logicalProjections].some(
    (projection) =>
      encodeRecallSessionProjection(projection, options).status ===
      RecallProjectionEncodingStatus.REQUIRES_RECONCILIATION,
  );
}

/** Applies a complete append delta to scalar projections and emits only newly eligible source spans. */
export function projectRecallSessionAppend(
  input: ProjectRecallSessionAppendInput,
): ProjectRecallSessionAppendResult {
  const appendedLogicalProjections = projectAppendRecords(input);
  if (appendedLogicalProjections === null) {
    return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
  }
  for (const marker of input.markers) {
    if (
      (marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION ||
        marker.trigger.kind === RecallWorkMarkerTrigger.BRANCH_EXIT) &&
      !appendedLogicalProjections.some((projection) =>
        markerAppliesToLogicalProjection(marker, projection),
      )
    ) {
      return reconciliation(RecallProjectionRepairReason.MALFORMED_GRAPH);
    }
  }
  const logicalProjections: LogicalSessionProjection[] = [];
  const newlyEligibleSpans: RecallEligibleSourceSpan[] = [];
  for (const projection of appendedLogicalProjections) {
    const reduced = reduceRecallEligibility({
      physicalProjection: input.physicalProjection,
      logicalProjection: projection,
      markers: input.markers.filter((marker) =>
        markerAppliesToLogicalProjection(marker, projection),
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
  if (
    projectionPayloadOverflows(
      physicalProjection,
      logicalProjections,
      input.maxProjectionPayloadBytes,
    )
  ) {
    return reconciliation(RecallProjectionRepairReason.PROJECTION_OVERFLOW);
  }
  return {
    status: RecallAppendProjectionStatus.PROJECTED,
    physicalProjection,
    logicalProjections,
    newlyEligibleSpans,
  };
}

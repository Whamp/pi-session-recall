import { createHash } from 'node:crypto';
import { stat } from 'node:fs/promises';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallEligibilityThreshold,
  RecallIncrementalTransferOutcomeKind,
  RecallProjectionEncodingStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';
import { materializeIncrementalRecallEligibleGraphView } from './materialize-incremental-recall-eligible-graph-view.js';
import { prepareIncrementalRecallTransfer } from './prepare-incremental-recall-transfer.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
} from './recall-session-projection.js';
import {
  readRecallSessionAppendDelta,
  type RecallSessionSourceRangeReader,
} from './read-recall-session-append-delta.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import { scheduleRecallWorkPlanEligibility } from './schedule-recall-work-plan-eligibility.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import { commitIncrementalRecallTransfer } from './commit-incremental-recall-transfer.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

/** Heavy dependencies and active-generation paths for one physical-session transfer plan. */
export interface TransferIncrementalRecallWorkPlanOptions {
  workPlan: RecallMarkerReplayWorkPlan;
  lockPath: string;
  evidenceDatabasePath: string;
  projectionDatabasePath: string;
  embeddingDimensions: number;
  chunkPolicy: RecallChunkPolicy;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  resolveProjectIdentity(sessionOrigin: string): Promise<ResolvedProjectIdentity | null>;
  embeddingCache: Pick<EmbeddingVectorCache, 'resolveEmbeddingVectors'>;
  readRange?: RecallSessionSourceRangeReader;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
}

/** Successful incremental transfer with the exact number of immutable documents committed. */
export interface CommittedIncrementalRecallWorkPlan {
  readonly kind: RecallIncrementalTransferOutcomeKind.COMMITTED;
  readonly committedDocumentCount: number;
}

/** Marker-backed work retained until one reconstructed quiet-period deadline. */
export interface DeferredIncrementalRecallWorkPlan {
  readonly kind: RecallIncrementalTransferOutcomeKind.DEFERRED;
  readonly threshold: RecallEligibilityThreshold;
  readonly readyAtEpochMilliseconds: number;
}

/** Durable outcome from one incremental transfer attempt. */
export type IncrementalRecallWorkPlanTransferOutcome =
  | CommittedIncrementalRecallWorkPlan
  | DeferredIncrementalRecallWorkPlan;

async function createInitialPhysicalProjection(
  workPlan: RecallMarkerReplayWorkPlan,
): Promise<PhysicalSessionProjection> {
  const firstMarker = workPlan.workItems[0]?.marker;
  if (firstMarker === undefined) {
    throw new Error('Recall incremental transfer requires at least one marker work item');
  }
  const metadata = await stat(firstMarker.physicalSessionPath, { bigint: true });
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(firstMarker.physicalSessionId),
    generationId: workPlan.targetGenerationId,
    physicalSessionId: firstMarker.physicalSessionId,
    sourcePath: firstMarker.physicalSessionPath,
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
      generationId: workPlan.targetGenerationId,
      coveredMarkerIds: [],
      runtimeSequences: [],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function assertSinglePhysicalSessionWorkPlan(workPlan: RecallMarkerReplayWorkPlan): void {
  const firstMarker = workPlan.workItems[0]?.marker;
  if (firstMarker === undefined) {
    throw new Error('Recall incremental transfer requires at least one marker work item');
  }
  for (const { marker } of workPlan.workItems) {
    if (
      marker.physicalSessionId !== firstMarker.physicalSessionId ||
      marker.physicalSessionPath !== firstMarker.physicalSessionPath
    ) {
      throw new Error('Recall incremental transfer work plan mixes physical sessions');
    }
  }
}

function readCurrentSessionProjections(
  options: TransferIncrementalRecallWorkPlanOptions,
  physicalProjectionId: string,
): {
  physicalProjection: PhysicalSessionProjection | undefined;
  logicalProjections: LogicalSessionProjection[];
} {
  const store = openZvecSessionProjectionStore({
    databasePath: options.projectionDatabasePath,
    generationId: options.workPlan.targetGenerationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const physicalProjection = store
      .fetchProjections([physicalProjectionId])
      .get(physicalProjectionId);
    if (
      physicalProjection === undefined ||
      physicalProjection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION
    ) {
      return { physicalProjection: undefined, logicalProjections: [] };
    }
    const logicalProjectionIds = physicalProjection.logicalSessionIds.map((logicalSessionId) =>
      createLogicalSessionProjectionId(physicalProjection.physicalSessionId, logicalSessionId),
    );
    const fetched = store.fetchProjections(logicalProjectionIds);
    const logicalProjections = logicalProjectionIds.map((projectionId) => {
      const projection = fetched.get(projectionId);
      if (
        projection === undefined ||
        projection.projectionKind !== RecallSessionProjectionKind.LOGICAL_SESSION
      ) {
        throw new Error(`Recall incremental logical projection missing: ${projectionId}`);
      }
      return projection;
    });
    return { physicalProjection, logicalProjections };
  } finally {
    store.close();
  }
}

function spansForLogicalProjection(
  spans: readonly RecallEligibleSourceSpan[],
  projection: LogicalSessionProjection,
): RecallEligibleSourceSpan[] {
  const sourceBoundaries = new Set(
    projection.entryDescriptors.map(({ startByte, endByte }) => `${startByte}:${endByte}`),
  );
  return spans.filter(({ startByte, endByte }) => sourceBoundaries.has(`${startByte}:${endByte}`));
}

function createDeferredIncrementalRecallWorkPlan(
  threshold: RecallEligibilityThreshold,
  readyAtEpochMilliseconds: number,
): DeferredIncrementalRecallWorkPlan {
  return {
    kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
    threshold,
    readyAtEpochMilliseconds,
  };
}

async function readSourceModifiedAtEpochMilliseconds(sourcePath: string): Promise<number> {
  const metadata = await stat(sourcePath, { bigint: true });
  const modifiedAtEpochMilliseconds = Number(metadata.mtimeNs / 1_000_000n);
  if (!Number.isSafeInteger(modifiedAtEpochMilliseconds) || modifiedAtEpochMilliseconds < 0) {
    throw new Error('Recall incremental source modified time invalid');
  }
  return modifiedAtEpochMilliseconds;
}

/** Transfers one physical session from durable marker work through observed checkpoint acknowledgement. */
export async function transferIncrementalRecallWorkPlan(
  options: TransferIncrementalRecallWorkPlanOptions,
): Promise<IncrementalRecallWorkPlanTransferOutcome> {
  assertSinglePhysicalSessionWorkPlan(options.workPlan);
  const firstMarker = options.workPlan.workItems[0]?.marker;
  if (firstMarker === undefined) {
    throw new Error('Recall incremental transfer requires at least one marker work item');
  }
  const physicalProjectionId = createPhysicalSessionProjectionId(firstMarker.physicalSessionId);
  const current = readCurrentSessionProjections(options, physicalProjectionId);
  const physicalProjection =
    current.physicalProjection ?? (await createInitialPhysicalProjection(options.workPlan));
  const appendDelta = await readRecallSessionAppendDelta(
    firstMarker.physicalSessionPath,
    physicalProjection,
    options.readRange ? { readRange: options.readRange } : {},
  );
  if (appendDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    throw new Error(
      `Recall incremental append requires reconciliation: ${appendDelta.repairReason}`,
    );
  }
  const sourceModifiedAtEpochMilliseconds = await readSourceModifiedAtEpochMilliseconds(
    firstMarker.physicalSessionPath,
  );
  const nowEpochMilliseconds = options.nowEpochMilliseconds ?? Date.now;
  const initialSchedule = scheduleRecallWorkPlanEligibility({
    workPlan: options.workPlan,
    sourceModifiedAtEpochMilliseconds,
    preparedDocumentCount: 0,
    nowEpochMilliseconds,
  });
  if (!initialSchedule.ready) {
    return createDeferredIncrementalRecallWorkPlan(
      initialSchedule.threshold,
      initialSchedule.readyAtEpochMilliseconds,
    );
  }
  const projected = projectRecallSessionAppend({
    physicalProjection,
    logicalProjections: current.logicalProjections,
    appendDelta,
    markers: options.workPlan.workItems.map(({ marker }) => marker),
    quiescenceObserved:
      initialSchedule.threshold === RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
  });
  if (projected.status !== RecallAppendProjectionStatus.PROJECTED) {
    throw new Error(
      `Recall incremental projection requires reconciliation: ${projected.repairReason}`,
    );
  }
  const eligibleSessions = await Promise.all(
    projected.logicalProjections.map(async (logicalProjection) => {
      const newlyEligibleSpans = spansForLogicalProjection(
        projected.newlyEligibleSpans,
        logicalProjection,
      );
      const graphView = await materializeIncrementalRecallEligibleGraphView({
        physicalProjection: projected.physicalProjection,
        logicalProjection,
        newlyEligibleSpans,
        appendDelta,
        ...(options.readRange ? { readRange: options.readRange } : {}),
      });
      return { graphView, logicalProjection, newlyEligibleSpans };
    }),
  );
  const prepared = await prepareIncrementalRecallTransfer({
    physicalProjection: projected.physicalProjection,
    eligibleSessions,
    workPlan: options.workPlan,
    chunkPolicy: options.chunkPolicy,
    loadTokenizer: () => options.loadTokenizer(),
    resolveProjectIdentity: (sessionOrigin) => options.resolveProjectIdentity(sessionOrigin),
    embeddingCache: options.embeddingCache,
    ...(options.signal ? { signal: options.signal } : {}),
  });
  if (prepared.status !== RecallProjectionEncodingStatus.ENCODED) {
    throw new Error(
      `Recall incremental transfer requires reconciliation: ${prepared.repairReason}`,
    );
  }
  const preparedSchedule = scheduleRecallWorkPlanEligibility({
    workPlan: options.workPlan,
    sourceModifiedAtEpochMilliseconds,
    preparedDocumentCount: prepared.documents.length,
    nowEpochMilliseconds,
  });
  if (!preparedSchedule.ready) {
    return createDeferredIncrementalRecallWorkPlan(
      preparedSchedule.threshold,
      preparedSchedule.readyAtEpochMilliseconds,
    );
  }
  const committed = await commitIncrementalRecallTransfer({
    prepared,
    lockPath: options.lockPath,
    evidenceDatabasePath: options.evidenceDatabasePath,
    projectionDatabasePath: options.projectionDatabasePath,
    embeddingDimensions: options.embeddingDimensions,
    ...(options.signal ? { signal: options.signal } : {}),
  });
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: committed.committedDocumentCount,
  };
}

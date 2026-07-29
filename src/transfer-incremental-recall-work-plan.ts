import { stat } from 'node:fs/promises';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import {
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallEligibilityThreshold,
  RecallIncrementalTransferOutcomeKind,
  RecallProjectionEncodingStatus,
  RecallSessionProjectionKind,
} from './enums.js';
import { createInitialRecallPhysicalProjection } from './create-recall-session-projection-baseline.js';
import { materializeIncrementalRecallEligibleGraphView } from './materialize-incremental-recall-eligible-graph-view.js';
import { prepareIncrementalRecallTransfer } from './prepare-incremental-recall-transfer.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
} from './recall-session-projection.js';
import {
  readRecallSessionAppendDelta,
  type RecallSessionSourceRangeReader,
} from './read-recall-session-append-delta.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  createRecallIncrementalDiagnosticMetrics,
  type RecallOperationDiagnostics,
} from './recall-operation-diagnostics.js';
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
  operationDiagnostics?: Pick<RecallOperationDiagnostics, 'recordIncrementalOperation'>;
  readRange?: RecallSessionSourceRangeReader;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
  monotonicMilliseconds?: () => number;
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
  physicalProjection?: PhysicalSessionProjection;
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
      return { logicalProjections: [] };
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
    current.physicalProjection ??
    (await createInitialRecallPhysicalProjection({
      physicalSessionId: firstMarker.physicalSessionId,
      physicalSessionPath: firstMarker.physicalSessionPath,
      generationId: options.workPlan.targetGenerationId,
    }));
  const appendDelta = await readRecallSessionAppendDelta(
    firstMarker.physicalSessionPath,
    physicalProjection,
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
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const preparationStartedAtMilliseconds = monotonicMilliseconds();
  let tokenizerMilliseconds = 0;
  const prepared = await prepareIncrementalRecallTransfer({
    physicalProjection: projected.physicalProjection,
    eligibleSessions,
    workPlan: options.workPlan,
    chunkPolicy: options.chunkPolicy,
    async loadTokenizer() {
      const loadStartedAtMilliseconds = monotonicMilliseconds();
      const tokenizer = await options.loadTokenizer();
      tokenizerMilliseconds += Math.max(monotonicMilliseconds() - loadStartedAtMilliseconds, 0);
      return {
        encodeConversationText(text: string) {
          const encodeStartedAtMilliseconds = monotonicMilliseconds();
          try {
            return tokenizer.encodeConversationText(text);
          } finally {
            tokenizerMilliseconds += Math.max(
              monotonicMilliseconds() - encodeStartedAtMilliseconds,
              0,
            );
          }
        },
      };
    },
    resolveProjectIdentity: (sessionOrigin) => options.resolveProjectIdentity(sessionOrigin),
    embeddingCache: options.embeddingCache,
    ...(options.signal ? { signal: options.signal } : {}),
  });
  if (prepared.status !== RecallProjectionEncodingStatus.ENCODED) {
    throw new Error(
      `Recall incremental transfer requires reconciliation: ${prepared.repairReason}`,
    );
  }
  const preparationMetrics = createRecallIncrementalDiagnosticMetrics();
  preparationMetrics.elapsedMilliseconds = Math.max(
    monotonicMilliseconds() - preparationStartedAtMilliseconds,
    0,
  );
  preparationMetrics.appendedByteCount = Math.max(
    appendDelta.appendCursorBytes - physicalProjection.appendCursorBytes,
    0,
  );
  preparationMetrics.parsedEntryCount = appendDelta.records.length;
  preparationMetrics.eligibleDocumentCount = prepared.documents.length;
  preparationMetrics.tokenizerMilliseconds = tokenizerMilliseconds;
  preparationMetrics.generationId = options.workPlan.targetGenerationId;
  options.operationDiagnostics?.recordIncrementalOperation({
    operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics: preparationMetrics,
  });
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
    ...(options.operationDiagnostics ? { operationDiagnostics: options.operationDiagnostics } : {}),
    ...(options.signal ? { signal: options.signal } : {}),
  });
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: committed.committedDocumentCount,
  };
}

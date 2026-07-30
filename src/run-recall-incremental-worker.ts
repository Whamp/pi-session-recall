import { access, open, readFile, stat } from 'node:fs/promises';
import { isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { completeRecallGenerationReplay } from './complete-recall-generation-replay.js';
import {
  coordinateRecallMarkerReplay,
  type RecallGenerationReplayCompletionPaths,
  type RecallMarkerReplayWorkPlan,
} from './coordinate-recall-marker-replay.js';
import {
  RecallBacklogFailureCategory,
  RecallConfirmedDeletionDecisionKind,
  RecallConfirmedDeletionHaltCategory,
  RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallEligibilityThreshold,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallMetadataSweepStatus,
  RecallSessionProjectionKind,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  decodeRecallBacklogSummary,
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  writeRecallBacklogSummary,
  RECALL_BACKLOG_SUMMARY_VERSION,
} from './recall-generation-state.js';
import {
  readRecallGenerationReplaySnapshot,
  RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
} from './recall-generation-replay-snapshot.js';
import {
  persistRecallIncrementalWorkerSchedule,
  readRecallIncrementalWorkerSchedule,
  signalRecallIncrementalWorkerWake,
  RECALL_INCREMENTAL_WORKER_SCHEDULE_VERSION,
  type RecallLargeTransferDeferral,
} from './recall-incremental-worker-schedule.js';
import {
  decideConfirmedSessionDeletion,
  type ConfirmedSessionDeletionReconciliationResult,
} from './confirmed-session-deletion-policy.js';
import {
  decodeRecallSessionProjection,
  mergeRecallMarkerCheckpoint,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import {
  createRecallWorkMarkerId,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarkerCodecOptions,
  type RecallWorkMarkerIdentity,
} from './recall-work-marker.js';
import { acknowledgeCoveredRecallMarkers } from './recall-marker-spool.js';
import { publishRecallWorkMarker } from './publish-recall-work-marker.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import {
  createRecallIncrementalDiagnosticMetrics,
  createRecallOperationDiagnostics,
  type RecallOperationDiagnostics,
} from './recall-operation-diagnostics.js';
import { recoverRecallGenerationCutover } from './recover-recall-generation-cutover.js';
import {
  RECALL_METADATA_SWEEP_CONTINUATION_FILENAME,
  scanRecallSessionMetadata,
  type KnownRecallSessionMetadataSource,
  type RecallSessionMetadataSweepResult,
} from './scan-recall-session-metadata.js';
import { scheduleRecallWorkPlanEligibility } from './schedule-recall-work-plan-eligibility.js';
import type {
  IncrementalRecallWorkPlanRequest,
  IncrementalRecallWorkPlanTransferOutcome,
} from './transfer-incremental-recall-work-plan.js';

/** Existing scalar sinks and executable callback protected by the detached-worker failure boundary. */
export interface RunRecallIncrementalWorkerDiagnosticBoundaryOptions {
  operationDiagnostics: Pick<
    RecallOperationDiagnostics,
    'recordDurableIncrementalFailure' | 'flush'
  >;
  persistFailure(): Promise<void>;
  run(): Promise<void>;
  monotonicMilliseconds?: () => number;
}

/** Records and atomically persists failures from anywhere in the detached worker executable. */
export async function runRecallIncrementalWorkerDiagnosticBoundary(
  options: RunRecallIncrementalWorkerDiagnosticBoundaryOptions,
): Promise<void> {
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const startedAtMilliseconds = monotonicMilliseconds();
  try {
    await options.run();
  } catch (error) {
    const metrics = createRecallIncrementalDiagnosticMetrics();
    metrics.elapsedMilliseconds = Math.max(monotonicMilliseconds() - startedAtMilliseconds, 0);
    options.operationDiagnostics.recordDurableIncrementalFailure({
      operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
      status: RecallDiagnosticStatus.FAILED,
      metrics,
      errorCategory: RecallDiagnosticErrorCategory.OPERATION_FAILED,
    });
    await Promise.allSettled([options.persistFailure()]);
    throw error;
  } finally {
    await options.operationDiagnostics.flush();
  }
}

/** Scalar sinks and generation state used to durably expose one confirmed deletion halt. */
export interface ReportRecallIncrementalWorkerDeletionHaltOptions {
  operationDiagnostics: Pick<
    RecallOperationDiagnostics,
    'recordDurableIncrementalFailure' | 'flush'
  >;
  generationId: string;
  generationState: RecallGenerationCutoverState | null;
  haltResult: ConfirmedSessionDeletionReconciliationResult;
  workPlan: RecallIncrementalWorkerBacklogObservation['workPlan'];
  nowEpochMilliseconds?: () => number;
  persistFailure(): Promise<void>;
}

/** Persists a privacy-safe deletion halt through diagnostics and the atomic backlog summary. */
export async function reportRecallIncrementalWorkerDeletionHalt(
  options: ReportRecallIncrementalWorkerDeletionHaltOptions,
): Promise<void> {
  const [deletionSafeguardCategory] = Object.keys(options.haltResult.haltCategoryCounts);
  const metrics = createRecallIncrementalDiagnosticMetrics();
  metrics.generationId = options.generationId;
  metrics.generationState = options.generationState;
  metrics.deletionSafeguardCategory = deletionSafeguardCategory ?? null;
  const pendingPhysicalSessionIds = new Set(
    options.workPlan.workItems.map(({ marker }) => marker.physicalSessionId),
  );
  const oldestMarkerCreatedAtEpochMilliseconds = options.workPlan.workItems.reduce(
    (oldest, { marker }) => Math.min(oldest, marker.createdAtEpochMilliseconds),
    Number.POSITIVE_INFINITY,
  );
  metrics.backlogPendingEligibleSessionCount = pendingPhysicalSessionIds.size;
  metrics.backlogOldestEligibleMarkerAgeMilliseconds = Number.isFinite(
    oldestMarkerCreatedAtEpochMilliseconds,
  )
    ? Math.max(
        0,
        (options.nowEpochMilliseconds?.() ?? Date.now()) - oldestMarkerCreatedAtEpochMilliseconds,
      )
    : null;
  options.operationDiagnostics.recordDurableIncrementalFailure({
    operationKind: RecallDiagnosticOperationKind.DELETION_RECONCILIATION,
    status: RecallDiagnosticStatus.FAILED,
    metrics,
    errorCategory: RecallDiagnosticErrorCategory.OPERATION_FAILED,
  });
  await Promise.allSettled([options.persistFailure(), options.operationDiagnostics.flush()]);
}

/** Explicit lightweight paths and lazy import boundary for one short-lived incremental worker. */
export interface RunRecallIncrementalWorkerOptions extends RecallWorkMarkerCodecOptions {
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  controlDirectory: string;
  targetGenerationId: string;
  generationRegistryPath?: string;
  retainedMarkerDirectory?: string;
  generationReplayCompletion?: Omit<
    RecallGenerationReplayCompletionPaths,
    'markerQuarantineDirectory'
  >;
  knownSources?: readonly KnownRecallSessionMetadataSource[];
  confirmedDeletionMaxMissingSourceCount?: number;
  confirmedDeletionMaxMissingSourceRatio?: number;
  loadKnownSourceInventory?: () => Promise<RecallKnownSourceInventory>;
  reconcileDeletion?: (
    metadataSweep: RecallSessionMetadataSweepResult,
    physicalProjections: readonly PhysicalSessionProjection[],
    missingSourceWorkPlans: readonly RecallMarkerReplayWorkPlan[],
    workPlan: RecallMarkerReplayWorkPlan,
  ) => Promise<Pick<
    ConfirmedSessionDeletionReconciliationResult,
    'sourceMissingRecordedCount'
  > | void>;
  persistedLargeTransferDeferrals?: readonly RecallLargeTransferDeferral[];
  metadataSweepRequested?: boolean;
  loadHeavyDependencies?: () => Promise<void>;
  transferWorkPlan?: (
    workPlan: RecallMarkerReplayWorkPlan,
  ) => Promise<IncrementalRecallWorkPlanTransferOutcome>;
  operationDiagnostics?: Pick<RecallOperationDiagnostics, 'recordIncrementalOperation'>;
  nowEpochMilliseconds?: () => number;
  monotonicMilliseconds?: () => number;
  scanSessionMetadata?: typeof scanRecallSessionMetadata;
}

/** Known active-generation source projections loaded only when a metadata sweep is requested. */
export interface RecallKnownSourceInventory {
  knownSources: readonly KnownRecallSessionMetadataSource[];
  physicalProjections: readonly PhysicalSessionProjection[];
}

/** One worker invocation result, retaining its unacknowledged work plan for later transfer. */
export interface RecallIncrementalWorkerResult {
  workPlan: RecallMarkerReplayWorkPlan;
  metadataSweep: RecallSessionMetadataSweepResult | null;
  heavyDependenciesLoaded: boolean;
  commitsFrozen: boolean;
  generationReplayCompleted: boolean | null;
  transferOutcomes: readonly IncrementalRecallWorkPlanTransferOutcome[];
  largeTransferDeferrals: readonly RecallLargeTransferDeferral[];
  metadataSweepFollowUpRequired: boolean;
  nextWakeAtEpochMilliseconds: number | null;
  replayBlockingFailureCategory: RecallBacklogFailureCategory | null;
}

type RecallIncrementalWorkerResultInput = Omit<
  RecallIncrementalWorkerResult,
  'metadataSweepFollowUpRequired' | 'nextWakeAtEpochMilliseconds' | 'replayBlockingFailureCategory'
>;

async function loadRecallIncrementalWorkerDependencies(): Promise<void> {
  await Promise.all([import('@huggingface/tokenizers'), import('@zvec/zvec')]);
}

async function hasRecallMetadataSweepContinuation(controlDirectory: string): Promise<boolean> {
  try {
    await access(joinRecallMetadataContinuationPath(controlDirectory));
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

function joinRecallMetadataContinuationPath(controlDirectory: string): string {
  return resolve(controlDirectory, RECALL_METADATA_SWEEP_CONTINUATION_FILENAME);
}

function workPlanRequestsRecallMetadataSweep(workPlan: RecallMarkerReplayWorkPlan): boolean {
  return workPlan.workItems.some(
    ({ marker }) => marker.trigger.kind === RecallWorkMarkerTrigger.ARRIVAL,
  );
}

/**
 * Reads the first line of a physical session JSONL file and extracts the logical session ID
 * from the session header record. Returns null if the header cannot be read or is not a valid
 * session header.
 */
async function readPhysicalSessionIdFromJsonlHeader(filePath: string): Promise<string | null> {
  const maximumHeaderBytes = 1024 * 1024;
  const handle = await open(filePath, 'r');
  try {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    while (totalBytes < maximumHeaderBytes) {
      const buffer = Buffer.allocUnsafe(Math.min(4096, maximumHeaderBytes - totalBytes));
      const { bytesRead } = await handle.read(buffer, 0, buffer.length, totalBytes);
      if (bytesRead === 0) {
        return null;
      }
      const chunk = buffer.subarray(0, bytesRead);
      const newlineIndex = chunk.indexOf(0x0a);
      if (newlineIndex === -1) {
        chunks.push(chunk);
        totalBytes += bytesRead;
        continue;
      }
      chunks.push(chunk.subarray(0, newlineIndex));
      const firstLine = Buffer.concat(chunks).toString('utf8');
      const trimmed = firstLine.endsWith('\r') ? firstLine.slice(0, -1) : firstLine;
      let parsed: unknown;
      try {
        parsed = JSON.parse(trimmed);
      } catch (error) {
        if (error instanceof SyntaxError) {
          return null;
        }
        throw error;
      }
      if (
        isUnknownRecord(parsed) &&
        (parsed.type === 'session' || parsed.type === 'v1_session') &&
        typeof parsed.id === 'string' &&
        parsed.id.length > 0
      ) {
        return parsed.id;
      }
      return null;
    }
    return null;
  } finally {
    await handle.close();
  }
}

/**
 * Publishes durable ARRIVAL markers for session files observed by a metadata sweep that are not
 * yet represented in the marker spool or the known-source inventory. Returns the count of markers
 * published.
 */
async function publishMetadataSweepArrivalMarkers(
  metadataSweep: RecallSessionMetadataSweepResult,
  options: Pick<
    RunRecallIncrementalWorkerOptions,
    'trustedSessionRoots' | 'markerSpoolDirectory' | 'knownSources' | 'loadKnownSourceInventory'
  >,
  existingPhysicalSessionIds: ReadonlySet<string>,
  nowEpochMilliseconds: number,
): Promise<number> {
  const unknownObservedFiles = metadataSweep.observedSessionMetadata.filter(
    (entry) => entry.physicalSessionId === null,
  );
  if (unknownObservedFiles.length === 0) {
    return 0;
  }
  const sessionRootDirectory = options.trustedSessionRoots[0] ?? '';
  let published = 0;
  const sweepRuntimeInstanceId = `metadata-sweep:${metadataSweep.sweepId}`;
  for (const [index, entry] of unknownObservedFiles.entries()) {
    const absolutePath = join(sessionRootDirectory, entry.relativePath);
    let physicalSessionId: string | null;
    try {
      physicalSessionId = await readPhysicalSessionIdFromJsonlHeader(absolutePath);
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        continue;
      }
      throw error;
    }
    if (physicalSessionId === null) {
      continue;
    }
    if (existingPhysicalSessionIds.has(physicalSessionId)) {
      continue;
    }
    const markerIdentity: RecallWorkMarkerIdentity = {
      version: RECALL_WORK_MARKER_VERSION,
      physicalSessionId,
      physicalSessionPath: absolutePath,
      runtimeInstanceId: sweepRuntimeInstanceId,
      runtimeSequence: index + 1,
      createdAtEpochMilliseconds: nowEpochMilliseconds,
      trigger: { kind: RecallWorkMarkerTrigger.ARRIVAL },
    };
    await publishRecallWorkMarker(
      { ...markerIdentity, markerId: createRecallWorkMarkerId(markerIdentity) },
      {
        markerSpoolDirectory: options.markerSpoolDirectory,
        trustedSessionRoots: options.trustedSessionRoots,
        // No-op signal: we are already inside the worker; the caller sets follow-up instead.
        workerSignal: { signalDetachedWorker() {} },
      },
    );
    published += 1;
  }
  return published;
}

function splitRecallWorkPlanByPhysicalSession(
  workPlan: RecallMarkerReplayWorkPlan,
): RecallMarkerReplayWorkPlan[] {
  const workItemsByPhysicalSource = new Map<string, RecallMarkerReplayWorkPlan['workItems']>();
  for (const workItem of workPlan.workItems) {
    const key = JSON.stringify([
      workItem.marker.physicalSessionId,
      workItem.marker.physicalSessionPath,
    ]);
    const existing = workItemsByPhysicalSource.get(key);
    if (existing === undefined) {
      workItemsByPhysicalSource.set(key, [workItem]);
    } else {
      existing.push(workItem);
    }
  }
  return [...workItemsByPhysicalSource.values()].map((workItems) => {
    const sourceMarkerIds = [
      ...new Set(workItems.flatMap(({ coveredMarkerIds }) => coveredMarkerIds)),
    ].toSorted();
    return {
      ...workPlan,
      discoveredMarkerCount: sourceMarkerIds.length,
      sourceMarkerIds,
      workItems,
    };
  });
}

function hasRecallMetadataReconciliationWork(
  metadataSweep: RecallSessionMetadataSweepResult | null,
): boolean {
  return (
    metadataSweep !== null &&
    (metadataSweep.status !== RecallMetadataSweepStatus.CONTINUATION_REQUIRED ||
      metadataSweep.observedSessionMetadata.length > 0 ||
      metadataSweep.observedKnownSourceIdentities.length > 0 ||
      metadataSweep.missingPhysicalSessionIds.length > 0)
  );
}

async function readWorkerSourceModifiedAtEpochMilliseconds(
  workPlan: RecallMarkerReplayWorkPlan,
): Promise<number | null> {
  const sourcePath = workPlan.workItems[0]?.marker.physicalSessionPath;
  if (sourcePath === undefined) {
    throw new Error('Recall worker source metadata requires one marker work item');
  }
  let metadata: Awaited<ReturnType<typeof stat>>;
  try {
    metadata = await stat(sourcePath, { bigint: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  const modifiedAtEpochMilliseconds = Number(metadata.mtimeNs / 1_000_000n);
  if (!Number.isSafeInteger(modifiedAtEpochMilliseconds) || modifiedAtEpochMilliseconds < 0) {
    throw new Error('Recall worker source modified time invalid');
  }
  return modifiedAtEpochMilliseconds;
}

/** Coordinates one bounded worker pass and loads tokenizer/zvec only after eligible work exists. */
export async function runRecallIncrementalWorker(
  options: RunRecallIncrementalWorkerOptions,
): Promise<RecallIncrementalWorkerResult> {
  const monotonicMilliseconds = options.monotonicMilliseconds ?? (() => performance.now());
  const startedAtMilliseconds = monotonicMilliseconds();
  const nowEpochMilliseconds = options.nowEpochMilliseconds ?? Date.now;
  const registry = options.generationRegistryPath
    ? await readRecallGenerationRegistry(options.generationRegistryPath)
    : null;
  if (
    registry?.activeGenerationId !== null &&
    registry?.activeGenerationId !== undefined &&
    registry.activeGenerationId !== options.targetGenerationId
  ) {
    throw new Error('Recall incremental worker target does not match the generation registry');
  }
  const activeGenerationEntry = registry?.generations.find(
    ({ generationId }) => generationId === registry.activeGenerationId,
  );
  let metadataSweepFollowUpRequired = options.metadataSweepRequested ?? false;
  let fixedReplayMarkerIds: readonly string[] | undefined;
  if (
    activeGenerationEntry?.state === RecallGenerationCutoverState.REPLAY_PENDING &&
    options.generationReplayCompletion !== undefined
  ) {
    const generationDirectory = await resolveRecallGenerationDirectory(
      options.generationReplayCompletion.generationRootDirectory,
      options.targetGenerationId,
    );
    const replaySnapshot = await readRecallGenerationReplaySnapshot(
      join(
        generationDirectory,
        activeGenerationEntry.replaySnapshotFileName ?? RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
      ),
    );
    if (replaySnapshot.generationId !== options.targetGenerationId) {
      throw new Error(
        `Recall incremental worker replay snapshot identity mismatch: expected ${options.targetGenerationId}, received ${replaySnapshot.generationId}`,
      );
    }
    fixedReplayMarkerIds = replaySnapshot.pendingMarkerIds;
  }
  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: options.markerSpoolDirectory,
    markerQuarantineDirectory: options.markerQuarantineDirectory,
    targetGenerationId: options.targetGenerationId,
    trustedSessionRoots: options.trustedSessionRoots,
    ...(fixedReplayMarkerIds === undefined ? {} : { fixedReplayMarkerIds }),
    ...(options.retainedMarkerDirectory
      ? { retainedMarkerDirectory: options.retainedMarkerDirectory }
      : {}),
    ...(options.generationReplayCompletion
      ? {
          generationReplayCompletion: {
            ...options.generationReplayCompletion,
            markerQuarantineDirectory: options.markerQuarantineDirectory,
          },
        }
      : {}),
  });
  function finishWorkerResult(
    result: RecallIncrementalWorkerResultInput,
  ): RecallIncrementalWorkerResult {
    const metrics = createRecallIncrementalDiagnosticMetrics();
    metrics.elapsedMilliseconds = Math.max(monotonicMilliseconds() - startedAtMilliseconds, 0);
    const oldestMarkerCreatedAt = workPlan.workItems.reduce(
      (oldest, { marker }) => Math.min(oldest, marker.createdAtEpochMilliseconds),
      Number.POSITIVE_INFINITY,
    );
    metrics.markerAgeMilliseconds = Number.isFinite(oldestMarkerCreatedAt)
      ? Math.max(nowEpochMilliseconds() - oldestMarkerCreatedAt, 0)
      : null;
    metrics.metadataSweepScannedFileCount = result.metadataSweep?.scannedFileCount ?? 0;
    metrics.metadataSweepObservedSessionCount = result.metadataSweep?.observedSessionFileCount ?? 0;
    metrics.metadataSweepElapsedMilliseconds = result.metadataSweep?.elapsedMilliseconds ?? 0;
    metrics.generationId = options.targetGenerationId;
    metrics.generationState =
      registry?.generations.find(({ generationId }) => generationId === options.targetGenerationId)
        ?.state ?? null;
    metrics.deletionSafeguardCategory =
      result.metadataSweep?.deletionConfirmationSuppressed === true
        ? result.metadataSweep.status
        : null;
    metrics.backlogPendingEligibleSessionCount = workPlan.workItems.length;
    metrics.backlogOldestEligibleMarkerAgeMilliseconds = metrics.markerAgeMilliseconds;
    options.operationDiagnostics?.recordIncrementalOperation({
      operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
      status: RecallDiagnosticStatus.SUCCEEDED,
      metrics,
    });
    const deferredDeadlines = result.transferOutcomes.flatMap((outcome) =>
      outcome.kind === RecallIncrementalTransferOutcomeKind.DEFERRED
        ? [outcome.readyAtEpochMilliseconds]
        : [],
    );
    const continuationDeadlines =
      result.metadataSweep?.status === RecallMetadataSweepStatus.CONTINUATION_REQUIRED ||
      metadataSweepFollowUpRequired
        ? [nowEpochMilliseconds()]
        : [];
    const ordinaryBacklogDeadlines =
      result.generationReplayCompleted === true && (workPlan.ordinaryBacklogMarkerCount ?? 0) > 0
        ? [nowEpochMilliseconds()]
        : [];
    const wakeDeadlines = [
      ...deferredDeadlines,
      ...continuationDeadlines,
      ...ordinaryBacklogDeadlines,
    ];
    return {
      ...result,
      metadataSweepFollowUpRequired,
      nextWakeAtEpochMilliseconds: wakeDeadlines.length === 0 ? null : Math.min(...wakeDeadlines),
      replayBlockingFailureCategory:
        workPlan.quarantineDiagnostics.length === 0
          ? null
          : RecallBacklogFailureCategory.MARKER_DECODE_FAILED,
    };
  }
  const physicalSourceStates = await Promise.all(
    splitRecallWorkPlanByPhysicalSession(workPlan).map(async (physicalWorkPlan) => ({
      workPlan: physicalWorkPlan,
      sourceModifiedAtEpochMilliseconds:
        await readWorkerSourceModifiedAtEpochMilliseconds(physicalWorkPlan),
    })),
  );
  const shouldSweepMetadata =
    workPlanRequestsRecallMetadataSweep(workPlan) ||
    metadataSweepFollowUpRequired ||
    physicalSourceStates.some(
      ({ sourceModifiedAtEpochMilliseconds }) => sourceModifiedAtEpochMilliseconds === null,
    ) ||
    (await hasRecallMetadataSweepContinuation(options.controlDirectory));
  const knownSourceInventory = shouldSweepMetadata
    ? await (options.loadKnownSourceInventory?.() ??
        Promise.resolve({
          knownSources: options.knownSources ?? [],
          physicalProjections: [],
        }))
    : null;
  const metadataSweep = shouldSweepMetadata
    ? await (options.scanSessionMetadata ?? scanRecallSessionMetadata)({
        sessionRootDirectory: options.trustedSessionRoots[0] ?? '',
        controlDirectory: options.controlDirectory,
        knownSources: knownSourceInventory?.knownSources ?? [],
        ...(options.confirmedDeletionMaxMissingSourceCount === undefined
          ? {}
          : {
              confirmedDeletionMaxMissingSourceCount:
                options.confirmedDeletionMaxMissingSourceCount,
            }),
        ...(options.confirmedDeletionMaxMissingSourceRatio === undefined
          ? {}
          : {
              confirmedDeletionMaxMissingSourceRatio:
                options.confirmedDeletionMaxMissingSourceRatio,
            }),
      })
    : null;
  if (metadataSweep !== null) {
    metadataSweepFollowUpRequired = false;
  }
  if (workPlan.workItems.length === 0 && !hasRecallMetadataReconciliationWork(metadataSweep)) {
    const activeEntry = registry?.generations.find(
      ({ generationId }) => generationId === options.targetGenerationId,
    );
    const generationReplayCompleted =
      activeEntry?.state === RecallGenerationCutoverState.REPLAY_PENDING &&
      options.generationReplayCompletion
        ? await completeRecallGenerationReplay({
            ...options.generationReplayCompletion,
            markerSpoolDirectory: options.markerSpoolDirectory,
            markerQuarantineDirectory: options.markerQuarantineDirectory,
          })
        : null;
    return finishWorkerResult({
      workPlan,
      metadataSweep,
      heavyDependenciesLoaded: false,
      commitsFrozen: false,
      generationReplayCompleted,
      transferOutcomes: [],
      largeTransferDeferrals: [],
    });
  }
  const missingPhysicalSessionIds = new Set(metadataSweep?.missingPhysicalSessionIds ?? []);
  const missingSourceWorkPlans = physicalSourceStates
    .filter(
      ({ workPlan: physicalWorkPlan, sourceModifiedAtEpochMilliseconds }) =>
        sourceModifiedAtEpochMilliseconds === null ||
        missingPhysicalSessionIds.has(
          physicalWorkPlan.workItems[0]?.marker.physicalSessionId ?? '',
        ),
    )
    .map(({ workPlan: physicalWorkPlan }) => physicalWorkPlan);
  if (
    metadataSweep !== null &&
    metadataSweep.status !== RecallMetadataSweepStatus.CONTINUATION_REQUIRED &&
    (knownSourceInventory?.physicalProjections.length ?? 0) > 0
  ) {
    const deletionResult = await options.reconcileDeletion?.(
      metadataSweep,
      knownSourceInventory?.physicalProjections ?? [],
      missingSourceWorkPlans,
      workPlan,
    );
    metadataSweepFollowUpRequired = (deletionResult?.sourceMissingRecordedCount ?? 0) > 0;
  }
  if (
    metadataSweep !== null &&
    metadataSweep.observedSessionMetadata.some((entry) => entry.physicalSessionId === null)
  ) {
    const existingPhysicalSessionIds = new Set([
      ...workPlan.workItems.map(({ marker }) => marker.physicalSessionId),
      ...(knownSourceInventory?.physicalProjections.map((p) => p.physicalSessionId) ?? []),
    ]);
    const publishedCount = await publishMetadataSweepArrivalMarkers(
      metadataSweep,
      options,
      existingPhysicalSessionIds,
      nowEpochMilliseconds(),
    );
    if (publishedCount > 0) {
      metadataSweepFollowUpRequired = true;
    }
  }
  const transferOutcomes: IncrementalRecallWorkPlanTransferOutcome[] = [];
  const largeTransferDeferrals: RecallLargeTransferDeferral[] = [];
  const readyPhysicalSourceStates: Array<{
    workPlan: RecallMarkerReplayWorkPlan;
    sourceModifiedAtEpochMilliseconds: number;
  }> = [];
  const observedNowEpochMilliseconds = nowEpochMilliseconds();
  for (const physicalSourceState of physicalSourceStates) {
    const { workPlan: physicalWorkPlan, sourceModifiedAtEpochMilliseconds } = physicalSourceState;
    if (
      sourceModifiedAtEpochMilliseconds === null ||
      missingSourceWorkPlans.includes(physicalWorkPlan)
    ) {
      continue;
    }
    const physicalSessionId = physicalWorkPlan.workItems[0]?.marker.physicalSessionId;
    const persistedLargeTransferDeferral = options.persistedLargeTransferDeferrals?.find(
      (candidate) =>
        candidate.physicalSessionId === physicalSessionId &&
        candidate.sourceModifiedAtEpochMilliseconds === sourceModifiedAtEpochMilliseconds &&
        candidate.sourceMarkerIds.length === physicalWorkPlan.sourceMarkerIds.length &&
        candidate.sourceMarkerIds.every(
          (markerId, index) => markerId === physicalWorkPlan.sourceMarkerIds[index],
        ),
    );
    if (
      persistedLargeTransferDeferral !== undefined &&
      observedNowEpochMilliseconds < persistedLargeTransferDeferral.readyAtEpochMilliseconds
    ) {
      largeTransferDeferrals.push(persistedLargeTransferDeferral);
      transferOutcomes.push({
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
        readyAtEpochMilliseconds: persistedLargeTransferDeferral.readyAtEpochMilliseconds,
      });
      continue;
    }
    const schedule = scheduleRecallWorkPlanEligibility({
      workPlan: physicalWorkPlan,
      sourceModifiedAtEpochMilliseconds,
      preparedDocumentCount: 0,
      nowEpochMilliseconds: () => observedNowEpochMilliseconds,
    });
    if (schedule.ready) {
      readyPhysicalSourceStates.push({
        workPlan: physicalWorkPlan,
        sourceModifiedAtEpochMilliseconds,
      });
    } else {
      transferOutcomes.push({
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: schedule.threshold,
        readyAtEpochMilliseconds: schedule.readyAtEpochMilliseconds,
      });
    }
  }
  if (readyPhysicalSourceStates.length === 0) {
    return finishWorkerResult({
      workPlan,
      metadataSweep,
      heavyDependenciesLoaded: false,
      commitsFrozen: false,
      generationReplayCompleted: null,
      transferOutcomes,
      largeTransferDeferrals,
    });
  }
  await (options.loadHeavyDependencies ?? loadRecallIncrementalWorkerDependencies)();
  for (const {
    workPlan: physicalWorkPlan,
    sourceModifiedAtEpochMilliseconds,
  } of readyPhysicalSourceStates) {
    const outcome = await options.transferWorkPlan?.(physicalWorkPlan);
    if (outcome !== undefined) {
      transferOutcomes.push(outcome);
      if (
        outcome.kind === RecallIncrementalTransferOutcomeKind.DEFERRED &&
        outcome.threshold === RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER
      ) {
        const physicalSessionId = physicalWorkPlan.workItems[0]?.marker.physicalSessionId;
        if (physicalSessionId === undefined) {
          throw new Error('Recall large transfer deferral requires one physical session');
        }
        largeTransferDeferrals.push({
          physicalSessionId,
          sourceModifiedAtEpochMilliseconds,
          sourceMarkerIds: [...physicalWorkPlan.sourceMarkerIds],
          threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
          readyAtEpochMilliseconds: outcome.readyAtEpochMilliseconds,
        });
      }
    }
  }
  const generationReplayCompleted =
    activeGenerationEntry?.state === RecallGenerationCutoverState.REPLAY_PENDING &&
    options.generationReplayCompletion
      ? await completeRecallGenerationReplay({
          ...options.generationReplayCompletion,
          markerSpoolDirectory: options.markerSpoolDirectory,
          markerQuarantineDirectory: options.markerQuarantineDirectory,
        })
      : null;
  return finishWorkerResult({
    workPlan,
    metadataSweep,
    heavyDependenciesLoaded: true,
    commitsFrozen: false,
    generationReplayCompleted,
    transferOutcomes,
    largeTransferDeferrals,
  });
}

async function loadTargetRecallKnownSourceInventory(
  projectionDatabasePath: string,
  generationId: string,
  sessionsDirectory: string,
): Promise<RecallKnownSourceInventory> {
  const { ZVecOpen } = await import('@zvec/zvec');
  const store = ZVecOpen(projectionDatabasePath, { readOnly: true });
  try {
    const records =
      store.stats.docCount === 0
        ? []
        : store.querySync({
            filter: `projectionKind = '${RecallSessionProjectionKind.PHYSICAL_SESSION}'`,
            topk: store.stats.docCount,
            outputFields: ['projectionJson'],
            includeVector: false,
          });
    const physicalProjections = records.map(({ id, fields }) => {
      if (typeof fields.projectionJson !== 'string') {
        throw new Error(`Recall target physical projection JSON missing for ${id}`);
      }
      let parsed: unknown;
      try {
        parsed = JSON.parse(fields.projectionJson);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall target physical projection JSON invalid for ${id}: ${message}`, {
          cause: error,
        });
      }
      if (
        typeof parsed !== 'object' ||
        parsed === null ||
        !('ingestionProjectionPayload' in parsed)
      ) {
        throw new Error(`Recall target physical ingestion projection missing for ${id}`);
      }
      const projection = decodeRecallSessionProjection(parsed.ingestionProjectionPayload, {
        expectedGenerationId: generationId,
      });
      if (projection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
        throw new Error(`Recall target physical projection kind mismatch for ${id}`);
      }
      return projection;
    });
    const knownSources = physicalProjections.map((projection) => {
      const sourceRelativePath = relative(sessionsDirectory, projection.sourcePath);
      if (sourceRelativePath.startsWith('..') || isAbsolute(sourceRelativePath)) {
        throw new Error('Recall target physical projection is outside the configured session root');
      }
      return { physicalSessionId: projection.physicalSessionId, relativePath: sourceRelativePath };
    });
    return { knownSources, physicalProjections };
  } finally {
    store.closeSync();
  }
}

function createTargetDeletionResult(): ConfirmedSessionDeletionReconciliationResult {
  return {
    halted: false,
    consideredPhysicalSessionCount: 0,
    sourceMissingRecordedCount: 0,
    sourceMissingClearedCount: 0,
    confirmedSourceDeletionCount: 0,
    removedEvidenceOccurrenceCount: 0,
    removedLogicalProjectionCount: 0,
    removedPhysicalProjectionCount: 0,
    acknowledgedCheckpointCount: 0,
    haltCategoryCounts: {},
  };
}

function haltTargetDeletionReconciliation(
  result: ConfirmedSessionDeletionReconciliationResult,
  haltCategory: RecallConfirmedDeletionHaltCategory,
): ConfirmedSessionDeletionReconciliationResult {
  return {
    ...result,
    halted: true,
    haltCategoryCounts: { [haltCategory]: 1 },
  };
}

async function reconcileTargetConfirmedSessionDeletion(options: {
  metadataSweep: RecallSessionMetadataSweepResult;
  physicalProjections: readonly PhysicalSessionProjection[];
  missingSourceWorkPlans: readonly RecallMarkerReplayWorkPlan[];
  workPlan: RecallMarkerReplayWorkPlan;
  sessionsDirectory: string;
  transferRequest(
    request: IncrementalRecallWorkPlanRequest,
  ): Promise<IncrementalRecallWorkPlanTransferOutcome>;
}): Promise<ConfirmedSessionDeletionReconciliationResult> {
  const result = createTargetDeletionResult();
  if (options.metadataSweep.status !== RecallMetadataSweepStatus.COMPLETE) {
    const haltCategory =
      options.metadataSweep.status === RecallMetadataSweepStatus.ROOT_UNAVAILABLE
        ? RecallConfirmedDeletionHaltCategory.ROOT_UNAVAILABLE
        : options.metadataSweep.status === RecallMetadataSweepStatus.PERMISSION_DENIED
          ? RecallConfirmedDeletionHaltCategory.PERMISSION_DENIED
          : options.metadataSweep.status === RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS
            ? RecallConfirmedDeletionHaltCategory.SUSPICIOUS_MASS_LOSS
            : RecallConfirmedDeletionHaltCategory.INCOMPLETE_SWEEP;
    return haltTargetDeletionReconciliation(result, haltCategory);
  }
  for (const projection of options.physicalProjections) {
    const observedIdentity = options.metadataSweep.observedKnownSourceIdentities.find(
      ({ physicalSessionId }) => physicalSessionId === projection.physicalSessionId,
    );
    const sourceWorkPlan = options.missingSourceWorkPlans.find(
      (candidate) =>
        candidate.workItems[0]?.marker.physicalSessionId === projection.physicalSessionId,
    );
    const matchingWorkItems = options.workPlan.workItems.filter(
      ({ marker }) => marker.physicalSessionId === projection.physicalSessionId,
    );
    const physicalWorkPlan: RecallMarkerReplayWorkPlan = sourceWorkPlan ?? {
      ...options.workPlan,
      sourceMarkerIds: matchingWorkItems.flatMap(({ coveredMarkerIds }) => coveredMarkerIds),
      workItems: matchingWorkItems,
    };
    const coveredProjection =
      physicalWorkPlan.workItems.length === 0
        ? projection
        : {
            ...projection,
            markerCheckpoint: mergeRecallMarkerCheckpoint({
              generationId: projection.generationId,
              current: projection.markerCheckpoint,
              coveredMarkerIds: physicalWorkPlan.workItems.flatMap(
                ({ coveredMarkerIds }) => coveredMarkerIds,
              ),
              runtimeSequences: physicalWorkPlan.workItems.map(({ marker }) => ({
                runtimeInstanceId: marker.runtimeInstanceId,
                sequence: marker.runtimeSequence,
              })),
            }),
          };
    const decision = decideConfirmedSessionDeletion({
      projection: coveredProjection,
      sweepId: options.metadataSweep.sweepId,
      sweepStatus: options.metadataSweep.status,
      observedAtEpochMilliseconds: Date.now(),
      sourceObservation:
        observedIdentity === undefined
          ? null
          : {
              sourceDevice: observedIdentity.sourceDevice,
              sourceInode: observedIdentity.sourceInode,
            },
    });
    if (decision.kind === RecallConfirmedDeletionDecisionKind.HALT) {
      return haltTargetDeletionReconciliation(result, decision.haltCategory);
    }
    if (decision.kind === RecallConfirmedDeletionDecisionKind.NO_CHANGE) {
      continue;
    }
    if ('nextProjection' in decision) {
      await options.transferRequest({
        physicalSessionProjectionUpdate: {
          workPlan: physicalWorkPlan,
          projection: decision.nextProjection,
          ...(decision.kind === RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION
            ? { acknowledgeMarkers: false }
            : {}),
        },
      });
      if (decision.kind === RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING) {
        result.sourceMissingRecordedCount += 1;
        result.consideredPhysicalSessionCount += 1;
        continue;
      }
      if (decision.kind === RecallConfirmedDeletionDecisionKind.CLEAR_SOURCE_MISSING) {
        result.sourceMissingClearedCount += 1;
        result.consideredPhysicalSessionCount += 1;
        continue;
      }
      result.confirmedSourceDeletionCount += 1;
    }
    const physicalSource = resolveRecallPhysicalSourceIdentity(
      options.sessionsDirectory,
      projection.sourcePath,
    );
    await options.transferRequest({
      confirmedPhysicalSourceDeletion: {
        targetGenerationId: projection.generationId,
        physicalSourceIdentity: physicalSource.physicalSourceIdentity,
      },
    });
    if (physicalWorkPlan.workItems.length > 0) {
      await acknowledgeCoveredRecallMarkers(physicalWorkPlan, coveredProjection.markerCheckpoint);
      result.acknowledgedCheckpointCount += 1;
    }
    result.consideredPhysicalSessionCount += 1;
    result.removedPhysicalProjectionCount += 1;
  }
  return result;
}

/** Scalar generation-state paths and clock used by one worker backlog refresh. */
export interface RecallIncrementalWorkerBacklogPaths {
  backlogSummaryPath: string;
  generationRegistryPath: string;
  targetGenerationId: string;
  nowEpochMilliseconds?: () => number;
}

/** Minimal pending work observation used to refresh scalar backlog count and age. */
export interface RecallIncrementalWorkerBacklogObservation {
  workPlan: {
    workItems: ReadonlyArray<{
      marker: {
        physicalSessionId: string;
        createdAtEpochMilliseconds: number;
      };
    }>;
  };
}

/** Atomically refreshes scalar backlog state after worker completion or executable failure. */
export async function writeRecallIncrementalWorkerBacklog(
  paths: RecallIncrementalWorkerBacklogPaths,
  failureCategory: RecallBacklogFailureCategory | null,
  result?: RecallIncrementalWorkerBacklogObservation,
): Promise<void> {
  const registry = await readRecallGenerationRegistry(paths.generationRegistryPath);
  const activeEntry = registry?.generations.find(
    ({ generationId }) => generationId === paths.targetGenerationId,
  );
  if (registry === null || activeEntry === undefined) {
    return;
  }
  const nowEpochMilliseconds = paths.nowEpochMilliseconds?.() ?? Date.now();
  const buildingEntry = registry.generations.find(
    ({ generationId }) => generationId === registry.buildingGenerationId,
  );
  const physicalSessionIds = new Set(
    result?.workPlan.workItems.map(({ marker }) => marker.physicalSessionId) ?? [],
  );
  const oldestMarkerCreatedAtEpochMilliseconds = result?.workPlan.workItems.reduce(
    (oldest, { marker }) => Math.min(oldest, marker.createdAtEpochMilliseconds),
    Number.POSITIVE_INFINITY,
  );
  await writeRecallBacklogSummary(paths.backlogSummaryPath, {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: physicalSessionIds.size,
    oldestEligibleMarkerAgeMilliseconds:
      oldestMarkerCreatedAtEpochMilliseconds === undefined ||
      !Number.isFinite(oldestMarkerCreatedAtEpochMilliseconds)
        ? null
        : Math.max(0, nowEpochMilliseconds - oldestMarkerCreatedAtEpochMilliseconds),
    activeGenerationId: paths.targetGenerationId,
    buildingGenerationId: registry.buildingGenerationId,
    generationState: buildingEntry?.state ?? activeEntry.state,
    activeGenerationAgeMilliseconds: Math.max(
      0,
      nowEpochMilliseconds - activeEntry.stateChangedAtEpochMilliseconds,
    ),
    rebuildAgeMilliseconds:
      buildingEntry === undefined
        ? null
        : Math.max(0, nowEpochMilliseconds - buildingEntry.rebuildStartedAtEpochMilliseconds),
    lastFailureCategory: failureCategory,
    observedAtEpochMilliseconds: nowEpochMilliseconds,
  });
}

/** Registry and optional work observation used for one durable detached-worker failure update. */
export interface RecallIncrementalWorkerFailureBacklogPaths {
  backlogSummaryPath: string;
  generationRegistryPath: string;
  targetGenerationId?: string;
  nowEpochMilliseconds?: () => number;
}

async function readPriorRecallIncrementalWorkerBacklog(
  backlogSummaryPath: string,
): Promise<ReturnType<typeof decodeRecallBacklogSummary> | null> {
  try {
    return decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

async function writeRecallIncrementalWorkerFailureFromRegistry(
  paths: RecallIncrementalWorkerFailureBacklogPaths,
  failureCategory: RecallBacklogFailureCategory,
  nowEpochMilliseconds: number,
  result?: RecallIncrementalWorkerBacklogObservation,
): Promise<void> {
  const registry = await readRecallGenerationRegistry(paths.generationRegistryPath);
  const targetGenerationId = paths.targetGenerationId ?? registry?.activeGenerationId;
  const targetGeneration = registry?.generations.find(
    ({ generationId }) => generationId === targetGenerationId,
  );
  if (targetGenerationId === null || targetGenerationId === undefined) {
    throw new Error('Recall incremental worker failure has no generation state');
  }
  if (targetGeneration === undefined) {
    throw new Error('Recall incremental worker failure generation is absent from the registry');
  }
  const priorSummary =
    result === undefined
      ? await readPriorRecallIncrementalWorkerBacklog(paths.backlogSummaryPath)
      : null;
  if (priorSummary !== null) {
    const buildingGeneration = registry?.generations.find(
      ({ generationId }) => generationId === registry.buildingGenerationId,
    );
    await writeRecallBacklogSummary(paths.backlogSummaryPath, {
      ...priorSummary,
      activeGenerationId: targetGenerationId,
      buildingGenerationId: registry?.buildingGenerationId ?? null,
      generationState: buildingGeneration?.state ?? targetGeneration.state,
      activeGenerationAgeMilliseconds: Math.max(
        0,
        nowEpochMilliseconds - targetGeneration.stateChangedAtEpochMilliseconds,
      ),
      rebuildAgeMilliseconds:
        buildingGeneration === undefined
          ? null
          : Math.max(
              0,
              nowEpochMilliseconds - buildingGeneration.rebuildStartedAtEpochMilliseconds,
            ),
      lastFailureCategory: failureCategory,
      observedAtEpochMilliseconds: nowEpochMilliseconds,
    });
    return;
  }
  await writeRecallIncrementalWorkerBacklog(
    {
      backlogSummaryPath: paths.backlogSummaryPath,
      generationRegistryPath: paths.generationRegistryPath,
      targetGenerationId,
      nowEpochMilliseconds: () => nowEpochMilliseconds,
    },
    failureCategory,
    result,
  );
}

async function writePriorRecallIncrementalWorkerFailure(
  backlogSummaryPath: string,
  failureCategory: RecallBacklogFailureCategory,
  nowEpochMilliseconds: number,
): Promise<void> {
  const summary = decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  await writeRecallBacklogSummary(backlogSummaryPath, {
    ...summary,
    lastFailureCategory: failureCategory,
    observedAtEpochMilliseconds: nowEpochMilliseconds,
  });
}

/** Atomically persists scalar worker failure state, preserving the prior summary if registry recovery fails. */
export async function writeRecallIncrementalWorkerFailureBacklog(
  paths: RecallIncrementalWorkerFailureBacklogPaths,
  failureCategory: RecallBacklogFailureCategory,
  result?: RecallIncrementalWorkerBacklogObservation,
): Promise<void> {
  const nowEpochMilliseconds = paths.nowEpochMilliseconds?.() ?? Date.now();
  const [registryWrite] = await Promise.allSettled([
    writeRecallIncrementalWorkerFailureFromRegistry(
      paths,
      failureCategory,
      nowEpochMilliseconds,
      result,
    ),
  ]);
  if (registryWrite?.status === 'fulfilled') {
    return;
  }

  const [priorSummaryWrite] = await Promise.allSettled([
    writePriorRecallIncrementalWorkerFailure(
      paths.backlogSummaryPath,
      failureCategory,
      nowEpochMilliseconds,
    ),
  ]);
  if (priorSummaryWrite?.status === 'rejected') {
    throw new AggregateError(
      [registryWrite?.reason, priorSummaryWrite.reason],
      'Recall incremental worker durable failure update failed',
    );
  }
}

/** Public service callback used when an operator process already owns configured inference. */
export interface RunConfiguredRecallIncrementalWorkerOptions {
  transferWorkPlan?(
    this: void,
    request: IncrementalRecallWorkPlanRequest,
  ): Promise<IncrementalRecallWorkPlanTransferOutcome>;
}

/** One generation-targeted configured worker pass and its bounded public result. */
export interface ConfiguredRecallIncrementalWorkerResult {
  activeGenerationId: string;
  result: RecallIncrementalWorkerResult;
}

/** Runs the production short-lived worker orchestration without acquiring process ownership. */
export async function runConfiguredRecallIncrementalWorker(
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
  operationDiagnostics: RecallOperationDiagnostics,
  options: RunConfiguredRecallIncrementalWorkerOptions = {},
): Promise<ConfiguredRecallIncrementalWorkerResult | null> {
  await recoverRecallGenerationCutover({
    activeGenerationPointerPath: config.activeGenerationPointerPath,
    generationRegistryPath: config.generationRegistryPath,
    generationRootDirectory: config.generationRootDirectory,
    backlogSummaryPath: config.backlogSummaryPath,
    lockPath: config.lockPath,
    embeddingDimensions: config.embeddingDimensions,
  });
  try {
    await access(config.activeGenerationPointerPath);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
  const activeSelection = await readRecallActiveGenerationSelection(
    config.activeGenerationPointerPath,
    config.generationRootDirectory,
  );
  const registry = await readRecallGenerationRegistry(config.generationRegistryPath);
  const schedulePath = resolve(config.markerControlDirectory, 'incremental-worker-schedule.json');
  const persistedSchedule = await readRecallIncrementalWorkerSchedule(schedulePath);
  let configuredRuntime: { dispose(): Promise<void> } | undefined;
  let productionTransferDependencies:
    | Promise<{
        transferRequest(
          request: IncrementalRecallWorkPlanRequest,
        ): Promise<IncrementalRecallWorkPlanTransferOutcome>;
      }>
    | undefined;
  const loadProductionTransferDependencies = () => {
    productionTransferDependencies ??= (async () => {
      if (options.transferWorkPlan !== undefined) {
        return { transferRequest: options.transferWorkPlan };
      }
      let rawManifest: unknown;
      try {
        rawManifest = JSON.parse(await readFile(activeSelection.manifestPath, 'utf8'));
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Recall incremental worker active generation manifest unreadable: ${message}`,
          { cause: error },
        );
      }
      if (!isUnknownRecord(rawManifest) || rawManifest.generationFormatVersion !== 1) {
        throw new Error(
          'Recall incremental worker active generation is not target format version 1',
        );
      }
      const inferenceRuntimeModule = await import('./configured-recall-inference-runtime.js');
      const runtime = await inferenceRuntimeModule.createConfiguredRecallInferenceRuntime(config);
      configuredRuntime = runtime;
      return {
        transferRequest: (request: IncrementalRecallWorkPlanRequest) =>
          runtime.service.transferIncrementalRecallWorkPlan(request),
      };
    })();
    return productionTransferDependencies;
  };
  let deletionReconciliationHalted = false;
  let result: RecallIncrementalWorkerResult;
  try {
    result = await runRecallIncrementalWorker({
      markerSpoolDirectory: config.markerSpoolDirectory,
      markerQuarantineDirectory: config.markerQuarantineDirectory,
      controlDirectory: config.markerControlDirectory,
      targetGenerationId: activeSelection.activeGenerationId,
      generationRegistryPath: config.generationRegistryPath,
      generationReplayCompletion: {
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRegistryPath: config.generationRegistryPath,
        backlogSummaryPath: config.backlogSummaryPath,
        generationRootDirectory: config.generationRootDirectory,
        lockPath: config.lockPath,
      },
      ...(registry?.rollbackGenerationId
        ? { retainedMarkerDirectory: resolve(config.markerControlDirectory, 'rollback-retained') }
        : {}),
      trustedSessionRoots: [config.sessionsDirectory],
      confirmedDeletionMaxMissingSourceCount: config.confirmedDeletionMaxMissingSourceCount,
      confirmedDeletionMaxMissingSourceRatio: config.confirmedDeletionMaxMissingSourceRatio,
      operationDiagnostics,
      persistedLargeTransferDeferrals: persistedSchedule?.largeTransferDeferrals ?? [],
      metadataSweepRequested: persistedSchedule?.metadataSweepRequested ?? false,
      async transferWorkPlan(workPlan) {
        const dependencies = await loadProductionTransferDependencies();
        return dependencies.transferRequest(workPlan);
      },
      loadKnownSourceInventory: () =>
        loadTargetRecallKnownSourceInventory(
          activeSelection.projectionDatabasePath,
          activeSelection.activeGenerationId,
          config.sessionsDirectory,
        ),
      async reconcileDeletion(
        metadataSweep,
        physicalProjections,
        missingSourceWorkPlans,
        workPlan,
      ) {
        const dependencies = await loadProductionTransferDependencies();
        const deletionResult = await reconcileTargetConfirmedSessionDeletion({
          metadataSweep,
          physicalProjections,
          missingSourceWorkPlans,
          workPlan,
          sessionsDirectory: config.sessionsDirectory,
          transferRequest: (request) => dependencies.transferRequest(request),
        });
        if (deletionResult.halted) {
          deletionReconciliationHalted = true;
          await reportRecallIncrementalWorkerDeletionHalt({
            operationDiagnostics,
            generationId: activeSelection.activeGenerationId,
            generationState:
              registry?.generations.find(
                ({ generationId }) => generationId === activeSelection.activeGenerationId,
              )?.state ?? null,
            haltResult: deletionResult,
            workPlan,
            persistFailure: () =>
              writeRecallIncrementalWorkerFailureBacklog(
                {
                  backlogSummaryPath: config.backlogSummaryPath,
                  generationRegistryPath: config.generationRegistryPath,
                  targetGenerationId: activeSelection.activeGenerationId,
                },
                RecallBacklogFailureCategory.CONFIRMED_DELETION_HALTED,
                { workPlan },
              ),
          });
        }
        return deletionResult;
      },
    });
    const remainingWorkPlan = await coordinateRecallMarkerReplay({
      markerSpoolDirectory: config.markerSpoolDirectory,
      markerQuarantineDirectory: config.markerQuarantineDirectory,
      targetGenerationId: activeSelection.activeGenerationId,
      trustedSessionRoots: [config.sessionsDirectory],
      ...(registry?.rollbackGenerationId
        ? { retainedMarkerDirectory: resolve(config.markerControlDirectory, 'rollback-retained') }
        : {}),
    });
    const shouldSignalWake = await persistRecallIncrementalWorkerSchedule({
      schedulePath,
      nowEpochMilliseconds: Date.now(),
      acknowledgedMetadataSweepRevision: persistedSchedule?.metadataSweepRevision ?? 0,
      schedule: {
        version: RECALL_INCREMENTAL_WORKER_SCHEDULE_VERSION,
        nextWakeAtEpochMilliseconds: result.nextWakeAtEpochMilliseconds,
        metadataSweepRequested: result.metadataSweepFollowUpRequired,
        largeTransferDeferrals: [...result.largeTransferDeferrals],
      },
    });
    if (shouldSignalWake && result.nextWakeAtEpochMilliseconds !== null) {
      signalRecallIncrementalWorkerWake({
        readyAtEpochMilliseconds: result.nextWakeAtEpochMilliseconds,
        workerOwnershipLockPath: config.workerOwnershipLockPath,
        workerExecutablePath: fileURLToPath(import.meta.url),
      });
    }
    await writeRecallIncrementalWorkerBacklog(
      {
        backlogSummaryPath: config.backlogSummaryPath,
        generationRegistryPath: config.generationRegistryPath,
        targetGenerationId: activeSelection.activeGenerationId,
      },
      deletionReconciliationHalted
        ? RecallBacklogFailureCategory.CONFIRMED_DELETION_HALTED
        : result.replayBlockingFailureCategory,
      { workPlan: remainingWorkPlan },
    );
  } finally {
    await configuredRuntime?.dispose();
  }
  return { activeGenerationId: activeSelection.activeGenerationId, result };
}

async function runRecallIncrementalWorkerExecutable(): Promise<void> {
  let failureVisibilityInitialized = false;
  try {
    const config = await loadRecallConversationConfig();
    const failureBacklogPaths: RecallIncrementalWorkerFailureBacklogPaths = {
      backlogSummaryPath: config.backlogSummaryPath,
      generationRegistryPath: config.generationRegistryPath,
    };
    const operationDiagnostics = createRecallOperationDiagnostics({
      mode: config.diagnosticsMode,
      activeLogPath: config.incrementalDiagnosticLogPath,
      retainedLogPath: config.incrementalDiagnosticLogPath.replace(/\.jsonl$/u, '.previous.jsonl'),
      notifyWarning(message) {
        process.emitWarning(message);
      },
      onPersistenceFailure: () =>
        writeRecallIncrementalWorkerFailureBacklog(
          failureBacklogPaths,
          RecallBacklogFailureCategory.DIAGNOSTICS_PERSISTENCE_FAILED,
        ),
    });
    failureVisibilityInitialized = true;
    await runRecallIncrementalWorkerDiagnosticBoundary({
      operationDiagnostics,
      persistFailure: () =>
        writeRecallIncrementalWorkerFailureBacklog(
          failureBacklogPaths,
          RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED,
        ),
      run: () => runConfiguredRecallIncrementalWorker(config, operationDiagnostics).then(() => {}),
    });
  } catch (error) {
    if (!failureVisibilityInitialized) {
      process.emitWarning(
        `Recall incremental worker failure sink initialization failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
      );
    }
    throw error;
  }
}

const executedModulePath = process.argv[1];
if (
  executedModulePath !== undefined &&
  resolve(executedModulePath) === fileURLToPath(import.meta.url)
) {
  runRecallIncrementalWorkerExecutable().catch(() => {
    process.exitCode = 1;
  });
}

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
  RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallEligibilityThreshold,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallMetadataSweepStatus,
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
import type { ConfirmedSessionDeletionReconciliationResult } from './reconcile-confirmed-session-deletion.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';
import {
  createRecallWorkMarkerId,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarkerCodecOptions,
  type RecallWorkMarkerIdentity,
} from './recall-work-marker.js';
import { publishRecallWorkMarker } from './publish-recall-work-marker.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { createLineageResolver, resolveProjectIdentity } from './resolve-project-identity.js';
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
import type { IncrementalRecallWorkPlanTransferOutcome } from './transfer-incremental-recall-work-plan.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

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
  const buildingInProgress = registry?.buildingGenerationId != null;
  const activeGenerationEntry = registry?.generations.find(
    ({ generationId }) => generationId === registry.activeGenerationId,
  );
  const activeGenerationIsLegacy =
    activeGenerationEntry?.state === RecallGenerationCutoverState.LEGACY_READ_ONLY;
  const commitsFrozen = buildingInProgress || activeGenerationIsLegacy;
  let metadataSweepFollowUpRequired = options.metadataSweepRequested ?? false;
  let fixedReplayMarkerIds: readonly string[] | undefined;
  if (
    activeGenerationEntry?.state === RecallGenerationCutoverState.REPLAY_PENDING &&
    options.generationReplayCompletion?.generationRootDirectory !== undefined
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
  if (commitsFrozen) {
    return finishWorkerResult({
      workPlan,
      metadataSweep: null,
      heavyDependenciesLoaded: false,
      commitsFrozen: true,
      generationReplayCompleted: null,
      transferOutcomes: [],
      largeTransferDeferrals: [],
    });
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

async function loadRecallKnownSourceInventory(
  projectionDatabasePath: string,
  generationId: string,
  sessionsDirectory: string,
): Promise<RecallKnownSourceInventory> {
  const { openZvecSessionProjectionStore } = await import('./zvec-session-projection-store.js');
  const store = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const physicalProjections = store.listPhysicalProjections();
    const knownSources = physicalProjections.map((projection) => {
      const relativePath = relative(sessionsDirectory, projection.sourcePath);
      if (relativePath.startsWith('..') || isAbsolute(relativePath)) {
        throw new Error('Recall active projection source is outside the configured session root');
      }
      return { physicalSessionId: projection.physicalSessionId, relativePath };
    });
    return { knownSources, physicalProjections };
  } finally {
    store.close();
  }
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

async function runConfiguredRecallIncrementalWorkerExecutable(
  config: Awaited<ReturnType<typeof loadRecallConversationConfig>>,
  operationDiagnostics: RecallOperationDiagnostics,
): Promise<void> {
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
      return;
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
  const resolveWorkerProjectIdentity = createLineageResolver(
    config.projectLineages,
    resolveProjectIdentity,
  );
  let configuredRuntime: { dispose(): Promise<void> } | undefined;
  let productionTransferDependencies:
    | Promise<{
        transferWorkPlan(
          workPlan: RecallMarkerReplayWorkPlan,
        ): Promise<IncrementalRecallWorkPlanTransferOutcome>;
      }>
    | undefined;
  const loadProductionTransferDependencies = () => {
    productionTransferDependencies ??= (async () => {
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
      if (isUnknownRecord(rawManifest) && rawManifest.generationFormatVersion === 1) {
        const inferenceRuntimeModule = await import('./configured-recall-inference-runtime.js');
        const runtime = await inferenceRuntimeModule.createConfiguredRecallInferenceRuntime(config);
        configuredRuntime = runtime;
        return {
          transferWorkPlan: (workPlan: RecallMarkerReplayWorkPlan) =>
            runtime.service.transferIncrementalRecallWorkPlan(workPlan),
        };
      }
      const [
        embeddingCacheModule,
        inferenceRuntimeModule,
        inferenceConfigModule,
        manifestModule,
        transferModule,
      ] = await Promise.all([
        import('./embedding-vector-cache.js'),
        import('./configured-recall-inference-runtime.js'),
        import('./recall-inference-configuration.js'),
        import('./recall-index-manifest.js'),
        import('./transfer-legacy-incremental-recall-work-plan.js'),
      ]);
      const manifest = await manifestModule.readRecallIndexManifest(activeSelection.manifestPath);
      if (manifest === null) {
        throw new Error('Recall incremental worker active generation manifest missing');
      }
      const inferenceConfigPath =
        inferenceRuntimeModule.resolveRecallInferenceConfigurationPath(config);
      const inferenceConfig = await inferenceConfigModule.readRecallInferenceConfiguration(
        inferenceConfigPath,
        { generationRegistryPath: config.generationRegistryPath },
      );
      let loadTokenizer: () => Promise<ConversationTextTokenizer>;
      let embeddings: { embedTexts(texts: string[], signal?: AbortSignal): Promise<number[][]> };
      if (inferenceConfig.embedding !== null) {
        const runtime = await inferenceRuntimeModule.createConfiguredRecallInferenceRuntime(
          config,
          {
            inferenceConfigurationPath: inferenceConfigPath,
          },
        );
        configuredRuntime = runtime;
        loadTokenizer = () => runtime.loadTokenizer();
        embeddings = {
          embedTexts: (texts, signal) => runtime.embeddingProvider.embedDocuments(texts, signal),
        };
      } else {
        const [embeddingClientModule, tokenizerModule] = await Promise.all([
          import('./local-embedding-client.js'),
          import('./octen-conversation-tokenizer.js'),
        ]);
        embeddings = embeddingClientModule.createLocalEmbeddingClient({
          baseUrl: config.embeddingBaseUrl,
          model: config.embeddingModel,
          dimensions: config.embeddingDimensions,
          batchSize: config.embeddingBatchSize,
        });
        loadTokenizer = () =>
          tokenizerModule.loadOctenConversationTokenizer({
            cacheDirectory: config.tokenizerCacheDirectory,
          });
      }
      const embeddingCache = embeddingCacheModule.createEmbeddingVectorCache({
        cacheDirectory: config.embeddingCacheDirectory,
        identity: embeddingCacheModule.createEmbeddingVectorCacheIdentity(manifest),
        embeddingRequestBatchSize: config.embeddingBatchSize,
        embeddings,
      });
      return {
        transferWorkPlan: (workPlan: RecallMarkerReplayWorkPlan) =>
          transferModule.transferLegacyIncrementalRecallWorkPlan({
            workPlan,
            lockPath: config.lockPath,
            evidenceDatabasePath: activeSelection.databasePath,
            projectionDatabasePath: activeSelection.projectionDatabasePath,
            embeddingDimensions: manifest.embedding.dimensions,
            chunkPolicy: manifest.chunkPolicy,
            loadTokenizer,
            resolveProjectIdentity: resolveWorkerProjectIdentity,
            embeddingCache,
            operationDiagnostics,
            nowEpochMilliseconds: Date.now,
          }),
      };
    })();
    return productionTransferDependencies;
  };
  let deletionReconciliationHalted = false;
  try {
    const result = await runRecallIncrementalWorker({
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
        return dependencies.transferWorkPlan(workPlan);
      },
      loadKnownSourceInventory: () =>
        loadRecallKnownSourceInventory(
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
        const { reconcileConfirmedSessionDeletion } =
          await import('./reconcile-confirmed-session-deletion.js');
        const deletionResult = await reconcileConfirmedSessionDeletion({
          metadataSweep,
          physicalProjections,
          activeGenerationPointerPath: config.activeGenerationPointerPath,
          generationRegistryPath: config.generationRegistryPath,
          generationRootDirectory: config.generationRootDirectory,
          lockPath: config.lockPath,
          embeddingDimensions: config.embeddingDimensions,
          markerWorkPlans: missingSourceWorkPlans,
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
      result,
    );
  } finally {
    await configuredRuntime?.dispose();
  }
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
      run: () => runConfiguredRecallIncrementalWorkerExecutable(config, operationDiagnostics),
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

import { access, stat } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { completeRecallGenerationReplay } from './complete-recall-generation-replay.js';
import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import {
  coordinateRecallMarkerReplay,
  type RecallGenerationReplayCompletionPaths,
  type RecallMarkerReplayWorkPlan,
} from './coordinate-recall-marker-replay.js';
import {
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallMetadataSweepStatus,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { readRecallIndexManifest } from './recall-index-manifest.js';
import { createLocalEmbeddingClient } from './local-embedding-client.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import {
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';
import type { RecallWorkMarkerCodecOptions } from './recall-work-marker.js';
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
import {
  transferIncrementalRecallWorkPlan,
  type IncrementalRecallWorkPlanTransferOutcome,
} from './transfer-incremental-recall-work-plan.js';

/** Explicit lightweight paths and lazy import boundary for one short-lived incremental worker. */
export interface RunRecallIncrementalWorkerOptions extends RecallWorkMarkerCodecOptions {
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  controlDirectory: string;
  targetGenerationId: string;
  generationRegistryPath?: string;
  retainedMarkerDirectory?: string;
  generationReplayCompletion?: RecallGenerationReplayCompletionPaths;
  knownSources?: readonly KnownRecallSessionMetadataSource[];
  confirmedDeletionMaxMissingSourceCount?: number;
  confirmedDeletionMaxMissingSourceRatio?: number;
  loadKnownSourceInventory?: () => Promise<RecallKnownSourceInventory>;
  reconcileDeletion?: (
    metadataSweep: RecallSessionMetadataSweepResult,
    physicalProjections: readonly PhysicalSessionProjection[],
  ) => Promise<void>;
  loadHeavyDependencies?: () => Promise<void>;
  transferWorkPlan?: (
    workPlan: RecallMarkerReplayWorkPlan,
  ) => Promise<IncrementalRecallWorkPlanTransferOutcome>;
  operationDiagnostics?: Pick<RecallOperationDiagnostics, 'recordIncrementalOperation'>;
  nowEpochMilliseconds?: () => number;
  monotonicMilliseconds?: () => number;
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
}

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
): Promise<number> {
  const sourcePath = workPlan.workItems[0]?.marker.physicalSessionPath;
  if (sourcePath === undefined) {
    throw new Error('Recall worker source metadata requires one marker work item');
  }
  const metadata = await stat(sourcePath, { bigint: true });
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
  const activeGenerationIsLegacy =
    registry?.generations.find(({ generationId }) => generationId === registry.activeGenerationId)
      ?.state === RecallGenerationCutoverState.LEGACY_READ_ONLY;
  const commitsFrozen = buildingInProgress || activeGenerationIsLegacy;
  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: options.markerSpoolDirectory,
    markerQuarantineDirectory: options.markerQuarantineDirectory,
    targetGenerationId: options.targetGenerationId,
    trustedSessionRoots: options.trustedSessionRoots,
    ...(options.retainedMarkerDirectory
      ? { retainedMarkerDirectory: options.retainedMarkerDirectory }
      : {}),
    ...(options.generationReplayCompletion
      ? { generationReplayCompletion: options.generationReplayCompletion }
      : {}),
  });
  function finishWorkerResult(
    result: RecallIncrementalWorkerResult,
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
    return result;
  }
  if (commitsFrozen) {
    return finishWorkerResult({
      workPlan,
      metadataSweep: null,
      heavyDependenciesLoaded: false,
      commitsFrozen: true,
      generationReplayCompleted: null,
      transferOutcomes: [],
    });
  }
  const shouldSweepMetadata =
    workPlanRequestsRecallMetadataSweep(workPlan) ||
    (await hasRecallMetadataSweepContinuation(options.controlDirectory));
  const knownSourceInventory = shouldSweepMetadata
    ? await (options.loadKnownSourceInventory?.() ??
        Promise.resolve({
          knownSources: options.knownSources ?? [],
          physicalProjections: [],
        }))
    : null;
  const metadataSweep = shouldSweepMetadata
    ? await scanRecallSessionMetadata({
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
          })
        : null;
    return finishWorkerResult({
      workPlan,
      metadataSweep,
      heavyDependenciesLoaded: false,
      commitsFrozen: false,
      generationReplayCompleted,
      transferOutcomes: [],
    });
  }
  const transferOutcomes: IncrementalRecallWorkPlanTransferOutcome[] = [];
  const readyPhysicalWorkPlans: RecallMarkerReplayWorkPlan[] = [];
  for (const physicalWorkPlan of splitRecallWorkPlanByPhysicalSession(workPlan)) {
    const sourceModifiedAtEpochMilliseconds =
      await readWorkerSourceModifiedAtEpochMilliseconds(physicalWorkPlan);
    const schedule = scheduleRecallWorkPlanEligibility({
      workPlan: physicalWorkPlan,
      sourceModifiedAtEpochMilliseconds,
      preparedDocumentCount: 0,
      nowEpochMilliseconds,
    });
    if (schedule.ready) {
      readyPhysicalWorkPlans.push(physicalWorkPlan);
    } else {
      transferOutcomes.push({
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: schedule.threshold,
        readyAtEpochMilliseconds: schedule.readyAtEpochMilliseconds,
      });
    }
  }
  if (readyPhysicalWorkPlans.length === 0 && !hasRecallMetadataReconciliationWork(metadataSweep)) {
    return finishWorkerResult({
      workPlan,
      metadataSweep,
      heavyDependenciesLoaded: false,
      commitsFrozen: false,
      generationReplayCompleted: null,
      transferOutcomes,
    });
  }
  await (options.loadHeavyDependencies ?? loadRecallIncrementalWorkerDependencies)();
  for (const physicalWorkPlan of readyPhysicalWorkPlans) {
    const outcome = await options.transferWorkPlan?.(physicalWorkPlan);
    if (outcome !== undefined) {
      transferOutcomes.push(outcome);
    }
  }
  if (
    metadataSweep !== null &&
    metadataSweep.status !== RecallMetadataSweepStatus.CONTINUATION_REQUIRED &&
    (knownSourceInventory?.physicalProjections.length ?? 0) > 0
  ) {
    await options.reconcileDeletion?.(
      metadataSweep,
      knownSourceInventory?.physicalProjections ?? [],
    );
  }
  return finishWorkerResult({
    workPlan,
    metadataSweep,
    heavyDependenciesLoaded: true,
    commitsFrozen: false,
    generationReplayCompleted: null,
    transferOutcomes,
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

async function runRecallIncrementalWorkerExecutable(): Promise<void> {
  const config = await loadRecallConversationConfig();
  await recoverRecallGenerationCutover({
    activeGenerationPointerPath: config.activeGenerationPointerPath,
    generationRegistryPath: config.generationRegistryPath,
    backlogSummaryPath: config.backlogSummaryPath,
    lockPath: config.lockPath,
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
  const operationDiagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.incrementalDiagnosticLogPath,
    retainedLogPath: config.incrementalDiagnosticLogPath.replace(/\.jsonl$/u, '.previous.jsonl'),
    notifyWarning(message) {
      process.emitWarning(message);
    },
  });
  const manifest = await readRecallIndexManifest(activeSelection.manifestPath);
  if (manifest === null) {
    throw new Error('Recall incremental worker active generation manifest missing');
  }
  const embeddings = createLocalEmbeddingClient({
    baseUrl: config.embeddingBaseUrl,
    model: config.embeddingModel,
    dimensions: config.embeddingDimensions,
    batchSize: config.embeddingBatchSize,
  });
  const embeddingCache = createEmbeddingVectorCache({
    cacheDirectory: config.embeddingCacheDirectory,
    identity: createEmbeddingVectorCacheIdentity(manifest),
    embeddingRequestBatchSize: config.embeddingBatchSize,
    embeddings,
  });
  const loadTokenizer = () =>
    loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory });
  const resolveWorkerProjectIdentity = createLineageResolver(
    config.projectLineages,
    resolveProjectIdentity,
  );
  await runRecallIncrementalWorker({
    markerSpoolDirectory: config.markerSpoolDirectory,
    markerQuarantineDirectory: config.markerQuarantineDirectory,
    controlDirectory: config.markerControlDirectory,
    targetGenerationId: activeSelection.activeGenerationId,
    generationRegistryPath: config.generationRegistryPath,
    generationReplayCompletion: {
      activeGenerationPointerPath: config.activeGenerationPointerPath,
      generationRegistryPath: config.generationRegistryPath,
      backlogSummaryPath: config.backlogSummaryPath,
      lockPath: config.lockPath,
    },
    ...(registry?.rollbackGenerationId
      ? { retainedMarkerDirectory: resolve(config.markerControlDirectory, 'rollback-retained') }
      : {}),
    trustedSessionRoots: [config.sessionsDirectory],
    confirmedDeletionMaxMissingSourceCount: config.confirmedDeletionMaxMissingSourceCount,
    confirmedDeletionMaxMissingSourceRatio: config.confirmedDeletionMaxMissingSourceRatio,
    operationDiagnostics,
    transferWorkPlan: (workPlan) =>
      transferIncrementalRecallWorkPlan({
        workPlan,
        lockPath: config.lockPath,
        evidenceDatabasePath: activeSelection.databasePath,
        projectionDatabasePath: activeSelection.projectionDatabasePath,
        embeddingDimensions: config.embeddingDimensions,
        chunkPolicy: manifest.chunkPolicy,
        loadTokenizer,
        resolveProjectIdentity: resolveWorkerProjectIdentity,
        embeddingCache,
        nowEpochMilliseconds: Date.now,
      }),
    loadKnownSourceInventory: () =>
      loadRecallKnownSourceInventory(
        activeSelection.projectionDatabasePath,
        activeSelection.activeGenerationId,
        config.sessionsDirectory,
      ),
    async reconcileDeletion(metadataSweep, physicalProjections) {
      const { formatConfirmedSessionDeletionResult, reconcileConfirmedSessionDeletion } =
        await import('./reconcile-confirmed-session-deletion.js');
      const result = await reconcileConfirmedSessionDeletion({
        metadataSweep,
        physicalProjections,
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRootDirectory: config.generationRootDirectory,
        lockPath: config.lockPath,
        embeddingDimensions: config.embeddingDimensions,
      });
      if (result.halted || result.confirmedSourceDeletionCount > 0) {
        process.emitWarning(
          `Recall confirmed deletion: ${formatConfirmedSessionDeletionResult(result)}`,
        );
      }
    },
  });
  await operationDiagnostics.flush();
}

const executedModulePath = process.argv[1];
if (
  executedModulePath !== undefined &&
  resolve(executedModulePath) === fileURLToPath(import.meta.url)
) {
  runRecallIncrementalWorkerExecutable().catch((error: unknown) => {
    process.emitWarning(
      `Recall incremental worker failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
    );
    process.exitCode = 1;
  });
}

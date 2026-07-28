import { access } from 'node:fs/promises';
import { isAbsolute, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { completeRecallGenerationReplay } from './complete-recall-generation-replay.js';
import {
  coordinateRecallMarkerReplay,
  type RecallGenerationReplayCompletionPaths,
  type RecallMarkerReplayWorkPlan,
} from './coordinate-recall-marker-replay.js';
import {
  RecallGenerationCutoverState,
  RecallMetadataSweepStatus,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';
import type { RecallWorkMarkerCodecOptions } from './recall-work-marker.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { recoverRecallGenerationCutover } from './recover-recall-generation-cutover.js';
import {
  RECALL_METADATA_SWEEP_CONTINUATION_FILENAME,
  scanRecallSessionMetadata,
  type KnownRecallSessionMetadataSource,
  type RecallSessionMetadataSweepResult,
} from './scan-recall-session-metadata.js';

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

function hasRecallIncrementalTransferWork(
  workPlan: RecallMarkerReplayWorkPlan,
  metadataSweep: RecallSessionMetadataSweepResult | null,
): boolean {
  return (
    workPlan.workItems.length > 0 ||
    (metadataSweep !== null &&
      (metadataSweep.status !== RecallMetadataSweepStatus.CONTINUATION_REQUIRED ||
        metadataSweep.observedSessionMetadata.length > 0 ||
        metadataSweep.observedKnownSourceIdentities.length > 0 ||
        metadataSweep.missingPhysicalSessionIds.length > 0))
  );
}

/** Coordinates one bounded worker pass and loads tokenizer/zvec only after eligible work exists. */
export async function runRecallIncrementalWorker(
  options: RunRecallIncrementalWorkerOptions,
): Promise<RecallIncrementalWorkerResult> {
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
  if (commitsFrozen) {
    return {
      workPlan,
      metadataSweep: null,
      heavyDependenciesLoaded: false,
      commitsFrozen: true,
      generationReplayCompleted: null,
    };
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
  if (!hasRecallIncrementalTransferWork(workPlan, metadataSweep)) {
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
    return {
      workPlan,
      metadataSweep,
      heavyDependenciesLoaded: false,
      commitsFrozen: false,
      generationReplayCompleted,
    };
  }
  await (options.loadHeavyDependencies ?? loadRecallIncrementalWorkerDependencies)();
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
  return {
    workPlan,
    metadataSweep,
    heavyDependenciesLoaded: true,
    commitsFrozen: false,
    generationReplayCompleted: null,
  };
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

import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallBacklogSummary,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;

/** Filesystem and marker operations retained outside the durable rollback transition. */
export interface RollbackRecallGenerationTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  rollbackRetentionMilliseconds?: number;
  nowEpochMilliseconds?: () => number;
  validateRollbackGeneration(generationId: string): Promise<void>;
  restoreRetainedMarkers(): Promise<number>;
  retainRecoveryRequired(): void;
}

/** Durable rollback outcome plus whether marker replay must be signaled after lock release. */
export interface RollbackRecallGenerationTransitionResult {
  result: {
    activeGenerationId: string;
    rollbackGenerationId: string;
    restoredMarkerCount: number;
  };
  replayRequired: boolean;
}

/** Durable paths and external marker proof required to complete generation replay. */
export interface CompleteRecallGenerationReplayTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  nowEpochMilliseconds?: () => number;
  proveReplayWorkComplete(): Promise<boolean>;
}

/**
 * Swaps the retained rollback generation into the active role using registry-first publication.
 * Directory validation and marker restoration remain external domain operations.
 */
export async function rollbackRecallGenerationTransition(
  options: RollbackRecallGenerationTransitionOptions,
): Promise<RollbackRecallGenerationTransitionResult> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry || !registry.rollbackGenerationId) {
    throw new Error('Recall generation rollback unavailable: no retained rollback generation');
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation rollback found pointer and registry disagreement');
  }
  if (registry.buildingGenerationId !== null) {
    throw new Error('Recall generation rollback unavailable while a replacement builds');
  }
  const rollbackEntry = registry.generations.find(
    ({ generationId }) => generationId === registry.rollbackGenerationId,
  );
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === registry.activeGenerationId,
  );
  if (
    !rollbackEntry ||
    !activeEntry ||
    rollbackEntry.state !== RecallGenerationCutoverState.ROLLBACK
  ) {
    throw new Error('Recall generation rollback registry roles invalid');
  }
  await options.validateRollbackGeneration(rollbackEntry.generationId);
  const rolledBackAt = options.nowEpochMilliseconds?.() ?? Date.now();
  if (
    rollbackEntry.retireAfterEpochMilliseconds !== undefined &&
    rollbackEntry.retireAfterEpochMilliseconds !== null &&
    rollbackEntry.retireAfterEpochMilliseconds <= rolledBackAt
  ) {
    throw new Error('Recall generation rollback unavailable: retention period expired');
  }
  const targetState =
    rollbackEntry.indexManifestVersion === 5
      ? RecallGenerationCutoverState.LEGACY_READ_ONLY
      : RecallGenerationCutoverState.REPLAY_PENDING;
  const activeReplacement: RecallGenerationRegistryEntry = {
    ...rollbackEntry,
    state: targetState,
    stateChangedAtEpochMilliseconds: rolledBackAt,
    retireAfterEpochMilliseconds: null,
  };
  const rollbackReplacement: RecallGenerationRegistryEntry = {
    ...activeEntry,
    state: RecallGenerationCutoverState.ROLLBACK,
    stateChangedAtEpochMilliseconds: rolledBackAt,
    retireAfterEpochMilliseconds:
      rolledBackAt +
      (options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS),
  };
  const targetPointer = createRecallActiveGenerationPointer(rollbackEntry.generationId);
  const nextRegistry = {
    ...registry,
    activeGenerationId: rollbackEntry.generationId,
    rollbackGenerationId: activeEntry.generationId,
    activePointerChecksum: targetPointer.checksum,
    generations: registry.generations.map((entry) => {
      if (entry.generationId === activeReplacement.generationId) {
        return activeReplacement;
      }
      if (entry.generationId === rollbackReplacement.generationId) {
        return rollbackReplacement;
      }
      return entry;
    }),
  };
  const restoredMarkerCount = await options.restoreRetainedMarkers();
  try {
    await writeRecallGenerationRegistry(options.generationRegistryPath, nextRegistry);
    await writeRecallActiveGenerationPointer(options.activeGenerationPointerPath, targetPointer);
    await writeRecallBacklogSummary(options.backlogSummaryPath, {
      version: RECALL_BACKLOG_SUMMARY_VERSION,
      pendingEligibleSessionCount: restoredMarkerCount,
      oldestEligibleMarkerAgeMilliseconds: null,
      activeGenerationId: rollbackEntry.generationId,
      buildingGenerationId: null,
      generationState: targetState,
      activeGenerationAgeMilliseconds: 0,
      rebuildAgeMilliseconds: null,
      lastFailureCategory: null,
      observedAtEpochMilliseconds: rolledBackAt,
    });
  } catch (error) {
    options.retainRecoveryRequired();
    throw error;
  }
  return {
    result: {
      activeGenerationId: rollbackEntry.generationId,
      rollbackGenerationId: activeEntry.generationId,
      restoredMarkerCount,
    },
    replayRequired: targetState === RecallGenerationCutoverState.REPLAY_PENDING,
  };
}

/** Reports whether the persisted active transition state requires detached marker replay. */
export async function recallGenerationTransitionRequiresReplaySignal(
  generationRegistryPath: string,
): Promise<boolean> {
  const registry = await readRecallGenerationRegistry(generationRegistryPath);
  const activeEntry = registry?.generations.find(
    ({ generationId }) => generationId === registry.activeGenerationId,
  );
  return activeEntry?.state === RecallGenerationCutoverState.REPLAY_PENDING;
}

function createActiveRecallGenerationBacklogSummary(
  activeGenerationId: string,
  observedAtEpochMilliseconds: number,
): RecallBacklogSummary {
  return {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 0,
    oldestEligibleMarkerAgeMilliseconds: null,
    activeGenerationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.ACTIVE,
    activeGenerationAgeMilliseconds: 0,
    rebuildAgeMilliseconds: null,
    lastFailureCategory: null,
    observedAtEpochMilliseconds,
  };
}

/**
 * Completes replay after external marker work is empty, publishing registry before backlog state.
 * Already-active generations repair their backlog summary without rechecking marker work.
 */
export async function completeRecallGenerationReplayTransition(
  options: CompleteRecallGenerationReplayTransitionOptions,
): Promise<boolean> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error(
      'Recall generation replay completion requires initialized pointer and registry',
    );
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation replay completion found pointer and registry disagreement');
  }
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  if (!activeEntry) {
    throw new Error('Recall generation replay completion active registry entry missing');
  }
  const completedAt = options.nowEpochMilliseconds?.() ?? Date.now();
  const activeBacklog = createActiveRecallGenerationBacklogSummary(
    pointer.activeGenerationId,
    completedAt,
  );
  if (activeEntry.state === RecallGenerationCutoverState.ACTIVE) {
    await writeRecallBacklogSummary(options.backlogSummaryPath, activeBacklog);
    return true;
  }
  if (activeEntry.state !== RecallGenerationCutoverState.REPLAY_PENDING) {
    return false;
  }
  if (!(await options.proveReplayWorkComplete())) {
    return false;
  }
  const activeReplacement: RecallGenerationRegistryEntry = {
    ...activeEntry,
    state: RecallGenerationCutoverState.ACTIVE,
    stateChangedAtEpochMilliseconds: completedAt,
  };
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...registry,
    generations: registry.generations.map((entry) =>
      entry.generationId === activeReplacement.generationId ? activeReplacement : entry,
    ),
  });
  await writeRecallBacklogSummary(options.backlogSummaryPath, activeBacklog);
  return true;
}

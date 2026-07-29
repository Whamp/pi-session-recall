import { RecallGenerationCutoverState } from './enums.js';
import {
  activateRecallReplacementInRegistry,
  createRecallActiveGenerationPointer,
  createReplayPendingActivationBacklogSummary,
  encodeRecallGenerationRegistry,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallBacklogSummary,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;

/** Validated build evidence required to create one READY replacement snapshot. */
export interface CreateReadyRecallGenerationTransitionOptions {
  registry: RecallGenerationRegistry;
  buildingEntry: RecallGenerationRegistryEntry;
  indexManifestFingerprint: string;
  rebuildMarkerWatermark: readonly string[];
  readyAtEpochMilliseconds: number;
}

/** READY entry and registry retained until the bounded cutover write window. */
export interface CreateReadyRecallGenerationTransitionResult {
  readyEntry: RecallGenerationRegistryEntry;
  readyRegistry: RecallGenerationRegistry;
}

/** Durable READY activation inputs and fault callbacks owned by the rebuild coordinator. */
export interface ActivateReadyRecallGenerationTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  expectedActivePointer: Awaited<ReturnType<typeof readRecallActiveGenerationPointer>>;
  expectedFrozenRegistry: RecallGenerationRegistry;
  readyRegistry: RecallGenerationRegistry;
  readyEntry: RecallGenerationRegistryEntry;
  rollbackRetentionMilliseconds?: number;
  activatedAtEpochMilliseconds: number;
  beforePointerSwap?(): Promise<void>;
  afterPointerSwap?(): Promise<void>;
  throwIfCancelled(): void;
  retainRecoveryRequired(): void;
}

/** Active and prior generation IDs produced by one completed durable pointer cutover. */
export interface ActivateReadyRecallGenerationTransitionResult {
  previousGenerationId: string | null;
  activeGenerationId: string;
}

/** Durable replay backlog publication after a replacement activation leaves the write window. */
export interface PublishRecallGenerationActivationBacklogTransitionOptions {
  backlogSummaryPath: string;
  readyEntry: RecallGenerationRegistryEntry;
  activatedAtEpochMilliseconds: number;
}

/** Durable paths and clock used to select expired validated generations for collection. */
export interface PrepareRetiredRecallGenerationCollectionTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  nowEpochMilliseconds?: () => number;
}

/** Generation IDs durably marked retired before their directories may be deleted. */
export interface PrepareRetiredRecallGenerationCollectionTransitionResult {
  candidateGenerationIds: string[];
}

/** Durable inputs for removing successfully deleted retired entries from the registry. */
export interface CompleteRetiredRecallGenerationCollectionTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  deletedGenerationIds: readonly string[];
}

/** Whether retained rollback markers may be deleted after registry completion. */
export interface CompleteRetiredRecallGenerationCollectionTransitionResult {
  removeRetainedMarkers: boolean;
}

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

/** Creates a READY replacement entry without publishing cutover state. */
export function createReadyRecallGenerationTransition(
  options: CreateReadyRecallGenerationTransitionOptions,
): CreateReadyRecallGenerationTransitionResult {
  if (
    options.registry.buildingGenerationId !== options.buildingEntry.generationId ||
    options.buildingEntry.state !== RecallGenerationCutoverState.BUILDING
  ) {
    throw new Error('Recall generation readiness requires the registered building generation');
  }
  const readyEntry: RecallGenerationRegistryEntry = {
    ...options.buildingEntry,
    state: RecallGenerationCutoverState.READY,
    indexManifestFingerprint: options.indexManifestFingerprint,
    stateChangedAtEpochMilliseconds: options.readyAtEpochMilliseconds,
    rebuildMarkerWatermark: [...options.rebuildMarkerWatermark],
    validatedAtEpochMilliseconds: options.readyAtEpochMilliseconds,
  };
  return {
    readyEntry,
    readyRegistry: {
      ...options.registry,
      generations: options.registry.generations.map((entry) =>
        entry.generationId === readyEntry.generationId ? readyEntry : entry,
      ),
    },
  };
}

/** Publishes READY registry, active pointer, then activated registry inside the cutover window. */
export async function activateReadyRecallGenerationTransition(
  options: ActivateReadyRecallGenerationTransitionOptions,
): Promise<ActivateReadyRecallGenerationTransitionResult> {
  const [currentPointer, currentRegistry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (
    currentPointer?.activeGenerationId !== options.expectedActivePointer?.activeGenerationId ||
    currentPointer?.checksum !== options.expectedActivePointer?.checksum ||
    currentRegistry === null ||
    encodeRecallGenerationRegistry(currentRegistry) !==
      encodeRecallGenerationRegistry(options.expectedFrozenRegistry)
  ) {
    throw new Error('Recall generation state changed before pointer cutover');
  }
  const replacementPointer = createRecallActiveGenerationPointer(options.readyEntry.generationId);
  const previousGenerationId = options.readyRegistry.activeGenerationId;
  const activatedRegistry = activateRecallReplacementInRegistry(
    options.readyRegistry,
    options.readyEntry,
    replacementPointer.checksum,
    options.activatedAtEpochMilliseconds,
    options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS,
  );
  await options.beforePointerSwap?.();
  options.throwIfCancelled();
  try {
    await writeRecallGenerationRegistry(options.generationRegistryPath, options.readyRegistry);
    await writeRecallActiveGenerationPointer(
      options.activeGenerationPointerPath,
      replacementPointer,
    );
    await options.afterPointerSwap?.();
    await writeRecallGenerationRegistry(options.generationRegistryPath, activatedRegistry);
  } catch (error) {
    options.retainRecoveryRequired();
    throw error;
  }
  return {
    previousGenerationId,
    activeGenerationId: options.readyEntry.generationId,
  };
}

/** Publishes replay-pending backlog only after the active pointer cutover completes. */
export async function publishRecallGenerationActivationBacklogTransition(
  options: PublishRecallGenerationActivationBacklogTransitionOptions,
): Promise<void> {
  await writeRecallBacklogSummary(
    options.backlogSummaryPath,
    createReplayPendingActivationBacklogSummary(
      options.readyEntry,
      options.activatedAtEpochMilliseconds,
    ),
  );
}

function isRecallGenerationCollectible(
  entry: RecallGenerationRegistryEntry,
  nowEpochMilliseconds: number,
): boolean {
  return (
    (entry.state === RecallGenerationCutoverState.ROLLBACK ||
      entry.state === RecallGenerationCutoverState.RETIRED) &&
    entry.validatedAtEpochMilliseconds !== undefined &&
    entry.validatedAtEpochMilliseconds !== null &&
    entry.retireAfterEpochMilliseconds !== undefined &&
    entry.retireAfterEpochMilliseconds !== null &&
    entry.retireAfterEpochMilliseconds <= nowEpochMilliseconds
  );
}

/** Marks validated expired rollback material retired before directory deletion begins. */
export async function prepareRetiredRecallGenerationCollectionTransition(
  options: PrepareRetiredRecallGenerationCollectionTransitionOptions,
): Promise<PrepareRetiredRecallGenerationCollectionTransitionResult> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error('Recall generation collection requires initialized pointer and registry');
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation collection found pointer and registry disagreement');
  }
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  if (
    activeEntry?.state !== RecallGenerationCutoverState.ACTIVE ||
    registry.buildingGenerationId !== null
  ) {
    return { candidateGenerationIds: [] };
  }
  const now = options.nowEpochMilliseconds?.() ?? Date.now();
  const candidateGenerationIds = registry.generations
    .filter(
      (entry) =>
        entry.generationId !== pointer.activeGenerationId &&
        isRecallGenerationCollectible(entry, now),
    )
    .map(({ generationId }) => generationId)
    .toSorted();
  if (candidateGenerationIds.length === 0) {
    return { candidateGenerationIds };
  }
  const candidateIds = new Set(candidateGenerationIds);
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...registry,
    rollbackGenerationId:
      registry.rollbackGenerationId !== null && candidateIds.has(registry.rollbackGenerationId)
        ? null
        : registry.rollbackGenerationId,
    generations: registry.generations.map((entry) =>
      candidateIds.has(entry.generationId)
        ? { ...entry, state: RecallGenerationCutoverState.RETIRED }
        : entry,
    ),
  });
  return { candidateGenerationIds };
}

/** Removes only successfully deleted retired entries and reports marker-retention ownership. */
export async function completeRetiredRecallGenerationCollectionTransition(
  options: CompleteRetiredRecallGenerationCollectionTransitionOptions,
): Promise<CompleteRetiredRecallGenerationCollectionTransitionResult> {
  if (options.deletedGenerationIds.length === 0) {
    return { removeRetainedMarkers: false };
  }
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error('Recall generation collection completion requires pointer and registry');
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error(
      'Recall generation collection completion found pointer and registry disagreement',
    );
  }
  const deletedIds = new Set(options.deletedGenerationIds);
  for (const generationId of deletedIds) {
    const entry = registry.generations.find((candidate) => candidate.generationId === generationId);
    if (!entry || entry.state !== RecallGenerationCutoverState.RETIRED) {
      throw new Error(
        `Recall generation collection completion entry is not retired: ${generationId}`,
      );
    }
  }
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...registry,
    generations: registry.generations.filter(({ generationId }) => !deletedIds.has(generationId)),
  });
  return { removeRetainedMarkers: registry.rollbackGenerationId === null };
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

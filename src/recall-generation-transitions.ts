import {
  RecallBacklogFailureCategory,
  RecallGenerationCutoverState,
  RECALL_INDEX_MANIFEST_VERSION,
} from './enums.js';
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
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallBacklogSummary,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import { RECALL_WORK_MARKER_VERSION } from './recall-work-marker.js';

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;

/** Durable crash state and ownership evidence used to recover an interrupted cutover. */
export interface RecoverRecallGenerationStateTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  pointer: Awaited<ReturnType<typeof readRecallActiveGenerationPointer>>;
  registry: RecallGenerationRegistry | null;
  backlog: RecallBacklogSummary | null;
  abandonedBuildingGeneration: boolean;
  recoveredAtEpochMilliseconds: number;
  rollbackRetentionMilliseconds?: number;
  retainRecoveryRequired(): void;
}

/** State publication result plus the active store that may require write-mode attestation. */
export interface RecoverRecallGenerationStateTransitionResult {
  stateChanged: boolean;
  activeGenerationId: string;
}

/** Exact version-5 identity permitted for explicit read-only legacy adoption. */
export interface AdoptLegacyRecallGenerationTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  generationId: string;
  indexManifestFingerprint: string;
  adoptedAtEpochMilliseconds: number;
}

/** Initial registry and pointer identity used to validate build start or resume. */
export interface InspectRecallGenerationBuildStartTransitionOptions {
  registry: RecallGenerationRegistry;
  activePointer: Awaited<ReturnType<typeof readRecallActiveGenerationPointer>>;
  generationId: string;
  resumeExistingGeneration: boolean;
}

/** Existing failed/building entry reused by an explicit resume, when requested. */
export interface InspectRecallGenerationBuildStartTransitionResult {
  resumableEntry?: RecallGenerationRegistryEntry;
}

/** Expected durable state checked immediately before snapshot capture and build freeze. */
export interface AssertRecallGenerationBuildStateUnchangedTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  expectedActivePointer: Awaited<ReturnType<typeof readRecallActiveGenerationPointer>>;
  expectedPersistedRegistry: RecallGenerationRegistry | null;
}

/** Durable inputs for freezing incremental commits behind one BUILDING generation. */
export interface StartRecallGenerationBuildTransitionOptions {
  generationRegistryPath: string;
  backlogSummaryPath: string;
  registry: RecallGenerationRegistry;
  generationId: string;
  resumableEntry?: RecallGenerationRegistryEntry;
  embeddingProfileId?: string;
  rebuildMarkerWatermark: readonly string[];
  startedAtEpochMilliseconds: number;
}

/** BUILDING entry and frozen registry published before model work starts. */
export interface StartRecallGenerationBuildTransitionResult {
  buildingEntry: RecallGenerationRegistryEntry;
  registry: RecallGenerationRegistry;
}

/** Durable state and original error used to fail one replacement build. */
export interface FailRecallGenerationBuildTransitionOptions {
  generationRegistryPath: string;
  backlogSummaryPath: string;
  registry: RecallGenerationRegistry;
  buildingEntry: RecallGenerationRegistryEntry;
  buildFailure: Error;
  pendingMarkerWatermark: readonly string[];
  rebuildStartedAtEpochMilliseconds: number;
  failedAtEpochMilliseconds: number;
}

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

/** Classifies and publishes every supported interrupted generation cutover state. */
export async function recoverRecallGenerationStateTransition(
  options: RecoverRecallGenerationStateTransitionOptions,
): Promise<RecoverRecallGenerationStateTransitionResult> {
  const { pointer, registry, backlog } = options;
  const buildingGenerationEntry = registry?.generations.find(
    ({ generationId }) =>
      generationId === registry.buildingGenerationId && registry.buildingGenerationId !== null,
  );
  const pointerAndRegistrySelectSameGeneration =
    pointer !== null &&
    registry?.activeGenerationId === pointer.activeGenerationId &&
    registry.activePointerChecksum === pointer.checksum;
  if (
    pointerAndRegistrySelectSameGeneration &&
    pointer !== null &&
    registry !== null &&
    buildingGenerationEntry?.state === RecallGenerationCutoverState.BUILDING &&
    ((backlog?.buildingGenerationId === buildingGenerationEntry.generationId &&
      backlog.lastFailureCategory === RecallBacklogFailureCategory.REBUILD_FAILED) ||
      options.abandonedBuildingGeneration)
  ) {
    const activeEntry = registry.generations.find(
      ({ generationId }) => generationId === registry.activeGenerationId,
    );
    if (activeEntry === undefined) {
      options.retainRecoveryRequired();
      throw new Error('Recall failed rebuild recovery active registry entry missing');
    }
    await writeRecallGenerationRegistry(options.generationRegistryPath, {
      ...registry,
      buildingGenerationId: null,
      generations: registry.generations.map((entry) =>
        entry.generationId === buildingGenerationEntry.generationId
          ? {
              ...entry,
              state: RecallGenerationCutoverState.FAILED,
              stateChangedAtEpochMilliseconds: options.recoveredAtEpochMilliseconds,
            }
          : entry,
      ),
    });
    await writeRecallBacklogSummary(options.backlogSummaryPath, {
      version: RECALL_BACKLOG_SUMMARY_VERSION,
      pendingEligibleSessionCount: backlog?.pendingEligibleSessionCount ?? 0,
      oldestEligibleMarkerAgeMilliseconds: backlog?.oldestEligibleMarkerAgeMilliseconds ?? null,
      activeGenerationId: pointer.activeGenerationId,
      buildingGenerationId: null,
      generationState: activeEntry.state,
      activeGenerationAgeMilliseconds: 0,
      rebuildAgeMilliseconds: null,
      lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
      observedAtEpochMilliseconds: options.recoveredAtEpochMilliseconds,
    });
    return { stateChanged: true, activeGenerationId: pointer.activeGenerationId };
  }
  if (
    pointerAndRegistrySelectSameGeneration &&
    pointer !== null &&
    buildingGenerationEntry?.state !== RecallGenerationCutoverState.READY
  ) {
    return { stateChanged: false, activeGenerationId: pointer.activeGenerationId };
  }
  if (
    pointerAndRegistrySelectSameGeneration &&
    pointer !== null &&
    registry !== null &&
    buildingGenerationEntry?.state === RecallGenerationCutoverState.READY &&
    buildingGenerationEntry.validatedAtEpochMilliseconds !== undefined &&
    buildingGenerationEntry.validatedAtEpochMilliseconds !== null
  ) {
    const replacementPointer = createRecallActiveGenerationPointer(
      buildingGenerationEntry.generationId,
    );
    const recoveredRegistry = activateRecallReplacementInRegistry(
      registry,
      buildingGenerationEntry,
      replacementPointer.checksum,
      options.recoveredAtEpochMilliseconds,
      options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS,
    );
    await writeRecallActiveGenerationPointer(
      options.activeGenerationPointerPath,
      replacementPointer,
    );
    await writeRecallGenerationRegistry(options.generationRegistryPath, recoveredRegistry);
    await writeRecallBacklogSummary(
      options.backlogSummaryPath,
      createReplayPendingActivationBacklogSummary(
        buildingGenerationEntry,
        options.recoveredAtEpochMilliseconds,
      ),
    );
    return { stateChanged: true, activeGenerationId: buildingGenerationEntry.generationId };
  }
  const registrySelectedEntry = registry?.generations.find(
    ({ generationId }) => generationId === registry.activeGenerationId,
  );
  const registryFirstCutover =
    registry !== null &&
    registrySelectedEntry !== undefined &&
    ((pointer === null &&
      registrySelectedEntry.state === RecallGenerationCutoverState.LEGACY_READ_ONLY) ||
      (pointer !== null &&
        registry.rollbackGenerationId === pointer.activeGenerationId &&
        (registrySelectedEntry.state === RecallGenerationCutoverState.REPLAY_PENDING ||
          registrySelectedEntry.state === RecallGenerationCutoverState.LEGACY_READ_ONLY)));
  if (registryFirstCutover) {
    const targetPointer = createRecallActiveGenerationPointer(registrySelectedEntry.generationId);
    await writeRecallActiveGenerationPointer(options.activeGenerationPointerPath, targetPointer);
    await writeRecallBacklogSummary(
      options.backlogSummaryPath,
      registrySelectedEntry.state === RecallGenerationCutoverState.REPLAY_PENDING
        ? createReplayPendingActivationBacklogSummary(
            registrySelectedEntry,
            options.recoveredAtEpochMilliseconds,
          )
        : {
            version: RECALL_BACKLOG_SUMMARY_VERSION,
            pendingEligibleSessionCount: 0,
            oldestEligibleMarkerAgeMilliseconds: null,
            activeGenerationId: registrySelectedEntry.generationId,
            buildingGenerationId: null,
            generationState: registrySelectedEntry.state,
            activeGenerationAgeMilliseconds: 0,
            rebuildAgeMilliseconds: null,
            lastFailureCategory: null,
            observedAtEpochMilliseconds: options.recoveredAtEpochMilliseconds,
          },
    );
    return { stateChanged: true, activeGenerationId: registrySelectedEntry.generationId };
  }
  const replacement = registry?.generations.find(
    ({ generationId }) => generationId === pointer?.activeGenerationId,
  );
  if (
    pointer === null ||
    registry === null ||
    registry.buildingGenerationId !== pointer.activeGenerationId ||
    replacement?.state !== RecallGenerationCutoverState.READY ||
    replacement.validatedAtEpochMilliseconds === undefined ||
    replacement.validatedAtEpochMilliseconds === null
  ) {
    options.retainRecoveryRequired();
    throw new Error(
      'Recall generation cutover recovery found an unsupported pointer and registry state',
    );
  }
  const recoveredRegistry = activateRecallReplacementInRegistry(
    registry,
    replacement,
    pointer.checksum,
    options.recoveredAtEpochMilliseconds,
    options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS,
  );
  await writeRecallGenerationRegistry(options.generationRegistryPath, recoveredRegistry);
  await writeRecallBacklogSummary(
    options.backlogSummaryPath,
    createReplayPendingActivationBacklogSummary(replacement, options.recoveredAtEpochMilliseconds),
  );
  return { stateChanged: true, activeGenerationId: replacement.generationId };
}

/** Publishes exact legacy adoption registry, pointer, then read-only backlog state. */
export async function adoptLegacyRecallGenerationTransition(
  options: AdoptLegacyRecallGenerationTransitionOptions,
): Promise<void> {
  const pointer = createRecallActiveGenerationPointer(options.generationId);
  const [existingPointer, existingRegistry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (existingPointer !== null && existingPointer.checksum !== pointer.checksum) {
    throw new Error('Recall legacy adoption journal conflicts with the active pointer');
  }
  if (
    existingRegistry !== null &&
    (existingRegistry.activeGenerationId !== options.generationId ||
      existingRegistry.activePointerChecksum !== pointer.checksum)
  ) {
    throw new Error('Recall legacy adoption journal conflicts with the generation registry');
  }
  if (existingRegistry === null) {
    await writeRecallGenerationRegistry(options.generationRegistryPath, {
      version: RECALL_GENERATION_REGISTRY_VERSION,
      activeGenerationId: options.generationId,
      buildingGenerationId: null,
      rollbackGenerationId: null,
      activePointerChecksum: pointer.checksum,
      generations: [
        {
          generationId: options.generationId,
          state: RecallGenerationCutoverState.LEGACY_READ_ONLY,
          indexManifestVersion: 5,
          markerSchemaVersion: null,
          sessionProjectionSchemaVersion: null,
          indexManifestFingerprint: options.indexManifestFingerprint,
          rebuildStartedAtEpochMilliseconds: options.adoptedAtEpochMilliseconds,
          stateChangedAtEpochMilliseconds: options.adoptedAtEpochMilliseconds,
          rebuildStartMarkerId: null,
          rebuildMarkerWatermark: [],
          validatedAtEpochMilliseconds: options.adoptedAtEpochMilliseconds,
          retireAfterEpochMilliseconds: null,
        },
      ],
    });
  }
  if (existingPointer === null) {
    await writeRecallActiveGenerationPointer(options.activeGenerationPointerPath, pointer);
  }
  await writeRecallBacklogSummary(options.backlogSummaryPath, {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 0,
    oldestEligibleMarkerAgeMilliseconds: null,
    activeGenerationId: options.generationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.LEGACY_READ_ONLY,
    activeGenerationAgeMilliseconds: 0,
    rebuildAgeMilliseconds: 0,
    lastFailureCategory: null,
    observedAtEpochMilliseconds: options.adoptedAtEpochMilliseconds,
  });
}

/** Validates active/building roles before a new build directory is created or resumed. */
export function inspectRecallGenerationBuildStartTransition(
  options: InspectRecallGenerationBuildStartTransitionOptions,
): InspectRecallGenerationBuildStartTransitionResult {
  const resumableEntry = options.resumeExistingGeneration
    ? options.registry.generations.find(({ generationId }) => generationId === options.generationId)
    : undefined;
  if (
    options.resumeExistingGeneration &&
    (!resumableEntry ||
      (resumableEntry.state !== RecallGenerationCutoverState.FAILED &&
        resumableEntry.state !== RecallGenerationCutoverState.BUILDING))
  ) {
    throw new Error(
      `Recall generation resume requires failed or abandoned building state: ${options.generationId}`,
    );
  }
  if (
    options.registry.buildingGenerationId !== null &&
    options.registry.buildingGenerationId !== resumableEntry?.generationId
  ) {
    throw new Error(
      `Recall generation rebuild already in progress: ${options.registry.buildingGenerationId}`,
    );
  }
  if (
    options.registry.activeGenerationId !== (options.activePointer?.activeGenerationId ?? null) ||
    options.registry.activePointerChecksum !== (options.activePointer?.checksum ?? null)
  ) {
    throw new Error('Recall generation registry and active pointer disagree before rebuild');
  }
  return resumableEntry ? { resumableEntry } : {};
}

/** Proves pointer and registry state remain unchanged before external snapshot capture. */
export async function assertRecallGenerationBuildStateUnchangedTransition(
  options: AssertRecallGenerationBuildStateUnchangedTransitionOptions,
): Promise<void> {
  const [currentPointer, currentRegistry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  const registryChanged =
    options.expectedPersistedRegistry === null
      ? currentRegistry !== null
      : currentRegistry === null ||
        encodeRecallGenerationRegistry(currentRegistry) !==
          encodeRecallGenerationRegistry(options.expectedPersistedRegistry);
  if (currentPointer?.checksum !== options.expectedActivePointer?.checksum || registryChanged) {
    throw new Error('Recall generation state changed before rebuild freeze');
  }
}

/** Publishes BUILDING backlog then registry, compensating backlog if registry publication fails. */
export async function startRecallGenerationBuildTransition(
  options: StartRecallGenerationBuildTransitionOptions,
): Promise<StartRecallGenerationBuildTransitionResult> {
  const buildingEntry: RecallGenerationRegistryEntry = {
    ...(options.resumableEntry ?? {
      generationId: options.generationId,
      indexManifestVersion: RECALL_INDEX_MANIFEST_VERSION,
      markerSchemaVersion: RECALL_WORK_MARKER_VERSION,
      sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
      rebuildStartedAtEpochMilliseconds: options.startedAtEpochMilliseconds,
    }),
    state: RecallGenerationCutoverState.BUILDING,
    ...(options.embeddingProfileId ? { embeddingProfileId: options.embeddingProfileId } : {}),
    indexManifestFingerprint: '0'.repeat(64),
    stateChangedAtEpochMilliseconds: options.startedAtEpochMilliseconds,
    rebuildStartMarkerId: options.rebuildMarkerWatermark[0] ?? null,
    rebuildMarkerWatermark: [...options.rebuildMarkerWatermark],
    validatedAtEpochMilliseconds: null,
    retireAfterEpochMilliseconds: null,
  };
  const frozenRegistry: RecallGenerationRegistry = {
    ...options.registry,
    buildingGenerationId: options.generationId,
    generations: options.resumableEntry
      ? options.registry.generations.map((entry) =>
          entry.generationId === buildingEntry.generationId ? buildingEntry : entry,
        )
      : [...options.registry.generations, buildingEntry],
  };
  if (frozenRegistry.activeGenerationId !== null) {
    await writeRecallBacklogSummary(options.backlogSummaryPath, {
      version: RECALL_BACKLOG_SUMMARY_VERSION,
      pendingEligibleSessionCount: options.rebuildMarkerWatermark.length,
      oldestEligibleMarkerAgeMilliseconds: null,
      activeGenerationId: frozenRegistry.activeGenerationId,
      buildingGenerationId: options.generationId,
      generationState: RecallGenerationCutoverState.BUILDING,
      activeGenerationAgeMilliseconds: 0,
      rebuildAgeMilliseconds: 0,
      lastFailureCategory: null,
      observedAtEpochMilliseconds: options.startedAtEpochMilliseconds,
    });
  }
  const [registryWrite] = await Promise.allSettled([
    writeRecallGenerationRegistry(options.generationRegistryPath, frozenRegistry),
  ]);
  if (registryWrite?.status === 'rejected') {
    if (frozenRegistry.activeGenerationId !== null) {
      const activeEntry = frozenRegistry.generations.find(
        ({ generationId }) => generationId === frozenRegistry.activeGenerationId,
      );
      await writeRecallBacklogSummary(options.backlogSummaryPath, {
        version: RECALL_BACKLOG_SUMMARY_VERSION,
        pendingEligibleSessionCount: options.rebuildMarkerWatermark.length,
        oldestEligibleMarkerAgeMilliseconds: null,
        activeGenerationId: frozenRegistry.activeGenerationId,
        buildingGenerationId: null,
        generationState: activeEntry?.state ?? RecallGenerationCutoverState.ACTIVE,
        activeGenerationAgeMilliseconds: 0,
        rebuildAgeMilliseconds: null,
        lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
        observedAtEpochMilliseconds: options.startedAtEpochMilliseconds,
      });
    }
    throw registryWrite.reason;
  }
  return { buildingEntry, registry: frozenRegistry };
}

/** Marks a BUILDING replacement failed while preserving every original publication error. */
export async function failRecallGenerationBuildTransition(
  options: FailRecallGenerationBuildTransitionOptions,
): Promise<Error> {
  let rebuildError = options.buildFailure;
  const failedEntry: RecallGenerationRegistryEntry = {
    ...options.buildingEntry,
    state: RecallGenerationCutoverState.FAILED,
    stateChangedAtEpochMilliseconds: options.failedAtEpochMilliseconds,
    rebuildMarkerWatermark: [...options.pendingMarkerWatermark],
  };
  const [registryWrite] = await Promise.allSettled([
    writeRecallGenerationRegistry(options.generationRegistryPath, {
      ...options.registry,
      buildingGenerationId: null,
      generations: options.registry.generations.map((entry) =>
        entry.generationId === failedEntry.generationId ? failedEntry : entry,
      ),
    }),
  ]);
  if (registryWrite?.status === 'rejected') {
    rebuildError = new AggregateError(
      [rebuildError, registryWrite.reason],
      'Recall replacement build and failure registry update failed',
    );
  }
  if (options.registry.activeGenerationId !== null) {
    try {
      await writeRecallBacklogSummary(options.backlogSummaryPath, {
        version: RECALL_BACKLOG_SUMMARY_VERSION,
        pendingEligibleSessionCount: options.pendingMarkerWatermark.length,
        oldestEligibleMarkerAgeMilliseconds: null,
        activeGenerationId: options.registry.activeGenerationId,
        buildingGenerationId:
          registryWrite?.status === 'rejected' ? options.buildingEntry.generationId : null,
        generationState: RecallGenerationCutoverState.FAILED,
        activeGenerationAgeMilliseconds: 0,
        rebuildAgeMilliseconds: Math.max(
          0,
          options.failedAtEpochMilliseconds - options.rebuildStartedAtEpochMilliseconds,
        ),
        lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
        observedAtEpochMilliseconds: options.failedAtEpochMilliseconds,
      });
    } catch (backlogError) {
      rebuildError = new AggregateError(
        [rebuildError, backlogError],
        'Recall replacement build and failure backlog update failed',
      );
    }
  }
  return rebuildError;
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

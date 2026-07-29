import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import { RecallGenerationCutoverState, RECALL_INDEX_MANIFEST_VERSION } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallActiveGenerationPointer,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import {
  activateReadyRecallGenerationTransition,
  assertRecallGenerationBuildStateUnchangedTransition,
  createReadyRecallGenerationTransition,
  failRecallGenerationBuildTransition,
  inspectRecallGenerationBuildStartTransition,
  publishRecallGenerationActivationBacklogTransition,
  startRecallGenerationBuildTransition,
} from './recall-generation-transitions.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  recallRebuildOwnershipLockPath,
  tryAcquireRecallRebuildOwnershipLock,
} from './recall-rebuild-ownership-lock.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import { RECALL_WORK_MARKER_VERSION } from './recall-work-marker.js';

/** Paths owned exclusively by one replacement generation build. */
export interface RecallGenerationBuildPaths {
  generationId: string;
  generationDirectory: string;
  databasePath: string;
  projectionDatabasePath: string;
  statePath: string;
  manifestPath: string;
}

/** Build result and optional explicit optimization kept open until the coordinator closes it. */
export interface RecallReplacementGenerationBuild<Result> {
  result: Result;
  optimize?: () => Promise<void>;
  close(): Promise<void>;
}

/** Validation evidence required before a replacement generation may become active. */
export interface RecallGenerationValidation {
  indexManifestFingerprint: string;
}

/** Fault-injection stages around the authoritative atomic pointer replacement. */
export enum RecallGenerationCutoverStage {
  BEFORE_POINTER_SWAP = 'before_pointer_swap',
  AFTER_POINTER_SWAP = 'after_pointer_swap',
}

/** Inputs for one side-by-side rebuild with a short pointer-only cutover window. */
export interface RebuildRecallGenerationOptions<Result, BuildSnapshot = undefined> {
  generationRootDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  lockPath: string;
  generationId?: string;
  /** Configured embedding semantics persisted with the replacement generation. */
  embeddingProfileId?: string;
  /** Reopens this generation's durable checkpoint after a stopped or crashed detached build. */
  resumeExistingGeneration?: boolean;
  rollbackRetentionMilliseconds?: number;
  workerSignal: RecallDetachedWorkerSignal;
  signal?: AbortSignal;
  nowEpochMilliseconds?: () => number;
  captureBuildSnapshot?: () => Promise<BuildSnapshot>;
  buildGeneration(
    paths: RecallGenerationBuildPaths,
    buildSnapshot?: BuildSnapshot,
  ): Promise<RecallReplacementGenerationBuild<Result>>;
  validateGeneration(
    paths: RecallGenerationBuildPaths,
    result: Result,
    buildSnapshot?: BuildSnapshot,
  ): Promise<RecallGenerationValidation>;
  onCutoverStage?: (stage: RecallGenerationCutoverStage) => Promise<void>;
}

/** Completed replacement result plus the old and new active generation identities. */
export interface RebuildRecallGenerationResult<Result> {
  result: Result;
  previousGenerationId: string | null;
  activeGenerationId: string;
  replayMarkerWatermark: string[];
}

const MARKER_FILE_PATTERN = /^([A-Za-z0-9_-]+)\.json$/u;

function createReplacementGenerationId(nowEpochMilliseconds: number): string {
  return `generation_${nowEpochMilliseconds}_${randomUUID().replaceAll('-', '')}`;
}

function createGenerationBuildPaths(
  generationRootDirectory: string,
  generationId: string,
): RecallGenerationBuildPaths {
  createRecallActiveGenerationPointer(generationId);
  const generationDirectory = resolve(generationRootDirectory, generationId);
  return {
    generationId,
    generationDirectory,
    databasePath: join(generationDirectory, 'zvec'),
    projectionDatabasePath: join(generationDirectory, 'session-projections'),
    statePath: join(generationDirectory, 'index-state.json'),
    manifestPath: join(generationDirectory, 'index-manifest.json'),
  };
}

async function listPendingRecallMarkerIds(markerSpoolDirectory: string): Promise<string[]> {
  let names: string[];
  try {
    names = await readdir(markerSpoolDirectory);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
  return names
    .map((name) => MARKER_FILE_PATTERN.exec(name)?.[1])
    .filter((markerId) => markerId !== undefined)
    .toSorted();
}

async function calculateFileSha256(path: string): Promise<string> {
  return createHash('sha256')
    .update(await readFile(path))
    .digest('hex');
}

async function detectManifestVersion(manifestPath: string): Promise<5 | 6> {
  const value: unknown = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (typeof value !== 'object' || value === null || !('manifestVersion' in value)) {
    throw new Error('Recall active generation manifest version missing');
  }
  if (value.manifestVersion !== 5 && value.manifestVersion !== RECALL_INDEX_MANIFEST_VERSION) {
    throw new Error(
      `Recall active generation manifest version unsupported: ${String(value.manifestVersion)}`,
    );
  }
  return value.manifestVersion;
}

async function createInitialGenerationRegistry(
  pointer: RecallActiveGenerationPointer | null,
  generationRootDirectory: string,
  nowEpochMilliseconds: number,
): Promise<RecallGenerationRegistry> {
  if (pointer === null) {
    return {
      version: RECALL_GENERATION_REGISTRY_VERSION,
      activeGenerationId: null,
      buildingGenerationId: null,
      rollbackGenerationId: null,
      activePointerChecksum: null,
      generations: [],
    };
  }
  const generationDirectory = await resolveRecallGenerationDirectory(
    generationRootDirectory,
    pointer.activeGenerationId,
  );
  const manifestPath = join(generationDirectory, 'index-manifest.json');
  const manifestVersion = await detectManifestVersion(manifestPath);
  const legacy = manifestVersion === 5;
  return {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: pointer.activeGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: pointer.activeGenerationId,
        state: legacy
          ? RecallGenerationCutoverState.LEGACY_READ_ONLY
          : RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: manifestVersion,
        markerSchemaVersion: legacy ? null : RECALL_WORK_MARKER_VERSION,
        sessionProjectionSchemaVersion: legacy ? null : RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: await calculateFileSha256(manifestPath),
        rebuildStartedAtEpochMilliseconds: nowEpochMilliseconds,
        stateChangedAtEpochMilliseconds: nowEpochMilliseconds,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: nowEpochMilliseconds,
        retireAfterEpochMilliseconds: null,
      },
    ],
  };
}

async function assertRecallCutoverFilesystem(
  generationRootDirectory: string,
  pointerPath: string,
): Promise<void> {
  await Promise.all([
    mkdir(generationRootDirectory, { recursive: true }),
    mkdir(dirname(pointerPath), { recursive: true }),
  ]);
  const [generationRootStats, pointerDirectoryStats] = await Promise.all([
    stat(generationRootDirectory),
    stat(dirname(pointerPath)),
  ]);
  if (generationRootStats.dev !== pointerDirectoryStats.dev) {
    throw new Error('Recall generation cutover requires pointer and generations on one filesystem');
  }
}

function normalizeRecallRebuildError(error: unknown, fallbackMessage: string): Error {
  return error instanceof Error ? error : new Error(fallbackMessage, { cause: error });
}

function throwIfRebuildCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }
  throw signal.reason instanceof Error
    ? signal.reason
    : new Error('Recall generation rebuild cancelled', { cause: signal.reason });
}

/** Builds and validates a replacement beside the active generation, then atomically cuts over. */
async function rebuildRecallGenerationUnderOwnership<Result, BuildSnapshot = undefined>(
  options: RebuildRecallGenerationOptions<Result, BuildSnapshot>,
): Promise<RebuildRecallGenerationResult<Result>> {
  const now = options.nowEpochMilliseconds?.() ?? Date.now();
  if (options.resumeExistingGeneration && options.generationId === undefined) {
    throw new Error('Recall generation resume requires an existing generation ID');
  }
  const generationId = options.generationId ?? createReplacementGenerationId(now);
  const paths = createGenerationBuildPaths(options.generationRootDirectory, generationId);
  await assertRecallCutoverFilesystem(
    options.generationRootDirectory,
    options.activeGenerationPointerPath,
  );
  const startingPointer = await readRecallActiveGenerationPointer(
    options.activeGenerationPointerPath,
  );
  const persistedStartingRegistry = await readRecallGenerationRegistry(
    options.generationRegistryPath,
  );
  let registry =
    persistedStartingRegistry ??
    (await createInitialGenerationRegistry(startingPointer, options.generationRootDirectory, now));
  const { resumableEntry } = inspectRecallGenerationBuildStartTransition({
    registry,
    activePointer: startingPointer,
    generationId,
    resumeExistingGeneration: options.resumeExistingGeneration === true,
  });
  if (options.resumeExistingGeneration) {
    const generationStats = await stat(paths.generationDirectory).catch((error: unknown) => {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return null;
      }
      throw error;
    });
    if (!generationStats?.isDirectory()) {
      throw new Error(`Recall resumable generation directory missing: ${generationId}`);
    }
  } else {
    await mkdir(paths.generationDirectory);
  }
  const frozenBuild = await coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: false,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async () => {
      await assertRecallGenerationBuildStateUnchangedTransition({
        activeGenerationPointerPath: options.activeGenerationPointerPath,
        generationRegistryPath: options.generationRegistryPath,
        expectedActivePointer: startingPointer,
        expectedPersistedRegistry: persistedStartingRegistry,
      });
      const [rebuildMarkerWatermark, buildSnapshot] = await Promise.all([
        listPendingRecallMarkerIds(options.markerSpoolDirectory),
        options.captureBuildSnapshot?.(),
      ]);
      const frozen = await startRecallGenerationBuildTransition({
        generationRegistryPath: options.generationRegistryPath,
        backlogSummaryPath: options.backlogSummaryPath,
        registry,
        generationId,
        ...(resumableEntry ? { resumableEntry } : {}),
        ...(options.embeddingProfileId ? { embeddingProfileId: options.embeddingProfileId } : {}),
        rebuildMarkerWatermark,
        startedAtEpochMilliseconds: now,
      });
      return { ...frozen, buildSnapshot, rebuildMarkerWatermark };
    },
  );
  const { buildingEntry, buildSnapshot, rebuildMarkerWatermark } = frozenBuild;
  registry = frozenBuild.registry;

  async function failRecallReplacementBuild(
    buildFailure: Error,
    pendingMarkerWatermark: readonly string[],
  ): Promise<never> {
    let rebuildError = await failRecallGenerationBuildTransition({
      generationRegistryPath: options.generationRegistryPath,
      backlogSummaryPath: options.backlogSummaryPath,
      registry,
      buildingEntry,
      buildFailure,
      pendingMarkerWatermark,
      rebuildStartedAtEpochMilliseconds: now,
      failedAtEpochMilliseconds: options.nowEpochMilliseconds?.() ?? Date.now(),
    });
    if (registry.activeGenerationId !== null) {
      try {
        options.workerSignal.signalDetachedWorker();
      } catch (signalError) {
        rebuildError = new AggregateError(
          [rebuildError, signalError],
          'Recall replacement build and worker restart failed',
        );
      }
    }
    throw rebuildError;
  }

  let build: RecallReplacementGenerationBuild<Result> | undefined;
  let buildResult: Result | undefined;
  let validation: RecallGenerationValidation | undefined;
  try {
    throwIfRebuildCancelled(options.signal);
    build = await options.buildGeneration(paths, buildSnapshot);
    buildResult = build.result;
    if (build.optimize) {
      await build.optimize();
    }
    throwIfRebuildCancelled(options.signal);
    await build.close();
    build = undefined;
    validation = await options.validateGeneration(paths, buildResult, buildSnapshot);
    await resolveRecallGenerationDirectory(options.generationRootDirectory, generationId);
  } catch (error) {
    let rebuildError = normalizeRecallRebuildError(error, 'Recall replacement build failed');
    if (build) {
      try {
        await build.close();
      } catch (closeError) {
        rebuildError = new AggregateError(
          [rebuildError, closeError],
          'Recall replacement build and close failed',
        );
      }
    }
    return failRecallReplacementBuild(rebuildError, rebuildMarkerWatermark);
  }

  let readyWatermark = rebuildMarkerWatermark;
  let readyEntry: RecallGenerationRegistryEntry;
  try {
    throwIfRebuildCancelled(options.signal);
    const readyAt = options.nowEpochMilliseconds?.() ?? Date.now();
    readyWatermark = [
      ...new Set([
        ...rebuildMarkerWatermark,
        ...(await listPendingRecallMarkerIds(options.markerSpoolDirectory)),
      ]),
    ].toSorted();
    const readiness = createReadyRecallGenerationTransition({
      registry,
      buildingEntry,
      indexManifestFingerprint: validation.indexManifestFingerprint,
      rebuildMarkerWatermark: readyWatermark,
      readyAtEpochMilliseconds: readyAt,
    });
    readyEntry = readiness.readyEntry;
    registry = readiness.readyRegistry;
    throwIfRebuildCancelled(options.signal);
  } catch (error) {
    const readinessError = normalizeRecallRebuildError(
      error,
      'Recall replacement readiness failed',
    );
    return failRecallReplacementBuild(readinessError, readyWatermark);
  }
  const activatedAt = options.nowEpochMilliseconds?.() ?? Date.now();
  let previousGenerationId: string | null = null;
  try {
    await coordinateRecallWriteWindow(
      {
        lockPath: options.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      async (writeWindow) => {
        throwIfRebuildCancelled(options.signal);
        const activation = await activateReadyRecallGenerationTransition({
          activeGenerationPointerPath: options.activeGenerationPointerPath,
          generationRegistryPath: options.generationRegistryPath,
          expectedActivePointer: startingPointer,
          expectedFrozenRegistry: frozenBuild.registry,
          readyRegistry: registry,
          readyEntry,
          ...(options.rollbackRetentionMilliseconds === undefined
            ? {}
            : { rollbackRetentionMilliseconds: options.rollbackRetentionMilliseconds }),
          activatedAtEpochMilliseconds: activatedAt,
          beforePointerSwap: () =>
            options.onCutoverStage?.(RecallGenerationCutoverStage.BEFORE_POINTER_SWAP) ??
            Promise.resolve(),
          afterPointerSwap: () =>
            options.onCutoverStage?.(RecallGenerationCutoverStage.AFTER_POINTER_SWAP) ??
            Promise.resolve(),
          throwIfCancelled: () => throwIfRebuildCancelled(options.signal),
          retainRecoveryRequired: () => writeWindow.retainRecoveryRequired(),
        });
        previousGenerationId = activation.previousGenerationId;
      },
    );
  } catch (error) {
    const currentPointer = await readRecallActiveGenerationPointer(
      options.activeGenerationPointerPath,
    );
    if (currentPointer?.checksum === startingPointer?.checksum) {
      const cutoverError = normalizeRecallRebuildError(error, 'Recall replacement cutover failed');
      return failRecallReplacementBuild(cutoverError, readyWatermark);
    }
    options.workerSignal.signalDetachedWorker();
    throw error;
  }
  const [backlogWrite] = await Promise.allSettled([
    publishRecallGenerationActivationBacklogTransition({
      backlogSummaryPath: options.backlogSummaryPath,
      readyEntry,
      activatedAtEpochMilliseconds: activatedAt,
    }),
  ]);
  options.workerSignal.signalDetachedWorker();
  if (backlogWrite?.status === 'rejected') {
    throw backlogWrite.reason;
  }
  return {
    result: buildResult,
    previousGenerationId,
    activeGenerationId: generationId,
    replayMarkerWatermark: readyWatermark,
  };
}

/** Builds one replacement while holding crash-released ownership across its full lifecycle. */
export async function rebuildRecallGeneration<Result, BuildSnapshot = undefined>(
  options: RebuildRecallGenerationOptions<Result, BuildSnapshot>,
): Promise<RebuildRecallGenerationResult<Result>> {
  const ownershipLock = await tryAcquireRecallRebuildOwnershipLock(
    recallRebuildOwnershipLockPath(options.lockPath),
  );
  if (ownershipLock === null) {
    throw new Error('Recall replacement generation build already in progress');
  }
  try {
    return await rebuildRecallGenerationUnderOwnership(options);
  } finally {
    await ownershipLock.release();
  }
}

import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import { RECALL_INDEX_MANIFEST_VERSION } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  type RecallActiveGenerationPointer,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import {
  activateReadyRecallGenerationTransition,
  assertRecallGenerationBuildStateUnchangedTransition,
  createReadyRecallGenerationTransition,
  failRecallGenerationBuildTransition,
  prepareRecallGenerationBuildStartTransition,
  publishRecallGenerationActivationBacklogTransition,
  startRecallGenerationBuildTransition,
  type RecallGenerationBuildStartActiveGeneration,
} from './recall-generation-transitions.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  recallRebuildOwnershipLockPath,
  tryAcquireRecallRebuildOwnershipLock,
} from './recall-rebuild-ownership-lock.js';

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

async function detectManifestVersion(manifestPath: string): Promise<6> {
  const value: unknown = JSON.parse(await readFile(manifestPath, 'utf8'));
  if (typeof value !== 'object' || value === null || !('manifestVersion' in value)) {
    throw new Error('Recall active generation manifest version missing');
  }
  if (value.manifestVersion !== RECALL_INDEX_MANIFEST_VERSION) {
    throw new Error(
      `Recall active generation manifest version unsupported: ${String(value.manifestVersion)}`,
    );
  }
  return value.manifestVersion;
}

async function inspectActiveGenerationForBuildStart(
  pointer: RecallActiveGenerationPointer | null,
  generationRootDirectory: string,
): Promise<RecallGenerationBuildStartActiveGeneration | undefined> {
  if (pointer === null) {
    return undefined;
  }
  const generationDirectory = await resolveRecallGenerationDirectory(
    generationRootDirectory,
    pointer.activeGenerationId,
  );
  const manifestPath = join(generationDirectory, 'index-manifest.json');
  return {
    indexManifestVersion: await detectManifestVersion(manifestPath),
    indexManifestFingerprint: await calculateFileSha256(manifestPath),
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
  const activeGeneration =
    persistedStartingRegistry === null
      ? await inspectActiveGenerationForBuildStart(startingPointer, options.generationRootDirectory)
      : undefined;
  const preparedBuildStart = prepareRecallGenerationBuildStartTransition({
    registry: persistedStartingRegistry,
    activePointer: startingPointer,
    ...(activeGeneration ? { activeGeneration } : {}),
    generationId,
    resumeExistingGeneration: options.resumeExistingGeneration === true,
    inspectedAtEpochMilliseconds: now,
  });
  let { registry } = preparedBuildStart;
  const { resumableEntry } = preparedBuildStart;
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
  let activation: Awaited<ReturnType<typeof activateReadyRecallGenerationTransition>>;
  try {
    activation = await coordinateRecallWriteWindow(
      {
        lockPath: options.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      async (writeWindow) => {
        throwIfRebuildCancelled(options.signal);
        return activateReadyRecallGenerationTransition({
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
      activation,
    }),
  ]);
  options.workerSignal.signalDetachedWorker();
  if (backlogWrite?.status === 'rejected') {
    throw normalizeRecallRebuildError(
      backlogWrite.reason,
      'Recall generation activation backlog publication failed',
    );
  }
  return {
    result: buildResult,
    previousGenerationId: activation.previousGenerationId,
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

import { createHash, randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import {
  RecallBacklogFailureCategory,
  RecallGenerationCutoverState,
  RECALL_INDEX_MANIFEST_VERSION,
} from './enums.js';
import {
  createRecallActiveGenerationPointer,
  encodeRecallGenerationRegistry,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallActiveGenerationPointer,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import { readNodeErrorCode } from './read-node-error-code.js';
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

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;
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

function replaceGenerationEntry(
  registry: RecallGenerationRegistry,
  replacement: RecallGenerationRegistryEntry,
): RecallGenerationRegistryEntry[] {
  return registry.generations.map((entry) =>
    entry.generationId === replacement.generationId ? replacement : entry,
  );
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
export async function rebuildRecallGeneration<Result, BuildSnapshot = undefined>(
  options: RebuildRecallGenerationOptions<Result, BuildSnapshot>,
): Promise<RebuildRecallGenerationResult<Result>> {
  const now = options.nowEpochMilliseconds?.() ?? Date.now();
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
  if (registry.buildingGenerationId !== null) {
    throw new Error(
      `Recall generation rebuild already in progress: ${registry.buildingGenerationId}`,
    );
  }
  if (
    registry.activeGenerationId !== (startingPointer?.activeGenerationId ?? null) ||
    registry.activePointerChecksum !== (startingPointer?.checksum ?? null)
  ) {
    throw new Error('Recall generation registry and active pointer disagree before rebuild');
  }
  await mkdir(paths.generationDirectory);
  const frozenBuild = await coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: false,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async () => {
      const [currentPointer, currentRegistry] = await Promise.all([
        readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
        readRecallGenerationRegistry(options.generationRegistryPath),
      ]);
      const registryChanged =
        persistedStartingRegistry === null
          ? currentRegistry !== null
          : currentRegistry === null ||
            encodeRecallGenerationRegistry(currentRegistry) !==
              encodeRecallGenerationRegistry(persistedStartingRegistry);
      if (currentPointer?.checksum !== startingPointer?.checksum || registryChanged) {
        throw new Error('Recall generation state changed before rebuild freeze');
      }
      const [rebuildMarkerWatermark, buildSnapshot] = await Promise.all([
        listPendingRecallMarkerIds(options.markerSpoolDirectory),
        options.captureBuildSnapshot?.(),
      ]);
      const buildingEntry: RecallGenerationRegistryEntry = {
        generationId,
        state: RecallGenerationCutoverState.BUILDING,
        indexManifestVersion: RECALL_INDEX_MANIFEST_VERSION,
        markerSchemaVersion: RECALL_WORK_MARKER_VERSION,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: '0'.repeat(64),
        rebuildStartedAtEpochMilliseconds: now,
        stateChangedAtEpochMilliseconds: now,
        rebuildStartMarkerId: rebuildMarkerWatermark[0] ?? null,
        rebuildMarkerWatermark,
        validatedAtEpochMilliseconds: null,
        retireAfterEpochMilliseconds: null,
      };
      const frozenRegistry: RecallGenerationRegistry = {
        ...registry,
        buildingGenerationId: generationId,
        generations: [...registry.generations, buildingEntry],
      };
      if (frozenRegistry.activeGenerationId !== null) {
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: rebuildMarkerWatermark.length,
          oldestEligibleMarkerAgeMilliseconds: null,
          activeGenerationId: frozenRegistry.activeGenerationId,
          buildingGenerationId: generationId,
          generationState: RecallGenerationCutoverState.BUILDING,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: 0,
          lastFailureCategory: null,
          observedAtEpochMilliseconds: now,
        });
      }
      const [freezeRegistryWrite] = await Promise.allSettled([
        writeRecallGenerationRegistry(options.generationRegistryPath, frozenRegistry),
      ]);
      if (freezeRegistryWrite?.status === 'rejected') {
        if (frozenRegistry.activeGenerationId !== null) {
          const activeEntry = frozenRegistry.generations.find(
            ({ generationId: candidateId }) => candidateId === frozenRegistry.activeGenerationId,
          );
          await writeRecallBacklogSummary(options.backlogSummaryPath, {
            version: RECALL_BACKLOG_SUMMARY_VERSION,
            pendingEligibleSessionCount: rebuildMarkerWatermark.length,
            oldestEligibleMarkerAgeMilliseconds: null,
            activeGenerationId: frozenRegistry.activeGenerationId,
            buildingGenerationId: null,
            generationState: activeEntry?.state ?? RecallGenerationCutoverState.ACTIVE,
            activeGenerationAgeMilliseconds: 0,
            rebuildAgeMilliseconds: null,
            lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
            observedAtEpochMilliseconds: now,
          });
        }
        throw freezeRegistryWrite.reason;
      }
      return { buildingEntry, buildSnapshot, rebuildMarkerWatermark, registry: frozenRegistry };
    },
  );
  const { buildingEntry, buildSnapshot, rebuildMarkerWatermark } = frozenBuild;
  registry = frozenBuild.registry;

  async function failRecallReplacementBuild(
    buildFailure: Error,
    pendingMarkerWatermark: readonly string[],
  ): Promise<never> {
    let rebuildError = buildFailure;
    const failedAt = options.nowEpochMilliseconds?.() ?? Date.now();
    const failedEntry: RecallGenerationRegistryEntry = {
      ...buildingEntry,
      state: RecallGenerationCutoverState.FAILED,
      stateChangedAtEpochMilliseconds: failedAt,
      rebuildMarkerWatermark: [...pendingMarkerWatermark],
    };
    const [registryWrite] = await Promise.allSettled([
      writeRecallGenerationRegistry(options.generationRegistryPath, {
        ...registry,
        buildingGenerationId: null,
        generations: replaceGenerationEntry(registry, failedEntry),
      }),
    ]);
    if (registryWrite?.status === 'rejected') {
      rebuildError = new AggregateError(
        [rebuildError, registryWrite.reason],
        'Recall replacement build and failure registry update failed',
      );
    }
    if (registry.activeGenerationId !== null) {
      try {
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: pendingMarkerWatermark.length,
          oldestEligibleMarkerAgeMilliseconds: null,
          activeGenerationId: registry.activeGenerationId,
          buildingGenerationId:
            registryWrite?.status === 'rejected' ? buildingEntry.generationId : null,
          generationState: RecallGenerationCutoverState.FAILED,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: Math.max(0, failedAt - now),
          lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
          observedAtEpochMilliseconds: failedAt,
        });
      } catch (backlogError) {
        rebuildError = new AggregateError(
          [rebuildError, backlogError],
          'Recall replacement build and failure backlog update failed',
        );
      }
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
    readyEntry = {
      ...buildingEntry,
      state: RecallGenerationCutoverState.READY,
      indexManifestFingerprint: validation.indexManifestFingerprint,
      stateChangedAtEpochMilliseconds: readyAt,
      rebuildMarkerWatermark: readyWatermark,
      validatedAtEpochMilliseconds: readyAt,
    };
    registry = {
      ...registry,
      generations: replaceGenerationEntry(registry, readyEntry),
    };
    throwIfRebuildCancelled(options.signal);
  } catch (error) {
    const readinessError = normalizeRecallRebuildError(
      error,
      'Recall replacement readiness failed',
    );
    return failRecallReplacementBuild(readinessError, readyWatermark);
  }
  const newPointer = createRecallActiveGenerationPointer(generationId);
  const activatedAt = options.nowEpochMilliseconds?.() ?? Date.now();
  const retentionMilliseconds =
    options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS;
  const previousGenerationId = registry.activeGenerationId;
  const finalizedEntries = registry.generations.map((entry): RecallGenerationRegistryEntry => {
    if (entry.generationId === generationId) {
      return {
        ...readyEntry,
        state: RecallGenerationCutoverState.REPLAY_PENDING,
        stateChangedAtEpochMilliseconds: activatedAt,
      };
    }
    if (entry.generationId === previousGenerationId) {
      return {
        ...entry,
        state: RecallGenerationCutoverState.ROLLBACK,
        stateChangedAtEpochMilliseconds: activatedAt,
        retireAfterEpochMilliseconds: activatedAt + retentionMilliseconds,
      };
    }
    if (entry.state === RecallGenerationCutoverState.ROLLBACK) {
      return { ...entry, state: RecallGenerationCutoverState.RETIRED };
    }
    return entry;
  });
  const finalRegistry: RecallGenerationRegistry = {
    ...registry,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: previousGenerationId,
    activePointerChecksum: newPointer.checksum,
    generations: finalizedEntries,
  };

  try {
    await coordinateRecallWriteWindow(
      {
        lockPath: options.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      async (writeWindow) => {
        throwIfRebuildCancelled(options.signal);
        const [currentPointer, currentRegistry] = await Promise.all([
          readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
          readRecallGenerationRegistry(options.generationRegistryPath),
        ]);
        if (
          currentPointer?.activeGenerationId !== startingPointer?.activeGenerationId ||
          currentPointer?.checksum !== startingPointer?.checksum ||
          currentRegistry === null ||
          encodeRecallGenerationRegistry(currentRegistry) !==
            encodeRecallGenerationRegistry(frozenBuild.registry)
        ) {
          throw new Error('Recall generation state changed before pointer cutover');
        }
        await options.onCutoverStage?.(RecallGenerationCutoverStage.BEFORE_POINTER_SWAP);
        throwIfRebuildCancelled(options.signal);
        try {
          await writeRecallGenerationRegistry(options.generationRegistryPath, registry);
          await writeRecallActiveGenerationPointer(options.activeGenerationPointerPath, newPointer);
          await options.onCutoverStage?.(RecallGenerationCutoverStage.AFTER_POINTER_SWAP);
          await writeRecallGenerationRegistry(options.generationRegistryPath, finalRegistry);
        } catch (error) {
          writeWindow.retainRecoveryRequired();
          throw error;
        }
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
    writeRecallBacklogSummary(options.backlogSummaryPath, {
      version: RECALL_BACKLOG_SUMMARY_VERSION,
      pendingEligibleSessionCount: readyWatermark.length,
      oldestEligibleMarkerAgeMilliseconds: null,
      activeGenerationId: generationId,
      buildingGenerationId: null,
      generationState: RecallGenerationCutoverState.REPLAY_PENDING,
      activeGenerationAgeMilliseconds: 0,
      rebuildAgeMilliseconds: Math.max(0, activatedAt - now),
      lastFailureCategory: null,
      observedAtEpochMilliseconds: activatedAt,
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

import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  RecallBackgroundIndexProcessState,
  RecallDiagnosticsMode,
  RecallFixedSnapshotBuildOperationPhase,
} from './enums.js';
import type {
  ConversationIndexCheckpoint,
  ConversationIndexProgress,
} from './incremental-session-indexer.js';
import { listRecallConversationSessionFiles } from './recall-conversation-corpus.js';
import { listPendingRecallMarkerIds } from './recall-generation-replay-markers.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import {
  assertRecallGenerationBuildStateUnchangedTransition,
  failRecallGenerationBuildTransition,
  prepareRecallGenerationBuildStartTransition,
  publishReadyRecallGenerationBuildTransition,
  startRecallGenerationBuildTransition,
} from './recall-generation-transitions.js';
import { readRecallInferenceConfiguration } from './recall-inference-configuration.js';
import {
  recallRebuildOwnershipLockPath,
  tryAcquireRecallRebuildOwnershipLock,
} from './recall-rebuild-ownership-lock.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const RECALL_BACKGROUND_INDEX_STATUS_VERSION = 1;
const MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH = 4096;
const RECALL_BACKGROUND_OPERATION_STALL_MILLISECONDS = 30_000;
const BACKGROUND_BUILD_ACTIVE_OPERATION_SCHEMA = Type.Object(
  {
    phase: Type.Enum(RecallFixedSnapshotBuildOperationPhase),
    startedAt: Type.String({ format: 'date-time' }),
    physicalSourceIdentity: Type.Optional(
      Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
    ),
    sessionsRootRelativePath: Type.Optional(
      Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
    ),
    sourceNumber: Type.Optional(Type.Integer({ minimum: 1 })),
    totalPhysicalSourceCount: Type.Optional(Type.Integer({ minimum: 0 })),
    batchStartIndex: Type.Optional(Type.Integer({ minimum: 0 })),
    batchRecordCount: Type.Optional(Type.Integer({ minimum: 0 })),
    totalRecordCount: Type.Optional(Type.Integer({ minimum: 0 })),
  },
  { additionalProperties: false },
);

const BACKGROUND_BUILD_COMPLETED_OPERATION_SCHEMA = Type.Object(
  {
    ...BACKGROUND_BUILD_ACTIVE_OPERATION_SCHEMA.properties,
    completedAt: Type.String({ format: 'date-time' }),
    durationMilliseconds: Type.Number({ minimum: 0 }),
  },
  { additionalProperties: false },
);

const ACTIVE_BACKGROUND_INDEX_PROCESS_STATES = new Set<RecallBackgroundIndexProcessState>([
  RecallBackgroundIndexProcessState.STARTING,
  RecallBackgroundIndexProcessState.RUNNING,
  RecallBackgroundIndexProcessState.STOPPING,
]);

const BACKGROUND_INDEX_STATUS_SCHEMA = Type.Object(
  {
    version: Type.Literal(RECALL_BACKGROUND_INDEX_STATUS_VERSION),
    buildId: Type.String({ minLength: 1 }),
    generationId: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    embeddingProfileId: Type.String({ minLength: 1 }),
    processId: Type.Integer({ minimum: 1 }),
    processState: Type.Enum(RecallBackgroundIndexProcessState),
    startedAt: Type.String({ format: 'date-time' }),
    updatedAt: Type.String({ format: 'date-time' }),
    heartbeatAt: Type.Optional(Type.String({ format: 'date-time' })),
    cpuProfileLogPath: Type.Optional(Type.String({ minLength: 1 })),
    operationLogPath: Type.Optional(Type.String({ minLength: 1 })),
    completedAt: Type.Union([Type.String({ format: 'date-time' }), Type.Null()]),
    progress: Type.Union([
      Type.Object(
        {
          scannedSessions: Type.Integer({ minimum: 0 }),
          totalSessions: Type.Integer({ minimum: 0 }),
          sessionPath: Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
        },
        { additionalProperties: false },
      ),
      Type.Null(),
    ]),
    activeOperation: Type.Optional(
      Type.Union([BACKGROUND_BUILD_ACTIVE_OPERATION_SCHEMA, Type.Null()]),
    ),
    latestCompletedOperation: Type.Optional(
      Type.Union([BACKGROUND_BUILD_COMPLETED_OPERATION_SCHEMA, Type.Null()]),
    ),
    stallDiagnostic: Type.Optional(
      Type.Union([
        Type.Object(
          {
            detectedAt: Type.String({ format: 'date-time' }),
            phase: Type.Enum(RecallFixedSnapshotBuildOperationPhase),
            operationElapsedMilliseconds: Type.Number({ minimum: 0 }),
            heartbeatLagMilliseconds: Type.Number({ minimum: 0 }),
          },
          { additionalProperties: false },
        ),
        Type.Null(),
      ]),
    ),
    latestCheckpoint: Type.Union([
      Type.Object(
        {
          checkpointedSessions: Type.Integer({ minimum: 0 }),
          totalSessions: Type.Integer({ minimum: 0 }),
          sessionPath: Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
          physicalSourceIdentity: Type.Optional(
            Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
          ),
        },
        { additionalProperties: false },
      ),
      Type.Null(),
    ]),
    latestActionableError: Type.Union([
      Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
      Type.Null(),
    ]),
  },
  { additionalProperties: false },
);

const BACKGROUND_INDEX_SERVICE_CONFIG_SCHEMA = Type.Object(
  {
    sessionsDirectory: Type.String({ minLength: 1 }),
    dataDirectory: Type.String({ minLength: 1 }),
    databasePath: Type.String({ minLength: 1 }),
    projectionDatabasePath: Type.String({ minLength: 1 }),
    statePath: Type.String({ minLength: 1 }),
    manifestPath: Type.String({ minLength: 1 }),
    tokenizerCacheDirectory: Type.String({ minLength: 1 }),
    lockPath: Type.String({ minLength: 1 }),
    markerSpoolDirectory: Type.String({ minLength: 1 }),
    markerQuarantineDirectory: Type.String({ minLength: 1 }),
    markerControlDirectory: Type.String({ minLength: 1 }),
    workerOwnershipLockPath: Type.String({ minLength: 1 }),
    generationRootDirectory: Type.String({ minLength: 1 }),
    activeGenerationPointerPath: Type.String({ minLength: 1 }),
    generationRegistryPath: Type.String({ minLength: 1 }),
    backlogSummaryPath: Type.String({ minLength: 1 }),
    incrementalDiagnosticLogPath: Type.String({ minLength: 1 }),
    backgroundIndexStatusPath: Type.Optional(Type.String({ minLength: 1 })),
    backgroundIndexRequestPath: Type.Optional(Type.String({ minLength: 1 })),
    diagnosticsMode: Type.Enum(RecallDiagnosticsMode),
    diagnosticLogPath: Type.String({ minLength: 1 }),
    retainedDiagnosticLogPath: Type.String({ minLength: 1 }),
    embeddingBaseUrl: Type.String({ minLength: 1 }),
    embeddingModel: Type.String({ minLength: 1 }),
    embeddingServedModelId: Type.String({ minLength: 1 }),
    embeddingArtifact: Type.String({ minLength: 1 }),
    embeddingQuantization: Type.String({ minLength: 1 }),
    embeddingPooling: Type.String({ minLength: 1 }),
    embeddingDimensions: Type.Integer({ minimum: 1 }),
    embeddingBatchSize: Type.Integer({ minimum: 1 }),
    rerankerBaseUrl: Type.String({ minLength: 1 }),
    rerankerModel: Type.String({ minLength: 1 }),
    queryPlannerBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    searchWriteWindowWaitMilliseconds: Type.Integer({ minimum: 1 }),
    confirmedDeletionMaxMissingSourceCount: Type.Integer({ minimum: 1 }),
    confirmedDeletionMaxMissingSourceRatio: Type.Number({ exclusiveMinimum: 0, maximum: 1 }),
    projectLineages: Type.Record(Type.String(), Type.Array(Type.String())),
    searchCandidateLimits: Type.Object(
      {
        dense: Type.Integer({ minimum: 1 }),
        lexical: Type.Integer({ minimum: 1 }),
        identifier: Type.Integer({ minimum: 1 }),
      },
      { additionalProperties: false },
    ),
    chunkPolicy: Type.Optional(
      Type.Object(
        {
          maxTokens: Type.Integer({ minimum: 1 }),
          overlapTokens: Type.Integer({ minimum: 0 }),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);

const BACKGROUND_INDEX_WORKER_REQUEST_SCHEMA = Type.Object(
  {
    version: Type.Literal(1),
    buildId: Type.String({ minLength: 1 }),
    statusPath: Type.String({ minLength: 1 }),
    generationId: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    resumeExistingGeneration: Type.Boolean(),
    serviceConfig: BACKGROUND_INDEX_SERVICE_CONFIG_SCHEMA,
    serviceFactory: Type.Object(
      {
        moduleUrl: Type.String({ minLength: 1 }),
        exportName: Type.String({ minLength: 1 }),
      },
      { additionalProperties: false },
    ),
  },
  { additionalProperties: false },
);

/** Operation currently occupying a detached fixed-snapshot worker. */
export interface RecallBackgroundIndexActiveOperation {
  phase: RecallFixedSnapshotBuildOperationPhase;
  startedAt: string;
  physicalSourceIdentity?: string;
  sessionsRootRelativePath?: string;
  sourceNumber?: number;
  totalPhysicalSourceCount?: number;
  batchStartIndex?: number;
  batchRecordCount?: number;
  totalRecordCount?: number;
}

/** Latest fixed-snapshot operation completed by a detached worker. */
export interface RecallBackgroundIndexCompletedOperation extends RecallBackgroundIndexActiveOperation {
  completedAt: string;
  durationMilliseconds: number;
}

/** Watchdog evidence that one named operation and the worker heartbeat stopped advancing. */
export interface RecallBackgroundIndexStallDiagnostic {
  detectedAt: string;
  phase: RecallFixedSnapshotBuildOperationPhase;
  operationElapsedMilliseconds: number;
  heartbeatLagMilliseconds: number;
}

/** One bounded status record for a detached staging index build. */
export interface RecallBackgroundIndexGenerationStatus {
  version: 1;
  buildId: string;
  generationId: string | null;
  embeddingProfileId: string;
  processId: number;
  processState: RecallBackgroundIndexProcessState;
  startedAt: string;
  updatedAt: string;
  heartbeatAt?: string;
  cpuProfileLogPath?: string;
  operationLogPath?: string;
  completedAt: string | null;
  progress: ConversationIndexProgress | null;
  activeOperation?: RecallBackgroundIndexActiveOperation | null;
  latestCompletedOperation?: RecallBackgroundIndexCompletedOperation | null;
  stallDiagnostic?: RecallBackgroundIndexStallDiagnostic | null;
  latestCheckpoint:
    | (ConversationIndexCheckpoint & {
        physicalSourceIdentity?: string;
      })
    | null;
  latestActionableError: string | null;
}

/** Import location for reconstructing configured inference inside the detached worker. */
export interface RecallBackgroundIndexServiceFactory {
  moduleUrl: string;
  exportName: string;
}

/** Serialized input read once by a detached background index worker. */
export interface RecallBackgroundIndexWorkerRequest {
  version: 1;
  buildId: string;
  statusPath: string;
  generationId: string | null;
  resumeExistingGeneration: boolean;
  serviceConfig: RecallConversationConfig;
  serviceFactory: RecallBackgroundIndexServiceFactory;
}

/** Minimal #59 generation registry projection required by detached-build control. */
export interface RecallBackgroundIndexGenerationControl {
  readIndexGenerationStatus(): Promise<{
    active: { generationId: string } | null;
    staging: {
      generationId: string;
      embeddingProfileId: string;
      status: 'building' | 'ready' | 'resumable';
    } | null;
  }>;
}

/** Paths and semantic identity needed to control one background replacement build. */
export interface RecallBackgroundIndexCoordinatorConfig {
  serviceConfig: RecallConversationConfig;
  generationService: RecallBackgroundIndexGenerationControl;
  statusPath: string;
  requestPath: string;
  embeddingProfileId: string;
  serviceFactory: RecallBackgroundIndexServiceFactory;
}

/** Target build callback inputs frozen before the detached worker begins model work. */
export interface RecallReplacementGenerationBuildInput {
  generationId: string;
  physicalSessionPaths: readonly string[];
  resumeExistingGeneration: boolean;
  signal?: AbortSignal;
}

/** Validated target result returned before its READY registry role is published. */
export interface RecallReplacementGenerationBuildResult<Result> {
  result: Result;
  indexManifestFingerprint: string;
}

/** Configured target builder invoked under crash-released replacement ownership. */
export interface RunRecallReplacementGenerationBuildOptions<Result> {
  config: RecallBackgroundIndexCoordinatorConfig;
  generationId: string;
  resumeExistingGeneration: boolean;
  signal?: AbortSignal;
  buildGeneration(
    input: RecallReplacementGenerationBuildInput,
  ): Promise<RecallReplacementGenerationBuildResult<Result>>;
}

function truncateBackgroundStatusText(value: string): string {
  return value.slice(0, MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH);
}

/** Prefers a pending embedding replacement identity for staging start/resume gates. */
export async function resolveBackgroundIndexEmbeddingProfileId(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<string> {
  try {
    const configuration = await readRecallInferenceConfiguration(
      join(dirname(config.serviceConfig.manifestPath), 'inference-configuration.json'),
      {
        generationRegistryPath: config.serviceConfig.generationRegistryPath,
      },
    );
    const pendingProfileId = configuration.pendingEmbeddingReplacement?.embeddingProfileId;
    if (pendingProfileId) {
      return pendingProfileId;
    }
  } catch (error) {
    if (readNodeErrorCode(error) !== 'ENOENT') {
      throw error;
    }
  }
  return config.embeddingProfileId;
}

function isProcessAlive(processId: number): boolean {
  try {
    process.kill(processId, 0);
    return true;
  } catch (error) {
    return readNodeErrorCode(error) === 'EPERM';
  }
}

async function writeAtomicJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(value)}\n`, 'utf8');
  await rename(temporaryPath, path);
}

/** Derives one build-scoped worker request path from the configured request base path. */
export function recallBackgroundIndexWorkerRequestPath(
  baseRequestPath: string,
  buildId: string,
): string {
  return `${baseRequestPath}.${buildId}`;
}

/** Reads one validated background worker request written by the conversation service. */
export async function readRecallBackgroundIndexWorkerRequest(
  requestPath: string,
): Promise<RecallBackgroundIndexWorkerRequest> {
  try {
    const parsed: unknown = JSON.parse(await readFile(requestPath, 'utf8'));
    const request = Value.Parse(BACKGROUND_INDEX_WORKER_REQUEST_SCHEMA, parsed);
    return {
      ...request,
      serviceConfig: {
        ...request.serviceConfig,
        projectLineages: normalizeRecallProjectLineages(request.serviceConfig.projectLineages),
      },
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall background index worker request invalid at ${requestPath}: ${message}`,
      {
        cause: error,
      },
    );
  }
}

/** Reads one validated bounded background build status record. */
export async function readRecallBackgroundIndexStatusRecord(
  statusPath: string,
): Promise<RecallBackgroundIndexGenerationStatus | null> {
  try {
    const parsed: unknown = JSON.parse(await readFile(statusPath, 'utf8'));
    return Value.Parse(BACKGROUND_INDEX_STATUS_SCHEMA, parsed);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall background index status invalid at ${statusPath}: ${message}`, {
      cause: error,
    });
  }
}

async function writeRecallBackgroundIndexStatusRecord(
  statusPath: string,
  status: RecallBackgroundIndexGenerationStatus,
): Promise<void> {
  const validated = Value.Parse(BACKGROUND_INDEX_STATUS_SCHEMA, status);
  await writeAtomicJson(statusPath, validated);
}

async function acquireBackgroundIndexControlLock(statusPath: string): Promise<() => Promise<void>> {
  const lockPath = `${statusPath}.control-lock`;
  while (true) {
    const ownership = await tryAcquireRecallRebuildOwnershipLock(lockPath);
    if (ownership) {
      return () => ownership.release();
    }
    await sleep(25);
  }
}

/** Serializes one worker status transition with concurrent status and control operations. */
export async function updateRecallBackgroundIndexStatusRecord(
  statusPath: string,
  buildId: string,
  processId: number,
  update: (current: RecallBackgroundIndexGenerationStatus) => RecallBackgroundIndexGenerationStatus,
): Promise<void> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(statusPath);
  try {
    const current = await readRecallBackgroundIndexStatusRecord(statusPath);
    if (!current || current.buildId !== buildId || current.processId !== processId) {
      return;
    }
    await writeRecallBackgroundIndexStatusRecord(statusPath, update(current));
  } finally {
    await releaseControlLock();
  }
}

async function refreshBackgroundIndexStatus(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<RecallBackgroundIndexGenerationStatus | null> {
  const status = await readRecallBackgroundIndexStatusRecord(config.statusPath);
  if (!status) {
    return null;
  }
  const generationStatus = await config.generationService.readIndexGenerationStatus();
  const selectedGenerationId =
    generationStatus.staging?.generationId ??
    (status.processState === RecallBackgroundIndexProcessState.SUCCEEDED
      ? generationStatus.active?.generationId
      : null) ??
    status.generationId;
  let refreshed =
    selectedGenerationId !== status.generationId
      ? { ...status, generationId: selectedGenerationId, updatedAt: new Date().toISOString() }
      : status;
  const observedAtEpochMilliseconds = Date.now();
  const activeOperation = refreshed.activeOperation;
  const heartbeatAt = refreshed.heartbeatAt ?? refreshed.updatedAt;
  const operationElapsedMilliseconds =
    activeOperation === null || activeOperation === undefined
      ? 0
      : Math.max(0, observedAtEpochMilliseconds - Date.parse(activeOperation.startedAt));
  const heartbeatLagMilliseconds = Math.max(
    0,
    observedAtEpochMilliseconds - Date.parse(heartbeatAt),
  );
  const stallDiagnostic =
    activeOperation !== null &&
    activeOperation !== undefined &&
    operationElapsedMilliseconds >= RECALL_BACKGROUND_OPERATION_STALL_MILLISECONDS &&
    heartbeatLagMilliseconds >= RECALL_BACKGROUND_OPERATION_STALL_MILLISECONDS
      ? {
          detectedAt: new Date(observedAtEpochMilliseconds).toISOString(),
          phase: activeOperation.phase,
          operationElapsedMilliseconds,
          heartbeatLagMilliseconds,
        }
      : null;
  if (JSON.stringify(refreshed.stallDiagnostic ?? null) !== JSON.stringify(stallDiagnostic)) {
    refreshed = { ...refreshed, stallDiagnostic };
  }
  if (
    ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(refreshed.processState) &&
    !isProcessAlive(refreshed.processId)
  ) {
    const completedAt = new Date().toISOString();
    refreshed = {
      ...refreshed,
      generationId: generationStatus.staging?.generationId ?? refreshed.generationId,
      processState: RecallBackgroundIndexProcessState.CRASHED,
      updatedAt: completedAt,
      completedAt,
      latestActionableError: truncateBackgroundStatusText(
        `Recall background index worker ${refreshed.processId} exited without a completion record; resume the staging generation`,
      ),
    };
  }
  if (refreshed !== status) {
    await writeRecallBackgroundIndexStatusRecord(config.statusPath, refreshed);
  }
  return refreshed;
}

function throwIfReplacementGenerationBuildCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }
  throw signal.reason instanceof Error
    ? signal.reason
    : new Error('Recall replacement generation build cancelled', { cause: signal.reason });
}

/** Builds and validates one inactive target generation without activating it. */
export async function runRecallReplacementGenerationBuild<Result>(
  options: RunRecallReplacementGenerationBuildOptions<Result>,
): Promise<Result> {
  const ownership = await tryAcquireRecallRebuildOwnershipLock(
    recallRebuildOwnershipLockPath(options.config.serviceConfig.lockPath),
  );
  if (ownership === null) {
    throw new Error('Recall replacement generation build already in progress');
  }
  const startedAtEpochMilliseconds = Date.now();
  let frozenBuild: Awaited<ReturnType<typeof startRecallGenerationBuildTransition>> | undefined;
  try {
    throwIfReplacementGenerationBuildCancelled(options.signal);
    const [activePointer, persistedRegistry] = await Promise.all([
      readRecallActiveGenerationPointer(options.config.serviceConfig.activeGenerationPointerPath),
      readRecallGenerationRegistry(options.config.serviceConfig.generationRegistryPath),
    ]);
    const prepared = prepareRecallGenerationBuildStartTransition({
      registry: persistedRegistry,
      activePointer,
      generationId: options.generationId,
      resumeExistingGeneration: options.resumeExistingGeneration,
      inspectedAtEpochMilliseconds: startedAtEpochMilliseconds,
    });
    const physicalSessionPaths = options.resumeExistingGeneration
      ? []
      : await listRecallConversationSessionFiles(options.config.serviceConfig.sessionsDirectory);
    const rebuildMarkerWatermark = await listPendingRecallMarkerIds(
      options.config.serviceConfig.markerSpoolDirectory,
    );
    frozenBuild = await coordinateRecallWriteWindow(
      {
        lockPath: options.config.serviceConfig.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      async () => {
        await assertRecallGenerationBuildStateUnchangedTransition({
          activeGenerationPointerPath: options.config.serviceConfig.activeGenerationPointerPath,
          generationRegistryPath: options.config.serviceConfig.generationRegistryPath,
          expectedActivePointer: activePointer,
          expectedPersistedRegistry: persistedRegistry,
        });
        return startRecallGenerationBuildTransition({
          generationRegistryPath: options.config.serviceConfig.generationRegistryPath,
          backlogSummaryPath: options.config.serviceConfig.backlogSummaryPath,
          registry: prepared.registry,
          generationId: options.generationId,
          ...(prepared.resumableEntry ? { resumableEntry: prepared.resumableEntry } : {}),
          embeddingProfileId: options.config.embeddingProfileId,
          rebuildMarkerWatermark,
          startedAtEpochMilliseconds,
        });
      },
    );
    if (frozenBuild === undefined) {
      throw new Error('Recall replacement generation BUILDING publication missing');
    }
    const publishedBuild = frozenBuild;
    throwIfReplacementGenerationBuildCancelled(options.signal);
    const built = await options.buildGeneration({
      generationId: options.generationId,
      physicalSessionPaths,
      resumeExistingGeneration: options.resumeExistingGeneration,
      ...(options.signal ? { signal: options.signal } : {}),
    });
    throwIfReplacementGenerationBuildCancelled(options.signal);
    const readyMarkerWatermark = [
      ...new Set([
        ...rebuildMarkerWatermark,
        ...(await listPendingRecallMarkerIds(options.config.serviceConfig.markerSpoolDirectory)),
      ]),
    ].toSorted();
    await coordinateRecallWriteWindow(
      {
        lockPath: options.config.serviceConfig.lockPath,
        allowRecovery: false,
        ...(options.signal ? { signal: options.signal } : {}),
      },
      () =>
        publishReadyRecallGenerationBuildTransition({
          activeGenerationPointerPath: options.config.serviceConfig.activeGenerationPointerPath,
          generationRegistryPath: options.config.serviceConfig.generationRegistryPath,
          backlogSummaryPath: options.config.serviceConfig.backlogSummaryPath,
          expectedActivePointer: activePointer,
          frozenRegistry: publishedBuild.registry,
          buildingEntry: publishedBuild.buildingEntry,
          indexManifestFingerprint: built.indexManifestFingerprint,
          rebuildMarkerWatermark: readyMarkerWatermark,
          readyAtEpochMilliseconds: Date.now(),
        }),
    );
    return built.result;
  } catch (error) {
    const buildFailure =
      error instanceof Error ? error : new Error('Recall replacement build failed');
    if (frozenBuild !== undefined) {
      const failure = await failRecallGenerationBuildTransition({
        generationRegistryPath: options.config.serviceConfig.generationRegistryPath,
        backlogSummaryPath: options.config.serviceConfig.backlogSummaryPath,
        registry: frozenBuild.registry,
        buildingEntry: frozenBuild.buildingEntry,
        buildFailure,
        pendingMarkerWatermark: await listPendingRecallMarkerIds(
          options.config.serviceConfig.markerSpoolDirectory,
        ),
        rebuildStartedAtEpochMilliseconds: startedAtEpochMilliseconds,
        failedAtEpochMilliseconds: Date.now(),
      });
      throw failure;
    }
    throw buildFailure;
  } finally {
    await ownership.release();
  }
}

async function spawnBackgroundIndexWorker(
  config: RecallBackgroundIndexCoordinatorConfig,
  generationId: string,
  resumeExistingGeneration: boolean,
): Promise<RecallBackgroundIndexGenerationStatus> {
  const embeddingProfileId = await resolveBackgroundIndexEmbeddingProfileId(config);
  const buildId = randomUUID();
  const requestPath = recallBackgroundIndexWorkerRequestPath(config.requestPath, buildId);
  const request = {
    version: 1,
    buildId,
    statusPath: config.statusPath,
    generationId,
    resumeExistingGeneration,
    serviceConfig: {
      ...config.serviceConfig,
      projectLineages: Object.fromEntries(config.serviceConfig.projectLineages),
    },
    serviceFactory: config.serviceFactory,
  };
  await writeAtomicJson(requestPath, request);

  const workerPath = fileURLToPath(new URL('./recall-background-index-worker.ts', import.meta.url));
  const cpuProfileLogPath = `${config.statusPath}.${buildId}.v8.log`;
  const operationLogPath = `${config.statusPath}.${buildId}.operations.jsonl`;
  const child = spawn(
    process.execPath,
    [
      '--prof',
      '--no-logfile-per-isolate',
      `--logfile=${cpuProfileLogPath}`,
      '--import',
      'tsx',
      workerPath,
      requestPath,
    ],
    {
      cwd: dirname(workerPath),
      detached: true,
      stdio: 'ignore',
    },
  );
  await new Promise<void>((resolve, reject) => {
    child.once('spawn', resolve);
    child.once('error', reject);
  });
  if (!child.pid) {
    throw new Error('Recall background index worker failed to report a process ID');
  }
  const startedAt = new Date().toISOString();
  const status: RecallBackgroundIndexGenerationStatus = {
    version: RECALL_BACKGROUND_INDEX_STATUS_VERSION,
    buildId,
    generationId,
    embeddingProfileId,
    processId: child.pid,
    processState: RecallBackgroundIndexProcessState.STARTING,
    startedAt,
    updatedAt: startedAt,
    heartbeatAt: startedAt,
    cpuProfileLogPath,
    operationLogPath,
    completedAt: null,
    progress: null,
    activeOperation: null,
    latestCompletedOperation: null,
    stallDiagnostic: null,
    latestCheckpoint: null,
    latestActionableError: null,
  };
  await writeRecallBackgroundIndexStatusRecord(config.statusPath, status);
  child.unref();
  return status;
}

/** Starts one new detached staging generation and rejects implicit replacement or resume. */
export async function startRecallBackgroundIndexGeneration(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<RecallBackgroundIndexGenerationStatus> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(config.statusPath);
  try {
    const current = await refreshBackgroundIndexStatus(config);
    if (current && ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(current.processState)) {
      throw new Error(
        `Recall background index build already ${current.processState} in process ${current.processId}`,
      );
    }
    const generationStatus = await config.generationService.readIndexGenerationStatus();
    if (generationStatus.staging) {
      throw new Error(
        `Recall staging generation ${generationStatus.staging.generationId} already exists; resume or discard it explicitly`,
      );
    }
    return await spawnBackgroundIndexWorker(
      config,
      `generation_${Date.now()}_${randomUUID().replaceAll('-', '')}`,
      false,
    );
  } finally {
    await releaseControlLock();
  }
}

/** Resumes the selected staging generation in one new detached worker process. */
export async function resumeRecallBackgroundIndexGeneration(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<RecallBackgroundIndexGenerationStatus> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(config.statusPath);
  try {
    const current = await refreshBackgroundIndexStatus(config);
    if (current && ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(current.processState)) {
      throw new Error(
        `Recall background index build already ${current.processState} in process ${current.processId}`,
      );
    }
    const generationStatus = await config.generationService.readIndexGenerationStatus();
    if (!generationStatus.staging) {
      throw new Error('Recall background index resume requires a staging generation');
    }
    if (generationStatus.staging.status !== 'resumable') {
      throw new Error(
        `Recall background index resume requires resumable state, received ${generationStatus.staging.status}`,
      );
    }
    const embeddingProfileId = await resolveBackgroundIndexEmbeddingProfileId(config);
    if (generationStatus.staging.embeddingProfileId !== embeddingProfileId) {
      throw new Error(
        `Recall staging generation uses embedding profile ${generationStatus.staging.embeddingProfileId}, not ${embeddingProfileId}`,
      );
    }
    return await spawnBackgroundIndexWorker(config, generationStatus.staging.generationId, true);
  } finally {
    await releaseControlLock();
  }
}

/** Reads live status and converts a dead active worker record into a resumable crash. */
export async function readRecallBackgroundIndexGenerationStatus(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<RecallBackgroundIndexGenerationStatus | null> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(config.statusPath);
  try {
    return await refreshBackgroundIndexStatus(config);
  } finally {
    await releaseControlLock();
  }
}

/** Requests graceful cancellation from the detached worker without deleting staging work. */
export async function stopRecallBackgroundIndexGeneration(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<RecallBackgroundIndexGenerationStatus> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(config.statusPath);
  try {
    const current = await refreshBackgroundIndexStatus(config);
    if (!current || !ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(current.processState)) {
      throw new Error('Recall background index stop requires a live background build');
    }
    const updatedAt = new Date().toISOString();
    const stopping: RecallBackgroundIndexGenerationStatus = {
      ...current,
      processState: RecallBackgroundIndexProcessState.STOPPING,
      updatedAt,
    };
    await writeRecallBackgroundIndexStatusRecord(config.statusPath, stopping);
    try {
      process.kill(current.processId, 'SIGTERM');
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ESRCH') {
        throw error;
      }
    }
    return stopping;
  } finally {
    await releaseControlLock();
  }
}

/** Marks the latest completed background build discarded after staging files are removed. */
export async function markRecallBackgroundIndexGenerationDiscarded(
  config: RecallBackgroundIndexCoordinatorConfig,
): Promise<void> {
  const releaseControlLock = await acquireBackgroundIndexControlLock(config.statusPath);
  try {
    const current = await refreshBackgroundIndexStatus(config);
    if (!current) {
      return;
    }
    if (ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(current.processState)) {
      throw new Error(
        `Recall staging generation is owned by ${current.processState} worker ${current.processId}; stop it before discard`,
      );
    }
    const completedAt = new Date().toISOString();
    await writeRecallBackgroundIndexStatusRecord(config.statusPath, {
      ...current,
      processState: RecallBackgroundIndexProcessState.DISCARDED,
      updatedAt: completedAt,
      completedAt,
      latestActionableError: null,
    });
  } finally {
    await releaseControlLock();
  }
}

/** Removes one build-scoped worker request after a terminal state; status remains as bounded evidence. */
export async function removeRecallBackgroundIndexWorkerRequest(requestPath: string): Promise<void> {
  await rm(requestPath, { force: true });
}

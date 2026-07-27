import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { RecallBackgroundIndexProcessState, RecallDiagnosticsMode } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type {
  ConversationIndexCheckpoint,
  ConversationIndexProgress,
} from './incremental-session-indexer.js';
import {
  preserveStagingRecallIndexGeneration,
  readRecallIndexGenerationStatus,
  type RecallIndexGenerationCoordinatorConfig,
} from './recall-index-generations.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const RECALL_BACKGROUND_INDEX_STATUS_VERSION = 1;
const MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH = 4096;
const ACTIVE_BACKGROUND_INDEX_PROCESS_STATES = new Set<RecallBackgroundIndexProcessState>([
  RecallBackgroundIndexProcessState.STARTING,
  RecallBackgroundIndexProcessState.RUNNING,
  RecallBackgroundIndexProcessState.STOPPING,
]);

const backgroundIndexStatusSchema = Type.Object(
  {
    version: Type.Literal(RECALL_BACKGROUND_INDEX_STATUS_VERSION),
    buildId: Type.String({ minLength: 1 }),
    generationId: Type.Union([Type.String({ minLength: 1 }), Type.Null()]),
    embeddingProfileId: Type.String({ minLength: 1 }),
    processId: Type.Integer({ minimum: 1 }),
    processState: Type.Enum(RecallBackgroundIndexProcessState),
    startedAt: Type.String({ format: 'date-time' }),
    updatedAt: Type.String({ format: 'date-time' }),
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
    latestCheckpoint: Type.Union([
      Type.Object(
        {
          checkpointedSessions: Type.Integer({ minimum: 0 }),
          totalSessions: Type.Integer({ minimum: 0 }),
          sessionPath: Type.String({ maxLength: MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH }),
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

const backgroundIndexServiceConfigSchema = Type.Object(
  {
    sessionsDirectory: Type.String({ minLength: 1 }),
    databasePath: Type.String({ minLength: 1 }),
    statePath: Type.String({ minLength: 1 }),
    manifestPath: Type.String({ minLength: 1 }),
    tokenizerCacheDirectory: Type.String({ minLength: 1 }),
    embeddingCacheDirectory: Type.String({ minLength: 1 }),
    lockPath: Type.String({ minLength: 1 }),
    generationsDirectory: Type.Optional(Type.String({ minLength: 1 })),
    activeGenerationPath: Type.Optional(Type.String({ minLength: 1 })),
    stagingGenerationPath: Type.Optional(Type.String({ minLength: 1 })),
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

const backgroundIndexWorkerRequestSchema = Type.Object(
  {
    version: Type.Literal(1),
    buildId: Type.String({ minLength: 1 }),
    statusPath: Type.String({ minLength: 1 }),
    serviceConfig: backgroundIndexServiceConfigSchema,
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
  completedAt: string | null;
  progress: ConversationIndexProgress | null;
  latestCheckpoint: ConversationIndexCheckpoint | null;
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
  serviceConfig: RecallConversationConfig;
  serviceFactory: RecallBackgroundIndexServiceFactory;
}

/** Paths and semantic identity needed to control one background staging build. */
export interface RecallBackgroundIndexCoordinatorConfig {
  serviceConfig: RecallConversationConfig;
  generationCoordinatorConfig: RecallIndexGenerationCoordinatorConfig;
  statusPath: string;
  requestPath: string;
  embeddingProfileId: string;
  serviceFactory: RecallBackgroundIndexServiceFactory;
}

function truncateBackgroundStatusText(value: string): string {
  return value.slice(0, MAX_RECALL_BACKGROUND_STATUS_TEXT_LENGTH);
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

/** Reads one validated background worker request written by the conversation service. */
export async function readRecallBackgroundIndexWorkerRequest(
  requestPath: string,
): Promise<RecallBackgroundIndexWorkerRequest> {
  try {
    const parsed: unknown = JSON.parse(await readFile(requestPath, 'utf8'));
    const request = Value.Parse(backgroundIndexWorkerRequestSchema, parsed);
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
    return Value.Parse(backgroundIndexStatusSchema, parsed);
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
  const validated = Value.Parse(backgroundIndexStatusSchema, status);
  await writeAtomicJson(statusPath, validated);
}

async function acquireBackgroundIndexControlLock(statusPath: string): Promise<() => Promise<void>> {
  const lockPath = `${statusPath}.control-lock`;
  const ownerPath = `${lockPath}/owner.json`;
  await mkdir(dirname(lockPath), { recursive: true });
  let unreadableOwnerCount = 0;
  while (true) {
    try {
      await mkdir(lockPath);
      await writeAtomicJson(ownerPath, { pid: process.pid });
      return () => rm(lockPath, { recursive: true, force: true });
    } catch (error) {
      if (readNodeErrorCode(error) !== 'EEXIST') {
        throw error;
      }
      let ownerProcessId: number | undefined;
      try {
        const ownerText = await readFile(ownerPath, 'utf8');
        try {
          const owner: unknown = JSON.parse(ownerText);
          if (isUnknownRecord(owner)) {
            const processId = owner.pid;
            ownerProcessId =
              typeof processId === 'number' && Number.isInteger(processId) ? processId : undefined;
          }
        } catch {
          ownerProcessId = undefined;
        }
      } catch (readError) {
        if (readNodeErrorCode(readError) !== 'ENOENT') {
          throw readError;
        }
      }
      if (ownerProcessId === undefined) {
        unreadableOwnerCount += 1;
        if (unreadableOwnerCount >= 4) {
          await rm(lockPath, { recursive: true, force: true });
          unreadableOwnerCount = 0;
          continue;
        }
      } else if (!isProcessAlive(ownerProcessId)) {
        await rm(lockPath, { recursive: true, force: true });
        unreadableOwnerCount = 0;
        continue;
      } else {
        unreadableOwnerCount = 0;
      }
      await sleep(25);
    }
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
  const generationStatus = await readRecallIndexGenerationStatus(
    config.generationCoordinatorConfig,
  );
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
  if (
    ACTIVE_BACKGROUND_INDEX_PROCESS_STATES.has(refreshed.processState) &&
    !isProcessAlive(refreshed.processId)
  ) {
    if (generationStatus.staging) {
      await preserveStagingRecallIndexGeneration(
        config.generationCoordinatorConfig,
        generationStatus.staging.generationId,
      );
    }
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

async function spawnBackgroundIndexWorker(
  config: RecallBackgroundIndexCoordinatorConfig,
  generationId: string | null,
): Promise<RecallBackgroundIndexGenerationStatus> {
  const buildId = randomUUID();
  const request = {
    version: 1,
    buildId,
    statusPath: config.statusPath,
    serviceConfig: {
      ...config.serviceConfig,
      projectLineages: Object.fromEntries(config.serviceConfig.projectLineages),
    },
    serviceFactory: config.serviceFactory,
  };
  await writeAtomicJson(config.requestPath, request);

  const workerPath = fileURLToPath(new URL('./recall-background-index-worker.ts', import.meta.url));
  const child = spawn(process.execPath, ['--import', 'tsx', workerPath, config.requestPath], {
    detached: true,
    stdio: 'ignore',
  });
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
    embeddingProfileId: config.embeddingProfileId,
    processId: child.pid,
    processState: RecallBackgroundIndexProcessState.STARTING,
    startedAt,
    updatedAt: startedAt,
    completedAt: null,
    progress: null,
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
    const generationStatus = await readRecallIndexGenerationStatus(
      config.generationCoordinatorConfig,
    );
    if (generationStatus.staging) {
      throw new Error(
        `Recall staging generation ${generationStatus.staging.generationId} already exists; resume or discard it explicitly`,
      );
    }
    return await spawnBackgroundIndexWorker(config, null);
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
    const generationStatus = await readRecallIndexGenerationStatus(
      config.generationCoordinatorConfig,
    );
    if (!generationStatus.staging) {
      throw new Error('Recall background index resume requires a staging generation');
    }
    if (generationStatus.staging.embeddingProfileId !== config.embeddingProfileId) {
      throw new Error(
        `Recall staging generation uses embedding profile ${generationStatus.staging.embeddingProfileId}, not ${config.embeddingProfileId}`,
      );
    }
    return await spawnBackgroundIndexWorker(config, generationStatus.staging.generationId);
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

/** Removes a worker request after a terminal state; status remains as bounded evidence. */
export async function removeRecallBackgroundIndexWorkerRequest(requestPath: string): Promise<void> {
  await rm(requestPath, { force: true });
}

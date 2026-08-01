import { spawn } from 'node:child_process';
import { readFile, writeFile } from 'node:fs/promises';
import { setTimeout as sleep } from 'node:timers/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  RecallBackgroundIndexProcessState,
  RecallFixedSnapshotBuildOperationState,
} from './enums.js';
import {
  readRecallBackgroundIndexStatusRecord,
  readRecallBackgroundIndexWorkerRequest,
  removeRecallBackgroundIndexWorkerRequest,
  updateRecallBackgroundIndexStatusRecord,
  type RecallBackgroundIndexGenerationStatus,
} from './recall-background-index-build.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import type { RecallConversationService } from './recall-conversation-service.js';
import type { RecallFixedSnapshotBuildOperation } from './recall-physical-source-generation.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_BACKGROUND_WORKER_HEARTBEAT_MILLISECONDS = 5_000;
const RECALL_BACKGROUND_WATCHDOG_FAILURE_MILLISECONDS = 30 * 60 * 1_000;

type RecallBackgroundIndexServiceFactory = (
  config: RecallConversationConfig,
) => RecallConversationService | Promise<RecallConversationService>;

function isRecallBackgroundIndexServiceFactory(
  value: unknown,
): value is RecallBackgroundIndexServiceFactory {
  return typeof value === 'function';
}

/** Decides whether a named build operation and the independent worker heartbeat are both frozen. */
export function shouldTerminateStalledRecallBackgroundIndexWorker(
  status: Readonly<RecallBackgroundIndexGenerationStatus>,
  observedAtEpochMilliseconds: number,
  failureThresholdMilliseconds: number,
): boolean {
  if (status.activeOperation === null || status.activeOperation === undefined) {
    return false;
  }
  const heartbeatAt = status.heartbeatAt ?? status.updatedAt;
  return (
    observedAtEpochMilliseconds - Date.parse(status.activeOperation.startedAt) >=
      failureThresholdMilliseconds &&
    observedAtEpochMilliseconds - Date.parse(heartbeatAt) >= failureThresholdMilliseconds
  );
}

async function runRecallBackgroundIndexWatchdog(
  statusPath: string,
  buildId: string,
  workerProcessId: number,
): Promise<void> {
  while (true) {
    await sleep(RECALL_BACKGROUND_WORKER_HEARTBEAT_MILLISECONDS);
    let status: RecallBackgroundIndexGenerationStatus | null;
    try {
      status = await readRecallBackgroundIndexStatusRecord(statusPath);
    } catch {
      continue;
    }
    if (
      status === null ||
      status.buildId !== buildId ||
      status.processId !== workerProcessId ||
      status.processState === RecallBackgroundIndexProcessState.SUCCEEDED ||
      status.processState === RecallBackgroundIndexProcessState.FAILED ||
      status.processState === RecallBackgroundIndexProcessState.CRASHED ||
      status.processState === RecallBackgroundIndexProcessState.STOPPED ||
      status.processState === RecallBackgroundIndexProcessState.DISCARDED
    ) {
      return;
    }
    const observedAtEpochMilliseconds = Date.now();
    if (
      !shouldTerminateStalledRecallBackgroundIndexWorker(
        status,
        observedAtEpochMilliseconds,
        RECALL_BACKGROUND_WATCHDOG_FAILURE_MILLISECONDS,
      )
    ) {
      continue;
    }
    const processStat = await readFile(`/proc/${workerProcessId}/stat`, 'utf8').catch(
      () => 'unavailable',
    );
    const diagnosticPath = `${statusPath}.${buildId}.stall-diagnostic.json`;
    try {
      await writeFile(
        diagnosticPath,
        `${JSON.stringify(
          {
            version: 1,
            detectedAt: new Date(observedAtEpochMilliseconds).toISOString(),
            buildId,
            workerProcessId,
            activeOperation: status.activeOperation,
            heartbeatAt: status.heartbeatAt ?? status.updatedAt,
            processStat,
            cpuProfileLogPath: `${statusPath}.${buildId}.v8.log`,
            action: 'SIGKILL',
            reason:
              'Recall rebuild operation and worker heartbeat both exceeded the 30-minute stall limit',
          },
          null,
          2,
        )}\n`,
        { encoding: 'utf8', flag: 'wx' },
      );
    } catch (error) {
      if (readNodeErrorCode(error) !== 'EEXIST') {
        throw error;
      }
    }
    try {
      process.kill(workerProcessId, 'SIGKILL');
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ESRCH') {
        throw error;
      }
    }
    return;
  }
}

async function startRecallBackgroundIndexWatchdog(
  statusPath: string,
  buildId: string,
): Promise<void> {
  const workerPath = fileURLToPath(import.meta.url);
  const child = spawn(
    process.execPath,
    ['--import', 'tsx', workerPath, '--watchdog', statusPath, buildId, String(process.pid)],
    { detached: true, stdio: 'ignore' },
  );
  await new Promise<void>((resolve, reject) => {
    child.once('spawn', resolve);
    child.once('error', reject);
  });
  child.unref();
}

async function waitForWorkerStartRecord(
  statusPath: string,
  buildId: string,
): Promise<RecallBackgroundIndexGenerationStatus> {
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const status = await readRecallBackgroundIndexStatusRecord(statusPath);
    if (status?.buildId === buildId && status.processId === process.pid) {
      return status;
    }
    await sleep(10);
  }
  throw new Error(`Recall background index worker ${process.pid} did not receive its start record`);
}

async function runRecallBackgroundIndexWorker(REQUEST_PATH: string): Promise<void> {
  const request = await readRecallBackgroundIndexWorkerRequest(REQUEST_PATH);
  const cancellation = new AbortController();
  const stopWorker = () => {
    cancellation.abort(new Error('Recall background index worker stop requested'));
  };
  process.once('SIGTERM', stopWorker);
  process.once('SIGINT', stopWorker);

  await waitForWorkerStartRecord(request.statusPath, request.buildId);
  await startRecallBackgroundIndexWatchdog(request.statusPath, request.buildId);
  let statusWrites = Promise.resolve();
  function queueStatusUpdate(
    update: (
      current: RecallBackgroundIndexGenerationStatus,
    ) => RecallBackgroundIndexGenerationStatus,
  ): void {
    statusWrites = statusWrites.then(() =>
      updateRecallBackgroundIndexStatusRecord(
        request.statusPath,
        request.buildId,
        process.pid,
        update,
      ),
    );
  }

  const runningAt = new Date().toISOString();
  queueStatusUpdate((current) => ({
    ...current,
    processState:
      current.processState === RecallBackgroundIndexProcessState.STARTING
        ? RecallBackgroundIndexProcessState.RUNNING
        : current.processState,
    updatedAt: runningAt,
    heartbeatAt: runningAt,
  }));
  const heartbeatTimer = setInterval(() => {
    queueStatusUpdate((current) => {
      if (
        current.processState !== RecallBackgroundIndexProcessState.STARTING &&
        current.processState !== RecallBackgroundIndexProcessState.RUNNING &&
        current.processState !== RecallBackgroundIndexProcessState.STOPPING
      ) {
        return current;
      }
      const heartbeatAt = new Date().toISOString();
      return { ...current, updatedAt: heartbeatAt, heartbeatAt };
    });
  }, RECALL_BACKGROUND_WORKER_HEARTBEAT_MILLISECONDS);
  heartbeatTimer.unref();

  try {
    const factoryModule = await Promise.resolve<unknown>(import(request.serviceFactory.moduleUrl));
    if (!isUnknownRecord(factoryModule)) {
      throw new Error(
        `Recall background index service factory module invalid: ${request.serviceFactory.moduleUrl}`,
      );
    }
    const factory = factoryModule[request.serviceFactory.exportName];
    if (!isRecallBackgroundIndexServiceFactory(factory)) {
      throw new Error(
        `Recall background index service factory export missing: ${request.serviceFactory.exportName} in ${request.serviceFactory.moduleUrl}`,
      );
    }
    const service = await factory(request.serviceConfig);
    if (request.generationId === null) {
      throw new Error('Recall background index worker request generation ID missing');
    }
    const result = await service.buildReplacementRecallGeneration({
      generationId: request.generationId,
      resumeExistingGeneration: request.resumeExistingGeneration,
      signal: cancellation.signal,
      async onBuildOperation(
        operation: Readonly<RecallFixedSnapshotBuildOperation>,
        state: RecallFixedSnapshotBuildOperationState,
      ) {
        const observedAt = new Date().toISOString();
        queueStatusUpdate((current) => {
          if (state === RecallFixedSnapshotBuildOperationState.COMPLETED) {
            const activeOperation = current.activeOperation;
            return {
              ...current,
              updatedAt: observedAt,
              activeOperation: null,
              latestCompletedOperation:
                activeOperation === null || activeOperation === undefined
                  ? (current.latestCompletedOperation ?? null)
                  : {
                      ...activeOperation,
                      completedAt: observedAt,
                      durationMilliseconds: Math.max(
                        0,
                        Date.parse(observedAt) - Date.parse(activeOperation.startedAt),
                      ),
                    },
            };
          }
          return {
            ...current,
            updatedAt: observedAt,
            activeOperation: {
              phase: operation.phase,
              startedAt: observedAt,
              ...(operation.physicalSourceIdentity === undefined
                ? {}
                : { physicalSourceIdentity: operation.physicalSourceIdentity.slice(0, 4096) }),
              ...(operation.sessionsRootRelativePath === undefined
                ? {}
                : {
                    sessionsRootRelativePath: operation.sessionsRootRelativePath.slice(0, 4096),
                  }),
              ...(operation.sourceNumber === undefined
                ? {}
                : { sourceNumber: operation.sourceNumber }),
              ...(operation.totalPhysicalSourceCount === undefined
                ? {}
                : { totalPhysicalSourceCount: operation.totalPhysicalSourceCount }),
              ...(operation.batchStartIndex === undefined
                ? {}
                : { batchStartIndex: operation.batchStartIndex }),
              ...(operation.batchRecordCount === undefined
                ? {}
                : { batchRecordCount: operation.batchRecordCount }),
              ...(operation.totalRecordCount === undefined
                ? {}
                : { totalRecordCount: operation.totalRecordCount }),
            },
          };
        });
        await statusWrites;
      },
      onPhysicalSourceCheckpoint(checkpoint) {
        queueStatusUpdate((current) => ({
          ...current,
          processState:
            current.processState === RecallBackgroundIndexProcessState.STARTING
              ? RecallBackgroundIndexProcessState.RUNNING
              : current.processState,
          updatedAt: new Date().toISOString(),
          progress: {
            scannedSessions: checkpoint.completedPhysicalSourceCount,
            totalSessions: checkpoint.totalPhysicalSourceCount,
            sessionPath: checkpoint.sessionsRootRelativePath.slice(0, 4096),
          },
          latestCheckpoint: {
            checkpointedSessions: checkpoint.completedPhysicalSourceCount,
            totalSessions: checkpoint.totalPhysicalSourceCount,
            sessionPath: checkpoint.sessionsRootRelativePath.slice(0, 4096),
            physicalSourceIdentity: checkpoint.physicalSourceIdentity.slice(0, 4096),
          },
        }));
      },
    });
    await statusWrites;
    const completedAt = new Date().toISOString();
    queueStatusUpdate((current) => ({
      ...current,
      generationId: result.generationId,
      processState: RecallBackgroundIndexProcessState.SUCCEEDED,
      updatedAt: completedAt,
      completedAt,
      activeOperation: null,
      latestActionableError: null,
    }));
    await statusWrites;
  } catch (error) {
    await statusWrites.catch(() => undefined);
    const completedAt = new Date().toISOString();
    const message = error instanceof Error ? error.message : String(error);
    queueStatusUpdate((current) => ({
      ...current,
      processState: cancellation.signal.aborted
        ? RecallBackgroundIndexProcessState.STOPPED
        : RecallBackgroundIndexProcessState.FAILED,
      updatedAt: completedAt,
      completedAt,
      latestActionableError: cancellation.signal.aborted ? null : message.slice(0, 4096),
    }));
    await statusWrites;
    if (!cancellation.signal.aborted) {
      process.exitCode = 1;
    }
  } finally {
    clearInterval(heartbeatTimer);
    process.removeListener('SIGTERM', stopWorker);
    process.removeListener('SIGINT', stopWorker);
    await removeRecallBackgroundIndexWorkerRequest(REQUEST_PATH).catch(() => undefined);
  }
}

const executablePath = process.argv[1];
if (executablePath && pathToFileURL(executablePath).href === import.meta.url) {
  const requestPath = process.argv[2];
  if (requestPath === '--watchdog') {
    const statusPath = process.argv[3];
    const buildId = process.argv[4];
    const workerProcessId = Number(process.argv[5]);
    if (!statusPath || !buildId || !Number.isSafeInteger(workerProcessId) || workerProcessId < 1) {
      process.exitCode = 1;
    } else {
      await runRecallBackgroundIndexWatchdog(statusPath, buildId, workerProcessId);
    }
  } else if (!requestPath) {
    process.exitCode = 1;
  } else {
    await runRecallBackgroundIndexWorker(requestPath);
  }
}

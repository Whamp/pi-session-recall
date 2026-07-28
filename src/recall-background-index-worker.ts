import { setTimeout as sleep } from 'node:timers/promises';

import { RecallBackgroundIndexProcessState, RecallManualMaintenanceTrigger } from './enums.js';
import {
  readRecallBackgroundIndexStatusRecord,
  readRecallBackgroundIndexWorkerRequest,
  removeRecallBackgroundIndexWorkerRequest,
  updateRecallBackgroundIndexStatusRecord,
  type RecallBackgroundIndexGenerationStatus,
} from './recall-background-index-build.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import type { RecallConversationService } from './recall-conversation-service.js';
import { isUnknownRecord } from './is-unknown-record.js';

type RecallBackgroundIndexServiceFactory = (
  config: RecallConversationConfig,
) => RecallConversationService | Promise<RecallConversationService>;

function isRecallBackgroundIndexServiceFactory(
  value: unknown,
): value is RecallBackgroundIndexServiceFactory {
  return typeof value === 'function';
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
  }));

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
    const result = await service.index({
      rebuild: true,
      manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
      optimize: true,
      ...(request.generationId ? { resumeGenerationId: request.generationId } : {}),
      signal: cancellation.signal,
      onProgress(progress) {
        queueStatusUpdate((current) => ({
          ...current,
          processState:
            current.processState === RecallBackgroundIndexProcessState.STARTING
              ? RecallBackgroundIndexProcessState.RUNNING
              : current.processState,
          updatedAt: new Date().toISOString(),
          progress: {
            ...progress,
            sessionPath: progress.sessionPath.slice(0, 4096),
          },
        }));
      },
      onCheckpoint(checkpoint) {
        queueStatusUpdate((current) => ({
          ...current,
          updatedAt: new Date().toISOString(),
          latestCheckpoint: {
            ...checkpoint,
            sessionPath: checkpoint.sessionPath.slice(0, 4096),
          },
        }));
      },
    });
    void result;
    await statusWrites;
    const generationStatus = await service.readIndexGenerationStatus();
    const completedAt = new Date().toISOString();
    queueStatusUpdate((current) => ({
      ...current,
      generationId: generationStatus.active?.generationId ?? current.generationId,
      processState: RecallBackgroundIndexProcessState.SUCCEEDED,
      updatedAt: completedAt,
      completedAt,
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
    process.removeListener('SIGTERM', stopWorker);
    process.removeListener('SIGINT', stopWorker);
    await removeRecallBackgroundIndexWorkerRequest(REQUEST_PATH).catch(() => undefined);
  }
}

const REQUEST_PATH = process.argv[2];
if (!REQUEST_PATH) {
  process.exitCode = 1;
} else {
  await runRecallBackgroundIndexWorker(REQUEST_PATH);
}

import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, rename, rm, stat } from 'node:fs/promises';
import { dirname } from 'node:path';

import {
  type RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallDiagnosticsMode,
  type RecallLifecycleTrigger,
} from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_DIAGNOSTIC_RECORD_VERSION = 1;
const DEFAULT_MAXIMUM_DIAGNOSTIC_LOG_BYTES = 10 * 1_024 * 1_024;
const PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE = {
  disabled: false,
  warningEmitted: false,
};
const SLOW_RECALL_DIAGNOSTIC_THRESHOLD_MILLISECONDS = 1_000;
const MAX_RECORDED_SESSION_PATH_CHARACTERS = 4_096;

/** Replaceable wall and monotonic time source for recall diagnostic durations. */
export interface RecallDiagnosticsClock {
  monotonicMilliseconds(): number;
  wallClockIsoTimestamp(): string;
}

/** Exclusive phase totals and bounded work counts for one live session reconciliation. */
export interface RecallLiveSessionDiagnosticMetrics {
  sourceByteSize: number | null;
  changed: boolean | null;
  skipped: boolean | null;
  writerLockWaitMilliseconds: number;
  manifestStorePreparationMilliseconds: number;
  physicalSessionPreparationMilliseconds: number;
  projectIdentityResolutionMilliseconds: number;
  embeddingCacheResolutionMilliseconds: number;
  embeddingServerRequestMilliseconds: number;
  databaseWriteMilliseconds: number;
  indexStateCheckpointMilliseconds: number;
  cacheHitCount: number;
  newEmbeddingCount: number;
  embeddingRequestCount: number;
  upsertedDocumentCount: number;
  deletedDocumentCount: number;
}

/** Bounded scalar completion data for one live session reconciliation. */
export interface RecallLiveSessionDiagnosticCompletion {
  status: RecallDiagnosticStatus;
  metrics: RecallLiveSessionDiagnosticMetrics;
  scannedSessionCount: number;
  indexedSessionCount: number;
  removedSessionCount: number;
  failedSessionCount: number;
  cacheHitCount: number;
  newEmbeddingCount: number;
  embeddingRequestCount: number;
  deletedDocumentCount: number;
  totalDocumentCount: number | null;
  errorCategory?: RecallDiagnosticErrorCategory;
}

/** One bounded privacy-safe JSONL record for a recall diagnostic operation. */
export interface RecallOperationDiagnosticRecord {
  version: number;
  timestamp: string;
  operationId: string;
  operationKind: RecallDiagnosticOperationKind;
  lifecycleTrigger: RecallLifecycleTrigger;
  processId: number;
  sessionPath: string;
  status: RecallDiagnosticStatus;
  errorCategory: RecallDiagnosticErrorCategory | null;
  sourceByteSize: number | null;
  changed: boolean | null;
  skipped: boolean | null;
  elapsedMilliseconds: number | null;
  writerLockWaitMilliseconds: number | null;
  manifestStorePreparationMilliseconds: number | null;
  physicalSessionPreparationMilliseconds: number | null;
  projectIdentityResolutionMilliseconds: number | null;
  embeddingCacheResolutionMilliseconds: number | null;
  embeddingServerRequestMilliseconds: number | null;
  databaseWriteMilliseconds: number | null;
  indexStateCheckpointMilliseconds: number | null;
  unattributedMilliseconds: number | null;
  scannedSessionCount: number | null;
  indexedSessionCount: number | null;
  removedSessionCount: number | null;
  failedSessionCount: number | null;
  upsertedDocumentCount: number | null;
  deletedDocumentCount: number | null;
  totalDocumentCount: number | null;
  cacheHitCount: number | null;
  newEmbeddingCount: number | null;
  embeddingRequestCount: number | null;
}

/** Handle that completes one already-started live session diagnostic without awaiting persistence. */
export interface RecallLiveSessionDiagnosticOperation {
  complete(completion: RecallLiveSessionDiagnosticCompletion): void;
}

/** Non-critical recall diagnostic recorder with an explicit test-only drain boundary. */
export interface RecallOperationDiagnostics {
  startLiveSessionReconciliation(input: {
    lifecycleTrigger: RecallLifecycleTrigger;
    sessionPath: string;
  }): RecallLiveSessionDiagnosticOperation;
  flush(): Promise<void>;
}

/** Configuration for bounded local recall diagnostic persistence. */
export interface RecallOperationDiagnosticsOptions {
  mode: RecallDiagnosticsMode;
  activeLogPath: string;
  retainedLogPath: string;
  clock?: RecallDiagnosticsClock;
  notifyWarning: (message: string) => void;
  maximumLogBytes?: number;
}

const SYSTEM_RECALL_DIAGNOSTICS_CLOCK: RecallDiagnosticsClock = {
  monotonicMilliseconds: () => performance.now(),
  wallClockIsoTimestamp: () => new Date().toISOString(),
};

/** Creates zeroed live session measurements whose phase totals remain non-overlapping. */
export function createRecallLiveSessionDiagnosticMetrics(): RecallLiveSessionDiagnosticMetrics {
  return {
    sourceByteSize: null,
    changed: null,
    skipped: null,
    writerLockWaitMilliseconds: 0,
    manifestStorePreparationMilliseconds: 0,
    physicalSessionPreparationMilliseconds: 0,
    projectIdentityResolutionMilliseconds: 0,
    embeddingCacheResolutionMilliseconds: 0,
    embeddingServerRequestMilliseconds: 0,
    databaseWriteMilliseconds: 0,
    indexStateCheckpointMilliseconds: 0,
    cacheHitCount: 0,
    newEmbeddingCount: 0,
    embeddingRequestCount: 0,
    upsertedDocumentCount: 0,
    deletedDocumentCount: 0,
  };
}

function createRecallDiagnosticStartRecord(input: {
  clock: RecallDiagnosticsClock;
  operationId: string;
  lifecycleTrigger: RecallLifecycleTrigger;
  sessionPath: string;
}): RecallOperationDiagnosticRecord {
  return {
    version: RECALL_DIAGNOSTIC_RECORD_VERSION,
    timestamp: input.clock.wallClockIsoTimestamp(),
    operationId: input.operationId,
    operationKind: RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
    lifecycleTrigger: input.lifecycleTrigger,
    processId: process.pid,
    sessionPath: input.sessionPath.slice(0, MAX_RECORDED_SESSION_PATH_CHARACTERS),
    status: RecallDiagnosticStatus.STARTED,
    errorCategory: null,
    sourceByteSize: null,
    changed: null,
    skipped: null,
    elapsedMilliseconds: null,
    writerLockWaitMilliseconds: null,
    manifestStorePreparationMilliseconds: null,
    physicalSessionPreparationMilliseconds: null,
    projectIdentityResolutionMilliseconds: null,
    embeddingCacheResolutionMilliseconds: null,
    embeddingServerRequestMilliseconds: null,
    databaseWriteMilliseconds: null,
    indexStateCheckpointMilliseconds: null,
    unattributedMilliseconds: null,
    scannedSessionCount: null,
    indexedSessionCount: null,
    removedSessionCount: null,
    failedSessionCount: null,
    upsertedDocumentCount: null,
    deletedDocumentCount: null,
    totalDocumentCount: null,
    cacheHitCount: null,
    newEmbeddingCount: null,
    embeddingRequestCount: null,
  };
}

function createRecallDiagnosticCompletionRecord(input: {
  startRecord: RecallOperationDiagnosticRecord;
  startedAtMilliseconds: number;
  clock: RecallDiagnosticsClock;
  completion: RecallLiveSessionDiagnosticCompletion;
}): RecallOperationDiagnosticRecord {
  const elapsedMilliseconds = Math.max(
    input.clock.monotonicMilliseconds() - input.startedAtMilliseconds,
    0,
  );
  const metrics = input.completion.metrics;
  const attributedMilliseconds =
    metrics.writerLockWaitMilliseconds +
    metrics.manifestStorePreparationMilliseconds +
    metrics.physicalSessionPreparationMilliseconds +
    metrics.projectIdentityResolutionMilliseconds +
    metrics.embeddingCacheResolutionMilliseconds +
    metrics.embeddingServerRequestMilliseconds +
    metrics.databaseWriteMilliseconds +
    metrics.indexStateCheckpointMilliseconds;
  return {
    ...input.startRecord,
    timestamp: input.clock.wallClockIsoTimestamp(),
    status: input.completion.status,
    errorCategory: input.completion.errorCategory ?? null,
    sourceByteSize: metrics.sourceByteSize,
    changed: metrics.changed,
    skipped: metrics.skipped,
    elapsedMilliseconds,
    writerLockWaitMilliseconds: metrics.writerLockWaitMilliseconds,
    manifestStorePreparationMilliseconds: metrics.manifestStorePreparationMilliseconds,
    physicalSessionPreparationMilliseconds: metrics.physicalSessionPreparationMilliseconds,
    projectIdentityResolutionMilliseconds: metrics.projectIdentityResolutionMilliseconds,
    embeddingCacheResolutionMilliseconds: metrics.embeddingCacheResolutionMilliseconds,
    embeddingServerRequestMilliseconds: metrics.embeddingServerRequestMilliseconds,
    databaseWriteMilliseconds: metrics.databaseWriteMilliseconds,
    indexStateCheckpointMilliseconds: metrics.indexStateCheckpointMilliseconds,
    unattributedMilliseconds: Math.max(elapsedMilliseconds - attributedMilliseconds, 0),
    scannedSessionCount: input.completion.scannedSessionCount,
    indexedSessionCount: input.completion.indexedSessionCount,
    removedSessionCount: input.completion.removedSessionCount,
    failedSessionCount: input.completion.failedSessionCount,
    upsertedDocumentCount: metrics.upsertedDocumentCount,
    deletedDocumentCount: input.completion.deletedDocumentCount,
    totalDocumentCount: input.completion.totalDocumentCount,
    cacheHitCount: input.completion.cacheHitCount,
    newEmbeddingCount: input.completion.newEmbeddingCount,
    embeddingRequestCount: input.completion.embeddingRequestCount,
  };
}

/** Creates asynchronous local JSONL diagnostics that never propagate persistence failures. */
export function createRecallOperationDiagnostics(
  options: RecallOperationDiagnosticsOptions,
): RecallOperationDiagnostics {
  const clock = options.clock ?? SYSTEM_RECALL_DIAGNOSTICS_CLOCK;
  const maximumLogBytes = options.maximumLogBytes ?? DEFAULT_MAXIMUM_DIAGNOSTIC_LOG_BYTES;
  let pendingWrite = Promise.resolve();

  async function rotateDiagnosticLogIfNeeded(recordByteLength: number): Promise<void> {
    let activeLogBytes: number;
    try {
      activeLogBytes = (await stat(options.activeLogPath)).size;
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return;
      }
      throw error;
    }
    if (activeLogBytes === 0 || activeLogBytes + recordByteLength <= maximumLogBytes) {
      return;
    }
    await rm(options.retainedLogPath, { force: true });
    await rename(options.activeLogPath, options.retainedLogPath);
  }

  async function appendDiagnosticRecordAfter(
    previousWrite: Promise<void>,
    record: RecallOperationDiagnosticRecord,
  ): Promise<void> {
    await previousWrite;
    if (PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE.disabled) {
      return;
    }
    try {
      const line = `${JSON.stringify(record)}\n`;
      await mkdir(dirname(options.activeLogPath), { recursive: true });
      await rotateDiagnosticLogIfNeeded(Buffer.byteLength(line));
      await appendFile(options.activeLogPath, line, 'utf8');
    } catch {
      PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE.disabled = true;
      if (!PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE.warningEmitted) {
        PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE.warningEmitted = true;
        try {
          options.notifyWarning(
            'Recall diagnostics disabled after local log persistence failed; recall behavior is unchanged.',
          );
        } catch (warningError) {
          // Warning delivery is non-critical for the same reason diagnostic persistence is.
          void warningError;
        }
      }
    }
  }

  function shouldPersistDiagnosticRecord(record: RecallOperationDiagnosticRecord): boolean {
    if (options.mode === RecallDiagnosticsMode.OFF) {
      return false;
    }
    if (options.mode === RecallDiagnosticsMode.ALL) {
      return true;
    }
    if (record.status === RecallDiagnosticStatus.STARTED) {
      return false;
    }
    return (
      record.status === RecallDiagnosticStatus.FAILED ||
      record.status === RecallDiagnosticStatus.CANCELLED ||
      (record.elapsedMilliseconds ?? 0) >= SLOW_RECALL_DIAGNOSTIC_THRESHOLD_MILLISECONDS
    );
  }

  function queueDiagnosticRecord(record: RecallOperationDiagnosticRecord): void {
    if (
      !shouldPersistDiagnosticRecord(record) ||
      PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE.disabled
    ) {
      return;
    }
    pendingWrite = appendDiagnosticRecordAfter(pendingWrite, record);
  }

  return {
    startLiveSessionReconciliation(input) {
      const startedAtMilliseconds = clock.monotonicMilliseconds();
      const startRecord = createRecallDiagnosticStartRecord({
        clock,
        operationId: randomUUID(),
        lifecycleTrigger: input.lifecycleTrigger,
        sessionPath: input.sessionPath,
      });
      queueDiagnosticRecord(startRecord);
      let completed = false;
      return {
        complete(completion) {
          if (completed) {
            return;
          }
          completed = true;
          queueDiagnosticRecord(
            createRecallDiagnosticCompletionRecord({
              startRecord,
              startedAtMilliseconds,
              clock,
              completion,
            }),
          );
        },
      };
    },
    async flush() {
      await pendingWrite;
    },
  };
}

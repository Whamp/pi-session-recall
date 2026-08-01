import { randomUUID } from 'node:crypto';
import { appendFile, mkdir, rename, rm, stat } from 'node:fs/promises';
import { dirname } from 'node:path';

import {
  type RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallDiagnosticsMode,
  type RecallLifecycleTrigger,
  RecallManualMaintenanceTrigger,
  type RecallSearchScope,
} from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const RECALL_DIAGNOSTIC_RECORD_VERSION = 2;
const DEFAULT_MAXIMUM_DIAGNOSTIC_LOG_BYTES = 10 * 1_024 * 1_024;
const PROCESS_RECALL_DIAGNOSTIC_PERSISTENCE_STATE = {
  disabled: false,
  warningEmitted: false,
};
const SLOW_RECALL_DIAGNOSTIC_THRESHOLD_MILLISECONDS = 1_000;
const MAX_RECORDED_SESSION_PATH_CHARACTERS = 4_096;

class RecallDiagnosticOperationId {
  readonly value: string;

  constructor(value: string) {
    this.value = value;
  }

  toJSON(): string {
    return this.value;
  }
}

function createRecallDiagnosticOperationId(): RecallDiagnosticOperationId {
  return new RecallDiagnosticOperationId(randomUUID());
}

/** Replaceable wall and monotonic time source for recall diagnostic durations. */
export interface RecallDiagnosticsClock {
  monotonicMilliseconds(): number;
  wallClockIsoTimestamp(): string;
}

/** Exclusive phase totals and bounded work counts for one recall index operation. */
export interface RecallIndexDiagnosticMetrics {
  sourceByteSize: number | null;
  changed: boolean | null;
  skipped: boolean | null;
  writerLockWaitMilliseconds: number;
  manifestStorePreparationMilliseconds: number;
  physicalSessionScanMilliseconds: number;
  physicalSessionPreparationMilliseconds: number;
  projectIdentityResolutionMilliseconds: number;
  embeddingCacheResolutionMilliseconds: number;
  embeddingServerRequestMilliseconds: number;
  databaseWriteMilliseconds: number;
  indexStateCheckpointMilliseconds: number;
  optimizationRan: boolean;
  optimizationMilliseconds: number;
  scannedSessionCount: number;
  indexedSessionCount: number;
  removedSessionCount: number;
  failedSessionCount: number;
  cacheHitCount: number;
  newEmbeddingCount: number;
  embeddingRequestCount: number;
  upsertedDocumentCount: number;
  deletedDocumentCount: number;
}

/** Bounded scalar completion data for one recall index operation. */
export interface RecallIndexDiagnosticCompletion {
  status: RecallDiagnosticStatus;
  metrics: RecallIndexDiagnosticMetrics;
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

/** Exclusive phase totals and bounded state for one public recall search. */
export interface RecallSearchDiagnosticMetrics {
  freshnessBarrierRan: boolean;
  embeddingModelVerificationMilliseconds: number;
  activeSessionFreshnessMilliseconds: number;
  queryEmbeddingMilliseconds: number;
  retrievalRankingMilliseconds: number;
  deepRerankMilliseconds: number;
}

/** Bounded scalar completion data for one public recall search. */
export interface RecallSearchDiagnosticCompletion {
  status: RecallDiagnosticStatus;
  metrics: RecallSearchDiagnosticMetrics;
  totalDocumentCount: number | null;
  errorCategory?: RecallDiagnosticErrorCategory;
}

/** Bounded scalar outcome for one physical session check inside a full index scan. */
export interface RecallPhysicalSessionDiagnostic {
  sessionPath: string;
  status: RecallDiagnosticStatus;
  metrics: RecallIndexDiagnosticMetrics;
  elapsedMilliseconds: number;
  indexedSessionCount: number;
  removedSessionCount: number;
  failedSessionCount: number;
  errorCategory?: RecallDiagnosticErrorCategory;
}

/** One bounded privacy-safe JSONL record for a recall diagnostic operation. */
export interface RecallOperationDiagnosticRecord {
  version: number;
  timestamp: string;
  operationId: RecallDiagnosticOperationId;
  parentOperationId: RecallDiagnosticOperationId | null;
  operationKind: RecallDiagnosticOperationKind;
  lifecycleTrigger: RecallLifecycleTrigger | null;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger | null;
  processId: number;
  sessionPath: string | null;
  searchMode: 'hybrid' | 'deep-rerank' | null;
  recallScope: RecallSearchScope | null;
  status: RecallDiagnosticStatus;
  errorCategory: RecallDiagnosticErrorCategory | null;
  sourceByteSize: number | null;
  changed: boolean | null;
  skipped: boolean | null;
  elapsedMilliseconds: number | null;
  writerLockWaitMilliseconds: number | null;
  manifestStorePreparationMilliseconds: number | null;
  physicalSessionScanMilliseconds: number | null;
  physicalSessionPreparationMilliseconds: number | null;
  projectIdentityResolutionMilliseconds: number | null;
  embeddingCacheResolutionMilliseconds: number | null;
  embeddingServerRequestMilliseconds: number | null;
  databaseWriteMilliseconds: number | null;
  indexStateCheckpointMilliseconds: number | null;
  optimizationRan: boolean | null;
  optimizationMilliseconds: number | null;
  embeddingModelVerificationMilliseconds: number | null;
  activeSessionFreshnessMilliseconds: number | null;
  queryEmbeddingMilliseconds: number | null;
  retrievalRankingMilliseconds: number | null;
  deepRerankMilliseconds: number | null;
  freshnessBarrierRan: boolean | null;
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
  complete(completion: RecallIndexDiagnosticCompletion): void;
}

/** Handle that completes one already-started search diagnostic without awaiting persistence. */
export interface RecallSearchDiagnosticOperation {
  complete(completion: RecallSearchDiagnosticCompletion): void;
}

/** Handle that completes one final database optimization diagnostic. */
export interface RecallOptimizationDiagnostic {
  complete(input: {
    status: RecallDiagnosticStatus;
    errorCategory?: RecallDiagnosticErrorCategory;
  }): void;
}

/** Handle that completes one explicitly requested full index diagnostic. */
export interface RecallManualIndexDiagnostic {
  recordPhysicalSessionCheck(completion: RecallPhysicalSessionDiagnostic): void;
  startOptimization(): RecallOptimizationDiagnostic;
  complete(completion: RecallIndexDiagnosticCompletion): void;
}

/** Non-critical recall diagnostic recorder with an explicit test-only drain boundary. */
export interface RecallOperationDiagnostics {
  startLiveSessionReconciliation(input: {
    lifecycleTrigger: RecallLifecycleTrigger;
    sessionPath: string;
  }): RecallLiveSessionDiagnosticOperation;
  startRecallSearch(input: {
    searchMode: 'hybrid' | 'deep-rerank';
    recallScope: RecallSearchScope;
  }): RecallSearchDiagnosticOperation;
  startManualIndexMaintenance(input: {
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger;
  }): RecallManualIndexDiagnostic;
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

/** Accumulates one physical session check into full-index diagnostic totals. */
export function accumulateRecallIndexMetrics(
  aggregateMetrics: RecallIndexDiagnosticMetrics,
  physicalSessionMetrics: RecallIndexDiagnosticMetrics,
): void {
  aggregateMetrics.sourceByteSize =
    (aggregateMetrics.sourceByteSize ?? 0) + (physicalSessionMetrics.sourceByteSize ?? 0);
  aggregateMetrics.physicalSessionPreparationMilliseconds +=
    physicalSessionMetrics.physicalSessionPreparationMilliseconds;
  aggregateMetrics.projectIdentityResolutionMilliseconds +=
    physicalSessionMetrics.projectIdentityResolutionMilliseconds;
  aggregateMetrics.embeddingCacheResolutionMilliseconds +=
    physicalSessionMetrics.embeddingCacheResolutionMilliseconds;
  aggregateMetrics.embeddingServerRequestMilliseconds +=
    physicalSessionMetrics.embeddingServerRequestMilliseconds;
  aggregateMetrics.databaseWriteMilliseconds += physicalSessionMetrics.databaseWriteMilliseconds;
  aggregateMetrics.scannedSessionCount += physicalSessionMetrics.scannedSessionCount;
  aggregateMetrics.indexedSessionCount += physicalSessionMetrics.indexedSessionCount;
  aggregateMetrics.removedSessionCount += physicalSessionMetrics.removedSessionCount;
  aggregateMetrics.failedSessionCount += physicalSessionMetrics.failedSessionCount;
  aggregateMetrics.cacheHitCount += physicalSessionMetrics.cacheHitCount;
  aggregateMetrics.newEmbeddingCount += physicalSessionMetrics.newEmbeddingCount;
  aggregateMetrics.embeddingRequestCount += physicalSessionMetrics.embeddingRequestCount;
  aggregateMetrics.upsertedDocumentCount += physicalSessionMetrics.upsertedDocumentCount;
  aggregateMetrics.deletedDocumentCount += physicalSessionMetrics.deletedDocumentCount;
}

/** Creates zeroed search measurements whose phase totals remain non-overlapping. */
export function createRecallSearchDiagnosticMetrics(): RecallSearchDiagnosticMetrics {
  return {
    freshnessBarrierRan: false,
    embeddingModelVerificationMilliseconds: 0,
    activeSessionFreshnessMilliseconds: 0,
    queryEmbeddingMilliseconds: 0,
    retrievalRankingMilliseconds: 0,
    deepRerankMilliseconds: 0,
  };
}

/** Creates zeroed index measurements whose phase totals remain non-overlapping. */
export function createRecallIndexMetrics(): RecallIndexDiagnosticMetrics {
  return {
    sourceByteSize: null,
    changed: null,
    skipped: null,
    writerLockWaitMilliseconds: 0,
    manifestStorePreparationMilliseconds: 0,
    physicalSessionScanMilliseconds: 0,
    physicalSessionPreparationMilliseconds: 0,
    projectIdentityResolutionMilliseconds: 0,
    embeddingCacheResolutionMilliseconds: 0,
    embeddingServerRequestMilliseconds: 0,
    databaseWriteMilliseconds: 0,
    indexStateCheckpointMilliseconds: 0,
    optimizationRan: false,
    optimizationMilliseconds: 0,
    scannedSessionCount: 0,
    indexedSessionCount: 0,
    removedSessionCount: 0,
    failedSessionCount: 0,
    cacheHitCount: 0,
    newEmbeddingCount: 0,
    embeddingRequestCount: 0,
    upsertedDocumentCount: 0,
    deletedDocumentCount: 0,
  };
}

function createRecallDiagnosticStartRecord(input: {
  clock: RecallDiagnosticsClock;
  operationId: RecallDiagnosticOperationId;
  parentOperationId: RecallDiagnosticOperationId | null;
  operationKind: RecallDiagnosticOperationKind;
  lifecycleTrigger: RecallLifecycleTrigger | null;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger | null;
  sessionPath: string | null;
  searchMode: 'hybrid' | 'deep-rerank' | null;
  recallScope: RecallSearchScope | null;
}): RecallOperationDiagnosticRecord {
  return {
    version: RECALL_DIAGNOSTIC_RECORD_VERSION,
    timestamp: input.clock.wallClockIsoTimestamp(),
    operationId: input.operationId,
    parentOperationId: input.parentOperationId,
    operationKind: input.operationKind,
    lifecycleTrigger: input.lifecycleTrigger,
    manualMaintenanceTrigger: input.manualMaintenanceTrigger,
    processId: process.pid,
    sessionPath: input.sessionPath?.slice(0, MAX_RECORDED_SESSION_PATH_CHARACTERS) ?? null,
    searchMode: input.searchMode,
    recallScope: input.recallScope,
    status: RecallDiagnosticStatus.STARTED,
    errorCategory: null,
    sourceByteSize: null,
    changed: null,
    skipped: null,
    elapsedMilliseconds: null,
    writerLockWaitMilliseconds: null,
    manifestStorePreparationMilliseconds: null,
    physicalSessionScanMilliseconds: null,
    physicalSessionPreparationMilliseconds: null,
    projectIdentityResolutionMilliseconds: null,
    embeddingCacheResolutionMilliseconds: null,
    embeddingServerRequestMilliseconds: null,
    databaseWriteMilliseconds: null,
    indexStateCheckpointMilliseconds: null,
    optimizationRan: null,
    optimizationMilliseconds: null,
    embeddingModelVerificationMilliseconds: null,
    activeSessionFreshnessMilliseconds: null,
    queryEmbeddingMilliseconds: null,
    retrievalRankingMilliseconds: null,
    deepRerankMilliseconds: null,
    freshnessBarrierRan: null,
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
  completion: RecallIndexDiagnosticCompletion;
}): RecallOperationDiagnosticRecord {
  const elapsedMilliseconds = Math.max(
    input.clock.monotonicMilliseconds() - input.startedAtMilliseconds,
    0,
  );
  const metrics = input.completion.metrics;
  const attributedMilliseconds =
    metrics.writerLockWaitMilliseconds +
    metrics.manifestStorePreparationMilliseconds +
    metrics.physicalSessionScanMilliseconds +
    metrics.physicalSessionPreparationMilliseconds +
    metrics.projectIdentityResolutionMilliseconds +
    metrics.embeddingCacheResolutionMilliseconds +
    metrics.embeddingServerRequestMilliseconds +
    metrics.databaseWriteMilliseconds +
    metrics.indexStateCheckpointMilliseconds +
    metrics.optimizationMilliseconds;
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
    physicalSessionScanMilliseconds: metrics.physicalSessionScanMilliseconds,
    physicalSessionPreparationMilliseconds: metrics.physicalSessionPreparationMilliseconds,
    projectIdentityResolutionMilliseconds: metrics.projectIdentityResolutionMilliseconds,
    embeddingCacheResolutionMilliseconds: metrics.embeddingCacheResolutionMilliseconds,
    embeddingServerRequestMilliseconds: metrics.embeddingServerRequestMilliseconds,
    databaseWriteMilliseconds: metrics.databaseWriteMilliseconds,
    indexStateCheckpointMilliseconds: metrics.indexStateCheckpointMilliseconds,
    optimizationRan: metrics.optimizationRan,
    optimizationMilliseconds: metrics.optimizationMilliseconds,
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

function createRecallPhysicalSessionCompletionRecord(input: {
  parentStartRecord: RecallOperationDiagnosticRecord;
  clock: RecallDiagnosticsClock;
  completion: RecallPhysicalSessionDiagnostic;
}): RecallOperationDiagnosticRecord {
  const metrics = input.completion.metrics;
  const elapsedMilliseconds = Math.max(input.completion.elapsedMilliseconds, 0);
  const attributedMilliseconds =
    metrics.physicalSessionPreparationMilliseconds +
    metrics.projectIdentityResolutionMilliseconds +
    metrics.embeddingCacheResolutionMilliseconds +
    metrics.embeddingServerRequestMilliseconds +
    metrics.databaseWriteMilliseconds +
    metrics.indexStateCheckpointMilliseconds;
  return {
    ...createRecallDiagnosticStartRecord({
      clock: input.clock,
      operationId: createRecallDiagnosticOperationId(),
      parentOperationId: input.parentStartRecord.operationId,
      operationKind: RecallDiagnosticOperationKind.PHYSICAL_SESSION_CHECK,
      lifecycleTrigger: null,
      manualMaintenanceTrigger: input.parentStartRecord.manualMaintenanceTrigger,
      sessionPath: input.completion.sessionPath,
      searchMode: null,
      recallScope: null,
    }),
    status: input.completion.status,
    errorCategory: input.completion.errorCategory ?? null,
    sourceByteSize: metrics.sourceByteSize,
    changed: metrics.changed,
    skipped: metrics.skipped,
    elapsedMilliseconds,
    physicalSessionPreparationMilliseconds: metrics.physicalSessionPreparationMilliseconds,
    projectIdentityResolutionMilliseconds: metrics.projectIdentityResolutionMilliseconds,
    embeddingCacheResolutionMilliseconds: metrics.embeddingCacheResolutionMilliseconds,
    embeddingServerRequestMilliseconds: metrics.embeddingServerRequestMilliseconds,
    databaseWriteMilliseconds: metrics.databaseWriteMilliseconds,
    indexStateCheckpointMilliseconds: metrics.indexStateCheckpointMilliseconds,
    unattributedMilliseconds: Math.max(elapsedMilliseconds - attributedMilliseconds, 0),
    scannedSessionCount: 1,
    indexedSessionCount: input.completion.indexedSessionCount,
    removedSessionCount: input.completion.removedSessionCount,
    failedSessionCount: input.completion.failedSessionCount,
    upsertedDocumentCount: metrics.upsertedDocumentCount,
    deletedDocumentCount: metrics.deletedDocumentCount,
    cacheHitCount: metrics.cacheHitCount,
    newEmbeddingCount: metrics.newEmbeddingCount,
    embeddingRequestCount: metrics.embeddingRequestCount,
  };
}

function createRecallOptimizationCompletionRecord(input: {
  startRecord: RecallOperationDiagnosticRecord;
  startedAtMilliseconds: number;
  clock: RecallDiagnosticsClock;
  status: RecallDiagnosticStatus;
  errorCategory?: RecallDiagnosticErrorCategory;
}): RecallOperationDiagnosticRecord {
  const elapsedMilliseconds = Math.max(
    input.clock.monotonicMilliseconds() - input.startedAtMilliseconds,
    0,
  );
  return {
    ...input.startRecord,
    timestamp: input.clock.wallClockIsoTimestamp(),
    status: input.status,
    errorCategory: input.errorCategory ?? null,
    elapsedMilliseconds,
    optimizationRan: true,
    optimizationMilliseconds: elapsedMilliseconds,
    unattributedMilliseconds: 0,
  };
}

function createRecallSearchDiagnosticCompletionRecord(input: {
  startRecord: RecallOperationDiagnosticRecord;
  startedAtMilliseconds: number;
  clock: RecallDiagnosticsClock;
  completion: RecallSearchDiagnosticCompletion;
}): RecallOperationDiagnosticRecord {
  const elapsedMilliseconds = Math.max(
    input.clock.monotonicMilliseconds() - input.startedAtMilliseconds,
    0,
  );
  const metrics = input.completion.metrics;
  const attributedMilliseconds =
    metrics.embeddingModelVerificationMilliseconds +
    metrics.activeSessionFreshnessMilliseconds +
    metrics.queryEmbeddingMilliseconds +
    metrics.retrievalRankingMilliseconds +
    metrics.deepRerankMilliseconds;
  return {
    ...input.startRecord,
    timestamp: input.clock.wallClockIsoTimestamp(),
    status: input.completion.status,
    errorCategory: input.completion.errorCategory ?? null,
    elapsedMilliseconds,
    embeddingModelVerificationMilliseconds: metrics.embeddingModelVerificationMilliseconds,
    activeSessionFreshnessMilliseconds: metrics.activeSessionFreshnessMilliseconds,
    queryEmbeddingMilliseconds: metrics.queryEmbeddingMilliseconds,
    retrievalRankingMilliseconds: metrics.retrievalRankingMilliseconds,
    deepRerankMilliseconds: metrics.deepRerankMilliseconds,
    freshnessBarrierRan: metrics.freshnessBarrierRan,
    unattributedMilliseconds: Math.max(elapsedMilliseconds - attributedMilliseconds, 0),
    totalDocumentCount: input.completion.totalDocumentCount,
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
    if (
      record.operationKind === RecallDiagnosticOperationKind.FULL_INDEX ||
      record.operationKind === RecallDiagnosticOperationKind.REBUILD
    ) {
      return true;
    }
    if (record.operationKind === RecallDiagnosticOperationKind.PHYSICAL_SESSION_CHECK) {
      return (
        record.status === RecallDiagnosticStatus.FAILED ||
        (record.elapsedMilliseconds ?? 0) >= SLOW_RECALL_DIAGNOSTIC_THRESHOLD_MILLISECONDS
      );
    }
    if (record.status === RecallDiagnosticStatus.STARTED) {
      return false;
    }
    return (
      record.status === RecallDiagnosticStatus.FAILED ||
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
        operationId: createRecallDiagnosticOperationId(),
        parentOperationId: null,
        operationKind: RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
        lifecycleTrigger: input.lifecycleTrigger,
        manualMaintenanceTrigger: null,
        sessionPath: input.sessionPath,
        searchMode: null,
        recallScope: null,
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
    startRecallSearch(input) {
      const startedAtMilliseconds = clock.monotonicMilliseconds();
      const startRecord = createRecallDiagnosticStartRecord({
        clock,
        operationId: createRecallDiagnosticOperationId(),
        parentOperationId: null,
        operationKind: RecallDiagnosticOperationKind.SEARCH,
        lifecycleTrigger: null,
        manualMaintenanceTrigger: null,
        sessionPath: null,
        searchMode: input.searchMode,
        recallScope: input.recallScope,
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
            createRecallSearchDiagnosticCompletionRecord({
              startRecord,
              startedAtMilliseconds,
              clock,
              completion,
            }),
          );
        },
      };
    },
    startManualIndexMaintenance(input) {
      const startedAtMilliseconds = clock.monotonicMilliseconds();
      const operationKind =
        input.manualMaintenanceTrigger === RecallManualMaintenanceTrigger.MANUAL_REBUILD
          ? RecallDiagnosticOperationKind.REBUILD
          : RecallDiagnosticOperationKind.FULL_INDEX;
      const startRecord = createRecallDiagnosticStartRecord({
        clock,
        operationId: createRecallDiagnosticOperationId(),
        parentOperationId: null,
        operationKind,
        lifecycleTrigger: null,
        manualMaintenanceTrigger: input.manualMaintenanceTrigger,
        sessionPath: null,
        searchMode: null,
        recallScope: null,
      });
      queueDiagnosticRecord(startRecord);
      let completed = false;
      return {
        recordPhysicalSessionCheck(completion) {
          queueDiagnosticRecord(
            createRecallPhysicalSessionCompletionRecord({
              parentStartRecord: startRecord,
              clock,
              completion,
            }),
          );
        },
        startOptimization() {
          const optimizationStartedAtMilliseconds = clock.monotonicMilliseconds();
          const optimizationStartRecord = createRecallDiagnosticStartRecord({
            clock,
            operationId: createRecallDiagnosticOperationId(),
            parentOperationId: startRecord.operationId,
            operationKind: RecallDiagnosticOperationKind.OPTIMIZATION,
            lifecycleTrigger: null,
            manualMaintenanceTrigger: startRecord.manualMaintenanceTrigger,
            sessionPath: null,
            searchMode: null,
            recallScope: null,
          });
          queueDiagnosticRecord(optimizationStartRecord);
          let optimizationCompleted = false;
          return {
            complete(input) {
              if (optimizationCompleted) {
                return;
              }
              optimizationCompleted = true;
              queueDiagnosticRecord(
                createRecallOptimizationCompletionRecord({
                  startRecord: optimizationStartRecord,
                  startedAtMilliseconds: optimizationStartedAtMilliseconds,
                  clock,
                  status: input.status,
                  ...(input.errorCategory ? { errorCategory: input.errorCategory } : {}),
                }),
              );
            },
          };
        },
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

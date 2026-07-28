import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallDiagnosticsMode,
  RecallManualMaintenanceTrigger,
  RecallSearchScope,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  createRecallIncrementalDiagnosticMetrics,
  createRecallIndexMetrics,
  createRecallOperationDiagnostics,
  createRecallSearchDiagnosticMetrics,
  readRecallOperationDiagnosticRecords,
  type RecallIndexDiagnosticCompletion,
} from './recall-operation-diagnostics.js';

function completeTestDiagnosticOperation(
  operation: { complete(completion: RecallIndexDiagnosticCompletion): void },
  status: RecallDiagnosticStatus,
  errorCategory?: RecallDiagnosticErrorCategory,
): void {
  operation.complete({
    status,
    metrics: createRecallIndexMetrics(),
    scannedSessionCount: 1,
    indexedSessionCount: 0,
    removedSessionCount: 0,
    failedSessionCount: status === RecallDiagnosticStatus.FAILED ? 1 : 0,
    cacheHitCount: 0,
    newEmbeddingCount: 0,
    embeddingRequestCount: 0,
    deletedDocumentCount: 0,
    totalDocumentCount: null,
    ...(errorCategory ? { errorCategory } : {}),
  });
}

void test('incremental diagnostics persist versioned scalar worker and write-window evidence', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-incremental-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.ALL,
    activeLogPath,
    retainedLogPath: join(directory, 'diagnostics.previous.jsonl'),
    notifyWarning() {
      assert.fail('successful incremental diagnostics must not warn');
    },
  });
  const metrics = createRecallIncrementalDiagnosticMetrics();
  Object.assign(metrics, {
    markerAgeMilliseconds: 7,
    metadataSweepScannedFileCount: 10_000,
    metadataSweepObservedSessionCount: 9_000,
    metadataSweepElapsedMilliseconds: 450,
    appendedByteCount: 4_096,
    parsedEntryCount: 12,
    eligibleDocumentCount: 8,
    tokenizerMilliseconds: 5,
    embeddingCacheHitCount: 6,
    embeddingCacheMissCount: 2,
    embeddingRequestCount: 1,
    lockWaitMilliseconds: 3,
    evidenceOpenMilliseconds: 4,
    evidenceWriteMilliseconds: 5,
    projectionOpenMilliseconds: 6,
    projectionCommitMilliseconds: 7,
    closeMilliseconds: 8,
    checkpointObservationMilliseconds: 9,
    markerAcknowledgementMilliseconds: 10,
    generationId: 'generation_acceptance',
    generationState: 'active',
    recoveryCategory: 'writer_reopen',
    deletionSafeguardCategory: 'mass_loss_suppressed',
    backlogPendingEligibleSessionCount: 11,
    backlogOldestEligibleMarkerAgeMilliseconds: 12,
    backlogFailureCategory: 'write_failed',
  });
  diagnostics.recordIncrementalOperation({
    operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics,
  });
  await diagnostics.flush();

  const records = await readRecallOperationDiagnosticRecords(activeLogPath);
  assert.equal(records.length, 1);
  assert.deepEqual(
    Object.fromEntries(
      Object.keys(metrics).map((field) => [field, Reflect.get(records[0] ?? {}, field)]),
    ),
    metrics,
  );
  assert.equal(records[0]?.version, 3);

  await writeFile(
    activeLogPath,
    `${JSON.stringify({ version: 2, operationKind: 'search', status: 'succeeded' })}\n`,
  );
  const legacyRecords = await readRecallOperationDiagnosticRecords(activeLogPath);
  assert.equal(legacyRecords[0]?.version, 2);
});

void test('slow diagnostics omit fast cancelled physical session checks', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-slow-cancelled-physical-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.SLOW,
    activeLogPath,
    retainedLogPath: join(directory, 'diagnostics.previous.jsonl'),
    clock: {
      monotonicMilliseconds: () => 1,
      wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
    },
    notifyWarning() {
      assert.fail('successful cancelled physical filtering must not warn');
    },
  });
  const manualOperation = diagnostics.startManualIndexMaintenance({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  });
  manualOperation.recordPhysicalSessionCheck({
    sessionPath: '/sessions/cancelled.jsonl',
    status: RecallDiagnosticStatus.CANCELLED,
    errorCategory: RecallDiagnosticErrorCategory.OPERATION_CANCELLED,
    metrics: createRecallIndexMetrics(),
    elapsedMilliseconds: 1,
    indexedSessionCount: 0,
    removedSessionCount: 0,
    failedSessionCount: 0,
  });
  completeTestDiagnosticOperation(manualOperation, RecallDiagnosticStatus.SUCCEEDED);
  await diagnostics.flush();

  const records = (await readFile(activeLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.ok(
    records.every((record) => record.operationKind === RecallDiagnosticOperationKind.FULL_INDEX),
  );
});

void test('diagnostic persistence rotates at its cap and retains one predecessor', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-rotating-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  let monotonicMilliseconds = 0;
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const retainedLogPath = join(directory, 'diagnostics.previous.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.ALL,
    activeLogPath,
    retainedLogPath,
    maximumLogBytes: 1_500,
    clock: {
      monotonicMilliseconds: () => monotonicMilliseconds,
      wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
    },
    notifyWarning() {
      assert.fail('successful diagnostic rotation must not warn');
    },
  });

  for (let index = 0; index < 4; index += 1) {
    const operation = diagnostics.startRecallSearch({
      searchMode: 'hybrid',
      recallScope: RecallSearchScope.GLOBAL,
    });
    monotonicMilliseconds += 1;
    operation.complete({
      status: RecallDiagnosticStatus.SUCCEEDED,
      metrics: createRecallSearchDiagnosticMetrics(),
      totalDocumentCount: index,
    });
  }
  await diagnostics.flush();

  const activeRecords = (await readFile(activeLogPath, 'utf8')).trimEnd().split('\n');
  const retainedRecords = (await readFile(retainedLogPath, 'utf8')).trimEnd().split('\n');
  assert.ok(activeRecords.length >= 1);
  assert.ok(retainedRecords.length >= 1);
  assert.ok([...activeRecords, ...retainedRecords].every((line) => line.startsWith('{')));
  assert.deepEqual((await readdir(directory)).sort(), [
    'diagnostics.jsonl',
    'diagnostics.previous.jsonl',
  ]);
});

void test('off diagnostics mode performs no diagnostic filesystem operations', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-off-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filesystemBlocker = join(directory, 'not-a-directory');
  await writeFile(filesystemBlocker, 'unchanged');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath: join(filesystemBlocker, 'diagnostics.jsonl'),
    retainedLogPath: join(filesystemBlocker, 'diagnostics.previous.jsonl'),
    notifyWarning() {
      assert.fail('off mode must not reach diagnostic failure handling');
    },
  });

  const searchOperation = diagnostics.startRecallSearch({
    searchMode: 'hybrid',
    recallScope: RecallSearchScope.GLOBAL,
  });
  searchOperation.complete({
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics: createRecallSearchDiagnosticMetrics(),
    totalDocumentCount: 1,
  });
  const manualOperation = diagnostics.startManualIndexMaintenance({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  });
  const physicalSessionMetrics = createRecallIndexMetrics();
  manualOperation.recordPhysicalSessionCheck({
    sessionPath: '/sessions/off-manual.jsonl',
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics: physicalSessionMetrics,
    elapsedMilliseconds: 1_000,
    indexedSessionCount: 0,
    removedSessionCount: 0,
    failedSessionCount: 0,
  });
  completeTestDiagnosticOperation(manualOperation, RecallDiagnosticStatus.SUCCEEDED);
  await diagnostics.flush();

  assert.equal(await readFile(filesystemBlocker, 'utf8'), 'unchanged');
});

void test('off diagnostics mode still persists mandatory detached-worker failures', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-off-durable-worker-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'incremental-diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath,
    retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
    notifyWarning() {
      assert.fail('successful mandatory failure persistence must not warn');
    },
  });
  diagnostics.recordDurableIncrementalFailure({
    operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
    status: RecallDiagnosticStatus.FAILED,
    metrics: createRecallIncrementalDiagnosticMetrics(),
    errorCategory: RecallDiagnosticErrorCategory.OPERATION_FAILED,
  });
  await diagnostics.flush();

  const [record] = await readRecallOperationDiagnosticRecords(activeLogPath);
  assert.equal(record?.operationKind, RecallDiagnosticOperationKind.INCREMENTAL_WORKER);
  assert.equal(record?.status, RecallDiagnosticStatus.FAILED);
});

void test('diagnostic persistence failure warns once and disables later writes', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-failed-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const filesystemBlocker = join(directory, 'not-a-directory');
  await writeFile(filesystemBlocker, 'unchanged');
  const warnings: string[] = [];

  for (let index = 0; index < 2; index += 1) {
    const diagnostics = createRecallOperationDiagnostics({
      mode: RecallDiagnosticsMode.ALL,
      activeLogPath: join(filesystemBlocker, 'diagnostics.jsonl'),
      retainedLogPath: join(filesystemBlocker, 'diagnostics.previous.jsonl'),
      notifyWarning(message) {
        warnings.push(message);
        throw new Error('warning delivery failed');
      },
    });
    const operation = diagnostics.startRecallSearch({
      searchMode: 'hybrid',
      recallScope: RecallSearchScope.GLOBAL,
    });
    operation.complete({
      status: RecallDiagnosticStatus.SUCCEEDED,
      metrics: createRecallSearchDiagnosticMetrics(),
      totalDocumentCount: index,
    });
    await diagnostics.flush();
  }

  assert.deepEqual(warnings, [
    'Recall diagnostics disabled after local log and fallback persistence failed.',
  ]);
  assert.equal(await readFile(filesystemBlocker, 'utf8'), 'unchanged');
});

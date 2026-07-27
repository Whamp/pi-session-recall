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
  RecallLifecycleTrigger,
  RecallManualMaintenanceTrigger,
  RecallSearchScope,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  createRecallIndexMetrics,
  createRecallOperationDiagnostics,
  createRecallSearchDiagnosticMetrics,
  type RecallDiagnosticsClock,
  type RecallLiveSessionDiagnosticOperation,
} from './recall-operation-diagnostics.js';

function readTestDiagnosticString(
  record: Record<string, unknown> | undefined,
  propertyName: string,
): string {
  const value = record?.[propertyName];
  if (typeof value !== 'string') {
    assert.fail(`Expected diagnostic ${propertyName} to be a string`);
  }
  return value;
}

function completeTestDiagnosticOperation(
  operation: RecallLiveSessionDiagnosticOperation,
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

void test('all diagnostics mode writes live reconciliation start and completion records', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-operation-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  let monotonicMilliseconds = 100;
  let wallClockSequence = 0;
  const clock: RecallDiagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => `2026-07-27T10:00:0${wallClockSequence++}.000Z`,
  };
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.ALL,
    activeLogPath,
    retainedLogPath: join(directory, 'diagnostics.previous.jsonl'),
    clock,
    notifyWarning() {
      assert.fail('successful diagnostic writes must not warn');
    },
  });
  const operation = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    sessionPath: '/sessions/active.jsonl',
  });
  const metrics = createRecallIndexMetrics();
  metrics.sourceByteSize = 512;
  metrics.changed = true;
  metrics.skipped = false;
  metrics.physicalSessionPreparationMilliseconds = 4;
  metrics.embeddingCacheResolutionMilliseconds = 5;
  metrics.embeddingServerRequestMilliseconds = 6;
  metrics.databaseWriteMilliseconds = 3;
  metrics.indexStateCheckpointMilliseconds = 2;
  metrics.upsertedDocumentCount = 7;
  monotonicMilliseconds = 125;
  operation.complete({
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics,
    scannedSessionCount: 1,
    indexedSessionCount: 1,
    removedSessionCount: 0,
    failedSessionCount: 0,
    cacheHitCount: 2,
    newEmbeddingCount: 5,
    embeddingRequestCount: 1,
    deletedDocumentCount: 0,
    totalDocumentCount: 7,
  });
  await diagnostics.flush();

  const records = (await readFile(activeLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.equal(records[0]?.status, RecallDiagnosticStatus.STARTED);
  assert.equal(records[1]?.status, RecallDiagnosticStatus.SUCCEEDED);
  const operationId = readTestDiagnosticString(records[0], 'operationId');
  assert.equal(operationId, records[1]?.operationId);
  assert.match(operationId, /^[0-9a-f-]{36}$/u);
  assert.equal(
    records[1]?.operationKind,
    RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
  );
  assert.equal(records[1]?.lifecycleTrigger, RecallLifecycleTrigger.AGENT_SETTLED);
  assert.equal(records[1]?.processId, process.pid);
  assert.equal(records[1]?.sessionPath, '/sessions/active.jsonl');
  assert.equal(records[1]?.sourceByteSize, 512);
  assert.equal(records[1]?.changed, true);
  assert.equal(records[1]?.skipped, false);
  assert.equal(records[1]?.elapsedMilliseconds, 25);
  assert.equal(records[1]?.physicalSessionPreparationMilliseconds, 4);
  assert.equal(records[1]?.embeddingCacheResolutionMilliseconds, 5);
  assert.equal(records[1]?.embeddingServerRequestMilliseconds, 6);
  assert.equal(records[1]?.databaseWriteMilliseconds, 3);
  assert.equal(records[1]?.indexStateCheckpointMilliseconds, 2);
  assert.equal(records[1]?.unattributedMilliseconds, 5);
  assert.equal(records[1]?.upsertedDocumentCount, 7);
  assert.equal(records[1]?.totalDocumentCount, 7);
});

void test('slow diagnostics mode retains the threshold boundary and failures only', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-slow-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  let monotonicMilliseconds = 0;
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.SLOW,
    activeLogPath,
    retainedLogPath: join(directory, 'diagnostics.previous.jsonl'),
    clock: {
      monotonicMilliseconds: () => monotonicMilliseconds,
      wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
    },
    notifyWarning() {
      assert.fail('successful diagnostic writes must not warn');
    },
  });

  const fast = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    sessionPath: '/sessions/fast.jsonl',
  });
  monotonicMilliseconds = 999;
  completeTestDiagnosticOperation(fast, RecallDiagnosticStatus.SUCCEEDED);
  await diagnostics.flush();
  await assert.rejects(() => readFile(activeLogPath), { code: 'ENOENT' });

  const threshold = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.SESSION_SHUTDOWN,
    sessionPath: '/sessions/threshold.jsonl',
  });
  monotonicMilliseconds = 1_999;
  completeTestDiagnosticOperation(threshold, RecallDiagnosticStatus.SUCCEEDED);
  await diagnostics.flush();

  const failed = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    sessionPath: '/sessions/failed.jsonl',
  });
  monotonicMilliseconds = 2_000;
  completeTestDiagnosticOperation(
    failed,
    RecallDiagnosticStatus.FAILED,
    RecallDiagnosticErrorCategory.OPERATION_FAILED,
  );
  await diagnostics.flush();

  const records = (await readFile(activeLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.deepEqual(
    records.map((record) => ({
      sessionPath: record.sessionPath,
      status: record.status,
      elapsedMilliseconds: record.elapsedMilliseconds,
    })),
    [
      {
        sessionPath: '/sessions/threshold.jsonl',
        status: RecallDiagnosticStatus.SUCCEEDED,
        elapsedMilliseconds: 1_000,
      },
      {
        sessionPath: '/sessions/failed.jsonl',
        status: RecallDiagnosticStatus.FAILED,
        elapsedMilliseconds: 1,
      },
    ],
  );
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
    const operation = diagnostics.startLiveSessionReconciliation({
      lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
      sessionPath: `/sessions/rotation-${index}.jsonl`,
    });
    monotonicMilliseconds += 1;
    completeTestDiagnosticOperation(operation, RecallDiagnosticStatus.SUCCEEDED);
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

void test('diagnostic operation IDs and records remain bounded for an oversized session path', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-bounded-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'diagnostics.jsonl');
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.ALL,
    activeLogPath,
    retainedLogPath: join(directory, 'diagnostics.previous.jsonl'),
    notifyWarning() {
      assert.fail('bounded diagnostic writes must not warn');
    },
  });
  const operation = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    sessionPath: `/sessions/${'oversized-private-path'.repeat(100_000)}.jsonl`,
  });
  completeTestDiagnosticOperation(operation, RecallDiagnosticStatus.SUCCEEDED);
  await diagnostics.flush();

  const lines = (await readFile(activeLogPath, 'utf8')).trimEnd().split('\n');
  assert.equal(lines.length, 2);
  assert.ok(lines.every((line) => Buffer.byteLength(line) < 8 * 1_024));
  const records = lines.map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(readTestDiagnosticString(records[0], 'operationId').length, 36);
  assert.equal(readTestDiagnosticString(records[1], 'operationId').length, 36);
  assert.equal(readTestDiagnosticString(records[1], 'sessionPath').length, 4_096);
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

  const operation = diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    sessionPath: '/sessions/off.jsonl',
  });
  completeTestDiagnosticOperation(operation, RecallDiagnosticStatus.SUCCEEDED);
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
    const operation = diagnostics.startLiveSessionReconciliation({
      lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
      sessionPath: `/sessions/failure-${index}.jsonl`,
    });
    completeTestDiagnosticOperation(operation, RecallDiagnosticStatus.SUCCEEDED);
    await diagnostics.flush();
  }

  assert.deepEqual(warnings, [
    'Recall diagnostics disabled after local log persistence failed; recall behavior is unchanged.',
  ]);
  assert.equal(await readFile(filesystemBlocker, 'utf8'), 'unchanged');
});

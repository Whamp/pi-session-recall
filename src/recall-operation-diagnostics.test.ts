import assert from 'node:assert/strict';
import {
  appendFile,
  mkdir,
  mkdtemp,
  open,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from 'node:fs/promises';
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
  type RecallOperationDiagnosticsFilesystem,
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

type RecallDiagnosticFilesystemFault =
  | 'file-open'
  | 'file-write'
  | 'file-sync'
  | 'file-close'
  | 'directory-sync';

interface FaultingRecallDiagnosticFilesystemOptions {
  events: string[];
  fault?: RecallDiagnosticFilesystemFault;
  reportCreated?: boolean;
}

function createFaultingRecallDiagnosticFilesystem(
  options: FaultingRecallDiagnosticFilesystemOptions,
): RecallOperationDiagnosticsFilesystem {
  return {
    async createDirectory(path) {
      await mkdir(path, { recursive: true });
    },
    async getFileSize(path) {
      return (await stat(path)).size;
    },
    async removeFile(path) {
      options.events.push('remove-file');
      await rm(path, { force: true });
    },
    async renameFile(sourcePath, destinationPath) {
      options.events.push('rename-file');
      await rename(sourcePath, destinationPath);
    },
    async appendFile(path, content) {
      options.events.push('ordinary-append');
      await appendFile(path, content, 'utf8');
    },
    async openAppendFile(path) {
      options.events.push('file-open');
      if (options.fault === 'file-open') {
        throw new Error('injected diagnostic file open failure');
      }
      const openedFile = await (async () => {
        try {
          return { file: await open(path, 'ax', 0o600), created: true };
        } catch (error) {
          if (!isUnknownRecord(error) || error.code !== 'EEXIST') {
            throw error;
          }
          return { file: await open(path, 'a', 0o600), created: false };
        }
      })();
      return {
        created: options.reportCreated ?? openedFile.created,
        file: {
          async appendFile(content) {
            options.events.push('file-write');
            if (options.fault === 'file-write') {
              throw new Error('injected diagnostic file write failure');
            }
            await openedFile.file.appendFile(content, 'utf8');
          },
          async sync() {
            options.events.push('file-sync');
            if (options.fault === 'file-sync') {
              throw new Error('injected diagnostic file sync failure');
            }
            await openedFile.file.sync();
          },
          async close() {
            options.events.push('file-close');
            await openedFile.file.close();
            if (options.fault === 'file-close') {
              throw new Error('injected diagnostic file close failure');
            }
          },
        },
      };
    },
    async syncDirectory(path) {
      options.events.push('directory-sync');
      if (options.fault === 'directory-sync') {
        throw new Error('injected diagnostic directory sync failure');
      }
      const directory = await open(path, 'r');
      try {
        await directory.sync();
      } finally {
        await directory.close();
      }
    },
  };
}

function recordTestDurableIncrementalFailure(
  diagnostics: ReturnType<typeof createRecallOperationDiagnostics>,
): void {
  diagnostics.recordDurableIncrementalFailure({
    operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
    status: RecallDiagnosticStatus.FAILED,
    metrics: createRecallIncrementalDiagnosticMetrics(),
    errorCategory: RecallDiagnosticErrorCategory.OPERATION_FAILED,
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

void test('durable diagnostic file sync failure uses the scalar backlog fallback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-diagnostic-file-sync-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'incremental-diagnostics.jsonl');
  const events: string[] = [];
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath,
    retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
    persistenceState: { disabled: false, persistenceFailureHandled: false },
    filesystem: createFaultingRecallDiagnosticFilesystem({
      events,
      fault: 'file-sync',
    }),
    async onPersistenceFailure() {
      events.push('backlog-fallback');
    },
    notifyWarning() {
      assert.fail('successful scalar backlog fallback must suppress warning');
    },
  });
  recordTestDurableIncrementalFailure(diagnostics);
  await diagnostics.flush();

  assert.deepEqual(events, [
    'file-open',
    'file-write',
    'file-sync',
    'file-close',
    'backlog-fallback',
  ]);
});

void test('durable diagnostic open, write, and close failures use the scalar backlog fallback', async (t) => {
  for (const fault of ['file-open', 'file-write', 'file-close'] as const) {
    const directory = await mkdtemp(join(tmpdir(), `recall-diagnostic-${fault}-failure-`));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const events: string[] = [];
    const diagnostics = createRecallOperationDiagnostics({
      mode: RecallDiagnosticsMode.OFF,
      activeLogPath: join(directory, 'incremental-diagnostics.jsonl'),
      retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
      persistenceState: { disabled: false, persistenceFailureHandled: false },
      filesystem: createFaultingRecallDiagnosticFilesystem({ events, fault }),
      async onPersistenceFailure() {
        events.push('backlog-fallback');
      },
      notifyWarning() {
        assert.fail('successful scalar backlog fallback must suppress warning');
      },
    });
    recordTestDurableIncrementalFailure(diagnostics);
    await diagnostics.flush();

    const expectedEvents =
      fault === 'file-open'
        ? ['file-open', 'backlog-fallback']
        : fault === 'file-write'
          ? ['file-open', 'file-write', 'file-close', 'backlog-fallback']
          : ['file-open', 'file-write', 'file-sync', 'file-close', 'backlog-fallback'];
    assert.deepEqual(events, expectedEvents);
  }
});

void test('durable diagnostic creation sync failure uses the scalar backlog fallback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-diagnostic-create-sync-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const events: string[] = [];
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath: join(directory, 'incremental-diagnostics.jsonl'),
    retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
    persistenceState: { disabled: false, persistenceFailureHandled: false },
    filesystem: createFaultingRecallDiagnosticFilesystem({
      events,
      fault: 'directory-sync',
    }),
    async onPersistenceFailure() {
      events.push('backlog-fallback');
    },
    notifyWarning() {
      assert.fail('successful scalar backlog fallback must suppress warning');
    },
  });
  recordTestDurableIncrementalFailure(diagnostics);
  await diagnostics.flush();

  assert.deepEqual(events, [
    'file-open',
    'file-write',
    'file-sync',
    'file-close',
    'directory-sync',
    'backlog-fallback',
  ]);
});

void test('durable diagnostic rotation sync failure uses the scalar backlog fallback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-diagnostic-rotation-sync-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'incremental-diagnostics.jsonl');
  const retainedLogPath = join(directory, 'incremental-diagnostics.previous.jsonl');
  await writeFile(activeLogPath, 'retained predecessor');
  const events: string[] = [];
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath,
    retainedLogPath,
    maximumLogBytes: 1,
    persistenceState: { disabled: false, persistenceFailureHandled: false },
    filesystem: createFaultingRecallDiagnosticFilesystem({
      events,
      fault: 'directory-sync',
      reportCreated: false,
    }),
    async onPersistenceFailure() {
      events.push('backlog-fallback');
    },
    notifyWarning() {
      assert.fail('successful scalar backlog fallback must suppress warning');
    },
  });
  recordTestDurableIncrementalFailure(diagnostics);
  await diagnostics.flush();

  assert.deepEqual(events, [
    'remove-file',
    'rename-file',
    'file-open',
    'file-write',
    'file-sync',
    'file-close',
    'directory-sync',
    'backlog-fallback',
  ]);
  assert.equal(await readFile(retainedLogPath, 'utf8'), 'retained predecessor');
});

void test('durable diagnostics warn once when local and scalar backlog persistence both fail', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-diagnostic-dual-sink-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const events: string[] = [];
  const persistenceState = { disabled: false, persistenceFailureHandled: false };
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.OFF,
    activeLogPath: join(directory, 'incremental-diagnostics.jsonl'),
    retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
    persistenceState,
    filesystem: createFaultingRecallDiagnosticFilesystem({
      events,
      fault: 'file-sync',
    }),
    async onPersistenceFailure() {
      events.push('backlog-fallback');
      throw new Error('injected scalar backlog fallback failure');
    },
    notifyWarning(message) {
      events.push(message);
      throw new Error('injected warning delivery failure');
    },
  });
  recordTestDurableIncrementalFailure(diagnostics);
  recordTestDurableIncrementalFailure(diagnostics);
  await diagnostics.flush();

  assert.deepEqual(events, [
    'file-open',
    'file-write',
    'file-sync',
    'file-close',
    'backlog-fallback',
    'Recall diagnostics disabled after local log and fallback persistence failed.',
  ]);
  assert.deepEqual(persistenceState, { disabled: true, persistenceFailureHandled: true });
});

void test('ordinary diagnostic timing records retain the unsynced append path', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-diagnostic-ordinary-append-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeLogPath = join(directory, 'incremental-diagnostics.jsonl');
  const events: string[] = [];
  const diagnostics = createRecallOperationDiagnostics({
    mode: RecallDiagnosticsMode.ALL,
    activeLogPath,
    retainedLogPath: join(directory, 'incremental-diagnostics.previous.jsonl'),
    persistenceState: { disabled: false, persistenceFailureHandled: false },
    filesystem: createFaultingRecallDiagnosticFilesystem({
      events,
      fault: 'file-sync',
    }),
    notifyWarning() {
      assert.fail('ordinary append must not reach durable file sync');
    },
  });
  diagnostics.recordIncrementalOperation({
    operationKind: RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
    status: RecallDiagnosticStatus.SUCCEEDED,
    metrics: createRecallIncrementalDiagnosticMetrics(),
  });
  await diagnostics.flush();

  assert.deepEqual(events, ['ordinary-append']);
  const [record] = await readRecallOperationDiagnosticRecords(activeLogPath);
  assert.equal(record?.status, RecallDiagnosticStatus.SUCCEEDED);
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

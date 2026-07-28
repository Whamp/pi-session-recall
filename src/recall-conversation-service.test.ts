import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import {
  RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallLifecycleTrigger,
  RecallManualMaintenanceTrigger,
  RecallProjectIdentitySource,
  RecallSearchScope,
} from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import {
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  readRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';
import {
  createRecallConversationService as createProductionRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import { createRecallOperationDiagnostics } from './recall-operation-diagnostics.js';
import {
  normalizeRecallProjectLineages,
  parseRepositoryIdentity,
} from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const EXEC_FILE_ASYNC = promisify(execFile);

const rawProjectLineageInput: Readonly<Record<string, readonly string[]>> = {};
// @ts-expect-error Recall service configuration requires validated project lineage identities.
const invalidServiceProjectLineages: RecallConversationConfig['projectLineages'] =
  rawProjectLineageInput;
void invalidServiceProjectLineages;

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
    };
  },
};

function createTestConfig(directory: string, sessionsDirectory: string) {
  return {
    sessionsDirectory,
    dataDirectory: directory,
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    embeddingCacheDirectory: join(directory, 'embedding-cache'),
    lockPath: join(directory, 'recall.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(directory, 'markers', 'pending'),
    markerQuarantineDirectory: join(directory, 'markers', 'quarantine'),
    markerControlDirectory: join(directory, 'markers', 'control'),
    workerOwnershipLockPath: join(directory, 'incremental-worker.lock'),
    generationRootDirectory: join(directory, 'generations'),
    activeGenerationPointerPath: join(directory, 'active-generation.json'),
    generationRegistryPath: join(directory, 'generation-registry.json'),
    backlogSummaryPath: join(directory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(directory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-request-model',
    embeddingServedModelId: 'test-served-model',
    embeddingArtifact: 'test-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker-model',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 1, lexical: 1, identifier: 1 },
  };
}

const PRESERVE_FUSION_ORDER_RERANKER: LocalRerankerClient = {
  async rerankDocuments(query, documents) {
    void query;
    const scores: number[] = [];
    for (let index = 0; index < documents.length; index += 1) {
      scores.push(1 - index / (documents.length + 1));
    }
    return scores;
  },
};

function createRecallConversationService(
  config: Parameters<typeof createProductionRecallConversationService>[0],
  dependencies: NonNullable<Parameters<typeof createProductionRecallConversationService>[1]>,
) {
  return createProductionRecallConversationService(config, {
    reranker: PRESERVE_FUSION_ORDER_RERANKER,
    ...dependencies,
  });
}

function readTestDiagnosticNumber(
  record: Record<string, unknown> | undefined,
  propertyName: string,
): number {
  const value = record?.[propertyName];
  if (typeof value !== 'number') {
    assert.fail(`Expected diagnostic ${propertyName} to be a number`);
  }
  return value;
}

void test('slow diagnostics retain fast manual incremental index start and completion summaries', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-manual-index-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'manual-index.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'manual-index-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/manual-index',
      },
      {
        type: 'message',
        id: 'manual-index-entry',
        parentId: null,
        timestamp: '2026-07-27T10:00:01.000Z',
        message: { role: 'assistant', content: 'manual index diagnostic evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful manual index diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        monotonicMilliseconds += texts.includes(RECALL_EMBEDDING_CANARY_TEXT) ? 7 : 13;
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
  });

  const result = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
    optimize: true,
  });
  await diagnostics.flush();

  assert.equal(result.totalChunks, 1);
  assert.equal(result.indexSummary.scannedSessions, 1);
  assert.equal(result.indexSummary.indexedSessions, 1);
  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.deepEqual(
    records.map((record) => ({
      operationKind: record.operationKind,
      manualMaintenanceTrigger: record.manualMaintenanceTrigger,
      status: record.status,
      elapsedMilliseconds: record.elapsedMilliseconds,
    })),
    [
      {
        operationKind: 'full_index',
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
        status: RecallDiagnosticStatus.STARTED,
        elapsedMilliseconds: null,
      },
      {
        operationKind: 'full_index',
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
        status: RecallDiagnosticStatus.SUCCEEDED,
        elapsedMilliseconds: 20,
      },
    ],
  );
  assert.equal(records[1]?.manifestStorePreparationMilliseconds, 0);
  assert.equal(records[1]?.embeddingServerRequestMilliseconds, 20);
  assert.equal(records[1]?.scannedSessionCount, 1);
  assert.equal(records[1]?.indexedSessionCount, 1);
  assert.equal(records[1]?.totalDocumentCount, 1);
  await assert.rejects(
    () =>
      service.index({
        rebuild: true,
        // @ts-expect-error Runtime guard rejects contradictory manual maintenance modes.
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
      }),
    /Recall manual maintenance trigger mismatch.*manual_rebuild.*manual_incremental_index/u,
  );
  await assert.rejects(
    () =>
      // @ts-expect-error Runtime guard rejects a manual rebuild trigger without rebuild mode.
      service.index({
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
      }),
    /Recall manual maintenance trigger mismatch.*manual_incremental_index.*manual_rebuild/u,
  );
});

void test('all diagnostics record changed and unchanged physical session checks', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-physical-check-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'physical-check.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'physical-check-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/physical-check',
      },
      {
        type: 'message',
        id: 'physical-check-entry',
        parentId: null,
        timestamp: '2026-07-27T10:00:01.000Z',
        message: {
          role: 'assistant',
          content: Array.from({ length: 30 }, (value, index) => {
            void value;
            return `physical-check-token-${index}`;
          }).join(' '),
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const sourceByteSize = (await stat(sessionPath)).size;
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
    chunkPolicy: { id: '2-0', maxTokens: 2, overlapTokens: 0 },
  };
  const diagnosticsClock = {
    monotonicMilliseconds: () => 10,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful physical check diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
  });

  const changed = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
    optimize: true,
  });
  const unchanged = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
    optimize: true,
  });
  await rm(sessionPath);
  const removed = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
    optimize: true,
  });
  await diagnostics.flush();

  assert.equal(changed.indexSummary.indexedSessions, 1);
  assert.ok(changed.totalChunks > 1);
  assert.equal(unchanged.indexSummary.indexedSessions, 0);
  assert.equal(removed.indexSummary.removedSessions, 1);
  assert.equal(removed.indexSummary.deletedChunks, changed.totalChunks);
  assert.equal(removed.totalChunks, 0);
  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const physicalSessionRecords = records.filter(
    (record) => record.operationKind === 'physical_session_check',
  );
  assert.equal(records.length, 13);
  assert.equal(physicalSessionRecords.length, 3);
  const optimizationRecords = records.filter((record) => record.operationKind === 'optimization');
  assert.deepEqual(
    optimizationRecords.map((record) => record.status),
    [
      RecallDiagnosticStatus.STARTED,
      RecallDiagnosticStatus.SUCCEEDED,
      RecallDiagnosticStatus.STARTED,
      RecallDiagnosticStatus.SUCCEEDED,
    ],
  );
  assert.ok(
    optimizationRecords.every((record) =>
      records.some(
        (parentRecord) =>
          parentRecord.operationId === record.parentOperationId &&
          parentRecord.operationKind === 'full_index',
      ),
    ),
  );
  const manualCompletionRecords = records.filter(
    (record) =>
      record.operationKind === 'full_index' && record.status === RecallDiagnosticStatus.SUCCEEDED,
  );
  assert.deepEqual(
    manualCompletionRecords.map((record) => ({
      optimizationRan: record.optimizationRan,
      optimizationMilliseconds: record.optimizationMilliseconds,
    })),
    [
      { optimizationRan: true, optimizationMilliseconds: 0 },
      { optimizationRan: false, optimizationMilliseconds: 0 },
      { optimizationRan: true, optimizationMilliseconds: 0 },
    ],
  );
  const changedPhysicalSessionRecord = physicalSessionRecords[0];
  assert.equal(changedPhysicalSessionRecord?.sessionPath, sessionPath);
  assert.equal(changedPhysicalSessionRecord?.sourceByteSize, sourceByteSize);
  assert.equal(changedPhysicalSessionRecord?.changed, true);
  assert.equal(changedPhysicalSessionRecord?.skipped, false);
  assert.equal(changedPhysicalSessionRecord?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(changedPhysicalSessionRecord?.elapsedMilliseconds, 0);
  assert.equal(changedPhysicalSessionRecord?.upsertedDocumentCount, changed.totalChunks);
  assert.equal(changedPhysicalSessionRecord?.cacheHitCount, 0);
  assert.ok(readTestDiagnosticNumber(changedPhysicalSessionRecord, 'newEmbeddingCount') > 1);
  assert.ok(readTestDiagnosticNumber(changedPhysicalSessionRecord, 'embeddingRequestCount') > 1);
  assert.deepEqual(
    {
      sessionPath: physicalSessionRecords[1]?.sessionPath,
      sourceByteSize: physicalSessionRecords[1]?.sourceByteSize,
      changed: physicalSessionRecords[1]?.changed,
      skipped: physicalSessionRecords[1]?.skipped,
      status: physicalSessionRecords[1]?.status,
      elapsedMilliseconds: physicalSessionRecords[1]?.elapsedMilliseconds,
      upsertedDocumentCount: physicalSessionRecords[1]?.upsertedDocumentCount,
      cacheHitCount: physicalSessionRecords[1]?.cacheHitCount,
      newEmbeddingCount: physicalSessionRecords[1]?.newEmbeddingCount,
      embeddingRequestCount: physicalSessionRecords[1]?.embeddingRequestCount,
    },
    {
      sessionPath,
      sourceByteSize,
      changed: false,
      skipped: true,
      status: RecallDiagnosticStatus.SUCCEEDED,
      elapsedMilliseconds: 0,
      upsertedDocumentCount: 0,
      cacheHitCount: 0,
      newEmbeddingCount: 0,
      embeddingRequestCount: 0,
    },
  );
  assert.deepEqual(
    {
      sessionPath: physicalSessionRecords[2]?.sessionPath,
      sourceByteSize: physicalSessionRecords[2]?.sourceByteSize,
      changed: physicalSessionRecords[2]?.changed,
      skipped: physicalSessionRecords[2]?.skipped,
      status: physicalSessionRecords[2]?.status,
      elapsedMilliseconds: physicalSessionRecords[2]?.elapsedMilliseconds,
      upsertedDocumentCount: physicalSessionRecords[2]?.upsertedDocumentCount,
      deletedDocumentCount: physicalSessionRecords[2]?.deletedDocumentCount,
      cacheHitCount: physicalSessionRecords[2]?.cacheHitCount,
      newEmbeddingCount: physicalSessionRecords[2]?.newEmbeddingCount,
      embeddingRequestCount: physicalSessionRecords[2]?.embeddingRequestCount,
    },
    {
      sessionPath,
      sourceByteSize: 0,
      changed: true,
      skipped: false,
      status: RecallDiagnosticStatus.SUCCEEDED,
      elapsedMilliseconds: 0,
      upsertedDocumentCount: 0,
      deletedDocumentCount: changed.totalChunks,
      cacheHitCount: 0,
      newEmbeddingCount: 0,
      embeddingRequestCount: 0,
    },
  );
});

void test('slow diagnostics retain only threshold physical session checks', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-slow-physical-checks-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const writePhysicalSession = async (filename: string, content: string) => {
    await writeFile(
      join(sessionsDirectory, filename),
      [
        {
          type: 'session',
          version: 3,
          id: `${filename}-session`,
          timestamp: '2026-07-27T10:00:00.000Z',
          cwd: '/workspace/slow-physical-checks',
        },
        {
          type: 'message',
          id: `${filename}-entry`,
          parentId: null,
          timestamp: '2026-07-27T10:00:01.000Z',
          message: { role: 'assistant', content },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );
  };
  await writePhysicalSession('a-fast.jsonl', 'fast physical session evidence');
  await writePhysicalSession('b-slow.jsonl', 'slow physical session evidence');
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful slow physical diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        if (texts.some((text) => text.includes('slow physical session evidence'))) {
          monotonicMilliseconds += 1_000;
        }
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
  });

  await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  });
  await diagnostics.flush();

  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const physicalSessionRecords = records.filter(
    (record) => record.operationKind === 'physical_session_check',
  );
  assert.equal(physicalSessionRecords.length, 1);
  assert.equal(physicalSessionRecords[0]?.sessionPath, join(sessionsDirectory, 'b-slow.jsonl'));
  assert.equal(physicalSessionRecords[0]?.elapsedMilliseconds, 1_000);
});

void test('manual index diagnostics report continued physical session parse failure', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-parse-failed-manual-index-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const privateParseSentinel = 'PRIVATE_PARSE_FAILURE_SENTINEL_26';
  const sessionPath = join(sessionsDirectory, 'parse-failure.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'parse-failure-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/parse-failure',
      },
      {
        type: 'message',
        id: 'parse-failure-entry',
        parentId: 'private-missing-parent-26',
        timestamp: '2026-07-27T10:00:01.000Z',
        message: { role: 'assistant', content: privateParseSentinel },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const sourceByteSize = (await stat(sessionPath)).size;
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    notifyWarning() {
      assert.fail('successful parse-failure diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    embeddings: {
      async embedTexts(texts) {
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
  });

  const result = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  });
  await diagnostics.flush();

  assert.equal(result.totalChunks, 0);
  assert.equal(result.indexSummary.scannedSessions, 1);
  assert.equal(result.indexSummary.indexedSessions, 0);
  assert.equal(result.indexSummary.failedSessions.length, 1);
  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  assert.doesNotMatch(diagnosticJsonl, new RegExp(privateParseSentinel, 'u'));
  assert.doesNotMatch(diagnosticJsonl, /private-missing-parent-26/u);
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const physicalRecord = records.find(
    (record) => record.operationKind === 'physical_session_check',
  );
  const completionRecord = records.find(
    (record) =>
      record.operationKind === 'full_index' && record.status === RecallDiagnosticStatus.FAILED,
  );
  assert.equal(physicalRecord?.sessionPath, sessionPath);
  assert.equal(physicalRecord?.sourceByteSize, sourceByteSize);
  assert.equal(physicalRecord?.status, RecallDiagnosticStatus.FAILED);
  assert.equal(completionRecord?.scannedSessionCount, 1);
  assert.equal(completionRecord?.failedSessionCount, 1);
});

void test('manual index diagnostics retain partial counts for fatal embedding failure', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-failed-manual-index-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const privateFailureSentinel = 'PRIVATE_FATAL_EMBEDDING_SENTINEL_26';
  await writeFile(
    join(sessionsDirectory, 'fatal-embedding.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'fatal-embedding-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/fatal-embedding',
      },
      {
        type: 'message',
        id: 'fatal-embedding-entry',
        parentId: null,
        timestamp: '2026-07-27T10:00:01.000Z',
        message: { role: 'assistant', content: privateFailureSentinel },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful failure diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        if (!texts.includes(RECALL_EMBEDDING_CANARY_TEXT)) {
          monotonicMilliseconds += 29;
          throw new Error('private fatal embedding model response sentinel 26');
        }
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
  });

  await assert.rejects(
    () =>
      service.index({
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
      }),
    /private fatal embedding model response sentinel 26/u,
  );
  await diagnostics.flush();

  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  assert.doesNotMatch(diagnosticJsonl, new RegExp(privateFailureSentinel, 'u'));
  assert.doesNotMatch(diagnosticJsonl, /private fatal embedding model response sentinel 26/u);
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const physicalRecord = records.find(
    (record) => record.operationKind === 'physical_session_check',
  );
  const completionRecord = records.find(
    (record) =>
      record.operationKind === 'full_index' && record.status === RecallDiagnosticStatus.FAILED,
  );
  assert.equal(physicalRecord?.status, RecallDiagnosticStatus.FAILED);
  assert.equal(physicalRecord?.elapsedMilliseconds, 29);
  assert.equal(completionRecord?.scannedSessionCount, 1);
  assert.equal(completionRecord?.indexedSessionCount, 0);
  assert.equal(completionRecord?.failedSessionCount, 1);
  assert.equal(completionRecord?.embeddingRequestCount, 1);
  assert.equal(completionRecord?.totalDocumentCount, null);
});

void test('manual rebuild diagnostics isolate final database optimization duration', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-rebuild-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'manual-rebuild.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'manual-rebuild-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/manual-rebuild',
      },
      {
        type: 'message',
        id: 'manual-rebuild-entry',
        parentId: null,
        timestamp: '2026-07-27T10:00:01.000Z',
        message: { role: 'assistant', content: 'manual rebuild diagnostic evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  let monotonicMilliseconds = 0;
  let lockDurationRecorded = false;
  let checkpointDurationRecorded = false;
  let scanTimingEnabled = false;
  let postStoreClockCallCount = 0;
  let storedDocumentCount = 0;
  let optimizationCallCount = 0;
  const diagnosticsClock = {
    monotonicMilliseconds() {
      if (!lockDurationRecorded && existsSync(config.lockPath)) {
        lockDurationRecorded = true;
        monotonicMilliseconds += 11;
      }
      if (!checkpointDurationRecorded && existsSync(config.statePath)) {
        checkpointDurationRecorded = true;
        monotonicMilliseconds += 19;
      }
      if (scanTimingEnabled) {
        postStoreClockCallCount += 1;
        if (postStoreClockCallCount === 3) {
          monotonicMilliseconds += 31;
          scanTimingEnabled = false;
        }
      }
      return monotonicMilliseconds;
    },
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful rebuild diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        monotonicMilliseconds += texts.includes(RECALL_EMBEDDING_CANARY_TEXT) ? 7 : 13;
        return texts.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      monotonicMilliseconds += 5;
      return tokenizer;
    },
    openStore() {
      monotonicMilliseconds += 3;
      scanTimingEnabled = true;
      return {
        upsertChunks(chunks) {
          monotonicMilliseconds += 17;
          storedDocumentCount += chunks.length;
        },
        deleteChunks(ids) {
          storedDocumentCount = Math.max(storedDocumentCount - ids.length, 0);
        },
        searchDenseCandidates() {
          return [];
        },
        searchLexicalCandidates() {
          return [];
        },
        searchIdentifierCandidates() {
          return [];
        },
        fetchConversationChunks() {
          return new Map();
        },
        fetchVectors() {
          return new Map();
        },
        groupDenseCandidates() {
          return [];
        },
        addColumn() {},
        alterColumn() {},
        createIndex() {},
        async optimize() {
          optimizationCallCount += 1;
          monotonicMilliseconds += 37;
        },
        close() {
          monotonicMilliseconds += 2;
        },
        count() {
          return storedDocumentCount;
        },
      };
    },
  });

  const result = await service.index({
    rebuild: true,
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
    optimize: true,
  });
  await diagnostics.flush();

  assert.equal(optimizationCallCount, 1);
  assert.equal(result.totalChunks, 1);
  assert.equal(result.indexSummary.indexedSessions, 1);
  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.equal(records[0]?.operationKind, 'rebuild');
  assert.equal(records[0]?.manualMaintenanceTrigger, RecallManualMaintenanceTrigger.MANUAL_REBUILD);
  assert.equal(records[0]?.status, RecallDiagnosticStatus.STARTED);
  assert.equal(records[1]?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(records[1]?.writerLockWaitMilliseconds, 11);
  assert.equal(records[1]?.manifestStorePreparationMilliseconds, 8);
  assert.equal(records[1]?.physicalSessionScanMilliseconds, 31);
  assert.equal(records[1]?.embeddingCacheResolutionMilliseconds, 0);
  assert.equal(records[1]?.embeddingServerRequestMilliseconds, 20);
  assert.equal(records[1]?.databaseWriteMilliseconds, 17);
  assert.equal(records[1]?.indexStateCheckpointMilliseconds, 19);
  assert.equal(records[1]?.optimizationRan, true);
  assert.equal(records[1]?.optimizationMilliseconds, 37);
  assert.equal(records[1]?.elapsedMilliseconds, 145);
  assert.equal(records[1]?.unattributedMilliseconds, 2);
});

void test('manual index diagnostics preserve optimization failure and release the writer lock', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-optimization-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'optimization-failure.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'optimization-failure-session',
        timestamp: '2026-07-27T10:00:00.000Z',
        cwd: '/workspace/optimization-failure',
      },
      {
        type: 'message',
        id: 'optimization-failure-entry',
        parentId: null,
        timestamp: '2026-07-27T10:00:01.000Z',
        message: { role: 'assistant', content: 'optimization failure source sentinel 26' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  let monotonicMilliseconds = 0;
  let storedDocumentCount = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:02.000Z',
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful optimization-failure diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        return texts.map(() => [1, 0, 0]);
      },
    },
    loadTokenizer: async () => tokenizer,
    openStore() {
      return {
        upsertChunks(chunks) {
          storedDocumentCount += chunks.length;
        },
        deleteChunks(ids) {
          storedDocumentCount = Math.max(storedDocumentCount - ids.length, 0);
        },
        searchDenseCandidates() {
          return [];
        },
        searchLexicalCandidates() {
          return [];
        },
        searchIdentifierCandidates() {
          return [];
        },
        fetchConversationChunks() {
          return new Map();
        },
        fetchVectors() {
          return new Map();
        },
        groupDenseCandidates() {
          return [];
        },
        addColumn() {},
        alterColumn() {},
        createIndex() {},
        async optimize() {
          monotonicMilliseconds += 43;
          throw new Error('private optimization model response sentinel 26');
        },
        close() {},
        count() {
          return storedDocumentCount;
        },
      };
    },
  });

  await assert.rejects(
    () =>
      service.index({
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
        optimize: true,
      }),
    /private optimization model response sentinel 26/u,
  );
  await diagnostics.flush();

  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  assert.doesNotMatch(diagnosticJsonl, /optimization failure source sentinel 26/u);
  assert.doesNotMatch(diagnosticJsonl, /private optimization model response sentinel 26/u);
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const completionRecord = records.find(
    (record) =>
      record.operationKind === 'full_index' && record.status === RecallDiagnosticStatus.FAILED,
  );
  const optimizationCompletionRecord = records.find(
    (record) =>
      record.operationKind === 'optimization' && record.status === RecallDiagnosticStatus.FAILED,
  );
  assert.equal(optimizationCompletionRecord?.optimizationRan, true);
  assert.equal(optimizationCompletionRecord?.optimizationMilliseconds, 43);
  assert.equal(optimizationCompletionRecord?.parentOperationId, completionRecord?.operationId);
  assert.equal(completionRecord?.scannedSessionCount, 1);
  assert.equal(completionRecord?.indexedSessionCount, 1);
  assert.equal(completionRecord?.failedSessionCount, 0);
  assert.equal(completionRecord?.optimizationRan, true);
  assert.equal(completionRecord?.optimizationMilliseconds, 43);
  assert.equal(completionRecord?.totalDocumentCount, null);
});

void test('live session reconciliation records private-safe costs and keeps changed evidence searchable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-live-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  const privateSentinel = 'PRIVATE_LIVE_RECONCILIATION_SENTINEL_24';
  const privateToolArgumentSentinel = 'PRIVATE_TOOL_ARGUMENT_SENTINEL_24';
  const privateToolResultSentinel = 'PRIVATE_TOOL_RESULT_SENTINEL_24';
  const privateVectorSentinel = 0.123456789;
  const entries: object[] = [
    {
      type: 'session',
      version: 3,
      id: 'diagnostic-session',
      timestamp: '2026-07-27T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'diagnostic-user',
      parentId: null,
      timestamp: '2026-07-27T10:01:00Z',
      message: { role: 'user', content: 'initial searchable evidence' },
    },
  ];
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds++,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful diagnostics must not warn');
    },
  });
  const embeddedInputs: string[] = [];
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        embeddedInputs.push(...texts);
        return texts.map((text) =>
          text === RECALL_EMBEDDING_CANARY_TEXT
            ? [0, 0, 1]
            : [privateVectorSentinel, 0.234567891, 0.345678912],
        );
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  entries.push(
    {
      type: 'message',
      id: 'diagnostic-assistant',
      parentId: 'diagnostic-user',
      timestamp: '2026-07-27T10:02:00Z',
      message: {
        role: 'assistant',
        content: [
          { type: 'text', text: `new searchable ${privateSentinel}` },
          {
            type: 'toolCall',
            id: 'diagnostic-tool-call',
            name: 'read',
            arguments: { path: privateToolArgumentSentinel },
          },
        ],
      },
    },
    {
      type: 'message',
      id: 'diagnostic-tool-result',
      parentId: 'diagnostic-assistant',
      timestamp: '2026-07-27T10:03:00Z',
      message: {
        role: 'toolResult',
        toolCallId: 'diagnostic-tool-call',
        toolName: 'read',
        content: [{ type: 'text', text: privateToolResultSentinel }],
        isError: false,
      },
    },
  );
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  const sourceBeforeReconciliation = await readFile(sessionPath);
  const sourceByteSize = (await stat(sessionPath)).size;

  await service.reconcileSession(sessionPath, {
    lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
    lockWaitMilliseconds: 250,
  });
  await diagnostics.flush();

  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.equal(records[0]?.status, RecallDiagnosticStatus.STARTED);
  const completion = records[1];
  assert.equal(
    completion?.operationKind,
    RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
  );
  assert.equal(completion?.lifecycleTrigger, RecallLifecycleTrigger.AGENT_SETTLED);
  assert.equal(completion?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(completion?.sourceByteSize, sourceByteSize);
  assert.equal(completion?.changed, true);
  assert.equal(completion?.skipped, false);
  assert.equal(completion?.scannedSessionCount, 1);
  assert.equal(completion?.indexedSessionCount, 1);
  assert.ok(readTestDiagnosticNumber(completion, 'upsertedDocumentCount') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'cacheHitCount') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'newEmbeddingCount') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'embeddingRequestCount') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'elapsedMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'writerLockWaitMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'manifestStorePreparationMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'physicalSessionPreparationMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'projectIdentityResolutionMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'embeddingCacheResolutionMilliseconds') >= 0);
  assert.ok(readTestDiagnosticNumber(completion, 'embeddingServerRequestMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'databaseWriteMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'indexStateCheckpointMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(completion, 'unattributedMilliseconds') >= 0);
  for (const privateValue of [
    privateSentinel,
    privateToolArgumentSentinel,
    privateToolResultSentinel,
    String(privateVectorSentinel),
  ]) {
    assert.doesNotMatch(diagnosticJsonl, new RegExp(privateValue, 'u'));
  }
  assert.deepEqual(await readFile(sessionPath), sourceBeforeReconciliation);

  const embeddingCountBeforeShutdown = embeddedInputs.length;
  await service.reconcileSession(sessionPath, {
    lifecycleTrigger: RecallLifecycleTrigger.SESSION_SHUTDOWN,
    lockWaitMilliseconds: 250,
  });
  await diagnostics.flush();
  const recordsAfterShutdown = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(recordsAfterShutdown.every(isUnknownRecord));
  assert.equal(recordsAfterShutdown.length, 4);
  const shutdownCompletion = recordsAfterShutdown[3];
  assert.equal(shutdownCompletion?.lifecycleTrigger, RecallLifecycleTrigger.SESSION_SHUTDOWN);
  assert.equal(shutdownCompletion?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(shutdownCompletion?.sourceByteSize, sourceByteSize);
  assert.equal(shutdownCompletion?.changed, false);
  assert.equal(shutdownCompletion?.skipped, true);
  assert.equal(shutdownCompletion?.upsertedDocumentCount, 0);
  assert.equal(shutdownCompletion?.cacheHitCount, 0);
  assert.equal(shutdownCompletion?.newEmbeddingCount, 0);
  assert.equal(shutdownCompletion?.embeddingRequestCount, 0);
  assert.equal(embeddedInputs.length, embeddingCountBeforeShutdown);
  assert.deepEqual(await readFile(sessionPath), sourceBeforeReconciliation);

  const search = await service.search(privateSentinel, 1, { scope: RecallSearchScope.GLOBAL });
  assert.equal(search.results[0]?.entryId.value, 'diagnostic-assistant');
});

void test('all diagnostics record recall search and its trusted active-session freshness barrier', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-search-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'search-diagnostic-session',
        timestamp: '2026-07-27T11:00:00Z',
        cwd: '/search-project',
      },
      {
        type: 'message',
        id: 'search-diagnostic-entry',
        parentId: null,
        timestamp: '2026-07-27T11:01:00Z',
        message: { role: 'assistant', content: 'bounded search diagnostic evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds++,
    wallClockIsoTimestamp: () => '2026-07-27T11:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful search diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  const sourceBeforeSearch = await readFile(sessionPath);

  const search = await service.search('bounded search diagnostic', 1, {
    mode: 'hybrid',
    scope: RecallSearchScope.GLOBAL,
    activeSessionPath: sessionPath,
  });
  await diagnostics.flush();

  assert.equal(search.results[0]?.entryId.value, 'search-diagnostic-entry');
  assert.deepEqual(await readFile(sessionPath), sourceBeforeSearch);
  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 4);
  const searchStart = records[0];
  const freshnessStart = records[1];
  const freshnessCompletion = records[2];
  const searchCompletion = records[3];
  assert.equal(searchStart?.operationKind, RecallDiagnosticOperationKind.SEARCH);
  assert.equal(searchStart?.status, RecallDiagnosticStatus.STARTED);
  assert.equal(searchStart?.searchMode, 'hybrid');
  assert.equal(searchStart?.recallScope, RecallSearchScope.GLOBAL);
  assert.equal(searchStart?.processId, process.pid);
  assert.equal(
    freshnessStart?.operationKind,
    RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
  );
  assert.equal(freshnessStart?.lifecycleTrigger, RecallLifecycleTrigger.ACTIVE_SESSION_FRESHNESS);
  assert.equal(
    freshnessCompletion?.lifecycleTrigger,
    RecallLifecycleTrigger.ACTIVE_SESSION_FRESHNESS,
  );
  assert.equal(searchCompletion?.operationId, searchStart?.operationId);
  assert.equal(searchCompletion?.operationKind, RecallDiagnosticOperationKind.SEARCH);
  assert.equal(searchCompletion?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(searchCompletion?.searchMode, 'hybrid');
  assert.equal(searchCompletion?.recallScope, RecallSearchScope.GLOBAL);
  assert.equal(searchCompletion?.freshnessBarrierRan, true);
  assert.ok(readTestDiagnosticNumber(searchCompletion, 'elapsedMilliseconds') > 0);
  assert.ok(
    readTestDiagnosticNumber(searchCompletion, 'embeddingModelVerificationMilliseconds') > 0,
  );
  assert.ok(readTestDiagnosticNumber(searchCompletion, 'activeSessionFreshnessMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(searchCompletion, 'queryEmbeddingMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(searchCompletion, 'retrievalRankingMilliseconds') > 0);
  assert.equal(searchCompletion?.deepRerankMilliseconds, 0);
  assert.ok(readTestDiagnosticNumber(searchCompletion, 'unattributedMilliseconds') >= 0);
});

void test('deep search diagnostics isolate reranker time and exclude private search evidence', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-private-search-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'private.jsonl');
  const querySentinel = 'PRIVATE_QUERY_SENTINEL_25';
  const sourceSentinel = 'PRIVATE_SOURCE_SENTINEL_25';
  const neighborSentinel = 'PRIVATE_RECALLED_NEIGHBOR_SENTINEL_25';
  const toolArgumentSentinel = 'PRIVATE_TOOL_ARGUMENT_SENTINEL_25';
  const toolResultSentinel = 'PRIVATE_TOOL_RESULT_SENTINEL_25';
  const queryVectorSentinel = 0.2525252525;
  const rerankerScoreSentinel = 0.9191919191;
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'private-search-session',
        timestamp: '2026-07-27T12:00:00Z',
        cwd: '/private-search-project',
      },
      {
        type: 'message',
        id: 'private-search-entry',
        parentId: null,
        timestamp: '2026-07-27T12:01:00Z',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: `${querySentinel} ${sourceSentinel} alpha beta gamma ${neighborSentinel}`,
            },
            {
              type: 'toolCall',
              id: 'private-search-tool-call',
              name: 'read',
              arguments: { path: toolArgumentSentinel },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'private-search-tool-result',
        parentId: 'private-search-entry',
        timestamp: '2026-07-27T12:02:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'private-search-tool-call',
          toolName: 'read',
          content: [{ type: 'text', text: toolResultSentinel }],
          isError: false,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  let monotonicMilliseconds = 0;
  let recordSearchCosts = false;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T12:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
    chunkPolicy: { maxTokens: 4, overlapTokens: 1 },
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful private search diagnostics must not warn');
    },
  });
  const rerankerDocuments: string[][] = [];
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (recordSearchCosts && text === RECALL_EMBEDDING_CANARY_TEXT) {
            monotonicMilliseconds += 7;
          } else if (recordSearchCosts && text === querySentinel) {
            monotonicMilliseconds += 11;
          }
          return text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [queryVectorSentinel, 0.5, 0];
        });
      },
    },
    reranker: {
      async rerankDocuments(receivedQuery, documents) {
        assert.equal(receivedQuery, querySentinel);
        rerankerDocuments.push([...documents]);
        monotonicMilliseconds += 13;
        return documents.map((document, index) =>
          index === 0 && document.includes(sourceSentinel) ? rerankerScoreSentinel : 0.1,
        );
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  monotonicMilliseconds = 0;
  recordSearchCosts = true;

  const search = await service.search(querySentinel, 3, {
    mode: 'deep-rerank',
    scope: RecallSearchScope.GLOBAL,
  });
  await diagnostics.flush();

  assert.ok(rerankerDocuments.flat().some((document) => document.includes(sourceSentinel)));
  const serializedSearch = JSON.stringify(search);
  assert.match(serializedSearch, new RegExp(sourceSentinel, 'u'));
  assert.match(serializedSearch, new RegExp(neighborSentinel, 'u'));
  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  const completion = records[1];
  assert.equal(completion?.operationKind, RecallDiagnosticOperationKind.SEARCH);
  assert.equal(completion?.searchMode, 'deep-rerank');
  assert.equal(completion?.recallScope, RecallSearchScope.GLOBAL);
  assert.equal(completion?.freshnessBarrierRan, false);
  assert.equal(completion?.embeddingModelVerificationMilliseconds, 7);
  assert.equal(completion?.activeSessionFreshnessMilliseconds, 0);
  assert.equal(completion?.queryEmbeddingMilliseconds, 11);
  assert.equal(completion?.retrievalRankingMilliseconds, 0);
  assert.equal(completion?.deepRerankMilliseconds, 13);
  assert.equal(completion?.elapsedMilliseconds, 31);
  assert.equal(completion?.unattributedMilliseconds, 0);
  for (const privateValue of [
    querySentinel,
    sourceSentinel,
    neighborSentinel,
    toolArgumentSentinel,
    toolResultSentinel,
    String(queryVectorSentinel),
    String(rerankerScoreSentinel),
  ]) {
    assert.doesNotMatch(diagnosticJsonl, new RegExp(privateValue, 'u'));
  }
});

void test('slow search diagnostics omit 999 milliseconds and retain the 1000 millisecond boundary', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-slow-search-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'slow-search.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'slow-search-session',
        timestamp: '2026-07-27T13:00:00Z',
        cwd: '/slow-search-project',
      },
      {
        type: 'message',
        id: 'slow-search-entry',
        parentId: null,
        timestamp: '2026-07-27T13:01:00Z',
        message: { role: 'assistant', content: 'slow search threshold evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  let monotonicMilliseconds = 0;
  let recordSearchCosts = false;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T13:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful slow search diagnostics must not warn');
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        for (const text of texts) {
          if (recordSearchCosts && text === 'fast-search') {
            monotonicMilliseconds += 999;
          }
          if (recordSearchCosts && text === 'threshold-search') {
            monotonicMilliseconds += 1_000;
          }
        }
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  monotonicMilliseconds = 0;
  recordSearchCosts = true;

  await service.search('fast-search', 1, { scope: RecallSearchScope.GLOBAL });
  await diagnostics.flush();
  await assert.rejects(() => readFile(config.diagnosticLogPath), { code: 'ENOENT' });

  monotonicMilliseconds = 0;
  await service.search('threshold-search', 1, { scope: RecallSearchScope.GLOBAL });
  await diagnostics.flush();

  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 1);
  assert.equal(records[0]?.operationKind, RecallDiagnosticOperationKind.SEARCH);
  assert.equal(records[0]?.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(records[0]?.elapsedMilliseconds, 1_000);
});

void test('slow diagnostics omit a fast unchanged live session reconciliation', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-slow-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'slow-diagnostic-session',
        timestamp: '2026-07-27T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'slow-diagnostic-entry',
        parentId: null,
        timestamp: '2026-07-27T10:01:00Z',
        message: { role: 'user', content: 'evidence survives a skipped reconciliation' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const diagnosticsClock = {
    monotonicMilliseconds: () => 0,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('omitted diagnostics must not warn');
    },
  });
  let contentEmbeddingCount = 0;
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        contentEmbeddingCount += texts.filter(
          (text) => text !== RECALL_EMBEDDING_CANARY_TEXT,
        ).length;
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  const embeddingCountBeforeReconciliation = contentEmbeddingCount;

  await service.reconcileSession(sessionPath, {
    lifecycleTrigger: RecallLifecycleTrigger.SESSION_SHUTDOWN,
    lockWaitMilliseconds: 250,
  });
  await diagnostics.flush();

  await assert.rejects(() => readFile(config.diagnosticLogPath), { code: 'ENOENT' });
  assert.equal(contentEmbeddingCount, embeddingCountBeforeReconciliation);
  const search = await service.search('survives skipped', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(search.results[0]?.entryId.value, 'slow-diagnostic-entry');
});

void test('live and freshness diagnostics classify caller-signal cancellation without changing the error', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-cancelled-live-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  const sessionHeader = {
    type: 'session',
    version: 3,
    id: 'cancelled-live-diagnostic-session',
    timestamp: '2026-07-27T10:00:00Z',
    cwd: '/project',
  };
  const initialEntries = [
    sessionHeader,
    {
      type: 'message',
      id: 'cancelled-live-initial-entry',
      parentId: null,
      timestamp: '2026-07-27T10:01:00Z',
      message: { role: 'user', content: 'initial cancellation evidence' },
    },
  ];
  await writeFile(
    sessionPath,
    initialEntries.map((entry) => JSON.stringify(entry)).join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    notifyWarning() {
      assert.fail('successful cancellation diagnostics must not warn');
    },
  });
  let pendingCancellation:
    | { abortController: AbortController; cancellationError: Error }
    | undefined;
  const service = createRecallConversationService(config, {
    diagnostics,
    embeddings: {
      async embedTexts(texts, signal) {
        if (pendingCancellation && texts.some((text) => text !== RECALL_EMBEDDING_CANARY_TEXT)) {
          assert.equal(signal, pendingCancellation.abortController.signal);
          pendingCancellation.abortController.abort(pendingCancellation.cancellationError);
          throw pendingCancellation.cancellationError;
        }
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  await service.index();
  await writeFile(
    sessionPath,
    [
      ...initialEntries,
      {
        type: 'message',
        id: 'cancelled-live-new-entry',
        parentId: 'cancelled-live-initial-entry',
        timestamp: '2026-07-27T10:02:00Z',
        message: { role: 'assistant', content: 'changed cancellation evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const directAbortController = new AbortController();
  const directCancellationError = new Error('direct live cancellation sentinel');
  pendingCancellation = {
    abortController: directAbortController,
    cancellationError: directCancellationError,
  };
  await assert.rejects(
    () =>
      service.reconcileSession(sessionPath, {
        lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
        signal: directAbortController.signal,
      }),
    (error) => error === directCancellationError,
  );

  const freshnessAbortController = new AbortController();
  const freshnessCancellationError = new Error('freshness cancellation sentinel');
  pendingCancellation = {
    abortController: freshnessAbortController,
    cancellationError: freshnessCancellationError,
  };
  await assert.rejects(
    () =>
      service.search('cancellation evidence', 1, {
        scope: RecallSearchScope.GLOBAL,
        activeSessionPath: sessionPath,
        signal: freshnessAbortController.signal,
      }),
    (error) => error === freshnessCancellationError,
  );
  await diagnostics.flush();

  const records = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  const liveCompletions = records.filter(
    (record) =>
      record.operationKind === RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION &&
      record.status !== RecallDiagnosticStatus.STARTED,
  );
  assert.deepEqual(
    liveCompletions.map((record) => ({
      lifecycleTrigger: record.lifecycleTrigger,
      status: record.status,
      errorCategory: record.errorCategory,
      failedSessionCount: record.failedSessionCount,
    })),
    [
      {
        lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
        status: RecallDiagnosticStatus.CANCELLED,
        errorCategory: RecallDiagnosticErrorCategory.OPERATION_CANCELLED,
        failedSessionCount: 0,
      },
      {
        lifecycleTrigger: RecallLifecycleTrigger.ACTIVE_SESSION_FRESHNESS,
        status: RecallDiagnosticStatus.CANCELLED,
        errorCategory: RecallDiagnosticErrorCategory.OPERATION_CANCELLED,
        failedSessionCount: 0,
      },
    ],
  );
  const searchCompletion = records.find(
    (record) =>
      record.operationKind === RecallDiagnosticOperationKind.SEARCH &&
      record.status !== RecallDiagnosticStatus.STARTED,
  );
  assert.equal(searchCompletion?.status, RecallDiagnosticStatus.CANCELLED);
  assert.equal(searchCompletion?.errorCategory, RecallDiagnosticErrorCategory.OPERATION_CANCELLED);
});

void test('failed live diagnostics preserve the reconciliation error, lock cleanup, and source bytes', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-failed-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  const sessionHeader = {
    type: 'session',
    version: 3,
    id: 'failed-diagnostic-session',
    timestamp: '2026-07-27T10:00:00Z',
    cwd: '/project',
  };
  await writeFile(
    sessionPath,
    [
      sessionHeader,
      {
        type: 'message',
        id: 'previous-valid-entry',
        parentId: null,
        timestamp: '2026-07-27T10:01:00Z',
        message: { role: 'user', content: 'stale evidence removed after corruption' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds++,
    wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful diagnostic persistence must not warn');
    },
  });
  let failContentEmbeddings = false;
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts) {
        if (failContentEmbeddings && texts.some((text) => text !== RECALL_EMBEDDING_CANARY_TEXT)) {
          throw new Error('embedding server unavailable');
        }
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  const privateFailureSentinel = 'PRIVATE_FAILED_RECONCILIATION_SENTINEL_24';
  await writeFile(
    sessionPath,
    [
      sessionHeader,
      {
        type: 'message',
        id: 'private-missing-parent-id',
        parentId: 'missing-parent',
        timestamp: '2026-07-27T10:02:00Z',
        message: { role: 'user', content: privateFailureSentinel },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const sourceBeforeReconciliation = await readFile(sessionPath);

  await assert.rejects(
    () =>
      service.reconcileSession(sessionPath, {
        lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
      }),
    /Recall active session reconciliation failed.*missing parent missing-parent/su,
  );
  await diagnostics.flush();

  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.equal(records[1]?.status, RecallDiagnosticStatus.FAILED);
  assert.equal(records[1]?.errorCategory, RecallDiagnosticErrorCategory.OPERATION_FAILED);
  assert.equal(records[1]?.sourceByteSize, sourceBeforeReconciliation.byteLength);
  assert.equal(records[1]?.changed, true);
  assert.equal(records[1]?.skipped, false);
  assert.doesNotMatch(diagnosticJsonl, /private-missing-parent-id|missing-parent/u);
  assert.doesNotMatch(diagnosticJsonl, new RegExp(privateFailureSentinel, 'u'));
  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  assert.deepEqual(await readFile(sessionPath), sourceBeforeReconciliation);
  const search = await service.search('stale evidence', 1, { scope: RecallSearchScope.GLOBAL });
  assert.deepEqual(search.results, []);

  await writeFile(
    sessionPath,
    [
      sessionHeader,
      {
        type: 'message',
        id: 'embedding-failure-entry',
        parentId: null,
        timestamp: '2026-07-27T10:03:00Z',
        message: { role: 'user', content: 'model outage evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  failContentEmbeddings = true;
  const sourceBeforeEmbeddingFailure = await readFile(sessionPath);
  await assert.rejects(
    () =>
      service.reconcileSession(sessionPath, {
        lifecycleTrigger: RecallLifecycleTrigger.SESSION_SHUTDOWN,
      }),
    /embedding server unavailable/u,
  );
  await diagnostics.flush();

  const recordsAfterEmbeddingFailure = (await readFile(config.diagnosticLogPath, 'utf8'))
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(recordsAfterEmbeddingFailure.every(isUnknownRecord));
  const liveReconciliationRecords = recordsAfterEmbeddingFailure.filter(
    (record) => record.operationKind === RecallDiagnosticOperationKind.LIVE_SESSION_RECONCILIATION,
  );
  assert.equal(liveReconciliationRecords.length, 4);
  const embeddingFailure = liveReconciliationRecords[3];
  assert.equal(embeddingFailure?.status, RecallDiagnosticStatus.FAILED);
  assert.equal(embeddingFailure?.embeddingRequestCount, 1);
  assert.ok(readTestDiagnosticNumber(embeddingFailure, 'embeddingServerRequestMilliseconds') > 0);
  assert.ok(readTestDiagnosticNumber(embeddingFailure, 'embeddingCacheResolutionMilliseconds') > 0);
  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  assert.deepEqual(await readFile(sessionPath), sourceBeforeEmbeddingFailure);
});

void test('slow diagnostics retain failed searches, omit fast cancellation, and preserve errors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-failed-search-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'failed-search.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'failed-search-session',
        timestamp: '2026-07-27T14:00:00Z',
        cwd: '/failed-search-project',
      },
      {
        type: 'message',
        id: 'failed-search-entry',
        parentId: null,
        timestamp: '2026-07-27T14:01:00Z',
        message: { role: 'assistant', content: 'failed search diagnostic evidence' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  let monotonicMilliseconds = 0;
  const diagnosticsClock = {
    monotonicMilliseconds: () => monotonicMilliseconds,
    wallClockIsoTimestamp: () => '2026-07-27T14:00:00.000Z',
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.SLOW,
  };
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: config.diagnosticLogPath,
    retainedLogPath: config.retainedDiagnosticLogPath,
    clock: diagnosticsClock,
    notifyWarning() {
      assert.fail('successful failed-search diagnostic writes must not warn');
    },
  });
  const rerankerError = new Error('private reranker model response sentinel 25');
  const cancellationError = new Error('private cancellation sentinel 25');
  const service = createRecallConversationService(config, {
    diagnostics,
    diagnosticsClock,
    embeddings: {
      async embedTexts(texts, signal) {
        if (signal?.aborted) {
          throw cancellationError;
        }
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    reranker: {
      async rerankDocuments() {
        monotonicMilliseconds += 17;
        throw rerankerError;
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  monotonicMilliseconds = 0;

  await assert.rejects(
    () =>
      service.search('reranker failure', 1, {
        mode: 'deep-rerank',
        scope: RecallSearchScope.GLOBAL,
      }),
    (error) => error === rerankerError,
  );
  const abortController = new AbortController();
  abortController.abort(cancellationError);
  await assert.rejects(
    () =>
      service.search('cancelled search', 1, {
        scope: RecallSearchScope.GLOBAL,
        signal: abortController.signal,
      }),
    (error) => error === cancellationError,
  );
  await writeFile(config.manifestPath, '{"private_manifest_sentinel_25":');
  let manifestError: unknown;
  try {
    await service.search('manifest failure', 1, { scope: RecallSearchScope.GLOBAL });
    assert.fail('invalid manifest search must fail');
  } catch (error) {
    manifestError = error;
  }
  assert.ok(manifestError instanceof Error);
  assert.match(manifestError.message, /Recall index manifest unreadable/u);
  await diagnostics.flush();

  const diagnosticJsonl = await readFile(config.diagnosticLogPath, 'utf8');
  const records = diagnosticJsonl
    .trimEnd()
    .split('\n')
    .map((line): unknown => JSON.parse(line));
  assert.ok(records.every(isUnknownRecord));
  assert.equal(records.length, 2);
  assert.deepEqual(
    records.map((record) => record.status),
    [RecallDiagnosticStatus.FAILED, RecallDiagnosticStatus.FAILED],
  );
  assert.equal(records[0]?.searchMode, 'deep-rerank');
  assert.equal(records[0]?.deepRerankMilliseconds, 17);
  assert.equal(records[1]?.embeddingModelVerificationMilliseconds, 0);
  assert.doesNotMatch(diagnosticJsonl, /private reranker model response sentinel 25/u);
  assert.doesNotMatch(diagnosticJsonl, /private cancellation sentinel 25/u);
  assert.doesNotMatch(diagnosticJsonl, /private_manifest_sentinel_25/u);
  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
});

void test('diagnostic write failure cannot change manual or live indexing', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-write-failed-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'active.jsonl');
  const entries: object[] = [
    {
      type: 'session',
      version: 3,
      id: 'write-failed-diagnostic-session',
      timestamp: '2026-07-27T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'write-failed-initial-entry',
      parentId: null,
      timestamp: '2026-07-27T10:01:00Z',
      message: { role: 'user', content: 'initial evidence' },
    },
  ];
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    diagnosticsMode: RecallDiagnosticsMode.ALL,
  };
  const filesystemBlocker = join(directory, 'not-a-directory');
  await writeFile(filesystemBlocker, 'unchanged');
  const warnings: string[] = [];
  const diagnostics = createRecallOperationDiagnostics({
    mode: config.diagnosticsMode,
    activeLogPath: join(filesystemBlocker, 'diagnostics.jsonl'),
    retainedLogPath: join(filesystemBlocker, 'diagnostics.previous.jsonl'),
    notifyWarning(message) {
      warnings.push(message);
    },
  });
  const service = createRecallConversationService(config, {
    diagnostics,
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  const manualResult = await service.index({
    manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
    optimize: true,
  });
  await diagnostics.flush();
  assert.equal(manualResult.indexSummary.indexedSessions, 1);
  assert.equal(manualResult.totalChunks, 1);
  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  assert.ok(await readRecallIndexManifest(config.manifestPath));
  const searchDuringDiagnosticFailure = await service.search('initial evidence', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  await diagnostics.flush();
  assert.equal(
    searchDuringDiagnosticFailure.results[0]?.entryId.value,
    'write-failed-initial-entry',
  );
  assert.deepEqual(warnings, [
    'Recall diagnostics disabled after local log persistence failed; recall behavior is unchanged.',
  ]);
  entries.push({
    type: 'message',
    id: 'write-failed-new-entry',
    parentId: 'write-failed-initial-entry',
    timestamp: '2026-07-27T10:02:00Z',
    message: { role: 'assistant', content: 'searchable despite diagnostic failure' },
  });
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  const sourceBeforeReconciliation = await readFile(sessionPath);

  const result = await service.reconcileSession(sessionPath, {
    lifecycleTrigger: RecallLifecycleTrigger.SESSION_SHUTDOWN,
  });
  await diagnostics.flush();

  assert.equal(result.indexSummary.indexedSessions, 1);
  assert.ok(result.totalChunks > 1);
  assert.deepEqual(warnings, [
    'Recall diagnostics disabled after local log persistence failed; recall behavior is unchanged.',
  ]);
  assert.equal(await readFile(filesystemBlocker, 'utf8'), 'unchanged');
  await assert.rejects(() => stat(config.lockPath), { code: 'ENOENT' });
  assert.deepEqual(await readFile(sessionPath), sourceBeforeReconciliation);
  const search = await service.search('despite diagnostic failure', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(search.results[0]?.entryId.value, 'write-failed-new-entry');
});

void test('recall search keeps ordinary reads stable and refreshes resumed or forked active sessions', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  const entries: object[] = [
    {
      type: 'session',
      version: 3,
      id: 'session-1',
      timestamp: '2026-07-24T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'queue-entry',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'We chose a durable queue for job delivery.' }],
      },
    },
    {
      type: 'message',
      id: 'ui-entry',
      parentId: 'queue-entry',
      timestamp: '2026-07-24T10:02:00Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'The navigation bar is blue.' }],
      },
    },
  ];
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');

  const embeddedInputs: string[] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddedInputs.push(...texts);
      return texts.map((text) => {
        if (text === RECALL_EMBEDDING_CANARY_TEXT) {
          return [0, 0, 1];
        }
        return text.toLowerCase().includes('queue') ? [1, 0, 0] : [0, 1, 0];
      });
    },
  };
  let tokenizerLoads = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
  });

  const indexed = await service.index();
  assert.equal(indexed.indexSummary.cacheHits, 0);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 2);
  assert.equal(indexed.indexSummary.embeddingRequestCount, 1);
  assert.equal(indexed.totalChunks, 2);

  const first = await service.search('What did we decide about job queues?', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(first.results[0]?.entryId.value, 'queue-entry');
  assert.equal(first.results[0]?.sessionPath, sessionPath);
  assert.equal(first.totalChunks, 2);

  entries.push({
    type: 'message',
    id: 'unindexed-entry',
    parentId: 'ui-entry',
    timestamp: '2026-07-24T10:03:00Z',
    message: { role: 'assistant', content: 'This must not be indexed by search.' },
  });
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  const second = await service.search('queue decision', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(second.totalChunks, 2);
  assert.equal(tokenizerLoads, 1);
  assert.deepEqual(embeddedInputs, [
    RECALL_EMBEDDING_CANARY_TEXT,
    'We chose a durable queue for job delivery.',
    'The navigation bar is blue.',
    RECALL_EMBEDDING_CANARY_TEXT,
    'What did we decide about job queues?',
    RECALL_EMBEDDING_CANARY_TEXT,
    'queue decision',
  ]);

  const refreshed = await service.search('must not be indexed by search', 1, {
    scope: RecallSearchScope.GLOBAL,
    activeSessionPath: sessionPath,
  });
  assert.equal(refreshed.results[0]?.entryId.value, 'unindexed-entry');
  assert.equal(refreshed.totalChunks, 3);

  const forkSessionPath = join(sessionsDirectory, 'fork.jsonl');
  await writeFile(
    forkSessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'fork-session',
        timestamp: '2026-07-24T11:00:00Z',
        cwd: '/project',
        parentSession: sessionPath,
      },
      {
        type: 'message',
        id: 'fork-entry',
        parentId: null,
        timestamp: '2026-07-24T11:01:00Z',
        message: { role: 'user', content: 'fork lifecycle marker' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const forked = await service.search('fork lifecycle marker', 1, {
    scope: RecallSearchScope.GLOBAL,
    activeSessionPath: forkSessionPath,
  });
  assert.equal(forked.results[0]?.entryId.value, 'fork-entry');
  assert.equal(forked.results[0]?.parentSessionPath, sessionPath);
  assert.equal(forked.totalChunks, 4);

  const lockPath = join(directory, 'recall.lock');
  await mkdir(lockPath);
  await writeFile(join(lockPath, 'owner.json'), `${JSON.stringify({ pid: process.pid })}\n`);
  await assert.rejects(
    () =>
      service.reconcileSession(sessionPath, {
        lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
        lockWaitMilliseconds: 10,
      }),
    /Recall conversation operation cancelled/,
  );
  await rm(lockPath, { recursive: true });

  const lockOwner = `${JSON.stringify({ pid: 999_999_999 })}\n`;
  await mkdir(lockPath);
  await writeFile(join(lockPath, 'owner.json'), lockOwner);
  await assert.rejects(
    () => service.search('must not clear a stale lock', 1, { scope: RecallSearchScope.GLOBAL }),
    /stale lock from dead process 999999999.*\/pi-session-recall-index.*read-only search did not remove the lock/,
  );
  assert.equal(await readFile(join(directory, 'recall.lock', 'owner.json'), 'utf8'), lockOwner);
});

void test('explicit project scope filters dense, lexical, and identifier candidates before channel limits', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-project-scope-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const projectDirectory = join(directory, 'selected-project');
  const unrelatedDirectory = join(directory, 'unrelated-project');
  await Promise.all([mkdir(sessionsDirectory), mkdir(projectDirectory), mkdir(unrelatedDirectory)]);
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: projectDirectory });
  await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/scoped.git'], {
    cwd: projectDirectory,
  });
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: unrelatedDirectory });
  await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', 'https://github.com/Whamp/other.git'], {
    cwd: unrelatedDirectory,
  });
  const writeSession = async (
    fileName: string,
    sessionId: string,
    sessionOrigin: string,
    entryId: string,
    content: string,
  ): Promise<void> => {
    await writeFile(
      join(sessionsDirectory, fileName),
      [
        {
          type: 'session',
          version: 3,
          id: sessionId,
          timestamp: '2026-07-24T10:00:00Z',
          cwd: sessionOrigin,
        },
        {
          type: 'message',
          id: entryId,
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'assistant', content },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );
  };
  await writeSession(
    'selected.jsonl',
    'selected-session',
    projectDirectory,
    'selected-entry',
    'queue readNodeErrorCode',
  );
  await writeSession(
    'unrelated.jsonl',
    'unrelated-session',
    unrelatedDirectory,
    'unrelated-entry',
    'queue queue queue readNodeErrorCode readNodeErrorCode',
  );
  const query = 'queue readNodeErrorCode';
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          if (text === query || text.includes('queue queue')) {
            return [1, 0, 0];
          }
          return [0.8, 0.2, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  await service.index();
  const projectSearch = await service.search(query, 3, {
    scope: RecallSearchScope.PROJECT,
    invocationDirectory: projectDirectory,
  });

  assert.deepEqual(
    projectSearch.results.map((result) => result.entryId.value),
    ['selected-entry'],
  );
  assert.ok(projectSearch.results[0]?.dense);
  assert.ok(projectSearch.results[0]?.lexical);
  assert.ok(projectSearch.results[0]?.identifier);
  assert.equal(projectSearch.searchPolicy.scope, 'project');
  assert.equal(
    projectSearch.searchPolicy.invocationProjectIdentity,
    'git-origin:github.com/Whamp/scoped',
  );
  assert.equal(projectSearch.results[0]?.evidenceRelation, 'same_repository');

  const globalSearch = await service.search(query, 1, {
    scope: RecallSearchScope.GLOBAL,
    invocationDirectory: projectDirectory,
  });
  assert.equal(globalSearch.results[0]?.entryId.value, 'unrelated-entry');
  assert.equal(globalSearch.searchPolicy.scope, 'global');
});

void test('configured project lineage admits exact, descendant, deleted, and Git-conflicting historical origins before channel limits', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-project-lineage-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const invocationDirectory = join(directory, 'successor');
  const prototypeRoot = join(directory, 'prototype');
  const prototypeDescendant = join(prototypeRoot, 'packages', 'app');
  const deletedRoot = join(directory, 'deleted-prototype');
  const unrelatedDirectory = `${prototypeRoot}-nearby`;
  await Promise.all([
    mkdir(sessionsDirectory),
    mkdir(invocationDirectory),
    mkdir(prototypeDescendant, { recursive: true }),
    mkdir(unrelatedDirectory),
  ]);
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: invocationDirectory });
  await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/successor.git'], {
    cwd: invocationDirectory,
  });
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: prototypeRoot });
  await EXEC_FILE_ASYNC(
    'git',
    ['remote', 'add', 'origin', 'git@github.com:Whamp/obsolete-prototype.git'],
    { cwd: prototypeRoot },
  );
  await EXEC_FILE_ASYNC('git', ['init'], { cwd: unrelatedDirectory });
  await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/unrelated.git'], {
    cwd: unrelatedDirectory,
  });
  const fixtures = [
    { file: 'root.jsonl', origin: prototypeRoot, entry: 'root-lineage' },
    { file: 'descendant.jsonl', origin: prototypeDescendant, entry: 'descendant-lineage' },
    { file: 'deleted.jsonl', origin: deletedRoot, entry: 'deleted-lineage' },
    { file: 'successor.jsonl', origin: invocationDirectory, entry: 'successor-entry' },
    { file: 'unrelated.jsonl', origin: unrelatedDirectory, entry: 'unrelated-entry' },
  ];
  for (const fixture of fixtures) {
    const content =
      fixture.entry === 'unrelated-entry'
        ? 'lineage queue queue queue LineageIdentifier LineageIdentifier'
        : fixture.entry === 'successor-entry'
          ? 'successor current evidence UniqueCurrentIdentifier'
          : `lineage queue LineageIdentifier ${fixture.entry}`;
    await writeFile(
      join(sessionsDirectory, fixture.file),
      [
        {
          type: 'session',
          version: 3,
          id: `${fixture.entry}-session`,
          timestamp: '2026-07-24T10:00:00Z',
          cwd: fixture.origin,
        },
        {
          type: 'message',
          id: fixture.entry,
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'assistant', content },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );
  }
  const query = 'lineage queue LineageIdentifier';
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    projectLineages: normalizeRecallProjectLineages({
      'git-origin:github.com/Whamp/successor': [prototypeRoot, deletedRoot],
    }),
    searchCandidateLimits: { dense: 3, lexical: 3, identifier: 3 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          return text.includes('queue queue') || text === query ? [1, 0, 0] : [0.8, 0.2, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  await service.index();
  const projectSearch = await service.search(query, 3, { invocationDirectory });

  assert.deepEqual(projectSearch.results.map((result) => result.entryId.value).toSorted(), [
    'deleted-lineage',
    'descendant-lineage',
    'root-lineage',
  ]);
  assert.ok(
    projectSearch.results.every(
      (result) => result.dense !== null && result.lexical !== null && result.identifier !== null,
    ),
  );
  assert.ok(
    projectSearch.results.every(
      (result) =>
        result.projectAttribution?.projectIdentity === 'git-origin:github.com/Whamp/successor' &&
        result.projectAttribution.identitySource ===
          RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE &&
        result.evidenceRelation === RecallEvidenceRelation.CONFIGURED_PROJECT_LINEAGE,
    ),
  );

  const searchFromHistoricalRoot = await service.search(
    'successor current evidence UniqueCurrentIdentifier',
    1,
    { invocationDirectory: prototypeRoot },
  );
  assert.equal(searchFromHistoricalRoot.results[0]?.entryId.value, 'successor-entry');
  assert.equal(
    searchFromHistoricalRoot.results[0]?.evidenceRelation,
    RecallEvidenceRelation.CONFIGURED_PROJECT_LINEAGE,
  );

  const globalSearch = await service.search(query, 1, {
    scope: RecallSearchScope.GLOBAL,
    invocationDirectory,
  });
  assert.equal(globalSearch.results[0]?.entryId.value, 'unrelated-entry');
});

void test('omitted scope admits only the exact non-Git session origin', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-non-git-scope-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const invocationDirectory = join(directory, "project'\\quoted");
  const nearbyDirectory = join(directory, "project'\\quoted-nearby");
  const descendantDirectory = join(invocationDirectory, 'descendant');
  const emptyInvocationDirectory = join(directory, 'empty-project');
  await Promise.all([
    mkdir(sessionsDirectory),
    mkdir(invocationDirectory),
    mkdir(nearbyDirectory),
    mkdir(emptyInvocationDirectory),
  ]);
  await mkdir(descendantDirectory);

  const sessionFixtures = [
    {
      fileName: 'exact.jsonl',
      sessionOrigin: invocationDirectory,
      entryId: 'exact-origin-entry',
      content: 'exact local project memory',
    },
    {
      fileName: 'nearby.jsonl',
      sessionOrigin: nearbyDirectory,
      entryId: 'nearby-origin-entry',
      content: 'exact local project memory from a similarly named nearby origin',
    },
    {
      fileName: 'parent.jsonl',
      sessionOrigin: directory,
      entryId: 'parent-origin-entry',
      content: 'exact local project memory from a parent origin',
    },
    {
      fileName: 'descendant.jsonl',
      sessionOrigin: descendantDirectory,
      entryId: 'descendant-origin-entry',
      content: 'exact local project memory from a descendant origin',
    },
  ];
  for (const { fileName, sessionOrigin, entryId, content } of sessionFixtures) {
    await writeFile(
      join(sessionsDirectory, fileName),
      [
        {
          type: 'session',
          version: 3,
          id: `${entryId}-session`,
          timestamp: '2026-07-24T10:00:00Z',
          cwd: sessionOrigin,
        },
        {
          type: 'message',
          id: entryId,
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'assistant', content },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );
  }

  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    searchCandidateLimits: { dense: 4, lexical: 4, identifier: 4 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  await service.index();
  const search = await service.search('exact local project memory', 4, {
    invocationDirectory,
  });

  assert.deepEqual(
    search.results.map((result) => result.entryId.value),
    ['exact-origin-entry'],
  );
  assert.equal(search.searchPolicy.scope, 'project');
  assert.equal(
    search.searchPolicy.invocationProjectIdentity,
    `non-git-session-origin:${invocationDirectory}`,
  );
  assert.equal(search.results[0]?.projectAttribution?.identitySource, 'non_git_session_origin');
  assert.equal(search.results[0]?.evidenceRelation, 'same_session_origin');

  const globalSearch = await service.search('exact local project memory', 4, {
    scope: RecallSearchScope.GLOBAL,
    invocationDirectory,
  });
  assert.deepEqual(globalSearch.results.map((result) => result.entryId.value).toSorted(), [
    'descendant-origin-entry',
    'exact-origin-entry',
    'nearby-origin-entry',
    'parent-origin-entry',
  ]);
  assert.equal(
    globalSearch.results.find((result) => result.entryId.value === 'exact-origin-entry')
      ?.evidenceRelation,
    'same_session_origin',
  );
  assert.ok(
    globalSearch.results
      .filter((result) => result.entryId.value !== 'exact-origin-entry')
      .every((result) => result.evidenceRelation === RecallEvidenceRelation.UNRESTRICTED_GLOBAL),
  );

  const emptyProjectSearch = await service.search('exact local project memory', 4, {
    invocationDirectory: emptyInvocationDirectory,
  });
  assert.deepEqual(emptyProjectSearch.results, []);
  assert.match(formatRecallSearchResults(emptyProjectSearch), /Retry with scope "global"/);
});

void test('indexing resolves each distinct session origin once and keeps unresolved origins globally searchable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-project-assignment-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionFixtures = [
    { file: 'one.jsonl', id: 'one', origin: '/historical/repository', content: 'memoized first' },
    { file: 'two.jsonl', id: 'two', origin: '/historical/repository', content: 'memoized second' },
    {
      file: 'missing.jsonl',
      id: 'missing',
      origin: '/deleted/repository',
      content: 'memoized missing',
    },
  ];
  for (const fixture of sessionFixtures) {
    await writeFile(
      join(sessionsDirectory, fixture.file),
      [
        {
          type: 'session',
          version: 3,
          id: `session-${fixture.id}`,
          timestamp: '2026-07-24T10:00:00Z',
          cwd: fixture.origin,
        },
        {
          type: 'message',
          id: `entry-${fixture.id}`,
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'assistant', content: fixture.content },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );
  }
  const resolvedOrigins: string[] = [];
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    searchCandidateLimits: { dense: 3, lexical: 3, identifier: 3 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
    async resolveProjectIdentity(sessionOrigin) {
      resolvedOrigins.push(sessionOrigin);
      return sessionOrigin === '/historical/repository'
        ? {
            projectIdentity: parseRepositoryIdentity('git-origin:github.com/Whamp/historical'),
            identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
          }
        : null;
    },
  });

  await service.index();
  const search = await service.search('memoized', 3, { scope: RecallSearchScope.GLOBAL });

  assert.deepEqual(resolvedOrigins.toSorted(), ['/deleted/repository', '/historical/repository']);
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-one')?.projectAttribution
      ?.projectIdentity,
    'git-origin:github.com/Whamp/historical',
  );
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-two')?.projectAttribution
      ?.identitySource,
    RecallProjectIdentitySource.GIT_ORIGIN,
  );
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-missing')?.projectAttribution,
    null,
  );
});

void test('lineage metadata rebuild rejects stale policy and reuses cached vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-lineage-metadata-rebuild-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const historicalRoot = '/relocated/repository';
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'metadata.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'metadata-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: historicalRoot,
      },
      {
        type: 'message',
        id: 'metadata-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'assistant', content: 'Lineage metadata must reuse this vector.' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const embeddedInputs: string[] = [];
  const dependencies = {
    embeddings: {
      async embedTexts(texts: string[]) {
        embeddedInputs.push(...texts);
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  };
  const firstService = createRecallConversationService(
    {
      ...createTestConfig(directory, sessionsDirectory),
      projectLineages: normalizeRecallProjectLineages({
        'git-origin:github.com/Whamp/before-relocation': [historicalRoot],
      }),
    },
    dependencies,
  );
  const changedService = createRecallConversationService(
    {
      ...createTestConfig(directory, sessionsDirectory),
      projectLineages: normalizeRecallProjectLineages({
        'git-origin:github.com/Whamp/after-relocation': [historicalRoot],
      }),
    },
    dependencies,
  );

  const first = await firstService.index();
  await assert.rejects(
    () => changedService.index(),
    /projectIdentity\.lineageDigest.*\/pi-session-recall-index --rebuild/s,
  );
  const rebuilt = await changedService.index({ rebuild: true });
  const search = await changedService.search('Lineage metadata', 1, {
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(first.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(rebuilt.indexSummary.cacheHits, 1);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(
    search.results[0]?.projectAttribution?.projectIdentity,
    'git-origin:github.com/Whamp/after-relocation',
  );
  assert.equal(search.results[0]?.projectAttribution?.identitySource, 'configured_project_lineage');
  assert.equal(
    embeddedInputs.filter((text) => text === 'Lineage metadata must reuse this vector.').length,
    1,
  );
});

void test('recall service builds a temporary index with an explicit chunk policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-chunk-policy-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'bounded.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'bounded-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/bounded-project',
      },
      {
        type: 'message',
        id: 'bounded-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'assistant', content: 'one two three four five' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    chunkPolicy: { maxTokens: 3, overlapTokens: 1 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => [text.length, 1, 0]);
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const indexed = await service.index();
  const manifest = await readRecallIndexManifest(config.manifestPath);

  assert.equal(indexed.totalChunks, 2);
  assert.equal(manifest?.chunkPolicy.maxTokens, 3);
  assert.equal(manifest?.chunkPolicy.overlapTokens, 1);
});

void test('recall service fuses bounded dense, lexical, and identifier candidates with component scores', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-hybrid-search-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'hybrid.jsonl');
  const denseContent =
    'An append-only queue prevents jobs from disappearing after process restarts.';
  const identifierContent =
    'The process probe uses readNodeErrorCode() when permission checks fail with EPERM.';
  const quotedPhraseContent = 'The release marker contains alpha beta as one exact phrase.';
  const separatedPhraseContent =
    'The release marker contains alpha with unrelated words before beta.';
  await writeFile(
    sessionPath,
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'hybrid-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/hybrid-project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'dense-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'assistant', content: denseContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'identifier-entry',
        parentId: 'dense-entry',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'assistant', content: identifierContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'quoted-phrase-entry',
        parentId: 'identifier-entry',
        timestamp: '2026-07-24T10:03:00Z',
        message: { role: 'assistant', content: quotedPhraseContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'separated-phrase-entry',
        parentId: 'quoted-phrase-entry',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: separatedPhraseContent },
      }),
    ].join('\n') + '\n',
  );

  const semanticParaphraseQuery = 'How did we make background task delivery resilient?';
  const identifierQuery = 'readNodeErrorCode';
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          if (
            text === identifierContent ||
            text === quotedPhraseContent ||
            text === separatedPhraseContent
          ) {
            return [0, 1, 0];
          }
          return [1, 0, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  const denseResult = await service.search(semanticParaphraseQuery, 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.deepEqual(denseResult.searchPolicy, {
    scope: RecallSearchScope.GLOBAL,
    invocationProjectIdentity: null,
    rankingMode: 'hybrid',
    rankFusionVersion: 2,
    reciprocalRankConstant: 60,
    rerankPolicyVersion: null,
    rerankerModel: null,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 1, lexical: 1, identifier: 1 },
  });
  assert.equal(denseResult.results[0]?.entryId.value, 'dense-entry');
  assert.equal(denseResult.results[0]?.sessionPath, sessionPath);
  assert.deepEqual(denseResult.results[0]?.dense, { rank: 1, cosineDistance: 0 });
  assert.equal(denseResult.results[0]?.lexical, null);
  assert.equal(denseResult.results[0]?.identifier, null);
  assert.equal(denseResult.results[0]?.fusedScore, 0.01639344262295082);

  const exactIdentifier = await service.search(identifierQuery, 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(exactIdentifier.results[0]?.entryId.value, 'identifier-entry');
  assert.equal(exactIdentifier.results[0]?.sessionPath, sessionPath);
  assert.equal(exactIdentifier.results[0]?.dense, null);
  assert.equal(exactIdentifier.results[0]?.lexical?.rank, 1);
  assert.ok((exactIdentifier.results[0]?.lexical?.fullTextScore ?? 0) > 0);
  assert.equal(exactIdentifier.results[0]?.identifier?.rank, 1);
  assert.ok((exactIdentifier.results[0]?.identifier?.fullTextScore ?? 0) > 0);
  assert.equal(exactIdentifier.results[0]?.fusedScore, 0.03278688524590164);
  assert.equal(new Set(exactIdentifier.results.map((result) => result.id)).size, 2);

  const wrongCase = await service.search('readnodeerrorcode', 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  const wrongCaseIdentifier = wrongCase.results.find(
    (result) => result.entryId.value === 'identifier-entry',
  );
  assert.equal(wrongCaseIdentifier?.lexical?.rank, 1);
  assert.equal(wrongCaseIdentifier?.identifier, null);
  assert.equal(wrongCase.results[0]?.fusedScore, wrongCase.results[1]?.fusedScore);
  assert.deepEqual(
    wrongCase.results.map((result) => result.id),
    wrongCase.results.map((result) => result.id).toSorted(),
  );

  const quotedPhrase = await service.search('Where did we write "alpha beta"?', 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(quotedPhrase.results[0]?.entryId.value, 'quoted-phrase-entry');
  assert.equal(quotedPhrase.results[0]?.lexical?.rank, 1);
  assert.equal(quotedPhrase.results[0]?.identifier?.rank, 1);
  assert.ok(
    !quotedPhrase.results.some((result) => result.entryId.value === 'separated-phrase-entry'),
  );
});

void test('recall service defaults to fused ranking and reranks only in explicit deep mode', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-reranked-pool-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const fusionFavorite = 'Exact fusion favorite evidence.';
  const rerankerFavorite = 'Semantically strongest Qwen evidence.';
  await writeFile(
    join(sessionsDirectory, 'reranked.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'reranked-session',
        timestamp: '2026-07-25T10:00:00Z',
        cwd: '/reranked-project',
      },
      {
        type: 'message',
        id: 'fusion-favorite',
        parentId: null,
        timestamp: '2026-07-25T10:01:00Z',
        message: { role: 'assistant', content: fusionFavorite },
      },
      {
        type: 'message',
        id: 'reranker-favorite',
        parentId: 'fusion-favorite',
        timestamp: '2026-07-25T10:02:00Z',
        message: { role: 'assistant', content: rerankerFavorite },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const query = 'fusion favorite';
  const rerankerInputs: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(receivedQuery, documents) {
      assert.equal(receivedQuery, query);
      rerankerInputs.push([...documents]);
      return [0.1, 0.9];
    },
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    searchCandidateLimits: { dense: 2, lexical: 2, identifier: 2 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          if (text === rerankerFavorite) {
            return [0.9, 0.1, 0];
          }
          return [1, 0, 0];
        });
      },
    },
    reranker,
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  const defaultSearch = await service.search(query, 1, { scope: RecallSearchScope.GLOBAL });

  assert.deepEqual(rerankerInputs, []);
  assert.equal(defaultSearch.results.length, 1);
  assert.equal(defaultSearch.results[0]?.entryId.value, 'fusion-favorite');
  assert.equal(defaultSearch.results[0]?.rerankerScore, null);
  assert.equal(defaultSearch.searchPolicy.rankingMode, 'hybrid');

  const deepSearch = await service.search(query, 1, {
    mode: 'deep-rerank',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.deepEqual(rerankerInputs, [[fusionFavorite, rerankerFavorite]]);
  assert.equal(deepSearch.results.length, 1);
  assert.equal(deepSearch.results[0]?.entryId.value, 'reranker-favorite');
  assert.equal(deepSearch.results[0]?.rerankerScore, 0.9);
  assert.equal(deepSearch.results[0]?.dense?.rank, 2);
  assert.ok(Number.isFinite(deepSearch.results[0]?.dense?.cosineDistance));
  assert.equal(deepSearch.results[0]?.lexical, null);
  assert.equal(deepSearch.results[0]?.identifier, null);
  assert.equal(deepSearch.searchPolicy.rankingMode, 'deep-rerank');
});

void test('recall service fails clearly when Qwen reranking is unavailable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-reranker-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'failure.jsonl'),
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'reranker-failure-session',
        timestamp: '2026-07-25T10:00:00Z',
        cwd: '/failure-project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'failure-entry',
        parentId: null,
        timestamp: '2026-07-25T10:01:00Z',
        message: { role: 'assistant', content: 'Evidence that must not bypass reranking.' },
      }),
    ].join('\n') + '\n',
  );
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    reranker: {
      async rerankDocuments() {
        throw new Error(
          'Recall reranker request failed at http://reranker.test/v1/rerank: unavailable',
        );
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  const defaultSearch = await service.search('must not require Qwen', 1, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(defaultSearch.results[0]?.entryId.value, 'failure-entry');

  await assert.rejects(
    () =>
      service.search('must use Qwen', 1, {
        mode: 'deep-rerank',
        scope: RecallSearchScope.GLOBAL,
      }),
    /Recall reranker request failed at http:\/\/reranker\.test\/v1\/rerank: unavailable/,
  );
});

void test('recall service retrieves context-dependent replies and reuses turn-context vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-turn-context-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'turn-context.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'turn-context-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/release-project',
      },
      {
        type: 'message',
        id: 'user-request',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Ship release Atlas to edge nodes.' },
      },
      {
        type: 'message',
        id: 'assistant-call',
        parentId: 'user-request',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'private deployment plan' },
            {
              type: 'toolCall',
              id: 'call-release',
              name: 'read',
              arguments: { path: 'release.json' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-result',
        parentId: 'assistant-call',
        timestamp: '2026-07-24T10:03:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-release',
          toolName: 'read',
          content: [{ type: 'text', text: 'RAW_RELEASE_TOOL_OUTPUT' }],
          isError: false,
        },
      },
      {
        type: 'message',
        id: 'assistant-reply',
        parentId: 'tool-result',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: 'Yes, do it.' },
      },
      {
        type: 'message',
        id: 'next-user',
        parentId: 'assistant-reply',
        timestamp: '2026-07-24T10:05:00Z',
        message: { role: 'user', content: 'Report when deployment finishes.' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const embeddingInputs: string[][] = [];
  const query = 'Atlas Yes';
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddingInputs.push([...texts]);
      return texts.map((text) => {
        if (text === RECALL_EMBEDDING_CANARY_TEXT) {
          return [0, 0, 1];
        }
        if (text === query || text.startsWith('User:\nShip release Atlas')) {
          return [1, 0, 0];
        }
        return [0, 1, 0];
      });
    },
  };
  const config = createTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const indexed = await service.index();
  const recalled = await service.search(query, 1, { scope: RecallSearchScope.GLOBAL });
  const turnContext = recalled.results[0];

  assert.equal(indexed.totalChunks, 7);
  assert.equal(indexed.indexSummary.cacheHits, 0);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 4);
  assert.equal(indexed.indexSummary.embeddingRequestCount, 1);
  assert.equal(turnContext?.documentKind, 'turn_context');
  assert.equal(turnContext?.evidenceKind, 'turn_context');
  assert.equal(turnContext?.role, 'turn');
  assert.deepEqual(
    turnContext?.contributingEntryIds.map((id) => id.value),
    ['user-request', 'assistant-reply'],
  );
  assert.equal(turnContext?.dense?.rank, 1);
  assert.equal(turnContext?.lexical?.rank, 1);
  assert.equal(turnContext?.identifier?.rank, 1);
  assert.ok(turnContext?.content.includes('Ship release Atlas'));
  assert.ok(turnContext?.content.includes('Yes, do it.'));
  assert.ok(!turnContext?.content.includes('RAW_RELEASE_TOOL_OUTPUT'));
  assert.ok(!turnContext?.content.includes('private deployment plan'));

  const requestsBeforeRebuild = embeddingInputs.length;
  await rm(config.databasePath, { recursive: true });
  await rm(config.statePath);
  const rebuilt = await service.index();

  assert.equal(rebuilt.totalChunks, 7);
  assert.equal(rebuilt.indexSummary.cacheHits, 4);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(embeddingInputs.length, requestsBeforeRebuild);
});

void test('recall service fuses lexical-only tool evidence with dense conversation results', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-tool-evidence-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'tools.jsonl');
  const conversationContent = 'We diagnosed a file permission failure.';
  const toolCommand = 'cat /tmp/locked-file';
  const toolResult =
    'EPERM readNodeErrorCode /tmp/locked-file https://example.test/failure?id=EPERM';
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'tool-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/tool-project',
      },
      {
        type: 'message',
        id: 'assistant-tools',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: conversationContent },
            { type: 'thinking', thinking: 'never retrieve this private plan' },
            {
              type: 'toolCall',
              id: 'call-tools',
              name: 'bash',
              arguments: { command: toolCommand },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'result-tools',
        parentId: 'assistant-tools',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-tools',
          toolName: 'bash',
          content: [{ type: 'text', text: toolResult }],
          isError: true,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const embeddedInputs: string[] = [];
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        embeddedInputs.push(...texts);
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const indexed = await service.index();
  const exactError = await service.search('EPERM readNodeErrorCode', 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  const toolEvidence = exactError.results.find((result) => result.documentKind === 'tool');
  const conversation = exactError.results.find((result) => result.documentKind === 'conversation');

  assert.equal(indexed.totalChunks, 4);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 1);
  assert.ok(toolEvidence);
  assert.equal(toolEvidence.entryId.value, 'result-tools');
  assert.equal(toolEvidence.evidenceKind, 'tool_result');
  assert.equal(toolEvidence.evidencePart, 'result');
  assert.equal(toolEvidence.toolCallId, 'call-tools');
  assert.equal(toolEvidence.toolName, 'bash');
  assert.equal(toolEvidence.toolCallEntryId?.value, 'assistant-tools');
  assert.equal(toolEvidence.toolResultEntryId?.value, 'result-tools');
  assert.equal(toolEvidence.toolError, true);
  assert.equal(toolEvidence.sessionPath, sessionPath);
  assert.equal(toolEvidence.dense, null);
  assert.equal(toolEvidence.lexical?.rank, 1);
  assert.equal(toolEvidence.identifier?.rank, 1);
  assert.ok(conversation?.dense);
  assert.equal(conversation?.documentKind, 'conversation');
  assert.deepEqual(embeddedInputs, [
    RECALL_EMBEDDING_CANARY_TEXT,
    conversationContent,
    RECALL_EMBEDDING_CANARY_TEXT,
    'EPERM readNodeErrorCode',
  ]);

  const exactCommand = await service.search(toolCommand, 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  const callArguments = exactCommand.results.find((result) => result.evidencePart === 'arguments');
  assert.equal(callArguments?.content, `{"command":"${toolCommand}"}`);
  assert.equal(callArguments?.toolResultEntryId?.value, 'result-tools');
  assert.ok(exactCommand.results.every((result) => !result.content.includes('private plan')));

  const exactUrl = await service.search('https://example.test/failure?id=EPERM', 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    exactUrl.results.find((result) => result.evidencePart === 'result')?.content,
    toolResult,
  );
  const exactToolName = await service.search('bash', 2, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    exactToolName.results.find((result) => result.evidencePart === 'name')?.content,
    'bash',
  );
});

void test('fresh zvec rebuild reuses unchanged cached chunk vectors without embedding requests', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-cache-rebuild-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await writeFile(
    sessionPath,
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'session-1',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Reuse this durable vector.' },
      }),
    ].join('\n') + '\n',
  );

  const embeddingInputs: string[][] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddingInputs.push([...texts]);
      return texts.map((text) =>
        text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [text.length, 1, 0],
      );
    },
  };
  const config = createTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const first = await service.index();
  assert.equal(first.indexSummary.cacheHits, 0);
  assert.equal(first.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(first.indexSummary.embeddingRequestCount, 1);
  assert.equal(embeddingInputs.length, 2);

  await rm(config.databasePath, { recursive: true });
  await rm(config.statePath);

  const rebuilt = await service.index();
  assert.equal(rebuilt.totalChunks, 1);
  assert.equal(rebuilt.indexSummary.cacheHits, 1);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(embeddingInputs.length, 2);
});

void test('schema migration keeps canonical cache identity across tolerated canary jitter', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-canary-jitter-rebuild-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'one.jsonl'),
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'canary-jitter-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'canary-jitter-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'assistant', content: 'Reuse this vector across serving slots.' },
      }),
    ].join('\n') + '\n',
  );
  let useJitteredCanary = false;
  let canaryRequests = 0;
  const config = createTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            canaryRequests += 1;
            return useJitteredCanary ? [1, 0.001, 0] : [1, 0, 0];
          }
          return [0, 1, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const first = await service.index();
  const firstManifest = await readRecallIndexManifest(config.manifestPath);
  assert.ok(firstManifest);
  await writeFile(
    config.manifestPath,
    JSON.stringify({ ...firstManifest, manifestVersion: firstManifest.manifestVersion - 1 }),
  );
  useJitteredCanary = true;
  const rebuilt = await service.index({ rebuild: true });
  const rebuiltManifest = await readRecallIndexManifest(config.manifestPath);

  assert.equal(first.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(rebuilt.indexSummary.cacheHits, 1);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(
    rebuiltManifest?.embedding.canaryFingerprint,
    firstManifest?.embedding.canaryFingerprint,
  );
  assert.equal(canaryRequests, 2);
});

void test('explicit indexing retries a transient embedding-canary failure in the same process', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-canary-retry-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let canaryRequests = 0;
  let tokenizerLoads = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts() {
        canaryRequests += 1;
        if (canaryRequests === 1) {
          throw new Error('temporary canary failure');
        }
        return [[0, 0, 1]];
      },
    },
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
  });

  await assert.rejects(() => service.index(), /temporary canary failure/);
  const retried = await service.index();

  assert.equal(retried.totalChunks, 0);
  assert.equal(canaryRequests, 2);
  assert.equal(tokenizerLoads, 1);
});

void test('recall search refuses a missing manifest before opening or mutating index state', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-missing-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let embeddingRequests = 0;
  let tokenizerLoads = 0;
  let storeOpens = 0;
  const embeddings: LocalEmbeddingClient = {
    async embedTexts() {
      embeddingRequests += 1;
      return [[0, 0, 1]];
    },
  };
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('store must not open without a manifest');
    },
  });

  await assert.rejects(
    () => service.search('must remain read only', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall index manifest missing.*\/pi-session-recall-index --rebuild/,
  );
  await assert.rejects(
    () =>
      service.reconcileSession(join(sessionsDirectory, 'active.jsonl'), {
        lifecycleTrigger: RecallLifecycleTrigger.AGENT_SETTLED,
      }),
    /Recall automatic session ingestion requires an existing index generation.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(embeddingRequests, 0);
  assert.equal(tokenizerLoads, 0);
  assert.equal(storeOpens, 0);
});

void test('recall search detects an embedding model swap in the same service process', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-live-model-swap-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let modelSwapped = false;
  let canaryRequests = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            canaryRequests += 1;
            return modelSwapped ? [0, 1, 0] : [0, 0, 1];
          }
          return [1, 0, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  await service.index();
  await service.search('before model swap', 1, { scope: RecallSearchScope.GLOBAL });
  modelSwapped = true;

  await assert.rejects(
    () => service.search('after model swap', 1, { scope: RecallSearchScope.GLOBAL }),
    /embedding\.canaryCosineSimilarity.*\/pi-session-recall-index --rebuild/s,
  );

  await service.index({ rebuild: true });
  const rebuiltManifest = await readRecallIndexManifest(join(directory, 'index-manifest.json'));
  assert.equal(
    rebuiltManifest?.embedding.canaryFingerprint,
    createRecallEmbeddingCanaryFingerprint([0, 1, 0], 3),
  );
  assert.equal(canaryRequests, 4);
});

void test('ordinary indexing detects a model swap before embedding new session content', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-index-model-swap-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let modelSwapped = false;
  let contentEmbeddingRequests = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return modelSwapped ? [0, 0, 1] : [1, 0, 0];
          }
          contentEmbeddingRequests += 1;
          return [0, 1, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();
  await writeFile(
    join(sessionsDirectory, 'new-after-swap.jsonl'),
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'new-after-swap',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'new-after-swap-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Never mix this new-model vector.' },
      }),
    ].join('\n') + '\n',
  );
  modelSwapped = true;

  await assert.rejects(
    () => service.index(),
    /embedding\.canaryCosineSimilarity.*\/pi-session-recall-index --rebuild/s,
  );
  assert.equal(contentEmbeddingRequests, 0);
});

void test('recall search reports an incompatible manifest before opening zvec', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-incompatible-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const actualManifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: 'mean',
    },
    canaryEmbedding: [0, 0, 1],
  });
  await writeRecallIndexManifest(config.manifestPath, actualManifest);
  let storeOpens = 0;
  let tokenizerLoads = 0;
  const embeddings: LocalEmbeddingClient = {
    async embedTexts() {
      return [[0, 0, 1]];
    },
  };
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('incompatible index must not open');
    },
  });

  await assert.rejects(
    () => service.search('incompatible', 1, { scope: RecallSearchScope.GLOBAL }),
    /embedding\.pooling.*expected "last", received "mean".*\/pi-session-recall-index --rebuild/s,
  );
  assert.equal(storeOpens, 0);
  assert.equal(tokenizerLoads, 0);
});

void test('explicit rebuild replaces incompatible index metadata while preserving vector cache', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-explicit-rebuild-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const incompatibleManifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: 'mean',
    },
    canaryEmbedding: [0, 0, 1],
  });
  await writeRecallIndexManifest(config.manifestPath, incompatibleManifest);
  await mkdir(config.databasePath, { recursive: true });
  await writeFile(config.statePath, '{"version":1,"sessions":{}}\n');
  await mkdir(config.embeddingCacheDirectory, { recursive: true });
  const cacheSentinelPath = join(config.embeddingCacheDirectory, 'preserve-me');
  await writeFile(cacheSentinelPath, 'durable vector cache');
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts() {
        return [[0, 0, 1]];
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const rebuilt = await service.index({ rebuild: true });
  const rebuiltManifest = await readRecallIndexManifest(config.manifestPath);

  assert.equal(rebuilt.totalChunks, 0);
  assert.equal(rebuiltManifest?.embedding.pooling, 'last');
  assert.equal(await readFile(cacheSentinelPath, 'utf8'), 'durable vector cache');
});

void test('explicit rebuild preserves the old generation when model preflight fails', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-rebuild-preflight-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const oldManifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: 'mean',
    },
    canaryEmbedding: [0, 0, 1],
  });
  await writeRecallIndexManifest(config.manifestPath, oldManifest);
  const oldState = '{"version":1,"sessions":{}}\n';
  await writeFile(config.statePath, oldState);
  await mkdir(config.databasePath, { recursive: true });
  const databaseSentinelPath = join(config.databasePath, 'old-generation');
  await writeFile(databaseSentinelPath, 'preserve old generation');
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts() {
        throw new Error('embedding preflight unavailable');
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  await assert.rejects(
    () =>
      service.index({
        rebuild: true,
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
      }),
    /embedding preflight unavailable/,
  );

  assert.deepEqual(await readRecallIndexManifest(config.manifestPath), oldManifest);
  assert.equal(await readFile(config.statePath, 'utf8'), oldState);
  assert.equal(await readFile(databaseSentinelPath, 'utf8'), 'preserve old generation');

  const invalidCanaryService = createRecallConversationService(config, {
    embeddings: {
      async embedTexts() {
        return [[0, 1]];
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await assert.rejects(
    () =>
      invalidCanaryService.index({
        rebuild: true,
        manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD,
      }),
    /canary dimension mismatch: expected 3, received 2/,
  );

  assert.deepEqual(await readRecallIndexManifest(config.manifestPath), oldManifest);
  assert.equal(await readFile(config.statePath, 'utf8'), oldState);
  assert.equal(await readFile(databaseSentinelPath, 'utf8'), 'preserve old generation');
});

void test('explicit indexing refuses unmanifested legacy state before tokenizer or zvec access', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-legacy-state-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const legacyState = '{"version":1,"sessions":{}}\n';
  await writeFile(config.statePath, legacyState);
  let embeddingRequests = 0;
  let tokenizerLoads = 0;
  let storeOpens = 0;
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts() {
        embeddingRequests += 1;
        return [[0, 0, 1]];
      },
    },
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('legacy index must not open');
    },
  });

  await assert.rejects(
    () => service.index(),
    /manifest missing.*existing index data.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(await readFile(config.statePath, 'utf8'), legacyState);
  assert.equal(embeddingRequests, 0);
  assert.equal(tokenizerLoads, 0);
  assert.equal(storeOpens, 0);
});

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { RecallDiagnosticsMode, RecallGenerationCutoverState } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  createRecallActiveGenerationPointer,
  RECALL_GENERATION_REGISTRY_VERSION,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import {
  createOctenEmbeddingModelProfile,
  type RecallEmbeddingModelProfile,
} from './recall-model-profiles.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
    };
  },
};

function createPhysicalSourceGenerationTestConfig(
  dataDirectory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    dataDirectory,
    databasePath: join(dataDirectory, 'legacy-zvec'),
    projectionDatabasePath: join(dataDirectory, 'legacy-session-projections'),
    statePath: join(dataDirectory, 'legacy-index-state.json'),
    manifestPath: join(dataDirectory, 'legacy-index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    lockPath: join(dataDirectory, 'operation.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(dataDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(dataDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(dataDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(dataDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(dataDirectory, 'generations'),
    activeGenerationPointerPath: join(dataDirectory, 'active-generation.json'),
    generationRegistryPath: join(dataDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(dataDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(dataDirectory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-embedding-model',
    embeddingServedModelId: 'test/embedding-model',
    embeddingArtifact: 'test-embedding-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker-model',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
}

function createToolOnlyLogicalSession(
  rawSessionId: string,
  entryPrefix: string,
  cwd: string,
  searchableToken: string,
): Record<string, unknown>[] {
  const callId = `${entryPrefix}-call`;
  return [
    {
      type: 'session',
      version: 3,
      id: rawSessionId,
      timestamp: '2026-08-01T00:00:00.000Z',
      cwd,
    },
    {
      type: 'message',
      id: `${entryPrefix}-assistant`,
      parentId: null,
      timestamp: '2026-08-01T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [
          {
            type: 'toolCall',
            id: callId,
            name: 'read',
            arguments: { path: `/tmp/${searchableToken}.txt` },
          },
        ],
      },
    },
    {
      type: 'message',
      id: `${entryPrefix}-result`,
      parentId: `${entryPrefix}-assistant`,
      timestamp: '2026-08-01T00:00:02.000Z',
      message: {
        role: 'toolResult',
        toolCallId: callId,
        toolName: 'read',
        isError: false,
        content: [{ type: 'text', text: `${searchableToken} source evidence` }],
      },
    },
  ];
}

async function writeJsonl(
  path: string,
  records: readonly Record<string, unknown>[],
): Promise<void> {
  await writeFile(path, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`);
}

void test('configured service builds and searches a stored-width dense subset beside lexical-only evidence', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-dense-source-generation-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'mixed.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'mixed-session',
      timestamp: '2026-08-02T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'mixed-assistant',
      parentId: null,
      timestamp: '2026-08-02T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [
          { type: 'text', text: 'The dense constellation records the retained design decision.' },
          {
            type: 'toolCall',
            id: 'mixed-call',
            name: 'read',
            arguments: { path: '/tmp/LEXICAL_ONLY_NEEDLE.txt' },
          },
        ],
      },
    },
    {
      type: 'message',
      id: 'mixed-result',
      parentId: 'mixed-assistant',
      timestamp: '2026-08-02T00:00:02.000Z',
      message: {
        role: 'toolResult',
        toolCallId: 'mixed-call',
        toolName: 'read',
        isError: false,
        content: [{ type: 'text', text: 'LEXICAL_ONLY_NEEDLE source evidence' }],
      },
    },
  ]);

  const baseProfile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-native-model',
      servedModelId: 'fixture/native-model',
      artifact: 'fixture-native-model.fp32',
      artifactSha256: 'a'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  const profile: RecallEmbeddingModelProfile = Object.freeze({
    ...baseProfile,
    canary: Object.freeze({
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'fixture stored-width canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    }),
  });
  const documentInputs: string[] = [];
  const queryInputs: string[] = [];
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        documentInputs.push(...documents);
        return documents.map((document) =>
          document.includes('dense constellation') ? [0, 4, 100] : [4, 0, 100],
        );
      },
      async embedQuery(query) {
        queryInputs.push(query);
        return [0, 5, 200];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'b'.repeat(64) }],
    },
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });

  const generationId = 'generation_stored_width';
  const created = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  assert.ok(created.storeCounts.dense > 0);
  assert.ok(documentInputs.some((input) => input.includes('dense constellation')));
  assert.ok(documentInputs.every((input) => !input.includes('LEXICAL_ONLY_NEEDLE')));
  assert.equal(existsSync(join(dataDirectory, 'embedding-cache')), false);

  const generationDirectory = join(config.generationRootDirectory, generationId);
  const manifest: unknown = JSON.parse(
    await readFile(join(generationDirectory, 'index-manifest.json'), 'utf8'),
  );
  assert.ok(isUnknownRecord(manifest));
  const embeddingProfile: unknown = manifest.embeddingProfile;
  assert.ok(isUnknownRecord(embeddingProfile));
  assert.equal(embeddingProfile.nativeDimensions, 3);
  assert.equal(embeddingProfile.storedDimensions, 2);
  assert.equal(embeddingProfile.reduction, 'first-n-then-l2');
  assert.ok(isUnknownRecord(embeddingProfile.canary));
  assert.equal(embeddingProfile.canary.expectedNativeDimensions, 3);
  assert.equal(embeddingProfile.canary.expectedStoredDimensions, 2);

  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  try {
    assert.equal(dense.schema.vectors()[0]?.dimension, 2);
    assert.equal(dense.stats.docCount, created.storeCounts.dense);
    const denseRows = await dense.query({
      topk: dense.stats.docCount,
      outputFields: [
        'evidenceOccurrenceId',
        'embeddingProfileId',
        'storedDimensions',
        'evidenceChecksum',
        'embeddingInputChecksum',
        'vectorChecksum',
      ],
      includeVector: true,
    });
    assert.ok(denseRows.every((row) => Object.keys(row.vectors.embedding ?? {}).length === 2));
    assert.ok(denseRows.every((row) => row.fields.storedDimensions === 2));
    assert.ok(denseRows.every((row) => row.fields.evidenceOccurrenceId === row.id));
    assert.ok(denseRows.every((row) => String(row.fields.evidenceChecksum).length === 64));
    assert.ok(denseRows.every((row) => String(row.fields.embeddingInputChecksum).length === 64));
    assert.ok(denseRows.every((row) => String(row.fields.vectorChecksum).length === 64));
  } finally {
    dense.closeSync();
  }

  const results = await service.searchRecallGenerationHybrid(
    generationId,
    'LEXICAL_ONLY_NEEDLE constellation',
    10,
  );
  assert.deepEqual(queryInputs, [
    'fixture stored-width canary',
    'fixture stored-width canary',
    'LEXICAL_ONLY_NEEDLE constellation',
  ]);
  assert.ok(results.some((result) => result.denseRank !== null));
  assert.ok(
    results.some(
      (result) => result.lexicalRank !== null && result.evidence.isDenseSearchable === false,
    ),
  );
  assert.ok(
    results.every((result) => result.evidence.evidenceOccurrenceId.startsWith('occurrence_')),
  );
  assert.ok(results.every((result) => result.evidence.sessionsRootRelativePath === 'mixed.jsonl'));

  const damagedDense = ZVecOpen(join(generationDirectory, 'dense'));
  try {
    const [row] = await damagedDense.query({
      topk: 1,
      outputFields: [
        'schemaVersion',
        'generationId',
        'evidenceOccurrenceId',
        'physicalSourceIdentity',
        'logicalSessionOccurrenceId',
        'embeddingProfileId',
        'storedDimensions',
        'evidenceChecksum',
        'embeddingInputChecksum',
        'vectorChecksum',
        'projectIdentity',
        'projectIdentityDigest',
      ],
      includeVector: true,
    });
    assert.ok(row);
    const [status] = damagedDense.upsertSync([
      {
        id: row.id,
        fields: { ...row.fields, evidenceChecksum: '0'.repeat(64) },
        vectors: row.vectors,
      },
    ]);
    assert.equal(status?.ok, true);
  } finally {
    damagedDense.closeSync();
  }
  await assert.rejects(
    service.openValidatedRecallGeneration(generationId),
    /Recall coherent generation dense evidence checksum mismatch/u,
  );
});

void test('configured service resumes one fixed source snapshot after an interrupted dense write', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-resume-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'snapshot.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'snapshot-session',
      timestamp: '2026-08-03T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'snapshot-message',
      parentId: null,
      timestamp: '2026-08-03T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'ORIGINAL_SNAPSHOT_NEEDLE remains recoverable.' }],
      },
    },
  ]);

  const profile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-native-model',
      servedModelId: 'fixture/native-model',
      artifact: 'fixture-native-model.fp32',
      artifactSha256: 'c'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  const embeddedDocuments: string[] = [];
  let interruptDenseWrite = true;
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        embeddedDocuments.push(...documents);
        return documents.map(() => [3, 4, 100]);
      },
      async embedQuery() {
        return [3, 4, 100];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'd'.repeat(64) }],
    },
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
    async fixedSnapshotBuildFault(stage) {
      if (stage === 'after-dense-write' && interruptDenseWrite) {
        interruptDenseWrite = false;
        throw new Error('fixture interrupted dense write');
      }
    },
  });

  const generationId = 'generation_fixed_snapshot_resume';
  await assert.rejects(
    service.createRecallGenerationFromPhysicalSources({
      generationId,
      physicalSessionPaths: [sourcePath],
    }),
    /fixture interrupted dense write/u,
  );
  const generationDirectory = join(config.generationRootDirectory, generationId);
  assert.equal(existsSync(join(generationDirectory, 'write-recovery.json')), true);
  assert.equal(existsSync(join(generationDirectory, 'validation-receipt.json')), false);

  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'changed-session',
      timestamp: '2026-08-04T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'changed-message',
      parentId: null,
      timestamp: '2026-08-04T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'LATER_SOURCE_CHANGE must remain outside the build.' }],
      },
    },
  ]);

  const addedSourcePath = join(sessionsDirectory, 'added-after-snapshot.jsonl');
  await writeJsonl(
    addedSourcePath,
    createToolOnlyLogicalSession(
      'added-session',
      'added',
      projectDirectory,
      'ADDED_AFTER_SNAPSHOT',
    ),
  );
  await assert.rejects(
    service.createRecallGenerationFromPhysicalSources({
      generationId,
      physicalSessionPaths: [sourcePath, addedSourcePath],
    }),
    /Recall fixed snapshot generation resume source snapshot mismatch/u,
  );
  await rm(sourcePath);

  const resumed = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  assert.deepEqual(await service.openValidatedRecallGeneration(generationId), resumed);
  assert.equal(existsSync(join(generationDirectory, 'write-recovery.json')), false);
  assert.equal(existsSync(join(generationDirectory, 'validation-receipt.json')), true);
  assert.ok(
    (await service.searchRecallGenerationLexical(generationId, 'ORIGINAL_SNAPSHOT_NEEDLE', 10))
      .length >= 1,
  );
  assert.deepEqual(
    await service.searchRecallGenerationLexical(generationId, 'LATER_SOURCE_CHANGE', 10),
    [],
  );
  assert.deepEqual(
    await service.searchRecallGenerationLexical(generationId, 'ADDED_AFTER_SNAPSHOT', 10),
    [],
  );
  assert.equal(
    embeddedDocuments.filter((document) => document.includes('ORIGINAL_SNAPSHOT_NEEDLE')).length,
    1,
  );

  await rm(join(generationDirectory, 'build-sources'), { recursive: true });
  assert.deepEqual(await service.openValidatedRecallGeneration(generationId), resumed);
});

void test('configured fixed-snapshot build waits for a transient checkpoint store lock', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-lock-'));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'checkpoint-lock.jsonl');
  let checkpointLock: ReturnType<typeof ZVecOpen> | undefined;
  let releaseTimer: ReturnType<typeof setTimeout> | undefined;
  t.after(async () => {
    if (releaseTimer !== undefined) {
      clearTimeout(releaseTimer);
    }
    checkpointLock?.closeSync();
    await rm(disposableRoot, { recursive: true, force: true });
  });
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'checkpoint-lock-session',
      timestamp: '2026-08-04T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'checkpoint-lock-message',
      parentId: null,
      timestamp: '2026-08-04T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'TRANSIENT_CHECKPOINT_LOCK remains recoverable.' }],
      },
    },
  ]);

  const profile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-native-model',
      servedModelId: 'fixture/native-model',
      artifact: 'fixture-native-model.fp32',
      artifactSha256: 'e'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  let checkpointLockCount = 0;
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        return documents.map(() => [3, 4, 100]);
      },
      async embedQuery() {
        return [3, 4, 100];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'f'.repeat(64) }],
    },
    loadTokenizer: async () => tokenizer,
    workerSignal: { signalDetachedWorker() {} },
    fixedSnapshotBuildFault(stage, context) {
      if (stage !== 'after-store-close') {
        return;
      }
      checkpointLockCount += 1;
      checkpointLock = ZVecOpen(join(context.generationDirectory, 'lexical-source'));
      releaseTimer = setTimeout(() => {
        checkpointLock?.closeSync();
        checkpointLock = undefined;
        releaseTimer = undefined;
      }, 75);
    },
  });

  const generationId = 'generation_transient_checkpoint_lock';
  const created = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  assert.equal(checkpointLockCount, 1);
  assert.equal(checkpointLock, undefined);
  assert.deepEqual(await service.openValidatedRecallGeneration(generationId), created);
});

void test('configured service resolves current-build duplicates before copying a validated compatible generation', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-vector-lanes-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'duplicates.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'duplicate-session',
      timestamp: '2026-08-05T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'duplicate-first',
      parentId: null,
      timestamp: '2026-08-05T00:00:01.000Z',
      message: { role: 'assistant', content: 'BUILD_LOCAL_DUPLICATE' },
    },
    {
      type: 'message',
      id: 'duplicate-second',
      parentId: 'duplicate-first',
      timestamp: '2026-08-05T00:00:02.000Z',
      message: { role: 'assistant', content: 'BUILD_LOCAL_DUPLICATE' },
    },
  ]);

  const profile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-native-model',
      servedModelId: 'fixture/native-model',
      artifact: 'fixture-native-model.fp32',
      artifactSha256: 'e'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  const embeddedDocuments: string[] = [];
  let permitRecomputation = true;
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        if (!permitRecomputation) {
          throw new Error('fixture recomputation forbidden');
        }
        embeddedDocuments.push(...documents);
        return documents.map(() => [6, 8, 100]);
      },
      async embedQuery() {
        return [6, 8, 100];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'f'.repeat(64) }],
    },
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });

  const sourceGenerationId = 'generation_validated_vector_source';
  await service.createRecallGenerationFromPhysicalSources({
    generationId: sourceGenerationId,
    physicalSessionPaths: [sourcePath],
  });
  assert.equal(
    embeddedDocuments.filter((document) => document === 'BUILD_LOCAL_DUPLICATE').length,
    1,
  );

  permitRecomputation = false;
  const copied = await service.createRecallGenerationFromPhysicalSources({
    generationId: 'generation_copied_vectors',
    physicalSessionPaths: [sourcePath],
    validatedVectorSourceGenerationId: sourceGenerationId,
  });
  assert.ok(copied.storeCounts.dense >= 2);
  assert.equal(existsSync(join(dataDirectory, 'embedding-cache')), false);
});

void test('configured service withholds validation receipt after a reopened path mismatch', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-validation-fault-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'validation.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(
    sourcePath,
    createToolOnlyLogicalSession(
      'validation-session',
      'validation',
      projectDirectory,
      'VALIDATION_PATH_NEEDLE',
    ),
  );

  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
    async fixedSnapshotBuildFault(stage, context) {
      if (stage !== 'before-validation-receipt') {
        return;
      }
      const lexicalSource = ZVecOpen(join(context.generationDirectory, 'lexical-source'));
      try {
        const [row] = await lexicalSource.query({
          filter: "recordKind = 'entry-anchor'",
          topk: 1,
          outputFields: lexicalSource.schema.fields().map(({ name }) => name),
          includeVector: false,
        });
        assert.ok(row);
        const [status] = lexicalSource.upsertSync([
          {
            id: row.id,
            fields: { ...row.fields, sessionsRootRelativePath: 'wrong/path.jsonl' },
          },
        ]);
        assert.equal(status?.ok, true);
      } finally {
        lexicalSource.closeSync();
      }
    },
  });

  const generationId = 'generation_validation_path_fault';
  await assert.rejects(
    service.createRecallGenerationFromPhysicalSources({
      generationId,
      physicalSessionPaths: [sourcePath],
    }),
    /Recall fixed snapshot generation lexical\/source validation row mismatch/u,
  );
  assert.equal(
    existsSync(join(config.generationRootDirectory, generationId, 'validation-receipt.json')),
    false,
  );
});

void test('configured service never receipts a build cancelled at validation', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-fixed-snapshot-cancelled-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'cancelled.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(
    sourcePath,
    createToolOnlyLogicalSession(
      'cancelled-session',
      'cancelled',
      projectDirectory,
      'CANCELLED_VALIDATION_NEEDLE',
    ),
  );

  const controller = new AbortController();
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
    fixedSnapshotBuildFault(stage) {
      if (stage === 'before-validation-receipt') {
        controller.abort();
      }
    },
  });

  const generationId = 'generation_cancelled_validation';
  await assert.rejects(
    service.createRecallGenerationFromPhysicalSources({
      generationId,
      physicalSessionPaths: [sourcePath],
      signal: controller.signal,
    }),
    /Recall fixed snapshot generation build cancelled/u,
  );
  assert.equal(
    existsSync(join(config.generationRootDirectory, generationId, 'validation-receipt.json')),
    false,
  );
});

void test('configured service keys lexical evidence, anchors, and projections by physical source identity', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-physical-source-generation-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const relocatedSessionsDirectory = join(disposableRoot, 'relocated-sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const firstSourcePath = join(sessionsDirectory, 'team', 'session.jsonl');
  const reusedSourcePath = join(sessionsDirectory, 'archive', 'session.jsonl');
  await Promise.all([
    mkdir(join(sessionsDirectory, 'team'), { recursive: true }),
    mkdir(join(sessionsDirectory, 'archive'), { recursive: true }),
    mkdir(relocatedSessionsDirectory),
    mkdir(dataDirectory),
    mkdir(projectDirectory),
  ]);
  await writeJsonl(
    firstSourcePath,
    createToolOnlyLogicalSession('colliding-raw-session', 'first', projectDirectory, 'alpha_token'),
  );
  await writeJsonl(reusedSourcePath, [
    ...createToolOnlyLogicalSession(
      'colliding-raw-session',
      'reused-first',
      projectDirectory,
      'beta_token',
    ),
    ...createToolOnlyLogicalSession(
      'colliding-raw-session',
      'reused-second',
      projectDirectory,
      'gamma_token',
    ),
  ]);

  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });
  const generationId = 'generation_physical_sources';
  const created = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [firstSourcePath, reusedSourcePath],
  });
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: created.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
    ],
  });
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);

  assert.deepEqual(created.storeCounts, {
    lexicalSource: 15,
    dense: 0,
    sessionProjection: 5,
  });
  assert.deepEqual(await service.openValidatedRecallGeneration(generationId), created);
  const firstSource = resolveRecallPhysicalSourceIdentity(sessionsDirectory, firstSourcePath);
  const reusedSource = resolveRecallPhysicalSourceIdentity(sessionsDirectory, reusedSourcePath);
  assert.notEqual(firstSource.physicalSourceIdentity, reusedSource.physicalSourceIdentity);
  assert.deepEqual(
    resolveRecallPhysicalSourceIdentity(
      relocatedSessionsDirectory,
      join(relocatedSessionsDirectory, 'team', 'session.jsonl'),
    ),
    firstSource,
  );

  const generationDirectory = join(config.generationRootDirectory, generationId);
  const recoveryRecordPath = join(generationDirectory, 'write-recovery.json');
  await writeFile(
    recoveryRecordPath,
    `${JSON.stringify({
      version: 1,
      generationId,
      operation: 'delete-physical-source',
      physicalSourceIdentity: firstSource.physicalSourceIdentity,
    })}\n`,
  );
  await assert.rejects(
    service.searchRecallGenerationLexical(generationId, 'alpha_token', 10),
    /Recall coherent generation recovery required/u,
  );
  await rm(recoveryRecordPath);

  const alphaMatches = await service.searchRecallGenerationLexical(generationId, 'alpha_token', 10);
  const betaMatches = await service.searchRecallGenerationLexical(generationId, 'beta_token', 10);
  const gammaMatches = await service.searchRecallGenerationLexical(generationId, 'gamma_token', 10);
  assert.ok(alphaMatches.length >= 1);
  assert.ok(betaMatches.length >= 1);
  assert.ok(gammaMatches.length >= 1);
  assert.ok(alphaMatches.every((match) => match.isDenseSearchable === false));
  assert.ok(alphaMatches.every((match) => match.rawSessionId === 'colliding-raw-session'));
  assert.ok(
    alphaMatches.every(
      (match) => match.physicalSourceIdentity === firstSource.physicalSourceIdentity,
    ),
  );
  assert.ok(
    betaMatches.every(
      (match) => match.physicalSourceIdentity === reusedSource.physicalSourceIdentity,
    ),
  );
  assert.notEqual(
    betaMatches[0]?.logicalSessionOccurrenceId,
    gammaMatches[0]?.logicalSessionOccurrenceId,
  );
  assert.ok(alphaMatches.every((match) => match.projectIdentity !== ''));
  assert.ok(alphaMatches.every((match) => match.sourceLineStart >= 2));
  assert.ok(alphaMatches.every((match) => match.evidenceOccurrenceId.startsWith('occurrence_')));

  const lexicalSource = ZVecOpen(join(generationDirectory, 'lexical-source'), { readOnly: true });
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  const projections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(lexicalSource.schema.vectors().length, 0);
    assert.equal(dense.stats.docCount, 0);
    assert.equal(dense.schema.vectors().length, 1);
    assert.equal(projections.stats.docCount, 5);
    const anchorRows = await lexicalSource.query({
      filter: `recordKind = 'entry-anchor' AND physicalSourceIdentity = '${firstSource.physicalSourceIdentity}'`,
      topk: 10,
      outputFields: [
        'entryAnchorId',
        'entryId',
        'parentEntryId',
        'sourceOrder',
        'entryStartByte',
        'entryEndByte',
        'branchPathLeafIds',
        'projectIdentity',
      ],
      includeVector: false,
    });
    assert.equal(anchorRows.length, 2);
    const resultAnchor = anchorRows.find((row) => row.fields.entryId === 'first-result');
    assert.ok(resultAnchor);
    assert.equal(resultAnchor.fields.parentEntryId, 'first-assistant');
    assert.equal(resultAnchor.fields.sourceOrder, 3);
    assert.ok(
      Number(resultAnchor.fields.entryEndByte) > Number(resultAnchor.fields.entryStartByte),
    );
    assert.deepEqual(resultAnchor.fields.branchPathLeafIds, ['first-result']);
    assert.notEqual(resultAnchor.fields.projectIdentity, '');

    const sourceProjectionRows = await projections.query({
      filter: `physicalSourceIdentity = '${reusedSource.physicalSourceIdentity}'`,
      topk: 10,
      outputFields: [
        'projectionKind',
        'physicalSourceIdentity',
        'logicalSessionOccurrenceId',
        'projectionJson',
      ],
      includeVector: false,
    });
    assert.equal(sourceProjectionRows.length, 3);
    assert.equal(
      sourceProjectionRows.filter((row) => row.fields.projectionKind === 'physical_session').length,
      1,
    );
    assert.equal(
      sourceProjectionRows.filter((row) => row.fields.projectionKind === 'logical_session').length,
      2,
    );
    const physicalProjectionRow = sourceProjectionRows.find(
      (row) => row.fields.projectionKind === 'physical_session',
    );
    assert.ok(physicalProjectionRow);
    const physicalProjection: unknown = JSON.parse(
      String(physicalProjectionRow.fields.projectionJson),
    );
    assert.ok(isUnknownRecord(physicalProjection));
    const expectedMembership: unknown = physicalProjection.expectedMembership;
    assert.ok(isUnknownRecord(expectedMembership));
    assert.ok(isUnknownRecord(expectedMembership.lexicalSource));
    assert.equal(expectedMembership.lexicalSource.count, 10);
    assert.match(String(expectedMembership.lexicalSource.digest), /^[a-f0-9]{64}$/u);
    assert.ok(isUnknownRecord(expectedMembership.dense));
    assert.equal(expectedMembership.dense.count, 0);
    assert.equal(String(expectedMembership.dense.digest).length, 64);
    assert.ok(isUnknownRecord(expectedMembership.sessionProjection));
    assert.equal(expectedMembership.sessionProjection.count, 3);
    assert.equal(String(expectedMembership.sessionProjection.digest).length, 64);
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
    projections.closeSync();
  }

  await service.transferIncrementalRecallWorkPlan({
    confirmedPhysicalSourceDeletion: {
      targetGenerationId: generationId,
      physicalSourceIdentity: firstSource.physicalSourceIdentity,
    },
  });
  assert.deepEqual(
    await service.searchRecallGenerationLexical(generationId, 'alpha_token', 10),
    [],
  );
  assert.ok(
    (await service.searchRecallGenerationLexical(generationId, 'beta_token', 10)).length >= 1,
  );

  const afterDeletionLexical = ZVecOpen(join(generationDirectory, 'lexical-source'), {
    readOnly: true,
  });
  const afterDeletionProjections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(afterDeletionLexical.stats.docCount, 10);
    assert.equal(afterDeletionProjections.stats.docCount, 3);
  } finally {
    afterDeletionLexical.closeSync();
    afterDeletionProjections.closeSync();
  }
});

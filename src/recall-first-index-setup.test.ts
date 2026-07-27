import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallDiagnosticsMode } from './enums.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createFirstIndexSetupTestConfig(
  dataDirectory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    databasePath: join(dataDirectory, 'zvec'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'embedding-cache'),
    lockPath: join(dataDirectory, 'operation.lock'),
    generationsDirectory: join(dataDirectory, 'index-generations'),
    activeGenerationPath: join(dataDirectory, 'active-generation.json'),
    stagingGenerationPath: join(dataDirectory, 'staging-generation.json'),
    backgroundIndexStatusPath: join(dataDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath: join(dataDirectory, 'background-index-request.json'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'http://unused-embedding.test/v1',
    embeddingModel: 'test-embedding',
    embeddingServedModelId: 'test-embedding',
    embeddingArtifact: 'test-embedding.gguf',
    embeddingQuantization: 'test',
    embeddingPooling: 'mean',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'unused-reranker',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
}

function createFirstIndexSession(sessionId: string, text: string): string {
  return (
    [
      {
        type: 'session',
        version: 3,
        id: sessionId,
        timestamp: '2026-08-01T10:00:00Z',
        cwd: '/first-index-project',
      },
      {
        type: 'message',
        id: `${sessionId}-assistant`,
        parentId: null,
        timestamp: '2026-08-01T10:01:00Z',
        message: { role: 'assistant', content: [{ type: 'text', text }] },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n'
  );
}

void test('conversation service inspects first-index corpus metadata without model work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-first-index-inspection-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  await mkdir(join(sessionsDirectory, 'nested'), { recursive: true });
  await writeFile(join(sessionsDirectory, 'first.jsonl'), '1234');
  await writeFile(join(sessionsDirectory, 'nested', 'second.jsonl'), '123456');
  await writeFile(join(sessionsDirectory, 'ignored.txt'), 'not a session');

  const service = createRecallConversationService(
    createFirstIndexSetupTestConfig(join(root, 'data'), sessionsDirectory),
    {
      embeddingProvider: {
        async embedQuery() {
          assert.fail('metadata inspection must not embed a query');
        },
        async embedDocuments() {
          assert.fail('metadata inspection must not embed documents');
        },
      },
      async loadTokenizer() {
        assert.fail('metadata inspection must not load a tokenizer');
      },
      openStore() {
        assert.fail('metadata inspection must not open zvec');
      },
    },
  );

  const inspection = await service.inspectConversationCorpus();

  assert.deepEqual(inspection, {
    sessionCount: 2,
    sourceByteSize: 10,
  });
});

void test('conversation service verifies selected embedding semantics without indexing', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-first-index-verification-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const profile: RecallEmbeddingModelProfile = {
    identity: {
      requestModel: 'verified-fixture',
      servedModelId: 'verified-fixture',
      artifact: 'verified-fixture.gguf',
      dimensions: 3,
      quantization: 'fixture',
      pooling: 'mean',
      normalization: 'l2',
    },
    queryInputPrefix: 'query: ',
    documentInputPrefix: 'document: ',
    canary: {
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'verify selected embedding',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    },
  };
  let queryCount = 0;
  let tokenizerLoadCount = 0;
  const service = createRecallConversationService(
    createFirstIndexSetupTestConfig(join(root, 'data'), join(root, 'sessions')),
    {
      embeddingProfile: profile,
      tokenizerIdentity: {
        model: 'verified-fixture',
        revision: 'verified-revision',
        library: { name: 'verified-tokenizer', version: '1' },
        encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
        assets: [{ fileName: 'verified.gguf', sha256: 'b'.repeat(64) }],
      },
      embeddingProvider: {
        async embedQuery() {
          queryCount += 1;
          return [1, 0, 0];
        },
        async embedDocuments() {
          assert.fail('embedding selection verification must not embed corpus documents');
        },
      },
      async loadTokenizer() {
        tokenizerLoadCount += 1;
        return { encodeConversationText: () => ({ ids: [1] }) };
      },
      openStore() {
        assert.fail('embedding selection verification must not open zvec');
      },
    },
  );

  const verification = await service.verifyEmbeddingCapability();

  assert.deepEqual(verification, {
    embeddingProfileId: verification.embeddingProfileId,
    model: 'verified-fixture',
    dimensions: 3,
    normalization: 'l2',
    tokenizerModel: 'verified-fixture',
  });
  assert.match(verification.embeddingProfileId, /^embedding-profile-[a-f0-9]{64}$/u);
  assert.equal(queryCount, 2);
  assert.equal(tokenizerLoadCount, 1);
});

void test('conversation service measures a bounded sample and full rebuild reuses its cached vectors', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-first-index-sample-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    join(sessionsDirectory, 'small.jsonl'),
    createFirstIndexSession('small-session', 'sample cache sentinel'),
  );
  await writeFile(
    join(sessionsDirectory, 'large.jsonl'),
    createFirstIndexSession('large-session', 'full build additional evidence with more bytes'),
  );

  const profile: RecallEmbeddingModelProfile = {
    identity: {
      requestModel: 'first-index-fixture',
      servedModelId: 'first-index-fixture',
      artifact: 'first-index-fixture.gguf',
      dimensions: 3,
      quantization: 'fixture',
      pooling: 'mean',
      normalization: 'l2',
    },
    queryInputPrefix: 'query: ',
    documentInputPrefix: 'document: ',
    canary: {
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'first index canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    },
  };
  const embeddedDocuments: string[] = [];
  let monotonicMilliseconds = 0;
  const dataDirectory = join(root, 'data');
  const config = createFirstIndexSetupTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    tokenizerIdentity: {
      model: 'first-index-fixture',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'a'.repeat(64) }],
    },
    embeddingProvider: {
      async embedQuery() {
        monotonicMilliseconds += 20;
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        monotonicMilliseconds += documents.length * 10;
        embeddedDocuments.push(...documents);
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      monotonicMilliseconds += 5;
      return {
        encodeConversationText(text) {
          return { ids: Array.from(text).map((_, index) => index) };
        },
      };
    },
    diagnosticsClock: {
      monotonicMilliseconds() {
        return monotonicMilliseconds;
      },
      wallClockIsoTimestamp() {
        return '2026-08-01T10:00:00.000Z';
      },
    },
  });

  const measurement = await service.measureFirstIndexSample({ maximumSessionCount: 1 });

  assert.equal(measurement.corpus.sessionCount, 2);
  assert.equal(measurement.sampledSessionCount, 1);
  assert.equal(measurement.newlyEmbeddedDocumentCount, 1);
  assert.equal(measurement.cacheHitCount, 0);
  assert.equal(measurement.coldStartMilliseconds, 45);
  assert.equal(measurement.measuredSampleMilliseconds, 10);
  assert.ok(measurement.sourceBytesPerSecond > 0);
  assert.ok(measurement.estimatedDurationMilliseconds.minimum > 0);
  assert.ok(
    measurement.estimatedDurationMilliseconds.maximum >=
      measurement.estimatedDurationMilliseconds.minimum,
  );
  assert.match(await readFile(config.stagingGenerationPath ?? '', 'utf8'), /"status":"resumable"/u);

  const rebuilt = await service.index({ rebuild: true });

  assert.equal(rebuilt.indexSummary.cacheHits, 1);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(embeddedDocuments.length, 2);
});

import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { LocalEmbeddingClient } from './local-embedding-client.js';
import {
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: text
        .split(/\s+/u)
        .filter(Boolean)
        .map((_, index) => index),
    };
  },
};

function createTestConfig(directory: string, sessionsDirectory: string) {
  return {
    sessionsDirectory,
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    lockPath: join(directory, 'recall.lock'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-request-model',
    embeddingServedModelId: 'test-served-model',
    embeddingArtifact: 'test-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
  };
}

void test('recall service indexes explicitly and searches a compatible index without updating it', async (t) => {
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
  assert.equal(indexed.indexSummary.embeddedChunks, 2);
  assert.equal(indexed.totalChunks, 2);

  const first = await service.search('What did we decide about job queues?', 1);
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
  const second = await service.search('queue decision', 1);
  assert.equal(second.totalChunks, 2);
  assert.equal(tokenizerLoads, 1);
  assert.deepEqual(embeddedInputs, [
    RECALL_EMBEDDING_CANARY_TEXT,
    'We chose a durable queue for job delivery.',
    'The navigation bar is blue.',
    'What did we decide about job queues?',
    'queue decision',
  ]);

  const lockOwner = `${JSON.stringify({ pid: 999_999_999 })}\n`;
  await mkdir(join(directory, 'recall.lock'));
  await writeFile(join(directory, 'recall.lock', 'owner.json'), lockOwner);
  await assert.rejects(
    () => service.search('must not clear a stale lock', 1),
    /read-only search did not remove the lock/,
  );
  assert.equal(await readFile(join(directory, 'recall.lock', 'owner.json'), 'utf8'), lockOwner);
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
    () => service.search('must remain read only', 1),
    /Recall index manifest missing.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(embeddingRequests, 0);
  assert.equal(tokenizerLoads, 0);
  assert.equal(storeOpens, 0);
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
    canaryFingerprint: createRecallEmbeddingCanaryFingerprint([0, 0, 1], 3),
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
    () => service.search('incompatible', 1),
    /embedding\.pooling.*expected "last", received "mean".*\/pi-session-recall-index --rebuild/s,
  );
  assert.equal(storeOpens, 0);
  assert.equal(tokenizerLoads, 0);
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

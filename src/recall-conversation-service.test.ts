import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  normalizeRecallProjectLineages,
  parseProjectIdentity,
} from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';

function sessionLines(entries: object[]): string {
  return `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`;
}

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createTestEmbeddingVector(text: string): number[] {
  if (/atlas|manual|zvec/iu.test(text)) {
    return [1, 0, 0];
  }
  if (/queue/iu.test(text)) {
    return [0, 1, 0];
  }
  return [0, 0, 1];
}

const TEST_EMBEDDING_PROVIDER: RecallEmbeddingProvider = {
  embedQuery(query) {
    return Promise.resolve(createTestEmbeddingVector(query));
  },
  embedDocuments(documents) {
    return Promise.resolve(documents.map(createTestEmbeddingVector));
  },
};

function createTestConfig(root: string): RecallConversationConfig {
  const data = join(root, 'recall');
  return {
    sessionsDirectory: join(root, 'sessions'),
    databasePath: join(data, 'zvec'),
    statePath: join(data, 'index-state.json'),
    manifestPath: join(data, 'index-manifest.json'),
    tokenizerCacheDirectory: join(data, 'tokenizers'),
    lockPath: join(data, 'operation.lock'),
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-test',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 4,
    embeddingStoredDimensions: 3,
    embeddingBatchSize: 8,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
}

async function writeConversationSession(
  path: string,
  content: string,
  sessionId = 'session-1',
): Promise<void> {
  await mkdir(join(path, '..'), { recursive: true });
  await writeFile(
    path,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: sessionId,
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: `user-${sessionId}`,
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content },
      },
    ]),
  );
}

void test('service builds one zvec collection and performs read-only hybrid search', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'Use one manually maintained zvec collection.');
  const openedModes: string[] = [];
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    openStore(mode) {
      openedModes.push(mode);
      return openZvecConversationStore({
        databasePath: config.databasePath,
        dimensions: config.embeddingStoredDimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      });
    },
  });

  const indexed = await service.index({ rebuild: true, optimize: true });
  const stateBeforeSearch = await readFile(config.statePath, 'utf8');
  const search = await service.search('manual zvec', 5, { scope: RecallSearchScope.GLOBAL });
  const stateAfterSearch = await readFile(config.statePath, 'utf8');

  assert.equal(indexed.indexSummary.indexedSessions, 1);
  assert.ok(indexed.totalChunks >= 1);
  assert.equal(search.results[0]?.sessionPath, sessionPath);
  assert.equal(search.results[0]?.sourceLineStart, 2);
  assert.equal(search.searchPolicy.rankingMode, 'hybrid');
  assert.deepEqual(search.searchPolicy.candidateLimits, { dense: 8, lexical: 8, identifier: 8 });
  assert.equal(stateAfterSearch, stateBeforeSearch);
  assert.deepEqual(openedModes, ['write', 'read']);
});

void test('search never catches up a session changed after explicit indexing', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-read-only-search-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'Original manual recall evidence.');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const before = await readFile(config.statePath, 'utf8');
  await writeConversationSession(sessionPath, 'NEW_UNINDEXED_SOURCE_MARKER');

  const search = await service.search('NEW_UNINDEXED_SOURCE_MARKER', 5, {
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(await readFile(config.statePath, 'utf8'), before);
  assert.ok(
    search.results.every((result) => !result.content.includes('NEW_UNINDEXED_SOURCE_MARKER')),
  );
});

void test('manual rebuild replaces incompatible stored-dimension manifests', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-rebuild-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'Atlas rebuild evidence.',
  );
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });

  const changedConfig = { ...config, embeddingNativeDimensions: 3, embeddingStoredDimensions: 2 };
  const changedProvider: RecallEmbeddingProvider = {
    async embedQuery() {
      return [1, 0];
    },
    async embedDocuments(documents) {
      return documents.map(() => [1, 0]);
    },
  };
  const changedService = createRecallConversationService(changedConfig, {
    embeddingProvider: changedProvider,
    loadTokenizer: async () => tokenizer,
  });

  await assert.rejects(
    changedService.search('Atlas', 5, { scope: RecallSearchScope.GLOBAL }),
    /embedding\.nativeDimensions[\s\S]*psr index --rebuild/,
  );
  const rebuilt = await changedService.index({ rebuild: true });
  assert.equal(rebuilt.indexSummary.indexedSessions, 1);
  const search = await changedService.search('Atlas', 5, { scope: RecallSearchScope.GLOBAL });
  assert.equal(search.results.length, 1);
});

void test('ordinary indexing preserves the existing generation when its manifest is incompatible', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-incompatible-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(join(config.sessionsDirectory, 'one.jsonl'), 'Atlas evidence.');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const manifestBefore = await readFile(config.manifestPath, 'utf8');
  const incompatible = createRecallConversationService(
    { ...config, embeddingStoredDimensions: 2 },
    {
      embeddingProvider: TEST_EMBEDDING_PROVIDER,
      loadTokenizer: async () => tokenizer,
    },
  );

  await assert.rejects(incompatible.index(), /psr index --rebuild/);
  assert.equal(await readFile(config.manifestPath, 'utf8'), manifestBefore);
});

void test('project scope filters every retrieval channel before final ranking', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-project-scope-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'Atlas evidence for this project.',
  );
  const projectIdentity = parseProjectIdentity('non-git-session-origin:/project');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    async resolveProjectIdentity(workingDirectory) {
      return workingDirectory === '/project'
        ? {
            projectIdentity,
            identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
          }
        : null;
    },
  });
  await service.index({ rebuild: true });

  const included = await service.search('Atlas', 5, {
    scope: RecallSearchScope.PROJECT,
    invocationDirectory: '/project',
  });
  const global = await service.search('Atlas', 5, { scope: RecallSearchScope.GLOBAL });

  assert.equal(included.results.length, 1);
  assert.equal(global.results.length, 1);
  await assert.rejects(
    service.search('Atlas', 5, {
      scope: RecallSearchScope.PROJECT,
      invocationDirectory: '/other',
    }),
    /could not resolve a project identity/,
  );
});

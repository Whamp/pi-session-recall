import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
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
    indexMaintenanceStatusPath: join(data, 'index-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(data, 'physical-session-ignore.json'),
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

async function writePhysicalSessionIgnoreState(
  config: RecallConversationConfig,
  ignoredPhysicalSessionPaths: readonly string[],
): Promise<void> {
  await mkdir(join(config.physicalSessionIgnoreStatePath, '..'), { recursive: true });
  await writeFile(
    config.physicalSessionIgnoreStatePath,
    `${JSON.stringify({ version: 1, ignoredPhysicalSessionPaths })}\n`,
    'utf8',
  );
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

void test('standalone optimization compacts the existing collection without indexing sessions', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-standalone-optimize-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'Force optimization fixture evidence.',
  );
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const progressKinds: string[] = [];

  const result = await service.optimize({
    onProgress(event) {
      progressKinds.push(event.kind);
    },
  });

  assert.ok(result.totalChunks > 0);
  assert.deepEqual(progressKinds, ['optimizing-collection', 'completed']);
});

void test('standalone optimization remains repeatable after later indexed writes', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-repeatable-optimize-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'first.jsonl'),
    'Initial repeatable optimization evidence.',
  );
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const firstOptimization = await service.optimize();

  await writeConversationSession(
    join(config.sessionsDirectory, 'second.jsonl'),
    'Later repeatable optimization evidence.',
  );
  const laterIndex = await service.index();
  const secondOptimization = await service.optimize();

  assert.equal(laterIndex.indexSummary.indexedSessions, 1);
  assert.ok(secondOptimization.totalChunks > firstOptimization.totalChunks);
});

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
    getCurrentTime: () => new Date('2026-07-25T12:00:00.000Z'),
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
  const statusBeforeSearch = await readFile(config.indexMaintenanceStatusPath, 'utf8');
  const search = await service.search('manual zvec', 5, { scope: RecallSearchScope.GLOBAL });
  const stateAfterSearch = await readFile(config.statePath, 'utf8');
  const statusAfterSearch = await readFile(config.indexMaintenanceStatusPath, 'utf8');

  assert.equal(indexed.indexSummary.indexedSessions, 1);
  assert.ok(indexed.totalChunks >= 1);
  assert.equal(search.results[0]?.sessionPath, sessionPath);
  assert.equal(search.results[0]?.sourceLineStart, 2);
  assert.equal(search.searchPolicy.rankingMode, 'hybrid');
  assert.deepEqual(search.searchPolicy.candidateLimits, { dense: 8, lexical: 8, identifier: 8 });
  assert.deepEqual(search.indexMaintenanceStatus, {
    version: 1,
    completedAt: '2026-07-25T12:00:00.000Z',
    scannedSessions: 1,
    failedSessions: 0,
  });
  assert.equal(statusBeforeSearch, `${JSON.stringify(search.indexMaintenanceStatus, null, 2)}\n`);
  assert.equal(stateAfterSearch, stateBeforeSearch);
  assert.equal(statusAfterSearch, statusBeforeSearch);
  assert.deepEqual(
    (await readdir(join(root, 'recall'))).filter((name) =>
      name.startsWith('index-maintenance-status.json.'),
    ),
    [],
  );
  assert.deepEqual(openedModes, ['write', 'read']);
});

void test('completed no-op Index maintenance refreshes its durable status', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-no-op-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'No-op maintenance evidence.',
  );
  const completedTimes = [
    new Date('2026-07-25T12:00:00.000Z'),
    new Date('2026-07-25T12:30:00.000Z'),
  ];
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    getCurrentTime() {
      const currentTime = completedTimes.shift();
      assert.ok(currentTime);
      return currentTime;
    },
  });

  await service.index({ rebuild: true });
  const noOp = await service.index();
  const search = await service.search('No-op', 5, { scope: RecallSearchScope.GLOBAL });

  assert.equal(noOp.indexSummary.scannedSessions, 1);
  assert.equal(noOp.indexSummary.indexedSessions, 0);
  assert.deepEqual(search.indexMaintenanceStatus, {
    version: 1,
    completedAt: '2026-07-25T12:30:00.000Z',
    scannedSessions: 1,
    failedSessions: 0,
  });
});

void test('completed failures publish status while fatal and interrupted passes preserve it', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-failure-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'damaged.jsonl');
  await mkdir(config.sessionsDirectory, { recursive: true });
  await writeFile(sessionPath, 'not JSON\n', 'utf8');
  await writeFile(join(config.sessionsDirectory, 'also-damaged.jsonl'), 'still not JSON\n', 'utf8');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T13:00:00.000Z'),
  });

  const completedWithFailure = await service.index({ rebuild: true });
  const completedStatus = await readFile(config.indexMaintenanceStatusPath, 'utf8');

  assert.equal(completedWithFailure.indexSummary.scannedSessions, 2);
  assert.equal(completedWithFailure.indexSummary.failedSessions.length, 2);
  assert.deepEqual(JSON.parse(completedStatus), {
    version: 1,
    completedAt: '2026-07-25T13:00:00.000Z',
    scannedSessions: 2,
    failedSessions: 2,
  });

  await writeConversationSession(sessionPath, 'A fatal embedding attempt follows.');
  const fatalService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments() {
        throw new Error('Fatal embedding failure');
      },
    },
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T14:00:00.000Z'),
  });
  await assert.rejects(fatalService.index(), /Fatal embedding failure/u);
  assert.equal(await readFile(config.indexMaintenanceStatusPath, 'utf8'), completedStatus);

  const abortController = new AbortController();
  const interruptedService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments(documents, signal) {
        void documents;
        abortController.abort(new Error('Index maintenance interrupted'));
        signal?.throwIfAborted();
        return [];
      },
    },
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T15:00:00.000Z'),
  });
  await assert.rejects(
    interruptedService.index({ signal: abortController.signal }),
    /Index maintenance interrupted/u,
  );
  assert.equal(await readFile(config.indexMaintenanceStatusPath, 'utf8'), completedStatus);
});

void test('aborted no-op maintenance preserves the prior completed status', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-aborted-no-op-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'No-op interruption evidence.',
  );
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T15:00:00.000Z'),
  });
  await service.index({ rebuild: true });
  const completedStatus = await readFile(config.indexMaintenanceStatusPath, 'utf8');
  const abortController = new AbortController();

  await assert.rejects(
    service.index({
      signal: abortController.signal,
      onProgress(event) {
        if (event.kind === 'maintenance-workset-planned') {
          abortController.abort(new Error('No-op maintenance interrupted'));
        }
      },
    }),
    /Recall conversation operation cancelled/u,
  );
  assert.equal(await readFile(config.indexMaintenanceStatusPath, 'utf8'), completedStatus);
});

void test('failed rebuild removes stale status and successful rebuild replaces it', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-rebuild-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'Rebuild maintenance evidence.',
  );
  const initialService = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T16:00:00.000Z'),
  });
  await initialService.index({ rebuild: true });

  const failingRebuild = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments() {
        throw new Error('Rebuild embedding failure');
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  await assert.rejects(failingRebuild.index({ rebuild: true }), /Rebuild embedding failure/u);
  await assert.rejects(readFile(config.indexMaintenanceStatusPath, 'utf8'), /ENOENT/u);

  const successfulRebuild = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
    getCurrentTime: () => new Date('2026-07-25T17:00:00.000Z'),
  });
  await successfulRebuild.index({ rebuild: true });
  assert.deepEqual(JSON.parse(await readFile(config.indexMaintenanceStatusPath, 'utf8')), {
    version: 1,
    completedAt: '2026-07-25T17:00:00.000Z',
    scannedSessions: 1,
    failedSessions: 0,
  });
});

void test('service maintenance reads persisted ignores and rebuild preserves them', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-ignore-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const ignoredPath = join(config.sessionsDirectory, 'ignored.jsonl');
  const eligiblePath = join(config.sessionsDirectory, 'eligible.jsonl');
  await writeConversationSession(ignoredPath, 'IGNORED_POLICY_EVIDENCE', 'ignored');
  await writeConversationSession(eligiblePath, 'ELIGIBLE_POLICY_EVIDENCE', 'eligible');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  await writePhysicalSessionIgnoreState(config, [ignoredPath]);

  const removed = await service.index();
  assert.equal(removed.indexSummary.removedSessions, 1);
  const searchAfterIgnore = await service.search('IGNORED_POLICY_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.ok(searchAfterIgnore.results.every((result) => result.sessionPath !== ignoredPath));
  const ignoreStateBeforeRebuild = await readFile(config.physicalSessionIgnoreStatePath, 'utf8');

  const rebuilt = await service.index({ rebuild: true });

  assert.equal(rebuilt.indexSummary.indexedSessions, 1);
  assert.equal(
    await readFile(config.physicalSessionIgnoreStatePath, 'utf8'),
    ignoreStateBeforeRebuild,
  );
  const stateText = await readFile(config.statePath, 'utf8');
  const manifestText = await readFile(config.manifestPath, 'utf8');
  assert.ok(stateText.includes(JSON.stringify(eligiblePath)));
  assert.ok(!stateText.includes(JSON.stringify(ignoredPath)));
  assert.ok(!stateText.includes('ignoredPhysicalSessionPaths'));
  assert.ok(!manifestText.includes('ignoredPhysicalSessionPaths'));
  assert.ok(!manifestText.includes('physical-session-ignore.json'));
});

void test('malformed ignore state aborts rebuild before deleting a working index generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-ignore-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'WORKING_GENERATION_EVIDENCE');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const stateBefore = await readFile(config.statePath, 'utf8');
  const manifestBefore = await readFile(config.manifestPath, 'utf8');
  await writeFile(config.physicalSessionIgnoreStatePath, '{"version":1}\n', 'utf8');

  await assert.rejects(service.index({ rebuild: true }), /Physical session ignore state invalid/u);

  assert.equal(await readFile(config.statePath, 'utf8'), stateBefore);
  assert.equal(await readFile(config.manifestPath, 'utf8'), manifestBefore);
  const search = await service.search('WORKING_GENERATION_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(search.results.length, 1);
});

void test('ignore mutation after service snapshot acquisition affects the next pass', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-ignore-snapshot-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'SNAPSHOT_POLICY_EVIDENCE');
  await writePhysicalSessionIgnoreState(config, [sessionPath]);
  let mutateAfterSnapshot = true;
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    async loadTokenizer() {
      if (mutateAfterSnapshot) {
        mutateAfterSnapshot = false;
        await writePhysicalSessionIgnoreState(config, []);
      }
      return tokenizer;
    },
  });

  const first = await service.index({ rebuild: true });
  assert.equal(first.indexSummary.indexedSessions, 0);
  assert.equal(first.totalChunks, 0);

  const second = await service.index();
  assert.equal(second.indexSummary.indexedSessions, 1);
  assert.ok(second.totalChunks > 0);
});

void test('search tolerates unavailable status without catching up changed sessions', async (t) => {
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
  const stateBeforeSearch = await readFile(config.statePath, 'utf8');
  await writeConversationSession(sessionPath, 'NEW_UNINDEXED_SOURCE_MARKER');
  const unavailableStatuses: Array<{ name: string; content: string | null }> = [
    { name: 'missing', content: null },
    { name: 'malformed JSON', content: '{\n' },
    {
      name: 'unsupported version',
      content:
        '{"version":2,"completedAt":"2026-07-25T12:00:00.000Z","scannedSessions":1,"failedSessions":0}\n',
    },
    {
      name: 'invalid count',
      content:
        '{"version":1,"completedAt":"2026-07-25T12:00:00.000Z","scannedSessions":-1,"failedSessions":0}\n',
    },
    {
      name: 'invalid timestamp',
      content:
        '{"version":1,"completedAt":"2026-02-30T12:00:00.000Z","scannedSessions":1,"failedSessions":0}\n',
    },
  ];

  for (const unavailableStatus of unavailableStatuses) {
    if (unavailableStatus.content === null) {
      await rm(config.indexMaintenanceStatusPath, { force: true });
    } else {
      await writeFile(config.indexMaintenanceStatusPath, unavailableStatus.content, 'utf8');
    }

    const search = await service.search('NEW_UNINDEXED_SOURCE_MARKER', 5, {
      scope: RecallSearchScope.GLOBAL,
    });

    assert.equal(search.indexMaintenanceStatus, null, unavailableStatus.name);
    assert.equal(await readFile(config.statePath, 'utf8'), stateBeforeSearch);
    assert.ok(
      search.results.every((result) => !result.content.includes('NEW_UNINDEXED_SOURCE_MARKER')),
      unavailableStatus.name,
    );
    if (unavailableStatus.content === null) {
      await assert.rejects(readFile(config.indexMaintenanceStatusPath, 'utf8'), /ENOENT/u);
    } else {
      assert.equal(
        await readFile(config.indexMaintenanceStatusPath, 'utf8'),
        unavailableStatus.content,
        unavailableStatus.name,
      );
    }
  }
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

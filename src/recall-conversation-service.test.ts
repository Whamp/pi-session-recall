import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, readdir, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';

import { RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import { openRecallCatalog } from './recall-catalog.js';
import { openDenseRecallConversationStore } from './dense-recall-conversation-store.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  normalizeRecallProjectLineages,
  parseProjectIdentity,
} from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

function sessionLines(entries: object[]): string {
  return `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`;
}

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createTestEmbeddingVector(text: string): number[] {
  const vector = Array.from({ length: 1_024 }, () => 0);
  if (/atlas|manual|zvec|compact/iu.test(text)) {
    vector[0] = 1;
  } else if (/queue/iu.test(text)) {
    vector[1] = 1;
  } else {
    vector[2] = 1;
  }
  return vector;
}

const TEST_EMBEDDING_PROVIDER: RecallEmbeddingProvider = {
  embedQuery(query) {
    return Promise.resolve(createTestEmbeddingVector(query));
  },
  embedDocuments(documents) {
    return Promise.resolve(documents.map(createTestEmbeddingVector));
  },
};

function createTestConfig(
  root: string,
  options: { databaseGenerations?: boolean } = {},
): RecallConversationConfig {
  const data = join(root, 'recall');
  return {
    sessionsDirectory: join(root, 'sessions'),
    databasePath: join(data, 'zvec'),
    catalogPath: join(data, 'recall-catalog.sqlite'),
    statePath: join(data, 'index-state.json'),
    manifestPath: join(data, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(data, 'index-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(data, 'physical-session-ignore.json'),
    tokenizerCacheDirectory: join(data, 'tokenizers'),
    lockPath: join(data, 'operation.lock'),
    ...(options.databaseGenerations
      ? { databaseGenerationRootPath: join(data, 'generations') }
      : {}),
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-test',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 1_024,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 8,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, invocation: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
}

function readCatalogSessionPaths(catalogPath: string): string[] {
  const catalog = openRecallCatalog(catalogPath);
  try {
    return catalog.listPhysicalSessionPaths();
  } finally {
    catalog.close();
  }
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

void test('successful rebuild atomically activates a candidate and rollback restores the previous database', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-activation-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'ORIGINAL_GENERATION_EVIDENCE');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });

  const first = await service.index({ rebuild: true });
  const firstActiveTarget = await readlink(join(root, 'recall', 'active'));
  assert.equal(first.databaseTransition.kind, 'candidate-activated');
  assert.equal(first.databaseTransition.previousAvailable, false);

  await writeConversationSession(sessionPath, 'REPLACEMENT_GENERATION_EVIDENCE');
  const second = await service.index({ rebuild: true });
  const secondActiveTarget = await readlink(join(root, 'recall', 'active'));
  assert.equal(second.databaseTransition.kind, 'candidate-activated');
  assert.equal(second.databaseTransition.previousAvailable, true);
  assert.notEqual(secondActiveTarget, firstActiveTarget);
  const replacementSearch = await service.search('REPLACEMENT_GENERATION_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(replacementSearch.results[0]?.content ?? '', /REPLACEMENT_GENERATION_EVIDENCE/u);

  const rollback = await service.rollback();
  assert.equal(rollback.kind, 'previous-restored');
  assert.equal(await readlink(join(root, 'recall', 'active')), firstActiveTarget);
  const restoredSearch = await service.search('ORIGINAL_GENERATION_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(restoredSearch.results[0]?.content ?? '', /ORIGINAL_GENERATION_EVIDENCE/u);
});

void test('deferred rebuild stays inactive until the staged database target is explicitly activated', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-staging-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'ACTIVE_BEFORE_CERTIFICATION');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const activeBeforeStaging = await readlink(join(root, 'recall', 'active'));
  await writeConversationSession(sessionPath, 'STAGED_FOR_CERTIFICATION');

  const staged = await service.index({ rebuild: true, deferActivation: true });

  assert.equal(staged.databaseTransition.kind, 'candidate-staged');
  assert.equal(await readlink(join(root, 'recall', 'active')), activeBeforeStaging);
  const activeSearch = await service.search('ACTIVE_BEFORE_CERTIFICATION', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(activeSearch.results[0]?.content ?? '', /ACTIVE_BEFORE_CERTIFICATION/u);
  if (staged.databaseTransition.kind !== 'candidate-staged') {
    assert.fail('Expected the rebuilt database to remain staged');
  }

  const activation = await service.activate(staged.databaseTransition.databaseTarget);

  assert.deepEqual(activation, { kind: 'staged-activated', previousAvailable: true });
  assert.equal(
    await readlink(join(root, 'recall', 'active')),
    staged.databaseTransition.databaseTarget,
  );
  const activatedSearch = await service.search('STAGED_FOR_CERTIFICATION', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(activatedSearch.results[0]?.content ?? '', /STAGED_FOR_CERTIFICATION/u);
});

void test('staged rebuild uses a separate construction lock and leaves active search available', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-construction-lock-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'READABLE_DURING_STAGED_CONSTRUCTION');
  const stableService = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await stableService.index({ rebuild: true });

  let releaseEmbedding: (() => void) | undefined;
  const embeddingReleased = new Promise<void>((resolveEmbedding) => {
    releaseEmbedding = resolveEmbedding;
  });
  let reportEmbeddingStarted: (() => void) | undefined;
  const embeddingStarted = new Promise<void>((resolveStarted) => {
    reportEmbeddingStarted = resolveStarted;
  });
  const stagingService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments(documents) {
        reportEmbeddingStarted?.();
        await embeddingReleased;
        return documents.map(createTestEmbeddingVector);
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  const staging = stagingService.index({ rebuild: true, deferActivation: true });
  await embeddingStarted;

  await assert.rejects(readFile(join(config.lockPath, 'owner.json')), { code: 'ENOENT' });
  const activeSearch = await stableService.search('READABLE_DURING_STAGED_CONSTRUCTION', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(activeSearch.results[0]?.content ?? '', /READABLE_DURING_STAGED_CONSTRUCTION/u);

  releaseEmbedding?.();
  await staging;
});

void test('resumed staged rebuild preserves completed sessions after a fatal embedding outage', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-resume-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  await writeConversationSession(
    join(config.sessionsDirectory, 'one.jsonl'),
    'COMPLETED_BEFORE_OUTAGE',
    'session-one',
  );
  await writeConversationSession(
    join(config.sessionsDirectory, 'two.jsonl'),
    'FAILED_DURING_OUTAGE',
    'session-two',
  );
  const failingService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      embedDocuments(documents) {
        if (documents.some((document) => document.includes('FAILED_DURING_OUTAGE'))) {
          throw new Error('Embedding endpoint unavailable');
        }
        return Promise.resolve(documents.map(createTestEmbeddingVector));
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  await assert.rejects(
    failingService.index({ rebuild: true, deferActivation: true }),
    /Embedding endpoint unavailable/u,
  );
  const resumedService = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });

  const resumed = await resumedService.index({
    rebuild: true,
    deferActivation: true,
    resumeCandidate: true,
  });

  assert.equal(resumed.indexSummary.indexedSessions, 1);
  assert.equal(resumed.databaseTransition.kind, 'candidate-staged');
  if (resumed.databaseTransition.kind !== 'candidate-staged') {
    assert.fail('Expected resumed candidate to remain staged');
  }
  await resumedService.activate(resumed.databaseTransition.databaseTarget);
  const completedSearch = await resumedService.search('COMPLETED_BEFORE_OUTAGE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  const recoveredSearch = await resumedService.search('FAILED_DURING_OUTAGE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.ok(
    completedSearch.results.some((result) => /COMPLETED_BEFORE_OUTAGE/u.test(result.content)),
  );
  assert.ok(recoveredSearch.results.some((result) => /FAILED_DURING_OUTAGE/u.test(result.content)));
});

void test('fatal and interrupted rebuilds leave the active database unchanged', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-failure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'STABLE_ACTIVE_EVIDENCE');
  const stableService = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await stableService.index({ rebuild: true });
  const stableTarget = await readlink(join(root, 'recall', 'active'));

  await writeConversationSession(sessionPath, 'FAILED_CANDIDATE_EVIDENCE');
  const failingService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments() {
        throw new Error('Candidate embedding failed');
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  await assert.rejects(failingService.index({ rebuild: true }), /Candidate embedding failed/u);
  assert.equal(await readlink(join(root, 'recall', 'active')), stableTarget);

  const abortController = new AbortController();
  const interruptedService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments(documents, signal) {
        void documents;
        abortController.abort(new Error('Candidate rebuild interrupted'));
        signal?.throwIfAborted();
        return [];
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  await assert.rejects(
    interruptedService.index({ rebuild: true, signal: abortController.signal }),
    /Candidate rebuild interrupted/u,
  );
  assert.equal(await readlink(join(root, 'recall', 'active')), stableTarget);
  const search = await stableService.search('STABLE_ACTIVE_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(search.results[0]?.content ?? '', /STABLE_ACTIVE_EVIDENCE/u);
});

void test('failed candidate stays inactive and the next rebuild removes stale candidates', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-stale-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'ACTIVE_BEFORE_FAILED_CANDIDATE');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  const activeBeforeFailure = await readlink(join(root, 'recall', 'active'));
  await writeFile(join(config.sessionsDirectory, 'damaged.jsonl'), 'not JSON\n', 'utf8');

  const failed = await service.index({ rebuild: true });
  assert.equal(failed.databaseTransition.kind, 'candidate-failed');
  assert.equal(await readlink(join(root, 'recall', 'active')), activeBeforeFailure);
  assert.equal(
    (await readdir(config.databaseGenerationRootPath ?? '')).filter((name) =>
      name.startsWith('candidate-'),
    ).length,
    1,
  );

  await rm(join(config.sessionsDirectory, 'damaged.jsonl'));
  await writeConversationSession(sessionPath, 'ACTIVE_AFTER_STALE_CLEANUP');
  const recovered = await service.index({ rebuild: true });
  assert.deepEqual(recovered.databaseTransition, {
    kind: 'candidate-activated',
    previousAvailable: true,
    staleCandidatesRemoved: 1,
  });
  assert.equal(
    (await readdir(config.databaseGenerationRootPath ?? '')).filter((name) =>
      name.startsWith('candidate-'),
    ).length,
    0,
  );
});

void test('rollback rejects missing previous databases without changing the active pointer', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-missing-previous-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'FIRST_GENERATION');
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await service.index({ rebuild: true });
  await assert.rejects(service.rollback(), /No previous recall database/u);

  const firstTarget = await readlink(join(root, 'recall', 'active'));
  await writeConversationSession(sessionPath, 'SECOND_GENERATION');
  await service.index({ rebuild: true });
  const secondTarget = await readlink(join(root, 'recall', 'active'));
  await rm(resolve(join(root, 'recall'), firstTarget), { recursive: true, force: true });

  await assert.rejects(service.rollback(), /Previous recall database is missing/u);
  assert.equal(await readlink(join(root, 'recall', 'active')), secondTarget);
});

void test('first generation rebuild adopts a legacy database as rollback state', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-legacy-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const legacyConfig = createTestConfig(root);
  const sessionPath = join(legacyConfig.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'LEGACY_DATABASE_EVIDENCE');
  const legacyService = createRecallConversationService(legacyConfig, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await legacyService.index({ rebuild: true });

  await writeConversationSession(sessionPath, 'GENERATION_DATABASE_EVIDENCE');
  const generationConfig = createTestConfig(root, { databaseGenerations: true });
  const generationService = createRecallConversationService(generationConfig, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  const rebuilt = await generationService.index({ rebuild: true });
  assert.equal(rebuilt.databaseTransition.kind, 'candidate-activated');
  assert.equal(rebuilt.databaseTransition.previousAvailable, true);

  await generationService.rollback();
  assert.equal(await readlink(join(root, 'recall', 'active')), '.');
  const search = await generationService.search('LEGACY_DATABASE_EVIDENCE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.match(search.results[0]?.content ?? '', /LEGACY_DATABASE_EVIDENCE/u);
});

void test('rebuild activation, rollback, and search respect the shared writer lock', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-lock-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await writeConversationSession(sessionPath, 'LOCKED_ACTIVE_EVIDENCE');
  const stableService = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });
  await stableService.index({ rebuild: true });
  const stableTarget = await readlink(join(root, 'recall', 'active'));

  await writeConversationSession(sessionPath, 'LOCKED_CANDIDATE_EVIDENCE');
  let releaseEmbedding: (() => void) | undefined;
  const embeddingReleased = new Promise<void>((resolveEmbedding) => {
    releaseEmbedding = resolveEmbedding;
  });
  let reportEmbeddingStarted: (() => void) | undefined;
  const embeddingStarted = new Promise<void>((resolveStarted) => {
    reportEmbeddingStarted = resolveStarted;
  });
  const rebuildingService = createRecallConversationService(config, {
    embeddingProvider: {
      ...TEST_EMBEDDING_PROVIDER,
      async embedDocuments(documents) {
        reportEmbeddingStarted?.();
        await embeddingReleased;
        return documents.map(createTestEmbeddingVector);
      },
    },
    loadTokenizer: async () => tokenizer,
  });
  const rebuild = rebuildingService.index({ rebuild: true });
  await embeddingStarted;
  await assert.rejects(
    stableService.search('LOCKED_ACTIVE_EVIDENCE', 5, { scope: RecallSearchScope.GLOBAL }),
    /Recall index write lock/u,
  );
  assert.equal(await readlink(join(root, 'recall', 'active')), stableTarget);
  releaseEmbedding?.();
  await rebuild;

  const activatedTarget = await readlink(join(root, 'recall', 'active'));
  const abortController = new AbortController();
  await mkdir(config.lockPath);
  await writeFile(
    join(config.lockPath, 'owner.json'),
    `${JSON.stringify({ pid: process.pid })}\n`,
    'utf8',
  );
  await assert.rejects(
    stableService.rollback({
      signal: abortController.signal,
      onProgress(event) {
        if (event.kind === 'waiting-for-write-lock') {
          abortController.abort(new Error('Rollback lock wait cancelled'));
        }
      },
    }),
    /Rollback lock wait cancelled|AbortError/u,
  );
  assert.equal(await readlink(join(root, 'recall', 'active')), activatedTarget);
});

void test('service source search reads raw output without opening the index or embedding', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-source-search-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'source.jsonl');
  await mkdir(config.sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'source-session',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'source-result',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'source-call',
          toolName: 'bash',
          isError: true,
          content: [{ type: 'text', text: 'RAW_RESULT_ONLY hardware MODEL-9000' }],
        },
      },
    ]),
  );
  const projectIdentity = parseProjectIdentity('non-git-session-origin:/project');
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      embedQuery() {
        throw new Error('source search must not embed');
      },
      embedDocuments() {
        throw new Error('source search must not embed');
      },
    },
    loadTokenizer: async () => {
      throw new Error('source search must not load the tokenizer');
    },
    openStore() {
      throw new Error('source search must not open the index');
    },
    resolveProjectIdentity: async () => ({
      projectIdentity,
      identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
    }),
  });

  const search = await service.searchSource('MODEL-9000', 5, {
    scope: RecallSearchScope.PROJECT,
    invocationDirectory: '/trusted/project',
  });

  assert.equal(search.results[0]?.sessionPath, sessionPath);
  assert.equal(search.results[0]?.entryId, 'source-result');
  assert.equal(search.results[0]?.sourceLineStart, 2);
  assert.match(search.results[0]?.text ?? '', /RAW_RESULT_ONLY hardware MODEL-9000/u);
  await assert.rejects(readFile(config.statePath, 'utf8'), { code: 'ENOENT' });
});

void test('rebuild activates dense conversations and compact Invocations for combined normal recall', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-service-compact-layout-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root, { databaseGenerations: true });
  const sessionPath = join(config.sessionsDirectory, 'compact.jsonl');
  await mkdir(config.sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'compact-session',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'compact-conversation',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: {
          role: 'user',
          content: 'Keep the compact catalog migration source-backed and searchable.',
        },
      },
      {
        type: 'message',
        id: 'compact-tool-call',
        parentId: 'compact-conversation',
        timestamp: '2026-08-10T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'toolCall',
              id: 'compact-call',
              name: 'read',
              arguments: { path: '/project/src/compact-catalog.ts' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'compact-result',
        parentId: 'compact-tool-call',
        timestamp: '2026-08-10T10:03:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'compact-call',
          toolName: 'read',
          content: [{ type: 'text', text: 'RAW_RESULT_MUST_STAY_SOURCE_BACKED' }],
          isError: false,
        },
      },
    ]),
  );
  const service = createRecallConversationService(config, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });

  const indexed = await service.index({ rebuild: true });
  const search = await service.search('compact catalog', 2, {
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(indexed.databaseTransition.kind, 'candidate-activated');
  assert.deepEqual(indexed.documentCounts, { dense: 1, invocations: 1 });
  assert.deepEqual(search.documentCounts, { dense: 1, invocations: 1 });
  assert.deepEqual(
    search.results.map((result) => result.resultKind),
    ['conversation', 'invocation'],
  );
  assert.ok(
    search.results.every(
      (result) => !JSON.stringify(result).includes('RAW_RESULT_MUST_STAY_SOURCE_BACKED'),
    ),
  );
});

void test('service builds one compact database and performs read-only combined search', async (t) => {
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
      return openDenseRecallConversationStore({
        databasePath: config.databasePath,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      });
    },
  });

  const indexed = await service.index({ rebuild: true });
  const catalogBeforeSearch = await readFile(config.catalogPath);
  const statusBeforeSearch = await readFile(config.indexMaintenanceStatusPath, 'utf8');
  const search = await service.search('manual zvec', 5, { scope: RecallSearchScope.GLOBAL });
  const catalogAfterSearch = await readFile(config.catalogPath);
  const statusAfterSearch = await readFile(config.indexMaintenanceStatusPath, 'utf8');

  assert.equal(indexed.indexSummary.indexedSessions, 1);
  assert.ok(indexed.totalChunks >= 1);
  assert.equal(search.results[0]?.sessionPath, sessionPath);
  assert.equal(search.results[0]?.sourceLineStart, 2);
  assert.equal(search.searchPolicy.rankingMode, 'compact');
  assert.deepEqual(search.searchPolicy.candidateLimits, { dense: 8, invocation: 8 });
  assert.deepEqual(search.indexMaintenanceStatus, {
    version: 1,
    completedAt: '2026-07-25T12:00:00.000Z',
    scannedSessions: 1,
    failedSessions: 0,
  });
  assert.equal(statusBeforeSearch, `${JSON.stringify(search.indexMaintenanceStatus, null, 2)}\n`);
  assert.deepEqual(catalogAfterSearch, catalogBeforeSearch);
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
  const manifestText = await readFile(config.manifestPath, 'utf8');
  assert.deepEqual(readCatalogSessionPaths(config.catalogPath), [eligiblePath]);
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
  const catalogBefore = await readFile(config.catalogPath);
  const manifestBefore = await readFile(config.manifestPath, 'utf8');
  await writeFile(config.physicalSessionIgnoreStatePath, '{"version":1}\n', 'utf8');

  await assert.rejects(service.index({ rebuild: true }), /Physical session ignore state invalid/u);

  assert.deepEqual(await readFile(config.catalogPath), catalogBefore);
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
  const catalogBeforeSearch = await readFile(config.catalogPath);
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
    assert.deepEqual(await readFile(config.catalogPath), catalogBeforeSearch);
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

void test('manual rebuild replaces an incompatible embedding-model manifest', async (t) => {
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

  const changedConfig = { ...config, embeddingServedModelId: 'Octen/Replacement-Embedding-4B' };
  const changedService = createRecallConversationService(changedConfig, {
    embeddingProvider: TEST_EMBEDDING_PROVIDER,
    loadTokenizer: async () => tokenizer,
  });

  await assert.rejects(
    changedService.search('Atlas', 5, { scope: RecallSearchScope.GLOBAL }),
    /embedding\.servedModelId[\s\S]*psr index --rebuild/,
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

void test('project scope filters both compact fast stores before mixed-result selection', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-project-scope-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTestConfig(root);
  const sessionPath = join(config.sessionsDirectory, 'one.jsonl');
  await mkdir(config.sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'project-scope-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'project-conversation',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Atlas evidence for this project.' },
      },
      {
        type: 'message',
        id: 'project-invocation',
        parentId: 'project-conversation',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'toolCall',
              id: 'project-call',
              name: 'read',
              arguments: { path: '/project/Atlas.ts' },
            },
          ],
        },
      },
    ]),
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

  assert.deepEqual(
    included.results.map((result) => result.resultKind),
    ['conversation', 'invocation'],
  );
  assert.deepEqual(
    global.results.map((result) => result.resultKind),
    ['conversation', 'invocation'],
  );
  await assert.rejects(
    service.search('Atlas', 5, {
      scope: RecallSearchScope.PROJECT,
      invocationDirectory: '/other',
    }),
    /could not resolve a project identity/,
  );
});

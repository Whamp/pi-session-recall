import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import {
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  readRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';
import { createRecallConversationService as createProductionRecallConversationService } from './recall-conversation-service.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const execFileAsync = promisify(execFile);

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
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    embeddingCacheDirectory: join(directory, 'embedding-cache'),
    lockPath: join(directory, 'recall.lock'),
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
    projectLineages: {},
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

  const lockOwner = `${JSON.stringify({ pid: 999_999_999 })}\n`;
  await mkdir(join(directory, 'recall.lock'));
  await writeFile(join(directory, 'recall.lock', 'owner.json'), lockOwner);
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
  await execFileAsync('git', ['init'], { cwd: projectDirectory });
  await execFileAsync('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/scoped.git'], {
    cwd: projectDirectory,
  });
  await execFileAsync('git', ['init'], { cwd: unrelatedDirectory });
  await execFileAsync('git', ['remote', 'add', 'origin', 'https://github.com/Whamp/other.git'], {
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
  await execFileAsync('git', ['init'], { cwd: invocationDirectory });
  await execFileAsync('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/successor.git'], {
    cwd: invocationDirectory,
  });
  await execFileAsync('git', ['init'], { cwd: prototypeRoot });
  await execFileAsync(
    'git',
    ['remote', 'add', 'origin', 'git@github.com:Whamp/obsolete-prototype.git'],
    { cwd: prototypeRoot },
  );
  await execFileAsync('git', ['init'], { cwd: unrelatedDirectory });
  await execFileAsync('git', ['remote', 'add', 'origin', 'git@github.com:Whamp/unrelated.git'], {
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
    projectLineages: {
      'git-origin:github.com/Whamp/successor': [prototypeRoot, deletedRoot],
    },
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
        result.projectIdentity === 'git-origin:github.com/Whamp/successor' &&
        result.projectIdentitySource === RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE &&
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
  const invocationDirectory = join(directory, 'project');
  const nearbyDirectory = join(directory, 'project-nearby');
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
  assert.equal(search.results[0]?.projectIdentitySource, 'non_git_session_origin');
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
            projectIdentity: 'git-origin:github.com/Whamp/historical',
            identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
          }
        : null;
    },
  });

  await service.index();
  const search = await service.search('memoized', 3, { scope: RecallSearchScope.GLOBAL });

  assert.deepEqual(resolvedOrigins.toSorted(), ['/deleted/repository', '/historical/repository']);
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-one')?.projectIdentity,
    'git-origin:github.com/Whamp/historical',
  );
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-two')?.projectIdentitySource,
    RecallProjectIdentitySource.GIT_ORIGIN,
  );
  assert.equal(
    search.results.find((result) => result.entryId.value === 'entry-missing')?.projectIdentity,
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
      projectLineages: {
        'git-origin:github.com/Whamp/before-relocation': [historicalRoot],
      },
    },
    dependencies,
  );
  const changedService = createRecallConversationService(
    {
      ...createTestConfig(directory, sessionsDirectory),
      projectLineages: {
        'git-origin:github.com/Whamp/after-relocation': [historicalRoot],
      },
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
  assert.equal(search.results[0]?.projectIdentity, 'git-origin:github.com/Whamp/after-relocation');
  assert.equal(search.results[0]?.projectIdentitySource, 'configured_project_lineage');
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
    rankFusionVersion: 1,
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

void test('explicit rebuild keeps canonical cache identity across tolerated canary jitter', async (t) => {
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

  await assert.rejects(() => service.index({ rebuild: true }), /embedding preflight unavailable/);

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
    () => invalidCanaryService.index({ rebuild: true }),
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

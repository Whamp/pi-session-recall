import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
  type EmbeddingVectorCache,
} from './embedding-vector-cache.js';
import {
  indexChangedConversationSession,
  indexChangedConversationSessions,
} from './incremental-session-indexer.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { RecallProjectIdentitySource } from './enums.js';
import { createRecallIndexManifest } from './recall-index-manifest.js';
import { parseProjectIdentity, type ResolvedProjectIdentity } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

class MemoryConversationStore implements ConversationChunkStore {
  readonly chunks = new Map<string, IndexedSessionConversationChunk>();
  readonly deleted: string[] = [];

  upsertChunks(chunks: IndexedSessionConversationChunk[]): void {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, chunk);
    }
  }
  deleteChunks(ids: string[]): void {
    this.deleted.push(...ids);
    for (const id of ids) {
      this.chunks.delete(id);
    }
  }
  count(): number {
    return this.chunks.size;
  }
}

function sessionLines(entries: object[]): string {
  return entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n';
}

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
    };
  },
};

function createTestEmbeddingCache(
  directory: string,
  embeddings: LocalEmbeddingClient,
): EmbeddingVectorCache {
  const manifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: 'test-request-model',
      servedModelId: 'test-served-model',
      artifact: 'test-model.fp32',
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
    },
    canaryEmbedding: [0, 0, 1],
  });
  return createEmbeddingVectorCache({
    cacheDirectory: join(directory, 'embedding-cache'),
    identity: createEmbeddingVectorCacheIdentity(manifest),
    embeddingRequestBatchSize: 8,
    embeddings,
  });
}

void test('incremental index embeds only new content and removes deleted sessions', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions', '--project--');
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  const statePath = join(directory, 'state.json');
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
      id: 'user-1',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content: 'Remember the durable queue' },
    },
  ];
  await writeFile(sessionPath, sessionLines(entries));

  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddedBatches.push([...texts]);
      return texts.map((text) => [text.length, 1, 0]);
    },
  };

  const embeddingCache = createTestEmbeddingCache(directory, embeddings);
  const first = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
  });
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.cacheHits, 0);
  assert.equal(first.newlyEmbeddedChunks, 1);
  assert.equal(first.embeddingRequestCount, 1);
  assert.equal(embeddedBatches.length, 1);

  const second = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
  });
  assert.equal(second.indexedSessions, 0);
  assert.equal(second.cacheHits, 0);
  assert.equal(second.newlyEmbeddedChunks, 0);
  assert.equal(second.embeddingRequestCount, 0);
  assert.equal(embeddedBatches.length, 1);

  entries.push({
    type: 'message',
    id: 'assistant-1',
    parentId: 'user-1',
    timestamp: '2026-07-24T10:02:00Z',
    message: {
      role: 'assistant',
      content: [{ type: 'text', text: 'It is documented in docs/queue.md' }],
    },
  });
  await writeFile(sessionPath, sessionLines(entries));
  const third = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
  });
  assert.equal(third.indexedSessions, 1);
  assert.equal(third.cacheHits, 1);
  assert.equal(third.newlyEmbeddedChunks, 2);
  assert.deepEqual(embeddedBatches.at(-1), [
    'It is documented in docs/queue.md',
    'User:\nRemember the durable queue\n\nAssistant:\nIt is documented in docs/queue.md',
  ]);
  assert.equal(store.count(), 3);
  const refreshedUserChunk = [...store.chunks.values()].find(
    (chunk) => chunk.documentKind === 'conversation' && chunk.entryId.value === 'user-1',
  );
  assert.equal(refreshedUserChunk?.currentLeafId?.value, 'assistant-1');
  assert.deepEqual(
    refreshedUserChunk?.branchPathLeafIds.map((id) => id.value),
    ['assistant-1'],
  );
  assert.deepEqual(
    refreshedUserChunk?.childEntryIds.map((id) => id.value),
    ['assistant-1'],
  );

  entries.push({
    type: 'message',
    id: 'user-2',
    parentId: 'assistant-1',
    timestamp: '2026-07-24T10:03:00Z',
    message: { role: 'user', content: 'What happened next?' },
  });
  await writeFile(sessionPath, sessionLines(entries));
  const fourth = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
  });
  assert.equal(fourth.indexedSessions, 1);
  assert.equal(fourth.cacheHits, 3);
  assert.equal(fourth.newlyEmbeddedChunks, 1);
  assert.equal(fourth.embeddingRequestCount, 1);
  assert.deepEqual(embeddedBatches.at(-1), ['What happened next?']);
  const refreshedTurnContext = [...store.chunks.values()].find(
    (chunk) => chunk.documentKind === 'turn_context',
  );
  assert.equal(refreshedTurnContext?.currentLeafId?.value, 'user-2');
  assert.deepEqual(
    refreshedTurnContext?.branchPathLeafIds.map((id) => id.value),
    ['user-2'],
  );
  assert.deepEqual(
    refreshedTurnContext?.contributingEntryIds.map((id) => id.value),
    ['user-1', 'assistant-1'],
  );

  await rm(sessionPath);
  const fifth = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
  });
  assert.equal(fifth.removedSessions, 1);
  assert.equal(store.count(), 0);
  assert.equal(store.deleted.length, 4);
});

void test('targeted incremental index reconciles one active session without scanning siblings', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-targeted-indexer-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const activeSessionPath = join(sessionsDirectory, 'active.jsonl');
  const unrelatedSessionPath = join(sessionsDirectory, 'unrelated.jsonl');
  const createSession = (sessionId: string, entryId: string, content: string): object[] => [
    {
      type: 'session',
      version: 3,
      id: sessionId,
      timestamp: '2026-07-24T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: entryId,
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content },
    },
  ];
  const activeEntries = createSession('active-session', 'active-entry', 'active marker');
  await Promise.all([
    writeFile(activeSessionPath, sessionLines(activeEntries)),
    writeFile(
      unrelatedSessionPath,
      sessionLines(createSession('unrelated-session', 'unrelated-entry', 'unrelated marker')),
    ),
  ]);
  const store = new MemoryConversationStore();
  const options = {
    sessionPath: activeSessionPath,
    statePath: join(directory, 'state.json'),
    store,
    embeddingCache: createTestEmbeddingCache(directory, {
      async embedTexts(texts) {
        return texts.map((text) => [text.length, 1, 0]);
      },
    }),
    tokenizer,
  };

  const first = await indexChangedConversationSession(options);
  assert.equal(first.scannedSessions, 1);
  assert.equal(first.indexedSessions, 1);
  assert.deepEqual(
    [...store.chunks.values()].map((chunk) => chunk.content),
    ['active marker'],
  );

  activeEntries.push({
    type: 'message',
    id: 'active-response',
    parentId: 'active-entry',
    timestamp: '2026-07-24T10:02:00Z',
    message: { role: 'assistant', content: 'fresh active response' },
  });
  await writeFile(activeSessionPath, sessionLines(activeEntries));
  const second = await indexChangedConversationSession(options);
  assert.equal(second.scannedSessions, 1);
  assert.equal(second.indexedSessions, 1);
  assert.ok([...store.chunks.values()].some((chunk) => chunk.content === 'fresh active response'));
  assert.ok([...store.chunks.values()].every((chunk) => !chunk.content.includes('unrelated')));

  await rm(activeSessionPath);
  const third = await indexChangedConversationSession(options);
  assert.equal(third.removedSessions, 1);
  assert.equal(store.count(), 0);
});

void test('incremental index reconciles every logical session in one physical reuse container', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-reuse-history-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'reuse.jsonl');
  const statePath = join(directory, 'state.json');
  const firstSegment = [
    {
      type: 'session',
      version: 3,
      id: 'logical-one',
      timestamp: '2026-01-10T10:00:00Z',
      cwd: '/logical/one',
    },
    {
      type: 'message',
      id: 'repeated-entry',
      parentId: null,
      timestamp: '2026-01-10T10:00:01Z',
      message: { role: 'user', content: 'first logical evidence' },
    },
  ];
  const secondSegment = [
    {
      type: 'session',
      version: 3,
      id: 'logical-two',
      timestamp: '2026-01-10T11:00:00Z',
      cwd: '/logical/two',
    },
    {
      type: 'message',
      id: 'repeated-entry',
      parentId: null,
      timestamp: '2026-01-10T11:00:01Z',
      message: { role: 'user', content: 'second logical evidence' },
    },
  ];
  await writeFile(sessionPath, sessionLines([...firstSegment, ...secondSegment]));
  const store = new MemoryConversationStore();
  const resolvedOrigins: string[] = [];
  const options = {
    sessionPath,
    statePath,
    store,
    embeddingCache: createTestEmbeddingCache(directory, {
      async embedTexts(texts) {
        return texts.map((text) => [text.length, 1, 0]);
      },
    }),
    tokenizer,
    async resolveProjectIdentity(sessionOrigin: string): Promise<ResolvedProjectIdentity> {
      resolvedOrigins.push(sessionOrigin);
      return {
        projectIdentity: parseProjectIdentity(`non-git-session-origin:${sessionOrigin}`),
        identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
      };
    },
  };

  const first = await indexChangedConversationSession(options);
  assert.equal(first.indexedSessions, 1);
  assert.deepEqual(resolvedOrigins, ['/logical/one', '/logical/two']);
  assert.deepEqual(
    [...store.chunks.values()]
      .map((chunk) => ({
        sessionId: chunk.sessionId.value,
        projectIdentity: chunk.projectAttribution?.projectIdentity,
      }))
      .sort((left, right) => left.sessionId.localeCompare(right.sessionId)),
    [
      {
        sessionId: 'logical-one',
        projectIdentity: 'non-git-session-origin:/logical/one',
      },
      {
        sessionId: 'logical-two',
        projectIdentity: 'non-git-session-origin:/logical/two',
      },
    ],
  );
  const stateContent = await readFile(statePath, 'utf8');
  assert.match(stateContent, /^\{"version":2,"importPolicyVersion":1,"sessions":/u);
  for (const chunkId of store.chunks.keys()) {
    assert.ok(stateContent.includes(chunkId));
  }

  await writeFile(sessionPath, sessionLines(firstSegment));
  const second = await indexChangedConversationSession(options);
  assert.equal(second.indexedSessions, 1);
  assert.equal(store.count(), 1);
  assert.equal([...store.chunks.values()][0]?.sessionId.value, 'logical-one');
  assert.ok(second.deletedChunks >= 1);
});

void test('incremental index rejects state from an older session import policy', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-old-import-policy-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const statePath = join(directory, 'state.json');
  await writeFile(statePath, '{"version":1,"sessions":{}}\n');

  await assert.rejects(
    () =>
      indexChangedConversationSession({
        sessionPath: join(directory, 'absent.jsonl'),
        statePath,
        store: new MemoryConversationStore(),
        embeddingCache: createTestEmbeddingCache(directory, {
          async embedTexts(texts) {
            return texts.map((text) => [text.length, 1, 0]);
          },
        }),
        tokenizer,
      }),
    /Recall index state invalid/u,
  );
});

void test('incremental index removes stale documents when a changed session graph is corrupt', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-corrupt-change-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'changed-to-corrupt.jsonl');
  const statePath = join(directory, 'state.json');
  const header = {
    type: 'session',
    version: 3,
    id: 'changed-to-corrupt',
    timestamp: '2026-01-10T10:00:00Z',
    cwd: '/project',
  };
  await writeFile(
    sessionPath,
    sessionLines([
      header,
      {
        type: 'message',
        id: 'valid-entry',
        parentId: null,
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'initially valid evidence' },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const options = {
    sessionPath,
    statePath,
    store,
    embeddingCache: createTestEmbeddingCache(directory, {
      async embedTexts(texts) {
        return texts.map((text) => [text.length, 1, 0]);
      },
    }),
    tokenizer,
  };
  await indexChangedConversationSession(options);
  assert.equal(store.count(), 1);

  await writeFile(
    sessionPath,
    sessionLines([
      header,
      {
        type: 'message',
        id: 'orphan-entry',
        parentId: 'missing-entry',
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'corrupt evidence must not remain searchable' },
      },
    ]),
  );
  const failed = await indexChangedConversationSession(options);

  assert.equal(failed.failedSessions.length, 1);
  assert.match(failed.failedSessions[0]?.error ?? '', /missing parent missing-entry/u);
  assert.equal(failed.deletedChunks, 1);
  assert.equal(store.count(), 0);
  assert.ok(!(await readFile(statePath, 'utf8')).includes('valid-entry'));
});

void test('incremental index never sends lexical-only tool evidence to embeddings', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-tools-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'tools.jsonl'),
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'session-tools',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'assistant-tools',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: 'I will inspect the file.' },
            { type: 'thinking', thinking: 'private plan' },
            {
              type: 'toolCall',
              id: 'call-tools',
              name: 'bash',
              arguments: { command: 'cat /tmp/locked-file' },
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
          content: [{ type: 'text', text: 'EPERM /tmp/locked-file' }],
          isError: true,
        },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddedBatches.push([...texts]);
      return texts.map((text) => [text.length, 1, 0]);
    },
  };

  const summary = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath: join(directory, 'state.json'),
    store,
    embeddingCache: createTestEmbeddingCache(directory, embeddings),
    tokenizer,
  });
  const toolChunks = [...store.chunks.values()].filter((chunk) => chunk.documentKind === 'tool');

  assert.deepEqual(embeddedBatches, [['I will inspect the file.']]);
  assert.equal(summary.newlyEmbeddedChunks, 1);
  assert.equal(toolChunks.length, 3);
  assert.ok(toolChunks.every((chunk) => chunk.embedding === undefined));
  assert.ok(toolChunks.every((chunk) => !chunk.content.includes('private plan')));
});

void test('incremental index fails fast when the local embedding model is unavailable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionHeader = {
    type: 'session',
    version: 3,
    id: 'session',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  };
  for (const name of ['one', 'two']) {
    await writeFile(
      join(sessionsDirectory, `${name}.jsonl`),
      sessionLines([
        { ...sessionHeader, id: name },
        {
          type: 'message',
          id: `${name}-entry`,
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'user', content: `Remember ${name}` },
        },
      ]),
    );
  }
  let embeddingCalls = 0;
  const embeddings: LocalEmbeddingClient = {
    async embedTexts() {
      embeddingCalls += 1;
      throw new Error('Recall embedding request failed (503): unavailable');
    },
  };

  const embeddingCache = createTestEmbeddingCache(directory, embeddings);
  await assert.rejects(
    () =>
      indexChangedConversationSessions({
        sessionsDirectory,
        statePath: join(directory, 'state.json'),
        store: new MemoryConversationStore(),
        embeddingCache,
        tokenizer,
      }),
    /Recall embedding request failed/,
  );
  assert.equal(embeddingCalls, 1);
});

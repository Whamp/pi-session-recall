import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import { createTestRecallEmbeddingProvider } from './recall-test-utils.js';
import { RecallProjectIdentitySource } from './enums.js';
import { parseProjectIdentity, type ResolvedProjectIdentity } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

class MemoryConversationStore implements ConversationChunkStore {
  readonly chunks = new Map<string, IndexedSessionConversationChunk>();
  readonly deleted: string[] = [];

  async upsertChunks(chunks: readonly IndexedSessionConversationChunk[]): Promise<void> {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, chunk);
    }
  }
  async deleteChunks(ids: readonly string[]): Promise<void> {
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
  const embeddingProvider = createTestRecallEmbeddingProvider(async (texts) => {
    embeddedBatches.push([...texts]);
    return texts.map((text) => [text.length, 1, 0]);
  });

  const first = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
  });
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.newlyEmbeddedChunks, 1);
  assert.equal(first.embeddingRequestCount, 1);
  assert.equal(embeddedBatches.length, 1);

  const second = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
  });
  assert.equal(second.indexedSessions, 0);
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
    embeddingProvider,
    tokenizer,
  });
  assert.equal(third.indexedSessions, 1);
  assert.equal(third.newlyEmbeddedChunks, 3);
  assert.ok(embeddedBatches.at(-1)?.includes('Remember the durable queue'));
  assert.ok(embeddedBatches.at(-1)?.includes('It is documented in docs/queue.md'));
  assert.ok(
    embeddedBatches
      .at(-1)
      ?.includes(
        'User:\nRemember the durable queue\n\nAssistant:\nIt is documented in docs/queue.md',
      ),
  );
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
    embeddingProvider,
    tokenizer,
  });
  assert.equal(fourth.indexedSessions, 1);
  assert.equal(fourth.newlyEmbeddedChunks, 4);
  assert.equal(fourth.embeddingRequestCount, 1);
  assert.ok(embeddedBatches.at(-1)?.includes('What happened next?'));
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
    embeddingProvider,
    tokenizer,
  });
  assert.equal(fifth.removedSessions, 1);
  assert.equal(store.count(), 0);
  assert.equal(store.deleted.length, 4);
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
    sessionsDirectory: directory,
    statePath,
    store,
    embeddingProvider: createTestRecallEmbeddingProvider(async (texts) => {
      return texts.map((text) => [text.length, 1, 0]);
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

  const first = await indexChangedConversationSessions(options);
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
  assert.match(stateContent, /^\{"version":2,"importPolicyVersion":3,"sessions":/u);
  for (const chunkId of store.chunks.keys()) {
    assert.ok(stateContent.includes(chunkId));
  }

  await writeFile(sessionPath, sessionLines(firstSegment));
  const second = await indexChangedConversationSessions(options);
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
      indexChangedConversationSessions({
        sessionsDirectory: directory,
        statePath,
        store: new MemoryConversationStore(),
        embeddingProvider: createTestRecallEmbeddingProvider(async (texts) => {
          return texts.map((text) => [text.length, 1, 0]);
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
    sessionsDirectory: directory,
    statePath,
    store,
    embeddingProvider: createTestRecallEmbeddingProvider(async (texts) => {
      return texts.map((text) => [text.length, 1, 0]);
    }),
    tokenizer,
  };
  await indexChangedConversationSessions(options);
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
  const failed = await indexChangedConversationSessions(options);

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
  const embeddingProvider = createTestRecallEmbeddingProvider(async (texts) => {
    embeddedBatches.push([...texts]);
    return texts.map((text) => [text.length, 1, 0]);
  });

  const summary = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath: join(directory, 'state.json'),
    store,
    embeddingProvider,
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
  const embeddingProvider = createTestRecallEmbeddingProvider(async () => {
    embeddingCalls += 1;
    throw new Error('Recall embedding request failed (503): unavailable');
  });

  await assert.rejects(
    () =>
      indexChangedConversationSessions({
        sessionsDirectory,
        statePath: join(directory, 'state.json'),
        store: new MemoryConversationStore(),
        embeddingProvider,
        tokenizer,
      }),
    /Recall embedding request failed/,
  );
  assert.equal(embeddingCalls, 1);
});

void test('managed incremental index with retireMissingSourcesImmediately false does not delete chunks for temporarily absent sessions', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-no-immediate-retire-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'present.jsonl');
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'present-session',
        timestamp: '2026-07-27T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-27T10:00:01Z',
        message: { role: 'user', content: 'Remember this important thing' },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const embeddingProvider = createTestRecallEmbeddingProvider(async (texts) => {
    return texts.map(() => [1, 0, 0]);
  });
  const statePath = join(directory, 'state.json');

  // First pass: index the session to establish state with chunks.
  const firstSummary = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
    retireMissingSourcesImmediately: false,
  });
  assert.equal(firstSummary.indexedSessions, 1);
  const chunkCountAfterFirstIndex = store.chunks.size;
  assert.ok(chunkCountAfterFirstIndex > 0, 'should have indexed at least one chunk');

  // Simulate a temporary absence: remove the session file.
  await rm(sessionPath);

  // Second pass with retireMissingSourcesImmediately: false — should NOT delete chunks.
  const secondSummary = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
    retireMissingSourcesImmediately: false,
  });

  // Stale-path loop is skipped: no removals.
  assert.equal(secondSummary.removedSessions, 0);
  assert.equal(secondSummary.deletedChunks, 0);
  assert.equal(store.deleted.length, 0);
  // Chunks from the first pass are still present.
  assert.equal(store.chunks.size, chunkCountAfterFirstIndex);
});

void test('incremental index with retireMissingSourcesImmediately default deletes chunks for absent sessions', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-indexer-immediate-retire-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'will-disappear.jsonl');
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'disappearing-session',
        timestamp: '2026-07-27T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'entry-2',
        parentId: null,
        timestamp: '2026-07-27T10:00:01Z',
        message: { role: 'user', content: 'Content that will be deleted' },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const embeddingProvider = createTestRecallEmbeddingProvider(async (texts) => {
    return texts.map(() => [1, 0, 0]);
  });
  const statePath = join(directory, 'state.json');

  // First pass: index the session.
  await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
  });
  assert.ok(store.chunks.size > 0);

  // Remove the session file.
  await rm(sessionPath);

  // Second pass with default behavior: should delete chunks immediately.
  const secondSummary = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingProvider,
    tokenizer,
  });

  assert.equal(secondSummary.removedSessions, 1);
  assert.ok(secondSummary.deletedChunks > 0);
  assert.ok(store.deleted.length > 0);
  assert.equal(store.chunks.size, 0);
});

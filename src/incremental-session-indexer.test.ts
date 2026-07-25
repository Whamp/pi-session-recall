import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import type {
  EmbeddedSessionConversationChunk,
  RecallSearchResult,
  ZvecConversationStore,
} from './zvec-conversation-store.js';

class MemoryConversationStore implements ZvecConversationStore {
  readonly chunks = new Map<string, EmbeddedSessionConversationChunk>();
  readonly deleted: string[] = [];

  upsertChunks(chunks: EmbeddedSessionConversationChunk[]): void {
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
  search(): RecallSearchResult[] {
    return [];
  }
  fetchChecksums(ids: string[]): Map<string, string> {
    const checksums = new Map<string, string>();
    for (const id of ids) {
      const chunk = this.chunks.get(id);
      if (chunk) {
        checksums.set(id, chunk.checksum);
      }
    }
    return checksums;
  }
  async optimize(): Promise<void> {}
  close(): void {}
  count(): number {
    return this.chunks.size;
  }
}

function sessionLines(entries: object[]): string {
  return entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n';
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

  const first = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddings,
  });
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.embeddedChunks, 1);
  assert.equal(embeddedBatches.length, 1);

  const second = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddings,
  });
  assert.equal(second.indexedSessions, 0);
  assert.equal(second.embeddedChunks, 0);
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
    embeddings,
  });
  assert.equal(third.indexedSessions, 1);
  assert.equal(third.embeddedChunks, 1);
  assert.deepEqual(embeddedBatches.at(-1), ['It is documented in docs/queue.md']);
  assert.equal(store.count(), 2);

  await rm(sessionPath);
  const fourth = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddings,
  });
  assert.equal(fourth.removedSessions, 1);
  assert.equal(store.count(), 0);
  assert.equal(store.deleted.length, 2);
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

  await assert.rejects(
    () =>
      indexChangedConversationSessions({
        sessionsDirectory,
        statePath: join(directory, 'state.json'),
        store: new MemoryConversationStore(),
        embeddings,
      }),
    /Recall embedding request failed/,
  );
  assert.equal(embeddingCalls, 1);
});

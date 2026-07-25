import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
  type EmbeddingVectorCache,
} from './embedding-vector-cache.js';
import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { createRecallIndexManifest } from './recall-index-manifest.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  EmbeddedSessionConversationChunk,
} from './zvec-conversation-store.js';

class MemoryConversationStore implements ConversationChunkStore {
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
      ids: text
        .split(/\s+/u)
        .filter(Boolean)
        .map((_, index) => index),
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
    canaryFingerprint: 'a'.repeat(64),
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
  assert.equal(third.cacheHits, 0);
  assert.equal(third.newlyEmbeddedChunks, 1);
  assert.deepEqual(embeddedBatches.at(-1), ['It is documented in docs/queue.md']);
  assert.equal(store.count(), 2);

  await rm(sessionPath);
  const fourth = await indexChangedConversationSessions({
    sessionsDirectory,
    statePath,
    store,
    embeddingCache,
    tokenizer,
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

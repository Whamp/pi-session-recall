import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { SessionConversationChunk } from './session-conversation-index.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';

const baseChunk = {
  sessionId: { value: 'session-1' },
  sessionPath: '/sessions/session-1.jsonl',
  cwd: '/project',
  sessionName: 'Architecture',
  entryId: { value: 'entry-1' },
  role: 'user',
  timestamp: '2026-07-24T10:00:00Z',
  chunkIndex: 0,
} satisfies Omit<SessionConversationChunk, 'id' | 'checksum' | 'content'>;

void test('zvec conversation search returns ranked text and exact session provenance', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-'));
  const store = openZvecConversationStore({
    databasePath: join(directory, 'collection'),
    dimensions: 3,
  });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });

  store.upsertChunks([
    {
      ...baseChunk,
      id: 'chunk-a',
      content: 'We chose a durable queue for job delivery.',
      checksum: 'sum-a',
      embedding: [1, 0, 0],
    },
    {
      ...baseChunk,
      id: 'chunk-b',
      entryId: { value: 'entry-2' },
      content: 'The UI uses a blue navigation bar.',
      checksum: 'sum-b',
      embedding: [0, 1, 0],
    },
  ]);

  const results = store.search([1, 0, 0], 1);

  assert.equal(results.length, 1);
  assert.equal(results[0]?.content, 'We chose a durable queue for job delivery.');
  assert.equal(results[0]?.sessionPath, '/sessions/session-1.jsonl');
  assert.equal(results[0]?.entryId.value, 'entry-1');
  assert.equal(results[0]?.role, 'user');
  assert.equal(typeof results[0]?.score, 'number');
});

void test('zvec conversation store rejects an embedding dimension change that requires reindexing', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-dimension-'));
  const databasePath = join(directory, 'collection');
  t.after(() => rm(directory, { recursive: true, force: true }));

  const initialStore = openZvecConversationStore({ databasePath, dimensions: 3 });
  initialStore.close();

  assert.throws(
    () => openZvecConversationStore({ databasePath, dimensions: 2 }),
    /Recall zvec dimension mismatch.*reindex/,
  );
});

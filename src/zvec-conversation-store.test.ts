import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { SessionConversationChunk } from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type EmbeddedSessionConversationChunk,
} from './zvec-conversation-store.js';

const baseChunk = {
  schemaVersion: 2,
  documentKind: 'conversation',
  summaryKind: null,
  evidenceKind: 'conversation',
  sessionId: { value: 'session-1' },
  sessionPath: '/sessions/session-1.jsonl',
  parentSessionPath: '/sessions/parent.jsonl',
  cwd: '/project',
  projectPath: '/project',
  sessionName: 'Architecture',
  entryId: { value: 'entry-1' },
  parentEntryId: { value: 'parent-1' },
  childEntryIds: [{ value: 'child-1' }],
  contributingEntryIds: [{ value: 'entry-1' }],
  currentLeafId: { value: 'leaf-1' },
  branchPathLeafIds: [{ value: 'leaf-1' }, { value: 'leaf-2' }],
  isOnActiveBranch: true,
  isVisibleInActiveContext: false,
  compactedByEntryIds: [{ value: 'compaction-1' }],
  compactionFirstKeptEntryId: null,
  branchSummaryFromEntryId: null,
  role: 'user',
  timestamp: '2026-07-24T10:00:00Z',
  sourceLineStart: 3,
  sourceLineEnd: 3,
  sourceBlockStart: 1,
  sourceBlockEnd: 2,
  characterStart: 10,
  characterEnd: 52,
  tokenStart: 4,
  tokenEnd: 14,
  tokenCount: 10,
  overlapTokenCount: 2,
  textRunId: 'text-run-1',
  textRunIndex: 2,
  chunkIndex: 0,
  chunkCount: 2,
  siblingIds: ['chunk-sibling'],
  previousSiblingId: null,
  nextSiblingId: 'chunk-sibling',
  toolCallId: null,
  toolName: null,
  toolCallEntryId: null,
  toolResultEntryId: null,
  toolError: null,
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

  const firstChunk: EmbeddedSessionConversationChunk = {
    ...baseChunk,
    id: 'chunk-a',
    content: 'We chose a durable queue for job delivery.',
    checksum: 'sum-a',
    embedding: [1, 0, 0],
  };
  store.upsertChunks([
    firstChunk,
    {
      ...baseChunk,
      id: 'chunk-b',
      entryId: { value: 'entry-2' },
      contributingEntryIds: [{ value: 'entry-2' }],
      content: 'The UI uses a blue navigation bar.',
      checksum: 'sum-b',
      embedding: [0, 1, 0],
    },
  ]);

  const results = store.search([1, 0, 0], 1);

  assert.equal(results.length, 1);
  const result = results[0];
  assert.ok(result);
  const { embedding, ...expectedChunk } = firstChunk;
  const { score, ...actualChunk } = result;
  assert.deepEqual(actualChunk, expectedChunk);
  assert.equal(typeof score, 'number');
  assert.deepEqual(embedding, [1, 0, 0]);
  assert.deepEqual(store.fetchVectors(['chunk-a']), new Map([['chunk-a', [1, 0, 0]]]));
  const groups = store.groupDenseCandidates([1, 0, 0], 'entryId', 2, 1);
  assert.equal(groups[0]?.groupByValue, 'entry-1');
  assert.equal(groups[0]?.docs[0]?.id, 'chunk-a');
  assert.throws(
    () => store.groupDenseCandidates([1, 0, 0], 'entryId', 201, 1),
    /dense grouping limits invalid/,
  );
  assert.throws(() => store.search([1, 0, 0], 201), /dense candidate limit invalid/);
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

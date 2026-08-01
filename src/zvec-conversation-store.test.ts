import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createTestSessionConversationChunk } from './recall-test-utils.js';
import type { SessionConversationChunk } from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type EmbeddedSessionConversationChunk,
  type LexicalSessionConversationChunk,
} from './zvec-conversation-store.js';

const baseChunk = {
  ...createTestSessionConversationChunk({
    id: 'base-chunk',
    content: 'base content',
    sessionId: { value: 'session-1' },
    sessionPath: '/sessions/session-1.jsonl',
    parentSessionPath: '/sessions/parent.jsonl',
    sessionName: 'Architecture',
    entryId: { value: 'entry-1' },
    parentEntryId: { value: 'parent-1' },
    childEntryIds: [{ value: 'child-1' }],
    contributingEntryIds: [{ value: 'entry-1' }],
    currentLeafId: { value: 'leaf-1' },
    branchPathLeafIds: [{ value: 'leaf-1' }, { value: 'leaf-2' }],
    isVisibleInActiveContext: false,
    compactedByEntryIds: [{ value: 'compaction-1' }],
    role: 'user',
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
    chunkCount: 2,
    siblingIds: ['chunk-sibling'],
    nextSiblingId: 'chunk-sibling',
  }),
  isDenseSearchable: true,
} satisfies SessionConversationChunk & { isDenseSearchable: true };

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
    content: 'We chose a durable queue; readNodeErrorCode() handles EPERM, during job delivery.',
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
    {
      ...baseChunk,
      id: 'chunk-c',
      entryId: { value: 'entry-3' },
      contributingEntryIds: [{ value: 'entry-3' }],
      content: 'The marker contains alpha beta as one exact phrase.',
      checksum: 'sum-c',
      embedding: [0, 0, 1],
    },
    {
      ...baseChunk,
      id: 'chunk-d',
      entryId: { value: 'entry-4' },
      contributingEntryIds: [{ value: 'entry-4' }],
      content: 'The marker contains alpha with unrelated words before beta.',
      checksum: 'sum-d',
      embedding: [-1, 0, 0],
    },
  ]);

  const results = store.searchDenseCandidates([1, 0, 0], 1);

  assert.equal(results.length, 1);
  const result = results[0];
  assert.ok(result);
  const { embedding, ...expectedChunk } = firstChunk;
  const { cosineDistance, ...actualChunk } = result;
  assert.deepEqual(actualChunk, expectedChunk);
  assert.equal(typeof cosineDistance, 'number');
  assert.deepEqual(embedding, [1, 0, 0]);
  assert.deepEqual(store.fetchVectors(['chunk-a']), new Map([['chunk-a', [1, 0, 0]]]));
  assert.deepEqual(
    store.fetchConversationChunks(['chunk-a', 'missing-chunk']),
    new Map([['chunk-a', expectedChunk]]),
  );
  const lexicalResults = store.searchLexicalCandidates('readnodeerrorcode', 2);
  assert.equal(lexicalResults[0]?.id, 'chunk-a');
  assert.ok((lexicalResults[0]?.fullTextScore ?? 0) > 0);
  const identifierResults = store.searchIdentifierCandidates('readNodeErrorCode EPERM', 2);
  assert.equal(identifierResults[0]?.id, 'chunk-a');
  assert.ok((identifierResults[0]?.fullTextScore ?? 0) > 0);
  assert.deepEqual(store.searchIdentifierCandidates('readnodeerrorcode', 2), []);
  assert.deepEqual(store.searchIdentifierCandidates('readNodeErrorCode missingIdentifier', 2), []);
  assert.deepEqual(
    store.searchLexicalCandidates('"alpha beta"', 10).map((candidate) => candidate.id),
    ['chunk-c'],
  );
  assert.deepEqual(
    store.searchIdentifierCandidates('"alpha beta"', 10).map((candidate) => candidate.id),
    ['chunk-c'],
  );
  assert.deepEqual(
    store
      .searchLexicalCandidates('find "alpha beta" in the marker', 10)
      .map((candidate) => candidate.id),
    ['chunk-c'],
  );
  const groups = store.groupDenseCandidates([1, 0, 0], 'entryId', 2, 1);
  assert.equal(groups[0]?.groupByValue, 'entry-1');
  assert.equal(groups[0]?.docs[0]?.id, 'chunk-a');
  assert.throws(
    () => store.groupDenseCandidates([1, 0, 0], 'entryId', 201, 1),
    /dense grouping limits invalid/,
  );
  assert.throws(
    () => store.searchDenseCandidates([1, 0, 0], 201),
    /candidate limit invalid \(dense\)/,
  );
  assert.throws(
    () => store.searchLexicalCandidates('queue', 201),
    /candidate limit invalid \(lexical\)/,
  );
  assert.throws(
    () => store.searchIdentifierCandidates('EPERM', 201),
    /candidate limit invalid \(identifier\)/,
  );
});

void test('zvec hybrid search returns atomic and turn-context document kinds', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-turn-context-'));
  const store = openZvecConversationStore({
    databasePath: join(directory, 'collection'),
    dimensions: 3,
  });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });
  const atomicChunk: EmbeddedSessionConversationChunk = {
    ...baseChunk,
    id: 'atomic-reply',
    checksum: 'atomic-checksum',
    entryId: { value: 'assistant-reply' },
    contributingEntryIds: [{ value: 'assistant-reply' }],
    role: 'assistant',
    content: 'Yes, do it.',
    embedding: [1, 0, 0],
  };
  const turnContextChunk: EmbeddedSessionConversationChunk = {
    ...baseChunk,
    documentKind: 'turn_context',
    evidenceKind: 'turn_context',
    id: 'turn-context-reply',
    checksum: 'turn-context-checksum',
    entryId: { value: 'user-request' },
    contributingEntryIds: [{ value: 'user-request' }, { value: 'assistant-reply' }],
    role: 'turn',
    sourceLineStart: 2,
    sourceLineEnd: 5,
    sourceBlockStart: null,
    sourceBlockEnd: null,
    content: 'User:\nShip release Atlas to edge nodes.\n\nAssistant:\nYes, do it.',
    embedding: [1, 0, 0],
  };

  store.upsertChunks([atomicChunk, turnContextChunk]);

  assert.deepEqual(
    store
      .searchDenseCandidates([1, 0, 0], 2)
      .map((candidate) => candidate.documentKind)
      .toSorted(),
    ['conversation', 'turn_context'],
  );
  const lexicalTurn = store.searchLexicalCandidates('Atlas', 2)[0];
  assert.equal(lexicalTurn?.id, 'turn-context-reply');
  assert.equal(lexicalTurn?.documentKind, 'turn_context');
  assert.equal(lexicalTurn?.evidenceKind, 'turn_context');
  assert.equal(lexicalTurn?.role, 'turn');
  assert.deepEqual(
    lexicalTurn?.contributingEntryIds.map((id) => id.value),
    ['user-request', 'assistant-reply'],
  );
  assert.ok((lexicalTurn?.fullTextScore ?? 0) > 0);
});

void test('zvec round-trips lexical-only tool evidence and excludes it from dense search', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-zvec-tools-'));
  const store = openZvecConversationStore({
    databasePath: join(directory, 'collection'),
    dimensions: 3,
  });
  t.after(async () => {
    store.close();
    await rm(directory, { recursive: true, force: true });
  });
  const toolChunk: LexicalSessionConversationChunk = {
    ...baseChunk,
    documentKind: 'tool',
    evidenceKind: 'tool_result',
    evidencePart: 'result',
    isDenseSearchable: false,
    id: 'tool-chunk',
    checksum: 'tool-checksum',
    entryId: { value: 'tool-result-entry' },
    contributingEntryIds: [{ value: 'tool-result-entry' }],
    role: 'tool',
    toolCallId: 'call-tools',
    toolName: 'bash',
    toolCallEntryId: { value: 'tool-call-entry' },
    toolResultEntryId: { value: 'tool-result-entry' },
    toolError: true,
    content: 'TOOL_ONLY_EPERM /tmp/locked-file',
  };

  store.upsertChunks([toolChunk]);

  assert.deepEqual(store.searchDenseCandidates([1, 0, 0], 10), []);
  const lexicalResult = store.searchIdentifierCandidates('TOOL_ONLY_EPERM', 10)[0];
  assert.ok(lexicalResult);
  const { fullTextScore, ...storedToolChunk } = lexicalResult;
  assert.ok(fullTextScore > 0);
  assert.deepEqual(storedToolChunk, toolChunk);
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

import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import {
  createTestRecallSearchResult,
  createTestSessionConversationChunk,
} from './recall-test-utils.js';
import { rankFusedRecallSearchResults } from './rank-recall-search-results.js';

function createRecallCandidate(
  id: string,
  content: string,
  overrides: Partial<RecallSearchResult> = {},
): RecallSearchResult {
  return createTestRecallSearchResult({
    id,
    content,
    isOnActiveBranch: false,
    isVisibleInActiveContext: false,
    timestamp: '2026-07-25T10:00:00Z',
    ...overrides,
  });
}

void test('hybrid ranking rejects weak dense-only matches without hiding exact or strong evidence', () => {
  const fusedScore = 1 / 61;
  const results = rankFusedRecallSearchResults(
    [
      createRecallCandidate('weak-dense', 'weak', {
        isOnActiveBranch: true,
        dense: { rank: 1, cosineDistance: 0.6473 },
        fusedScore,
      }),
      createRecallCandidate('exact', 'exact', {
        isOnActiveBranch: true,
        dense: null,
        lexical: { rank: 1, fullTextScore: 14.25 },
        fusedScore,
      }),
      createRecallCandidate('strong-dense', 'strong', {
        dense: { rank: 2, cosineDistance: 0.49 },
        fusedScore: 1 / 62,
      }),
    ],
    5,
    () => new Map(),
  );

  assert.deepEqual(
    results.map((result) => result.id),
    ['exact', 'strong-dense'],
  );
});

void test('hybrid ranking suppresses overlapping sibling slots and retains exact provenance', () => {
  const shared = {
    sessionId: { value: 'shared-session' },
    sessionPath: '/sessions/shared.jsonl',
    entryId: { value: 'shared-entry' },
    contributingEntryIds: [{ value: 'shared-entry' }],
    textRunId: 'shared-run',
    chunkCount: 2,
  };
  const first = createRecallCandidate('first', 'alpha beta gamma', {
    ...shared,
    characterStart: 0,
    characterEnd: 16,
    tokenStart: 0,
    tokenEnd: 3,
    tokenCount: 3,
    chunkIndex: 0,
    siblingIds: ['second'],
    nextSiblingId: 'second',
    fusedScore: 0.04,
  });
  const second = createRecallCandidate('second', 'gamma delta epsilon', {
    ...shared,
    characterStart: 11,
    characterEnd: 30,
    tokenStart: 2,
    tokenEnd: 5,
    tokenCount: 3,
    overlapTokenCount: 1,
    chunkIndex: 1,
    siblingIds: ['first'],
    previousSiblingId: 'first',
    fusedScore: 0.03,
  });

  const results = rankFusedRecallSearchResults([first, second], 5, () => new Map());

  assert.equal(results.length, 1);
  assert.equal(results[0]?.id, 'first');
  assert.deepEqual(
    results[0]?.duplicateOccurrences.map((item) => item.id),
    ['second'],
  );
});

void test('hybrid ranking suppresses exact cross-session copies without conflating summaries', () => {
  const original = createRecallCandidate('original', 'same text', {
    checksum: 'same-checksum',
    sessionPath: '/sessions/original.jsonl',
    fusedScore: 0.04,
  });
  const copy = createRecallCandidate('copy', 'same text', {
    checksum: 'same-checksum',
    sessionPath: '/sessions/copy.jsonl',
    fusedScore: 0.03,
  });
  const summary = createRecallCandidate('summary', 'same text', {
    checksum: 'same-checksum',
    sessionPath: '/sessions/summary.jsonl',
    documentKind: 'summary',
    summaryKind: 'compaction',
    evidenceKind: 'compaction_summary',
    role: 'summary',
    fusedScore: 0.02,
  });

  const results = rankFusedRecallSearchResults([copy, summary, original], 5, () => new Map());

  assert.deepEqual(
    results.map((result) => result.id),
    ['original', 'summary'],
  );
  assert.deepEqual(
    results[0]?.duplicateOccurrences.map((item) => item.id),
    ['copy'],
  );
});

void test('hybrid ranking applies a small active-branch prior without hiding stronger evidence', () => {
  const results = rankFusedRecallSearchResults(
    [
      createRecallCandidate('active', 'active', {
        isOnActiveBranch: true,
        fusedScore: 0.02,
      }),
      createRecallCandidate('stronger-abandoned', 'abandoned', {
        fusedScore: 0.04,
      }),
    ],
    5,
    () => new Map(),
  );

  assert.deepEqual(
    results.map((result) => result.id),
    ['stronger-abandoned', 'active'],
  );
  assert.equal(results[1]?.activeBranchPrior, 0.01);
});

void test('hybrid ranking expands only exact contiguous siblings from one visible text run', () => {
  const common = {
    sessionId: { value: 'session' },
    sessionPath: '/sessions/source.jsonl',
    entryId: { value: 'entry' },
    contributingEntryIds: [{ value: 'entry' }],
    textRunId: 'run',
    textRunIndex: 0,
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
    chunkCount: 3,
  };
  const previous = createTestSessionConversationChunk({
    id: 'previous',
    content: 'alpha beta gamma',
    ...common,
    characterStart: 0,
    characterEnd: 16,
    tokenStart: 0,
    tokenEnd: 3,
    tokenCount: 3,
    chunkIndex: 0,
    siblingIds: ['winner'],
    nextSiblingId: 'winner',
  });
  const winner = createRecallCandidate('winner', 'gamma delta', {
    ...common,
    characterStart: 11,
    characterEnd: 22,
    tokenStart: 2,
    tokenEnd: 4,
    tokenCount: 2,
    overlapTokenCount: 1,
    chunkIndex: 1,
    siblingIds: ['previous', 'next'],
    previousSiblingId: 'previous',
    nextSiblingId: 'next',
  });
  const next = createTestSessionConversationChunk({
    id: 'next',
    content: 'delta epsilon',
    ...common,
    characterStart: 17,
    characterEnd: 30,
    tokenStart: 3,
    tokenEnd: 5,
    tokenCount: 2,
    overlapTokenCount: 1,
    chunkIndex: 2,
    siblingIds: ['winner'],
    previousSiblingId: 'winner',
  });

  const results = rankFusedRecallSearchResults(
    [winner],
    5,
    (ids) =>
      new Map(
        ids.flatMap((id) => {
          const chunk = id === 'previous' ? previous : id === 'next' ? next : undefined;
          return chunk ? [[id, chunk]] : [];
        }),
      ),
  );

  assert.equal(results[0]?.neighborContext?.content, 'alpha beta gamma delta epsilon');
  assert.deepEqual(
    results[0]?.neighborContext?.chunks.map((chunk) => chunk.id),
    ['previous', 'winner', 'next'],
  );
});

void test('hybrid ranking refuses neighbor expansion across a source geometry gap', () => {
  const winner = createRecallCandidate('winner', 'gamma delta', {
    sessionPath: '/sessions/source.jsonl',
    entryId: { value: 'entry' },
    contributingEntryIds: [{ value: 'entry' }],
    textRunId: 'run',
    chunkCount: 2,
    chunkIndex: 1,
    characterStart: 20,
    characterEnd: 31,
    tokenStart: 2,
    tokenEnd: 4,
    overlapTokenCount: 1,
    siblingIds: ['previous'],
    previousSiblingId: 'previous',
  });
  const previous = createTestSessionConversationChunk({
    ...winner,
    id: 'previous',
    content: 'alpha beta gamma',
    chunkIndex: 0,
    characterStart: 0,
    characterEnd: 16,
    tokenStart: 0,
    tokenEnd: 3,
    overlapTokenCount: 0,
    siblingIds: ['winner'],
    previousSiblingId: null,
    nextSiblingId: 'winner',
  });

  const results = rankFusedRecallSearchResults(
    [winner],
    5,
    () => new Map([['previous', previous]]),
  );

  assert.equal(results[0]?.neighborContext, null);
});

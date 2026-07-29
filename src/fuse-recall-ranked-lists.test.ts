import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallRankedListSource } from './enums.js';
import { fuseRecallRankedLists } from './fuse-recall-ranked-lists.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';

function createTestConversationChunk(id: string) {
  return createTestSessionConversationChunk({
    id,
    sessionId: { value: 'session-1' },
    sessionPath: '/sessions/session-1.jsonl',
    sessionName: 'Hybrid ranking',
    characterEnd: id.length,
    tokenEnd: 1,
    tokenCount: 1,
  });
}

void test('ranked-list fusion enforces list and fused-pool limits while retaining weighted evidence', () => {
  const first = createTestConversationChunk('a');
  const second = createTestConversationChunk('b');
  const third = createTestConversationChunk('c');

  const fused = fuseRecallRankedLists(
    [
      {
        source: RecallRankedListSource.DENSE,
        query: 'submitted query',
        weight: 2,
        candidateLimit: 2,
        higherNativeScoresRankFirst: false,
        candidates: [
          { document: third, nativeScore: 0.3 },
          { document: first, nativeScore: 0.1 },
          { document: second, nativeScore: 0.2 },
        ],
      },
      {
        source: RecallRankedListSource.LEXICAL,
        query: 'submitted query',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: true,
        candidates: [
          { document: second, nativeScore: 8 },
          { document: third, nativeScore: 7 },
        ],
      },
    ],
    2,
  );

  assert.deepEqual(
    fused.map((result) => result.id),
    ['b', 'a'],
  );
  assert.deepEqual(fused[0]?.rankedListEvidence, [
    {
      source: RecallRankedListSource.DENSE,
      query: 'submitted query',
      rank: 2,
      nativeScore: 0.2,
      weight: 2,
    },
    {
      source: RecallRankedListSource.LEXICAL,
      query: 'submitted query',
      rank: 1,
      nativeScore: 8,
      weight: 1,
    },
  ]);
  assert.deepEqual(fused[0]?.dense, { rank: 2, cosineDistance: 0.2 });
  assert.deepEqual(fused[0]?.lexical, { rank: 1, fullTextScore: 8 });
  assert.equal(fused[0]?.fusedScore, 2 / 62 + 1 / 61);
  assert.ok(!fused.some((result) => result.id === 'c'));
});

void test('ranked-list fusion counts a repeated document only at its first provider rank', () => {
  const repeated = createTestConversationChunk('repeated');
  const next = createTestConversationChunk('next');

  const fused = fuseRecallRankedLists(
    [
      {
        source: RecallRankedListSource.LEXICAL,
        query: 'provider duplicate',
        weight: 1,
        candidateLimit: 3,
        higherNativeScoresRankFirst: true,
        candidates: [
          { document: repeated, nativeScore: 5 },
          { document: repeated, nativeScore: 4 },
          { document: next, nativeScore: 3 },
        ],
      },
    ],
    3,
  );

  assert.equal(fused[0]?.fusedScore, 1 / 61);
  assert.deepEqual(fused[0]?.rankedListEvidence, [
    {
      source: RecallRankedListSource.LEXICAL,
      query: 'provider duplicate',
      rank: 1,
      nativeScore: 5,
      weight: 1,
    },
  ]);
  assert.deepEqual(fused[1]?.lexical, { rank: 3, fullTextScore: 3 });
});

void test('query-planned fusion applies one QMD bonus from each document best rank', () => {
  const first = createTestConversationChunk('bonus-first');
  const second = createTestConversationChunk('bonus-second');
  const fused = fuseRecallRankedLists(
    [
      {
        source: RecallRankedListSource.DENSE,
        query: 'submitted query',
        weight: 2,
        candidateLimit: 2,
        higherNativeScoresRankFirst: false,
        candidates: [
          { document: first, nativeScore: 0.1 },
          { document: second, nativeScore: 0.2 },
        ],
      },
      {
        source: RecallRankedListSource.PLANNED_VEC,
        query: 'semantic reformulation',
        weight: 1,
        candidateLimit: 2,
        higherNativeScoresRankFirst: false,
        candidates: [
          { document: first, nativeScore: 0.15 },
          { document: second, nativeScore: 0.25 },
        ],
      },
    ],
    2,
    { rankOne: 0.05, rankTwoOrThree: 0.02 },
  );

  assert.equal(fused[0]?.id, 'bonus-first');
  assert.equal(fused[0]?.topRankBonus, 0.05);
  assert.equal(fused[0]?.fusedScore, 2 / 61 + 1 / 61 + 0.05);
  assert.equal(fused[1]?.topRankBonus, 0.02);
  assert.equal(fused[1]?.fusedScore, 2 / 62 + 1 / 62 + 0.02);
});

void test('recall rank fusion rejects a non-finite score from a one-item channel', () => {
  assert.throws(
    () =>
      fuseRecallRankedLists(
        [
          {
            source: RecallRankedListSource.DENSE,
            query: 'invalid score',
            weight: 1,
            candidateLimit: 1,
            higherNativeScoresRankFirst: false,
            candidates: [{ document: createTestConversationChunk('a'), nativeScore: Number.NaN }],
          },
        ],
        1,
      ),
    /candidate score invalid \(dense\).*expected a finite number/,
  );
});

void test('recall rank fusion deduplicates channels and resolves equal scores by document ID', () => {
  const first = createTestConversationChunk('a');
  const second = createTestConversationChunk('b');
  const fused = fuseRecallRankedLists(
    [
      {
        source: RecallRankedListSource.DENSE,
        query: 'tie',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: false,
        candidates: [{ document: first, nativeScore: 0 }],
      },
      {
        source: RecallRankedListSource.LEXICAL,
        query: 'tie',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: true,
        candidates: [{ document: second, nativeScore: 2 }],
      },
    ],
    2,
  );

  assert.deepEqual(
    fused.map((result) => result.id),
    ['a', 'b'],
  );
  assert.equal(fused[0]?.fusedScore, 0.01639344262295082);
  assert.equal(fused[1]?.fusedScore, 0.01639344262295082);

  const duplicate = fuseRecallRankedLists(
    [
      {
        source: RecallRankedListSource.DENSE,
        query: 'duplicate',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: false,
        candidates: [{ document: second, nativeScore: 0.25 }],
      },
      {
        source: RecallRankedListSource.LEXICAL,
        query: 'duplicate',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: true,
        candidates: [{ document: second, nativeScore: 4 }],
      },
      {
        source: RecallRankedListSource.IDENTIFIER,
        query: 'duplicate',
        weight: 1,
        candidateLimit: 1,
        higherNativeScoresRankFirst: true,
        candidates: [{ document: second, nativeScore: 3 }],
      },
    ],
    2,
  );
  assert.equal(duplicate.length, 1);
  assert.deepEqual(
    {
      dense: duplicate[0]?.dense,
      lexical: duplicate[0]?.lexical,
      identifier: duplicate[0]?.identifier,
      fusedScore: duplicate[0]?.fusedScore,
    },
    {
      dense: { rank: 1, cosineDistance: 0.25 },
      lexical: { rank: 1, fullTextScore: 4 },
      identifier: { rank: 1, fullTextScore: 3 },
      fusedScore: 0.04918032786885246,
    },
  );
});

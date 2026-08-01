import assert from 'node:assert/strict';
import test from 'node:test';

import { fuseRecallSearchCandidates } from './fuse-recall-search-candidates.js';
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

void test('recall rank fusion rejects a non-finite score from a one-item channel', () => {
  assert.throws(
    () =>
      fuseRecallSearchCandidates(
        {
          denseCandidates: [{ ...createTestConversationChunk('a'), cosineDistance: Number.NaN }],
          lexicalCandidates: [],
          identifierCandidates: [],
        },
        1,
      ),
    /candidate score invalid \(dense\).*expected a finite number/,
  );
});

void test('recall rank fusion deduplicates channels and resolves equal scores by document ID', () => {
  const first = createTestConversationChunk('a');
  const second = createTestConversationChunk('b');
  const fused = fuseRecallSearchCandidates(
    {
      denseCandidates: [{ ...first, cosineDistance: 0 }],
      lexicalCandidates: [{ ...second, fullTextScore: 2 }],
      identifierCandidates: [],
    },
    2,
  );

  assert.deepEqual(
    fused.map((result) => result.id),
    ['a', 'b'],
  );
  assert.equal(fused[0]?.fusedScore, 0.01639344262295082);
  assert.equal(fused[1]?.fusedScore, 0.01639344262295082);

  const duplicate = fuseRecallSearchCandidates(
    {
      denseCandidates: [{ ...second, cosineDistance: 0.25 }],
      lexicalCandidates: [{ ...second, fullTextScore: 4 }],
      identifierCandidates: [{ ...second, fullTextScore: 3 }],
    },
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

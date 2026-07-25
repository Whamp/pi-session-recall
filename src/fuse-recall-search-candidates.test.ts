import assert from 'node:assert/strict';
import test from 'node:test';

import { fuseRecallSearchCandidates } from './fuse-recall-search-candidates.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

function createTestConversationChunk(id: string): SessionConversationChunk {
  return {
    schemaVersion: 3,
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    isDenseSearchable: true,
    id,
    checksum: `checksum-${id}`,
    sessionId: { value: 'session-1' },
    sessionPath: '/sessions/session-1.jsonl',
    parentSessionPath: null,
    cwd: '/project',
    projectPath: '/project',
    sessionName: 'Hybrid ranking',
    entryId: { value: `entry-${id}` },
    parentEntryId: null,
    childEntryIds: [],
    contributingEntryIds: [{ value: `entry-${id}` }],
    currentLeafId: { value: `entry-${id}` },
    branchPathLeafIds: [{ value: `entry-${id}` }],
    isOnActiveBranch: true,
    isVisibleInActiveContext: true,
    compactedByEntryIds: [],
    compactionFirstKeptEntryId: null,
    branchSummaryFromEntryId: null,
    role: 'assistant',
    timestamp: '2026-07-24T10:00:00Z',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
    characterStart: 0,
    characterEnd: id.length,
    tokenStart: 0,
    tokenEnd: 1,
    tokenCount: 1,
    overlapTokenCount: 0,
    textRunId: `run-${id}`,
    textRunIndex: 0,
    chunkIndex: 0,
    chunkCount: 1,
    siblingIds: [],
    previousSiblingId: null,
    nextSiblingId: null,
    toolCallId: null,
    toolName: null,
    toolCallEntryId: null,
    toolResultEntryId: null,
    toolError: null,
    content: `content ${id}`,
  };
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

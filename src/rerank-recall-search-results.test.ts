import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import { rerankRecallSearchResults } from './rerank-recall-search-results.js';

function createRecallCandidate(
  id: string,
  content: string,
  overrides: Partial<RecallSearchResult> = {},
): RecallSearchResult {
  return {
    schemaVersion: 4,
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    isDenseSearchable: true,
    id,
    checksum: `checksum-${id}`,
    sessionId: { value: `session-${id}` },
    sessionPath: `/sessions/${id}.jsonl`,
    parentSessionPath: null,
    cwd: '/project',
    projectPath: '/project',
    sessionName: `Session ${id}`,
    entryId: { value: `entry-${id}` },
    parentEntryId: null,
    childEntryIds: [],
    contributingEntryIds: [{ value: `entry-${id}` }],
    currentLeafId: { value: `entry-${id}` },
    branchPathLeafIds: [{ value: `entry-${id}` }],
    isOnActiveBranch: false,
    isVisibleInActiveContext: false,
    compactedByEntryIds: [],
    compactionFirstKeptEntryId: null,
    branchSummaryFromEntryId: null,
    role: 'assistant',
    timestamp: '2026-07-25T10:00:00Z',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
    characterStart: 0,
    characterEnd: content.length,
    tokenStart: 0,
    tokenEnd: 4,
    tokenCount: 4,
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
    content,
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    fusedScore: 0.02,
    ...overrides,
  };
}

void test('recall reranking sends every candidate kind as original text and preserves component scores', async () => {
  const candidates = [
    createRecallCandidate('conversation', 'Original atomic conversation.'),
    createRecallCandidate('turn', 'User:\nShip Atlas.\n\nAssistant:\nDone.', {
      documentKind: 'turn_context',
      evidenceKind: 'turn_context',
      role: 'turn',
      dense: null,
      lexical: { rank: 1, fullTextScore: 4.5 },
    }),
    createRecallCandidate('tool', 'EPERM /tmp/locked-file', {
      documentKind: 'tool',
      evidenceKind: 'tool_result',
      evidencePart: 'result',
      isDenseSearchable: false,
      role: 'tool',
      dense: null,
      identifier: { rank: 1, fullTextScore: 8.25 },
    }),
    createRecallCandidate('compaction', 'Compacted queue decision.', {
      documentKind: 'summary',
      summaryKind: 'compaction',
      evidenceKind: 'compaction_summary',
      role: 'summary',
    }),
    createRecallCandidate('branch', 'Abandoned polling branch.', {
      documentKind: 'summary',
      summaryKind: 'branch',
      evidenceKind: 'branch_summary',
      role: 'summary',
    }),
  ];
  const rerankerInputs: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      assert.equal(query, 'How did Atlas fail?');
      rerankerInputs.push([...documents]);
      return [0.1, 0.95, 0.8, 0.6, 0.4];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'How did Atlas fail?',
    candidates,
    resultLimit: 5,
    reranker,
    fetchConversationChunks() {
      return new Map();
    },
  });

  assert.deepEqual(rerankerInputs, [candidates.map((candidate) => candidate.content)]);
  assert.deepEqual(
    results.map((result) => result.id),
    ['turn', 'tool', 'compaction', 'branch', 'conversation'],
  );
  assert.equal(results[0]?.rerankerScore, 0.95);
  assert.deepEqual(results[0]?.lexical, { rank: 1, fullTextScore: 4.5 });
  assert.deepEqual(results[1]?.identifier, { rank: 1, fullTextScore: 8.25 });
  assert.equal(results[2]?.summaryKind, 'compaction');
  assert.equal(results[3]?.summaryKind, 'branch');
});

void test('recall reranking suppresses overlapping sibling slots and preserves the duplicate candidate', async () => {
  const first = createRecallCandidate('first', 'alpha beta gamma', {
    sessionId: { value: 'shared-session' },
    sessionPath: '/sessions/shared.jsonl',
    entryId: { value: 'shared-entry' },
    contributingEntryIds: [{ value: 'shared-entry' }],
    textRunId: 'shared-run',
    characterStart: 0,
    characterEnd: 16,
    tokenStart: 0,
    tokenEnd: 3,
    tokenCount: 3,
    chunkIndex: 0,
    chunkCount: 2,
    siblingIds: ['second'],
    nextSiblingId: 'second',
    fusedScore: 0.04,
  });
  const second = createRecallCandidate('second', 'gamma delta epsilon', {
    sessionId: { value: 'shared-session' },
    sessionPath: '/sessions/shared.jsonl',
    entryId: { value: 'shared-entry' },
    contributingEntryIds: [{ value: 'shared-entry' }],
    textRunId: 'shared-run',
    characterStart: 11,
    characterEnd: 30,
    tokenStart: 2,
    tokenEnd: 5,
    tokenCount: 3,
    overlapTokenCount: 1,
    chunkIndex: 1,
    chunkCount: 2,
    siblingIds: ['first'],
    previousSiblingId: 'first',
    fusedScore: 0.03,
    lexical: { rank: 2, fullTextScore: 3.25 },
  });
  const unrelated = createRecallCandidate('unrelated', 'different evidence', {
    fusedScore: 0.02,
  });
  const rerankerInputs: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      rerankerInputs.push([...documents]);
      return [0.9, 0.2];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'gamma',
    candidates: [second, unrelated, first],
    resultLimit: 3,
    reranker,
    fetchConversationChunks() {
      return new Map();
    },
  });

  assert.deepEqual(rerankerInputs, [['alpha beta gamma', 'different evidence']]);
  assert.deepEqual(
    results.map((result) => result.id),
    ['first', 'unrelated'],
  );
  assert.equal(results[0]?.duplicateOccurrences.length, 1);
  assert.equal(results[0]?.duplicateOccurrences[0]?.id, 'second');
  assert.deepEqual(results[0]?.duplicateOccurrences[0]?.lexical, {
    rank: 2,
    fullTextScore: 3.25,
  });
  assert.equal(results[0]?.duplicateOccurrences[0]?.sessionPath, '/sessions/shared.jsonl');
  assert.equal(results[0]?.duplicateOccurrences[0]?.characterStart, 11);
  assert.equal(results[0]?.duplicateOccurrences[0]?.characterEnd, 30);
});

void test('recall reranking keeps reciprocal siblings whose overlap text does not match', async () => {
  const sharedGeometry = {
    sessionId: { value: 'mismatched-session' },
    sessionPath: '/sessions/mismatched.jsonl',
    entryId: { value: 'mismatched-entry' },
    contributingEntryIds: [{ value: 'mismatched-entry' }],
    textRunId: 'mismatched-run',
    textRunIndex: 0,
    chunkCount: 2,
  };
  const first = createRecallCandidate('mismatched-first', 'alpha beta', {
    ...sharedGeometry,
    characterStart: 0,
    characterEnd: 10,
    tokenStart: 0,
    tokenEnd: 2,
    tokenCount: 2,
    chunkIndex: 0,
    siblingIds: ['mismatched-second'],
    nextSiblingId: 'mismatched-second',
  });
  const second = createRecallCandidate('mismatched-second', 'WRONG gamma', {
    ...sharedGeometry,
    characterStart: 6,
    characterEnd: 17,
    tokenStart: 1,
    tokenEnd: 3,
    tokenCount: 2,
    overlapTokenCount: 1,
    chunkIndex: 1,
    siblingIds: ['mismatched-first'],
    previousSiblingId: 'mismatched-first',
  });
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      assert.deepEqual(documents, ['alpha beta', 'WRONG gamma']);
      return [0.9, 0.8];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'gamma',
    candidates: [first, second],
    resultLimit: 2,
    reranker,
    fetchConversationChunks() {
      return new Map();
    },
  });

  assert.deepEqual(
    results.map((result) => result.id),
    ['mismatched-first', 'mismatched-second'],
  );
  assert.ok(results.every((result) => result.duplicateOccurrences.length === 0));
});

void test('recall reranking suppresses exact cross-session copies without conflating summaries or checksum collisions', async () => {
  const copiedContent = 'Preserve exact source provenance.';
  const firstCopy = createRecallCandidate('copy-a', copiedContent, {
    checksum: 'shared-checksum',
    fusedScore: 0.04,
  });
  const secondCopy = createRecallCandidate('copy-b', copiedContent, {
    checksum: 'shared-checksum',
    isOnActiveBranch: true,
    isVisibleInActiveContext: true,
    fusedScore: 0.03,
  });
  const syntheticSummary = createRecallCandidate('summary-copy', copiedContent, {
    checksum: 'shared-checksum',
    documentKind: 'summary',
    summaryKind: 'compaction',
    evidenceKind: 'compaction_summary',
    role: 'summary',
    fusedScore: 0.02,
  });
  const checksumCollision = createRecallCandidate('checksum-collision', 'Different source text.', {
    checksum: 'shared-checksum',
    fusedScore: 0.01,
  });
  const rerankerInputs: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      rerankerInputs.push([...documents]);
      return [0.9, 0.8, 0.7];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'source provenance',
    candidates: [firstCopy, secondCopy, syntheticSummary, checksumCollision],
    resultLimit: 4,
    reranker,
    fetchConversationChunks() {
      return new Map();
    },
  });

  assert.deepEqual(rerankerInputs, [[copiedContent, copiedContent, 'Different source text.']]);
  assert.deepEqual(
    results.map((result) => result.id),
    ['copy-a', 'summary-copy', 'checksum-collision'],
  );
  assert.equal(results[0]?.duplicateOccurrences.length, 1);
  assert.equal(results[0]?.isOnActiveBranch, false);
  assert.equal(results[0]?.activeBranchPrior, 0.01);
  assert.equal(results[0]?.duplicateOccurrences[0]?.id, 'copy-b');
  assert.equal(results[0]?.duplicateOccurrences[0]?.sessionPath, '/sessions/copy-b.jsonl');
  assert.equal(results[0]?.duplicateOccurrences[0]?.isOnActiveBranch, true);
  assert.equal(results[1]?.documentKind, 'summary');
  assert.equal(results[1]?.summaryKind, 'compaction');
});

void test('recall reranking favors an active branch without hiding a stronger abandoned match', async () => {
  const abandoned = createRecallCandidate('abandoned', 'Abandoned branch evidence.', {
    isOnActiveBranch: false,
    fusedScore: 0.04,
  });
  const active = createRecallCandidate('active', 'Active branch evidence.', {
    isOnActiveBranch: true,
    isVisibleInActiveContext: true,
    fusedScore: 0.03,
  });
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      assert.deepEqual(documents, ['Abandoned branch evidence.', 'Active branch evidence.']);
      return [0.5, 0.495];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'branch evidence',
    candidates: [abandoned, active],
    resultLimit: 2,
    reranker,
    fetchConversationChunks() {
      return new Map();
    },
  });

  assert.deepEqual(
    results.map((result) => result.id),
    ['active', 'abandoned'],
  );
  assert.equal(results[0]?.rerankerScore, 0.495);
  assert.equal(results[0]?.activeBranchPrior, 0.01);
  assert.equal(results[0]?.rankingScore, 0.505);
  assert.equal(results[1]?.rerankerScore, 0.5);
  assert.equal(results[1]?.activeBranchPrior, 0);
  assert.equal(results[1]?.rankingScore, 0.5);
  assert.equal(results[1]?.isOnActiveBranch, false);
});

void test('recall reranking expands a winning atomic chunk through valid same-run neighbors', async () => {
  const sharedGeometry = {
    sessionId: { value: 'expansion-session' },
    sessionPath: '/sessions/expansion.jsonl',
    entryId: { value: 'expansion-entry' },
    contributingEntryIds: [{ value: 'expansion-entry' }],
    textRunId: 'expansion-run',
    textRunIndex: 0,
    chunkCount: 3,
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
  };
  const previous = createRecallCandidate('previous', 'alpha beta gamma', {
    ...sharedGeometry,
    characterStart: 0,
    characterEnd: 16,
    tokenStart: 0,
    tokenEnd: 3,
    tokenCount: 3,
    overlapTokenCount: 0,
    chunkIndex: 0,
    siblingIds: ['winner'],
    nextSiblingId: 'winner',
  });
  const winner = createRecallCandidate('winner', 'gamma delta', {
    ...sharedGeometry,
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
  const next = createRecallCandidate('next', 'delta epsilon', {
    ...sharedGeometry,
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
  const fetchedIds: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      assert.deepEqual(documents, ['gamma delta']);
      return [0.9];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'delta',
    candidates: [winner],
    resultLimit: 1,
    reranker,
    fetchConversationChunks(ids) {
      fetchedIds.push([...ids]);
      return new Map([
        ['previous', previous],
        ['next', next],
      ]);
    },
  });

  assert.deepEqual(fetchedIds, [['previous', 'next']]);
  assert.equal(results[0]?.content, 'gamma delta');
  assert.equal(results[0]?.neighborContext?.content, 'alpha beta gamma delta epsilon');
  assert.deepEqual(
    results[0]?.neighborContext?.chunks.map((chunk) => ({
      id: chunk.id,
      characterStart: chunk.characterStart,
      characterEnd: chunk.characterEnd,
    })),
    [
      { id: 'previous', characterStart: 0, characterEnd: 16 },
      { id: 'winner', characterStart: 11, characterEnd: 22 },
      { id: 'next', characterStart: 17, characterEnd: 30 },
    ],
  );
});

void test('recall neighbor expansion rejects a source-offset gap between reciprocal siblings', async () => {
  const sharedGeometry = {
    sessionId: { value: 'gapped-session' },
    sessionPath: '/sessions/gapped.jsonl',
    entryId: { value: 'gapped-entry' },
    contributingEntryIds: [{ value: 'gapped-entry' }],
    textRunId: 'gapped-run',
    textRunIndex: 0,
    chunkCount: 2,
  };
  const winner = createRecallCandidate('gapped-winner', 'alpha beta', {
    ...sharedGeometry,
    characterStart: 0,
    characterEnd: 10,
    tokenStart: 0,
    tokenEnd: 2,
    tokenCount: 2,
    chunkIndex: 0,
    siblingIds: ['gapped-next'],
    nextSiblingId: 'gapped-next',
  });
  const gappedNext = createRecallCandidate('gapped-next', 'gamma delta', {
    ...sharedGeometry,
    characterStart: 12,
    characterEnd: 23,
    tokenStart: 2,
    tokenEnd: 4,
    tokenCount: 2,
    overlapTokenCount: 0,
    chunkIndex: 1,
    siblingIds: ['gapped-winner'],
    previousSiblingId: 'gapped-winner',
  });
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      assert.deepEqual(documents, ['alpha beta']);
      return [0.9];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'alpha',
    candidates: [winner],
    resultLimit: 1,
    reranker,
    fetchConversationChunks() {
      return new Map([['gapped-next', gappedNext]]);
    },
  });

  assert.equal(results[0]?.neighborContext, null);
});

void test('recall neighbor expansion rejects reciprocal pointers across entry, role, and evidence boundaries', async () => {
  const winner = createRecallCandidate('guarded-winner', 'middle evidence', {
    sessionId: { value: 'guarded-session' },
    sessionPath: '/sessions/guarded.jsonl',
    entryId: { value: 'guarded-entry' },
    contributingEntryIds: [{ value: 'guarded-entry' }],
    textRunId: 'guarded-run',
    chunkIndex: 1,
    chunkCount: 3,
    characterStart: 10,
    characterEnd: 25,
    tokenStart: 2,
    tokenEnd: 5,
    siblingIds: ['wrong-role', 'tool-neighbor'],
    previousSiblingId: 'wrong-role',
    nextSiblingId: 'tool-neighbor',
  });
  const wrongRole = createRecallCandidate('wrong-role', 'wrong role evidence', {
    sessionId: { value: 'guarded-session' },
    sessionPath: '/sessions/guarded.jsonl',
    entryId: { value: 'different-entry' },
    contributingEntryIds: [{ value: 'different-entry' }],
    role: 'user',
    textRunId: 'guarded-run',
    chunkIndex: 0,
    chunkCount: 3,
    characterStart: 0,
    characterEnd: 15,
    tokenStart: 0,
    tokenEnd: 3,
    siblingIds: ['guarded-winner'],
    nextSiblingId: 'guarded-winner',
  });
  const toolNeighbor = createRecallCandidate('tool-neighbor', 'tool output evidence', {
    sessionId: { value: 'guarded-session' },
    sessionPath: '/sessions/guarded.jsonl',
    entryId: { value: 'guarded-entry' },
    contributingEntryIds: [{ value: 'guarded-entry' }],
    documentKind: 'tool',
    evidenceKind: 'tool_result',
    evidencePart: 'result',
    isDenseSearchable: false,
    role: 'tool',
    textRunId: 'guarded-run',
    chunkIndex: 2,
    chunkCount: 3,
    characterStart: 20,
    characterEnd: 40,
    tokenStart: 4,
    tokenEnd: 7,
    overlapTokenCount: 1,
    siblingIds: ['guarded-winner'],
    previousSiblingId: 'guarded-winner',
    toolCallId: 'call-guarded',
    toolName: 'bash',
    toolResultEntryId: { value: 'guarded-entry' },
    toolError: false,
  });
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      assert.deepEqual(documents, ['middle evidence']);
      return [0.9];
    },
  };

  const results = await rerankRecallSearchResults({
    query: 'middle',
    candidates: [winner],
    resultLimit: 1,
    reranker,
    fetchConversationChunks() {
      return new Map([
        ['wrong-role', wrongRole],
        ['tool-neighbor', toolNeighbor],
      ]);
    },
  });

  assert.equal(results[0]?.neighborContext, null);
});

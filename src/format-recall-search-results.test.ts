import assert from 'node:assert/strict';
import test from 'node:test';

import { formatRecallSearchResults } from './format-recall-search-results.js';
import type { RecallSearchResult } from './fuse-recall-search-candidates.js';

const result = {
  schemaVersion: 2,
  documentKind: 'conversation',
  summaryKind: null,
  evidenceKind: 'conversation',
  id: 'chunk-1',
  checksum: 'sum-1',
  sessionId: { value: 'session-1' },
  sessionPath: '/sessions/one.jsonl',
  parentSessionPath: null,
  cwd: '/project',
  projectPath: '/project',
  sessionName: 'Queue design',
  entryId: { value: 'entry-1' },
  parentEntryId: null,
  childEntryIds: [],
  contributingEntryIds: [{ value: 'entry-1' }],
  currentLeafId: { value: 'entry-1' },
  branchPathLeafIds: [{ value: 'entry-1' }],
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
  characterEnd: 66,
  tokenStart: 0,
  tokenEnd: 10,
  tokenCount: 10,
  overlapTokenCount: 0,
  textRunId: 'text-run-1',
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
  content: 'The durable queue decision and its tradeoffs are documented here.',
  dense: { rank: 1, cosineDistance: 0.01234 },
  lexical: { rank: 2, fullTextScore: 0.87654 },
  identifier: null,
  fusedScore: 0.03252,
} satisfies RecallSearchResult;

void test('recall results include concise excerpts and exact source provenance', () => {
  const output = formatRecallSearchResults(
    {
      totalChunks: 42,
      results: [result],
      searchPolicy: {
        rankFusionVersion: 1,
        reciprocalRankConstant: 60,
        candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
      },
    },
    40,
  );

  assert.match(output, /1\. Queue design/);
  assert.match(output, /2026-07-24T10:00:00Z · assistant · \/project/);
  assert.match(output, /Source: \/sessions\/one\.jsonl#entry-1/);
  assert.match(output, /fused RRF 0\.0325/);
  assert.match(output, /dense #1 cosine distance 0\.0123/);
  assert.match(output, /lexical #2 FTS 0\.8765/);
  assert.match(output, /…/);
  assert.ok(!output.includes('Incremental index'));
  assert.ok(!output.includes('checksum'));
});

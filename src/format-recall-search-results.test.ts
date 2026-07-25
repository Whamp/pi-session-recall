import assert from 'node:assert/strict';
import test from 'node:test';

import { formatRecallSearchResults } from './format-recall-search-results.js';
import type { RecallSearchResult } from './zvec-conversation-store.js';

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
  score: 0.98765,
} satisfies RecallSearchResult;

void test('recall results include concise excerpts and exact source provenance', () => {
  const output = formatRecallSearchResults(
    {
      totalChunks: 42,
      results: [result],
    },
    40,
  );

  assert.match(output, /1\. Queue design/);
  assert.match(output, /2026-07-24T10:00:00Z · assistant · \/project/);
  assert.match(output, /Source: \/sessions\/one\.jsonl#entry-1/);
  assert.match(output, /cosine distance 0\.9877/);
  assert.match(output, /…/);
  assert.ok(!output.includes('Incremental index'));
  assert.ok(!output.includes('checksum'));
});

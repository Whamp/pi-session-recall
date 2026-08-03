import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { createTestRankedRecallSearchResult } from './recall-test-utils.js';
import type { RankedRecallSearchResult } from './rank-recall-search-results.js';

function createSearchPolicy(scope = RecallSearchScope.GLOBAL) {
  return {
    scope,
    invocationProjectIdentity: null,
    rankingMode: 'hybrid' as const,
    rankFusionVersion: 2,
    reciprocalRankConstant: 60,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
}

const result = createTestRankedRecallSearchResult({
  id: 'chunk-1',
  sessionPath: '/sessions/one.jsonl',
  sessionName: 'Queue design',
  entryId: { value: 'entry-1' },
  contributingEntryIds: [{ value: 'entry-1' }],
  sourceLineStart: 142,
  sourceLineEnd: 146,
  sourceBlockStart: 1,
  sourceBlockEnd: 2,
  content: 'The durable queue decision and its tradeoffs are documented here.',
  dense: { rank: 1, cosineDistance: 0.01234 },
  lexical: { rank: 2, fullTextScore: 0.87654 },
  fusedScore: 0.03252,
  activeBranchPrior: 0.01,
  rankingScore: 0.04252,
});

void test('recall output includes concise evidence and an agent-readable JSONL source locator', () => {
  const output = formatRecallSearchResults(
    {
      totalChunks: 42,
      results: [result],
      indexMaintenanceStatus: null,
      searchPolicy: createSearchPolicy(),
    },
    40,
  );

  assert.match(output, /deterministic fusion v2 \(RRF k=60\)/);
  assert.match(output, /Source: \/sessions\/one\.jsonl:142-146#entry-1/);
  assert.match(output, /ranking 0\.0425/);
  assert.match(output, /dense #1 cosine distance 0\.0123/);
  assert.match(output, /lexical #2 FTS 0\.8765/);
  assert.match(output, /…/);
  assert.doesNotMatch(output, /Qwen|rerank/iu);
});

void test('empty project recall recommends an explicit global retry', () => {
  const output = formatRecallSearchResults({
    totalChunks: 42,
    results: [],
    indexMaintenanceStatus: null,
    searchPolicy: createSearchPolicy(RecallSearchScope.PROJECT),
  });

  assert.match(output, /No matching past conversations found/);
  assert.match(output, /Retry with scope "global"/);
});

void test('turn-context output names every contributing entry and exact source lines', () => {
  const turn = {
    ...result,
    documentKind: 'turn_context',
    evidenceKind: 'turn_context',
    role: 'turn',
    entryId: { value: 'user-request' },
    contributingEntryIds: [{ value: 'user-request' }, { value: 'assistant-reply' }],
    sourceLineStart: 2,
    sourceLineEnd: 5,
    content: 'User: Ship Atlas. Assistant: Done.',
  } satisfies RankedRecallSearchResult;

  const output = formatRecallSearchResults({
    totalChunks: 1,
    results: [{ ...turn, evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL }],
    indexMaintenanceStatus: null,
    searchPolicy: createSearchPolicy(),
  });

  assert.match(output, /turn context/);
  assert.match(output, /Contributing entries: user-request → assistant-reply/);
  assert.match(output, /Source: \/sessions\/one\.jsonl:2-5#user-request/);
});

void test('tool evidence output preserves call linkage and source geometry', () => {
  const toolResult = {
    ...result,
    documentKind: 'tool',
    evidenceKind: 'tool_result',
    evidencePart: 'result',
    isDenseSearchable: false,
    id: 'tool-result-chunk',
    entryId: { value: 'result-entry' },
    role: 'tool',
    toolCallId: 'call-tools',
    toolName: 'bash',
    toolCallEntryId: { value: 'call-entry' },
    toolResultEntryId: { value: 'result-entry' },
    toolError: true,
    content: 'EPERM readNodeErrorCode /tmp/locked-file',
    dense: null,
  } satisfies RankedRecallSearchResult;

  const output = formatRecallSearchResults({
    totalChunks: 1,
    results: [{ ...toolResult, evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL }],
    indexMaintenanceStatus: null,
    searchPolicy: createSearchPolicy(),
  });

  assert.match(output, /tool_result\/result · bash · call call-tools · error/);
  assert.match(output, /Call source: call-entry · Result source: result-entry/);
  assert.match(output, /Source: \/sessions\/one\.jsonl:142-146#result-entry/);
});

void test('recall output retains stitched chunks and suppressed source occurrences', () => {
  const duplicate = {
    ...result,
    id: 'copy',
    sessionPath: '/sessions/copy.jsonl',
    entryId: { value: 'copy-entry' },
    isOnActiveBranch: false,
  };
  const expanded = {
    ...result,
    duplicateOccurrences: [duplicate],
    neighborContext: { content: 'complete stitched context', chunks: [result] },
  } satisfies RankedRecallSearchResult;

  const output = formatRecallSearchResults({
    totalChunks: 2,
    results: [{ ...expanded, evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL }],
    indexMaintenanceStatus: null,
    searchPolicy: createSearchPolicy(),
  });

  assert.match(output, /complete stitched context/);
  assert.match(
    output,
    /Duplicate occurrence: abandoned branch · \/sessions\/copy\.jsonl#copy-entry/,
  );
  assert.match(output, /Expanded chunks: chunk-1/);
});

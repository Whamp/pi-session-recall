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
    rankingMode: 'compact' as const,
    mixedResultPolicyVersion: 1,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, invocation: 8 },
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
      documentCounts: { dense: 42, invocations: 0 },
      results: [result],
      indexMaintenanceStatus: null,
      searchPolicy: createSearchPolicy(),
    },
    40,
  );

  assert.match(output, /compact mixed retrieval v1/);
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
    documentCounts: { dense: 42, invocations: 0 },
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
    documentCounts: { dense: 1, invocations: 0 },
    results: [{ ...turn, evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL }],
    indexMaintenanceStatus: null,
    searchPolicy: createSearchPolicy(),
  });

  assert.match(output, /turn context/);
  assert.match(output, /Contributing entries: user-request → assistant-reply/);
  assert.match(output, /Source: \/sessions\/one\.jsonl:2-5#user-request/);
});

void test('compact Invocation output names the tool and exact source locator without raw output', () => {
  const output = formatRecallSearchResults({
    totalChunks: 3,
    documentCounts: { dense: 3, invocations: 7 },
    results: [
      {
        resultKind: 'invocation',
        kind: 'tool_call',
        toolName: 'read',
        toolCallId: 'call-compact',
        sessionPath: '/sessions/compact.jsonl',
        sessionId: 'compact-session',
        entryId: 'compact-entry',
        sourceLineStart: 12,
        sourceLineEnd: 12,
        sourceBlockIndex: 1,
        timestamp: '2026-08-10T10:00:00Z',
        sessionOrigin: '/project',
        projectAttribution: null,
        isError: false,
        searchableText: 'tool="read"\npath="/project/src/compact-catalog.ts"',
        content: 'tool="read"\npath="/project/src/compact-catalog.ts"',
        rank: -7.5,
        evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
      },
    ],
    indexMaintenanceStatus: null,
    searchPolicy: {
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
      rankingMode: 'compact',
      mixedResultPolicyVersion: 1,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, invocation: 8 },
    },
  });

  assert.match(output, /3 dense documents and 7 compact Invocations/u);
  assert.match(output, /read Invocation/u);
  assert.match(output, /path="\/project\/src\/compact-catalog\.ts"/u);
  assert.match(output, /Source: \/sessions\/compact\.jsonl:12-12#compact-entry/u);
  assert.doesNotMatch(output, /RAW_RESULT/u);
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
    documentCounts: { dense: 2, invocations: 0 },
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

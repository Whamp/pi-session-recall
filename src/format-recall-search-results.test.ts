import assert from 'node:assert/strict';
import test from 'node:test';

import { formatRecallSearchResults } from './format-recall-search-results.js';
import { createTestRankedRecallSearchResult } from './recall-test-utils.js';
import type { RankedRecallSearchResult } from './rank-recall-search-results.js';

const result = createTestRankedRecallSearchResult({
  id: 'chunk-1',
  checksum: 'sum-1',
  sessionPath: '/sessions/one.jsonl',
  sessionName: 'Queue design',
  entryId: { value: 'entry-1' },
  contributingEntryIds: [{ value: 'entry-1' }],
  currentLeafId: { value: 'entry-1' },
  branchPathLeafIds: [{ value: 'entry-1' }],
  tokenEnd: 10,
  tokenCount: 10,
  textRunId: 'text-run-1',
  content: 'The durable queue decision and its tradeoffs are documented here.',
  dense: { rank: 1, cosineDistance: 0.01234 },
  lexical: { rank: 2, fullTextScore: 0.87654 },
  fusedScore: 0.03252,
  rerankerScore: 0.91234,
  activeBranchPrior: 0.01,
  rankingScore: 0.92234,
});

void test('recall results include concise excerpts and exact source provenance', () => {
  const output = formatRecallSearchResults(
    {
      totalChunks: 42,
      results: [result],
      searchPolicy: {
        rankingMode: 'deep-rerank',
        rankFusionVersion: 1,
        reciprocalRankConstant: 60,
        rerankPolicyVersion: 1,
        rerankerModel: 'qwen3-rerank',
        activeBranchPrior: 0.01,
        candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
      },
    },
    40,
  );

  assert.match(output, /Qwen qwen3-rerank policy v1 \(active prior \+0\.0100\)/);
  assert.match(output, /1\. Queue design/);
  assert.match(
    output,
    /2026-07-24T10:00:00Z · assistant · atomic conversation · active branch · \/project/,
  );
  assert.match(output, /Source: \/sessions\/one\.jsonl#entry-1/);
  assert.match(output, /ranking 0\.9223/);
  assert.match(output, /Qwen reranker 0\.9123/);
  assert.match(output, /active prior \+0\.0100/);
  assert.match(output, /fused RRF 0\.0325/);
  assert.match(output, /dense #1 cosine distance 0\.0123/);
  assert.match(output, /lexical #2 FTS 0\.8765/);
  assert.match(output, /active branch/);
  assert.match(output, /…/);
  assert.ok(!output.includes('Incremental index'));
  assert.ok(!output.includes('checksum'));
});

void test('hybrid recall output does not claim Qwen reranking ran', () => {
  const output = formatRecallSearchResults({
    totalChunks: 42,
    results: [{ ...result, rerankerScore: null, rankingScore: result.fusedScore + 0.01 }],
    searchPolicy: {
      rankingMode: 'hybrid',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: null,
      rerankerModel: null,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  });

  assert.match(output, /deterministic fusion v1/);
  assert.match(output, /without Qwen reranking/);
  assert.ok(!output.includes('Qwen reranker 0.'));
});

void test('turn-context results identify their kind and every contributing entry', () => {
  const turnContextResult = {
    ...result,
    schemaVersion: 4,
    documentKind: 'turn_context',
    evidenceKind: 'turn_context',
    id: 'turn-context-chunk',
    entryId: { value: 'user-request' },
    contributingEntryIds: [{ value: 'user-request' }, { value: 'assistant-reply' }],
    role: 'turn',
    sourceLineStart: 2,
    sourceLineEnd: 5,
    content: 'User:\nShip release Atlas.\n\nAssistant:\nYes, do it.',
  } satisfies RankedRecallSearchResult;

  const output = formatRecallSearchResults({
    totalChunks: 1,
    results: [turnContextResult],
    searchPolicy: {
      rankingMode: 'deep-rerank',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: 1,
      rerankerModel: 'qwen3-rerank',
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
    },
  });

  assert.match(output, /turn · turn context · active branch · \/project/);
  assert.match(output, /Contributing entries: user-request → assistant-reply/);
  assert.match(output, /Source: \/sessions\/one\.jsonl#user-request/);
});

void test('tool evidence results identify the exact call relationship and source', () => {
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
    results: [toolResult],
    searchPolicy: {
      rankingMode: 'deep-rerank',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: 1,
      rerankerModel: 'qwen3-rerank',
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
    },
  });

  assert.match(output, /tool · tool_result\/result · bash · call call-tools · error/);
  assert.match(output, /Call source: call-entry · Result source: result-entry/);
  assert.match(output, /Source: \/sessions\/one\.jsonl#result-entry/);
});

void test('recall results format stitched neighbors and every suppressed provenance occurrence', () => {
  const previous = {
    ...result,
    id: 'previous-chunk',
    characterStart: 0,
    characterEnd: 16,
    chunkIndex: 0,
    content: 'alpha beta gamma',
  };
  const winner = {
    ...result,
    id: 'winner-chunk',
    characterStart: 11,
    characterEnd: 22,
    chunkIndex: 1,
    content: 'gamma delta',
  };
  const next = {
    ...result,
    id: 'next-chunk',
    characterStart: 17,
    characterEnd: 30,
    chunkIndex: 2,
    content: 'delta epsilon',
  };
  const duplicate = {
    ...result,
    id: 'copied-chunk',
    sessionPath: '/sessions/copied.jsonl',
    entryId: { value: 'copied-entry' },
    isOnActiveBranch: false,
    characterStart: 40,
    characterEnd: 62,
    fusedScore: 0.02,
  };
  const expandedResult = {
    ...winner,
    duplicateOccurrences: [duplicate],
    neighborContext: {
      content: 'alpha beta gamma delta epsilon',
      chunks: [previous, winner, next],
    },
  } satisfies RankedRecallSearchResult;

  const output = formatRecallSearchResults({
    totalChunks: 4,
    results: [expandedResult],
    searchPolicy: {
      rankingMode: 'deep-rerank',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: 1,
      rerankerModel: 'qwen3-rerank',
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
    },
  });

  assert.match(output, /alpha beta gamma delta epsilon/);
  assert.match(
    output,
    /Expanded chunks: previous-chunk \[characters 0-16\] → winner-chunk \[characters 11-22\] → next-chunk \[characters 17-30\]/,
  );
  assert.match(
    output,
    /Duplicate occurrence: abandoned branch · \/sessions\/copied\.jsonl#copied-entry · document copied-chunk · characters 40-62 · fused RRF 0\.0200/,
  );
  assert.match(output, /Source: \/sessions\/one\.jsonl#entry-1/);
});

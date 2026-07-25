import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
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
        scope: RecallSearchScope.GLOBAL,
        invocationProjectIdentity: null,
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
    /2026-07-24T10:00:00Z · assistant · atomic conversation · active branch · session origin \/project · unrestricted global evidence/,
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

void test('project-scoped output explains invocation identity, session origin, and repository relationship', () => {
  const projectIdentity = 'git-origin:github.com/Whamp/pi-session-recall';
  const output = formatRecallSearchResults({
    totalChunks: 42,
    results: [
      {
        ...result,
        cwd: '/workspace/pi-session-recall',
        projectIdentity,
        projectIdentitySource: RecallProjectIdentitySource.GIT_ORIGIN,
        evidenceRelation: RecallEvidenceRelation.SAME_REPOSITORY,
      },
    ],
    searchPolicy: {
      scope: RecallSearchScope.PROJECT,
      invocationProjectIdentity: projectIdentity,
      rankingMode: 'hybrid',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: null,
      rerankerModel: null,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  });

  assert.match(output, /project scope/);
  assert.match(output, /git-origin:github\.com\/Whamp\/pi-session-recall/);
  assert.match(output, /session origin \/workspace\/pi-session-recall/);
  assert.match(output, /same repository/);
});

void test('configured-lineage output explains the historical session origin relation', () => {
  const projectIdentity = 'git-origin:github.com/Whamp/successor';
  const output = formatRecallSearchResults({
    totalChunks: 1,
    results: [
      {
        ...result,
        cwd: '/historical/prototype/packages/app',
        projectIdentity,
        projectIdentitySource: RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE,
        evidenceRelation: RecallEvidenceRelation.CONFIGURED_PROJECT_LINEAGE,
      },
    ],
    searchPolicy: {
      scope: RecallSearchScope.PROJECT,
      invocationProjectIdentity: projectIdentity,
      rankingMode: 'hybrid',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: null,
      rerankerModel: null,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  });

  assert.match(output, /session origin \/historical\/prototype\/packages\/app/);
  assert.match(output, /configured project lineage/);
});

void test('empty project recall recommends an explicit global retry without widening scope', () => {
  const output = formatRecallSearchResults({
    totalChunks: 42,
    results: [],
    searchPolicy: {
      scope: RecallSearchScope.PROJECT,
      invocationProjectIdentity: 'non-git-session-origin:/workspace/local-project',
      rankingMode: 'hybrid',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: null,
      rerankerModel: null,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  });

  assert.match(output, /No matching past conversations found/);
  assert.match(output, /Retry with scope "global"/);
  assert.match(output, /project scope was not broadened automatically/);
});

void test('hybrid recall output does not claim Qwen reranking ran', () => {
  const output = formatRecallSearchResults({
    totalChunks: 42,
    results: [{ ...result, rerankerScore: null, rankingScore: result.fusedScore + 0.01 }],
    searchPolicy: {
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
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
    schemaVersion: 6,
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
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
      rankingMode: 'deep-rerank',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: 1,
      rerankerModel: 'qwen3-rerank',
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 40, lexical: 40, identifier: 40 },
    },
  });

  assert.match(
    output,
    /turn · turn context · active branch · session origin \/project · unrestricted global evidence/,
  );
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
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
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
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
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

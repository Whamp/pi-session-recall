import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallSearchScope } from './enums.js';
import recallExtension, { createPiRecallToolDetails, searchPiRecall } from './recall-extension.js';
import type { RecallConversationService } from './recall-conversation-service.js';
import { createTestRankedRecallSearchResult } from './recall-test-utils.js';

function createEmptySearch(scope: RecallSearchScope, invocationProjectIdentity: null = null) {
  return {
    results: [],
    totalChunks: 0,
    searchPolicy: {
      scope,
      invocationProjectIdentity,
      rankingMode: 'hybrid' as const,
      rankFusionVersion: 2,
      reciprocalRankConstant: 60,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    },
  };
}

void test('Pi recall applies trusted cwd and project-default scope without a write path', async () => {
  const calls: unknown[] = [];
  const service = {
    async search(query, limit, options) {
      calls.push({ query, limit, options });
      return createEmptySearch(options?.scope ?? RecallSearchScope.PROJECT);
    },
    async index() {
      throw new Error('search adapter must not index');
    },
  } satisfies RecallConversationService;

  await searchPiRecall(service, { query: '  queue decision  ' }, { cwd: '/trusted/project' });
  await searchPiRecall(
    service,
    { query: 'global decision', limit: 3, scope: 'global' },
    { cwd: '/trusted/project' },
  );

  assert.deepEqual(calls, [
    {
      query: 'queue decision',
      limit: 5,
      options: {
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/project',
      },
    },
    {
      query: 'global decision',
      limit: 3,
      options: {
        scope: RecallSearchScope.GLOBAL,
        invocationDirectory: '/trusted/project',
      },
    },
  ]);
});

void test('Pi tool details retain line, block, character, and contributing-entry provenance', () => {
  const result = createTestRankedRecallSearchResult({
    id: 'source-result',
    sessionPath: '/sessions/source.jsonl',
    entryId: { value: 'source-entry' },
    contributingEntryIds: [{ value: 'source-entry' }, { value: 'context-entry' }],
    sourceLineStart: 20,
    sourceLineEnd: 24,
    sourceBlockStart: 1,
    sourceBlockEnd: 3,
    characterStart: 8,
    characterEnd: 88,
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
  });

  const details = createPiRecallToolDetails({
    results: [result],
    totalChunks: 1,
    searchPolicy: createEmptySearch(RecallSearchScope.GLOBAL).searchPolicy,
  });

  assert.deepEqual(details.sources[0], {
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    sessionOrigin: '/project',
    projectIdentity: null,
    projectIdentitySource: null,
    sessionPath: '/sessions/source.jsonl',
    entryId: 'source-entry',
    contributingEntryIds: ['source-entry', 'context-entry'],
    sourceLineStart: 20,
    sourceLineEnd: 24,
    sourceBlockStart: 1,
    sourceBlockEnd: 3,
    characterStart: 8,
    characterEnd: 88,
    isOnActiveBranch: true,
    rankingScore: 0.02,
    activeBranchPrior: 0,
    fusedScore: 0.02,
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    duplicateOccurrences: [],
    expandedChunks: [],
  });
});

void test('Pi extension registers only the read-only recall tool and directs maintenance to psr', async () => {
  let registeredToolCount = 0;
  let registeredName = '';
  let registeredDescription = '';
  let registeredParameters: unknown;
  await recallExtension({
    registerTool(tool) {
      registeredToolCount += 1;
      registeredName = tool.name;
      registeredDescription = tool.description;
      registeredParameters = tool.parameters;
    },
  });

  assert.equal(registeredToolCount, 1);
  assert.equal(registeredName, 'pi-session-recall');
  assert.match(registeredDescription, /run `psr index` explicitly/);
  assert.doesNotMatch(registeredDescription, /Qwen|query.plann|background/iu);
  const schemaText = JSON.stringify(registeredParameters);
  assert.match(schemaText, /query/);
  assert.match(schemaText, /scope/);
  assert.doesNotMatch(schemaText, /mode|rebuild/);
});

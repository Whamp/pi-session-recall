import assert from 'node:assert/strict';
import test from 'node:test';

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

import { RecallRankedListSource, RecallSearchScope } from './enums.js';
import recallExtension, { createPiRecallToolDetails, searchPiRecall } from './recall-extension.js';
import type {
  RecallConversationSearchOptions,
  RecallConversationService,
} from './recall-conversation-service.js';
import { createTestRankedRecallSearchResult } from './recall-test-utils.js';

void test('Pi session recall registers collision-free tool guidance and index command', async () => {
  const toolNames: string[] = [];
  const toolDescriptions: string[] = [];
  const toolGuidelines: string[] = [];
  const commandNames: string[] = [];
  const commandDescriptions: string[] = [];
  const toolParameterSchemas: string[] = [];
  const registrar: Pick<ExtensionAPI, 'registerTool' | 'registerCommand'> = {
    registerTool(definition) {
      toolNames.push(definition.name);
      toolDescriptions.push(definition.description);
      toolGuidelines.push(...(definition.promptGuidelines ?? []));
      toolParameterSchemas.push(JSON.stringify(definition.parameters));
    },
    registerCommand(name, definition) {
      commandNames.push(name);
      commandDescriptions.push(definition.description ?? '');
    },
  };

  await recallExtension(registrar);

  assert.deepEqual(toolNames, ['pi-session-recall']);
  assert.deepEqual(commandNames, ['pi-session-recall-index']);
  assert.ok(!toolNames.includes('recall'));
  assert.ok(!commandNames.includes('recall-index'));
  assert.match(commandDescriptions[0] ?? '', /quality gate/);
  assert.match(commandDescriptions[0] ?? '', /--rebuild/);
  assert.match(
    toolDescriptions[0] ?? '',
    /dense, lexical, and case-preserving identifier retrieval/,
  );
  assert.match(toolDescriptions[0] ?? '', /defaults to project scope/);
  assert.match(toolDescriptions[0] ?? '', /defaults to deterministic hybrid ranking/);
  assert.match(toolDescriptions[0] ?? '', /deep-rerank.*Qwen/);
  assert.match(toolDescriptions[0] ?? '', /labels active and abandoned branches/);
  assert.match(toolDescriptions[0] ?? '', /valid same-run atomic neighbors/);
  assert.match(toolParameterSchemas[0] ?? '', /project/);
  assert.match(toolParameterSchemas[0] ?? '', /global/);
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('projectPath'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('invocationDirectory'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('activeSessionPath'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('lifecycleTrigger'));
  assert.ok(
    toolGuidelines.some(
      (guideline) =>
        guideline.includes('Use pi-session-recall') &&
        guideline.includes('conversation or detail from a past session'),
    ),
  );
});

void test('Pi recall runtime never registers automatic whole-session maintenance', async () => {
  const lifecycleEvents: string[] = [];
  const registrar: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'> = {
    on(event) {
      lifecycleEvents.push(event);
    },
    registerTool() {},
    registerCommand() {},
  };

  await recallExtension(registrar);

  assert.deepEqual(lifecycleEvents, []);
});

void test('Pi recall tool details retain ranked-list evidence and every explicit limit', () => {
  const evidence = {
    source: RecallRankedListSource.LEXICAL,
    query: 'lease token',
    rank: 1,
    nativeScore: 8.5,
    weight: 1,
  };
  const details = createPiRecallToolDetails({
    totalChunks: 9,
    results: [
      createTestRankedRecallSearchResult({
        id: 'tool-details-result',
        rankedListEvidence: [evidence],
      }),
    ],
    searchPolicy: {
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
      rankingMode: 'hybrid',
      rankFusionVersion: 2,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: null,
      rerankerModel: null,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      fusedPoolLimit: 24,
      rerankPoolLimit: 24,
      finalResultLimit: 5,
    },
  });

  assert.deepEqual(details.searchPolicy, {
    scope: RecallSearchScope.GLOBAL,
    invocationProjectIdentity: null,
    rankingMode: 'hybrid',
    rankFusionVersion: 2,
    reciprocalRankConstant: 60,
    rerankPolicyVersion: null,
    rerankerModel: null,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    fusedPoolLimit: 24,
    rerankPoolLimit: 24,
    finalResultLimit: 5,
  });
  assert.deepEqual(details.sources[0]?.rankedListEvidence, [evidence]);
});

void test('Pi recall tool adapter propagates trusted cwd with project default and explicit global scope', async () => {
  const calls: Array<{
    query: string;
    limit: number;
    options: RecallConversationSearchOptions;
  }> = [];
  const service: RecallConversationService = {
    async search(query, limit, options) {
      if (!options) {
        throw new Error('Pi recall adapter test expected search options');
      }
      calls.push({ query, limit, options });
      return {
        totalChunks: 0,
        results: [],
        searchPolicy: {
          scope: options?.scope ?? RecallSearchScope.PROJECT,
          invocationProjectIdentity: null,
          rankingMode: options?.mode ?? 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
          rerankPolicyVersion: null,
          rerankerModel: null,
          activeBranchPrior: 0.01,
          candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
          fusedPoolLimit: 24,
          rerankPoolLimit: 24,
          finalResultLimit: limit,
        },
      };
    },
    async index() {
      return {
        totalChunks: 0,
        indexSummary: {
          scannedSessions: 0,
          indexedSessions: 0,
          removedSessions: 0,
          cacheHits: 0,
          newlyEmbeddedChunks: 0,
          embeddingRequestCount: 0,
          deletedChunks: 0,
          failedSessions: [],
        },
      };
    },
    async reconcileSession() {
      return {
        totalChunks: 0,
        indexSummary: {
          scannedSessions: 1,
          indexedSessions: 0,
          removedSessions: 0,
          cacheHits: 0,
          newlyEmbeddedChunks: 0,
          embeddingRequestCount: 0,
          deletedChunks: 0,
          failedSessions: [],
        },
      };
    },
  };
  const context = {
    cwd: '/trusted/invocation',
    sessionManager: {
      getSessionFile() {
        return '/sessions/active.jsonl';
      },
    },
  };

  await searchPiRecall(service, { query: 'project query', mode: 'hybrid' }, context, 5);
  await searchPiRecall(
    service,
    { query: 'global query', mode: 'deep-rerank', scope: 'global', limit: 2 },
    context,
    5,
  );

  assert.deepEqual(calls, [
    {
      query: 'project query',
      limit: 5,
      options: {
        mode: 'hybrid',
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/invocation',
      },
    },
    {
      query: 'global query',
      limit: 2,
      options: {
        mode: 'deep-rerank',
        scope: RecallSearchScope.GLOBAL,
        invocationDirectory: '/trusted/invocation',
      },
    },
  ]);
});

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
  assert.match(toolDescriptions[0] ?? '', /query-planned.*configured query planner/);
  assert.match(toolDescriptions[0] ?? '', /labels active and abandoned branches/);
  assert.match(toolDescriptions[0] ?? '', /valid same-run atomic neighbors/);
  assert.match(toolParameterSchemas[0] ?? '', /project/);
  assert.match(toolParameterSchemas[0] ?? '', /global/);
  assert.match(toolParameterSchemas[0] ?? '', /query-planned/);
  assert.match(toolParameterSchemas[0] ?? '', /lex/);
  assert.match(toolParameterSchemas[0] ?? '', /vec/);
  assert.match(toolParameterSchemas[0] ?? '', /hyde/);
  assert.match(toolParameterSchemas[0] ?? '', /"maxItems":10/);
  assert.match(toolParameterSchemas[0] ?? '', /omit.*configured query planner/);
  assert.match(toolParameterSchemas[0] ?? '', /intent/);
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
      rerankerIdentity: null,
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
    rerankerIdentity: null,
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    fusedPoolLimit: 24,
    rerankPoolLimit: 24,
    finalResultLimit: 5,
  });
  assert.deepEqual(details.sources[0]?.rankedListEvidence, [evidence]);
  assert.ok(!('topRankBonus' in (details.sources[0] ?? {})));
  assert.ok(!('retrievalPositionRank' in (details.sources[0] ?? {})));
});

void test('Pi recall tool details expose query-plan and position-aware ranking evidence', () => {
  const queryPlan = {
    source: 'agent' as const,
    plannerIdentity: null,
    intent: 'recover the accepted decision',
    plannedQueries: [{ type: 'vec' as const, query: 'Which design was accepted?' }],
    rankedLists: [
      {
        source: RecallRankedListSource.PLANNED_VEC,
        query: 'Which design was accepted?',
        weight: 1,
        candidateLimit: 20,
        admittedCandidateCount: 7,
      },
    ],
    fusionPolicy: {
      reciprocalRankConstant: 60,
      submittedQueryListWeight: 2,
      plannedQueryListWeight: 1,
      rankOneBonus: 0.05,
      rankTwoOrThreeBonus: 0.02,
    },
    rerankerProfile: {
      model: 'qwen3-rerank',
      policyVersion: 2,
      fusedRankBlend: [
        { firstRank: 1, lastRank: 3, retrievalWeight: 0.75, rerankerWeight: 0.25 },
        { firstRank: 4, lastRank: 10, retrievalWeight: 0.6, rerankerWeight: 0.4 },
        { firstRank: 11, lastRank: null, retrievalWeight: 0.4, rerankerWeight: 0.6 },
      ],
    },
  };
  const details = createPiRecallToolDetails({
    totalChunks: 9,
    results: [
      createTestRankedRecallSearchResult({
        id: 'query-planned-details-result',
        topRankBonus: 0.05,
        retrievalPositionRank: 2,
        retrievalPositionScore: 0.5,
        retrievalScoreWeight: 0.75,
        rerankerScoreWeight: 0.25,
      }),
    ],
    searchPolicy: {
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
      rankingMode: 'query-planned',
      rankFusionVersion: 2,
      reciprocalRankConstant: 60,
      rerankPolicyVersion: 2,
      rerankerModel: 'qwen3-rerank',
      rerankerIdentity: {
        profileId: 'qwen-reranking:qwen3-rerank',
        adapterId: 'custom-injected-reranking-v1',
        cacheIdentity: 'qwen-reranking:qwen3-rerank:custom-injected-reranking-v1',
      },
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 20, lexical: 20, identifier: 20 },
      fusedPoolLimit: 40,
      rerankPoolLimit: 40,
      finalResultLimit: 5,
      queryPlan,
    },
  });

  assert.deepEqual(details.searchPolicy.queryPlan, queryPlan);
  assert.deepEqual(
    {
      topRankBonus: details.sources[0]?.topRankBonus,
      retrievalPositionRank: details.sources[0]?.retrievalPositionRank,
      retrievalPositionScore: details.sources[0]?.retrievalPositionScore,
      retrievalScoreWeight: details.sources[0]?.retrievalScoreWeight,
      rerankerScoreWeight: details.sources[0]?.rerankerScoreWeight,
    },
    {
      topRankBonus: 0.05,
      retrievalPositionRank: 2,
      retrievalPositionScore: 0.5,
      retrievalScoreWeight: 0.75,
      rerankerScoreWeight: 0.25,
    },
  );
});

void test('Pi recall tool adapter propagates trusted cwd with project default and explicit global scope', async () => {
  const calls: Array<{
    query: string;
    limit: number;
    options: RecallConversationSearchOptions;
  }> = [];
  const service: RecallConversationService = {
    async verifyEmbeddingCapability() {
      throw new Error('Pi recall adapter test does not verify embeddings');
    },
    async inspectConversationCorpus() {
      throw new Error('Pi recall adapter test does not inspect the corpus');
    },
    async measureFirstIndexSample() {
      throw new Error('Pi recall adapter test does not measure indexing');
    },
    async verifyRerankingCapability() {
      throw new Error('Pi recall adapter test does not configure reranking');
    },
    async verifyQueryPlanningCapability() {
      throw new Error('Pi recall adapter test does not configure query planning');
    },
    async search(query, limit, options) {
      if (!options) {
        throw new Error('Pi recall adapter test expected search options');
      }
      calls.push({ query, limit, options });
      return {
        totalChunks: 0,
        results: [],
        candidateAdmission: [],
        searchPolicy: {
          scope: options?.scope ?? RecallSearchScope.PROJECT,
          invocationProjectIdentity: null,
          rankingMode: options?.mode ?? 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
          rerankPolicyVersion: null,
          rerankerModel: null,
          rerankerIdentity: null,
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
    async startBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not start background indexing');
    },
    async resumeBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not resume background indexing');
    },
    async readBackgroundIndexGenerationStatus() {
      return null;
    },
    async stopBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not stop background indexing');
    },
    async readIndexGenerationStatus() {
      return { active: null, staging: null };
    },
    async discardStagingIndexGeneration() {
      return false;
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
  const plan = [{ type: 'hyde', query: 'Expected answer passage.' }] as const;
  await searchPiRecall(
    service,
    {
      query: 'planned query',
      mode: 'query-planned',
      plan,
      intent: 'recover the accepted decision',
    },
    context,
    5,
  );
  await searchPiRecall(
    service,
    {
      query: 'planner-generated query',
      mode: 'query-planned',
      intent: 'recover a model-generated plan',
    },
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
    {
      query: 'planned query',
      limit: 5,
      options: {
        mode: 'query-planned',
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/invocation',
        plan,
        intent: 'recover the accepted decision',
      },
    },
    {
      query: 'planner-generated query',
      limit: 5,
      options: {
        mode: 'query-planned',
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/invocation',
        intent: 'recover a model-generated plan',
      },
    },
  ]);
});

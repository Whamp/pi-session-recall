import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallInferenceBackend } from './enums.js';

import { RecallDiagnosticsMode, RecallSearchScope } from './enums.js';
import { createRecallQueryPlanningExecutionIdentity } from './recall-inference-capabilities.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  type RecallQueryPlanningModelProfile,
} from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

void test('conversation service verifies replacement planners without rebuilding vectors or planning search', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-qmd-planner-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'planner.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'planner-session',
        timestamp: '2026-07-28T10:00:00Z',
        cwd: '/planner-project',
      },
      {
        type: 'message',
        id: 'planner-evidence',
        parentId: null,
        timestamp: '2026-07-28T10:01:00Z',
        message: { role: 'assistant', content: 'Source provenance remains searchable evidence.' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const config = {
    sessionsDirectory,
    dataDirectory: directory,
    databasePath: join(directory, 'zvec'),
    projectionDatabasePath: join(directory, 'session-projections'),
    markerSpoolDirectory: join(directory, 'markers', 'pending'),
    markerQuarantineDirectory: join(directory, 'markers', 'quarantine'),
    markerControlDirectory: join(directory, 'markers', 'control'),
    workerOwnershipLockPath: join(directory, 'incremental-worker.lock'),
    generationRootDirectory: join(directory, 'generations'),
    activeGenerationPointerPath: join(directory, 'active-generation.json'),
    generationRegistryPath: join(directory, 'generation-registry.json'),
    backlogSummaryPath: join(directory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(directory, 'incremental-diagnostics.jsonl'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    lockPath: join(directory, 'recall.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'fixture-embedding-model',
    embeddingServedModelId: 'fixture-embedding-model',
    embeddingArtifact: 'fixture-embedding.gguf',
    embeddingQuantization: 'fixture',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'unused-reranker',
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 2, lexical: 2, identifier: 2 },
  };
  const embeddingProvider = {
    async embedQuery(query: string) {
      return query === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0];
    },
    async embedDocuments(documents: readonly string[]) {
      return documents.map(() => [1, 0, 0]);
    },
  };
  const recommendedProfile = createRecommendedQmdQueryPlanningModelProfile();
  let firstPlannerCalls = 0;
  const firstAdapterId = 'fixture-query-planning-v1';
  const firstPlanner = {
    executionIdentity: createRecallQueryPlanningExecutionIdentity(
      recommendedProfile,
      firstAdapterId,
      'first-planner-configuration',
      RecallInferenceBackend.CUSTOM,
      7_000,
    ),
    async planRecallQuery() {
      firstPlannerCalls += 1;
      return [
        { type: 'lex' as const, query: 'Copper Finch evidence' },
        { type: 'vec' as const, query: 'how Copper Finch is retained' },
      ];
    },
  };
  const firstService = createRecallConversationService(config, {
    embeddingProvider,
    queryPlanningProfile: recommendedProfile,
    queryPlanner: firstPlanner,
    loadTokenizer: async () => TOKENIZER,
  });
  await firstService.index({ rebuild: true });

  const firstVerification = await firstService.verifyQueryPlanningCapability();
  const firstSearch = await firstService.search('source provenance', 1, {
    mode: 'hybrid',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(firstPlannerCalls, 1);
  assert.equal(firstSearch.results[0]?.entryId.value, 'planner-evidence');
  assert.equal(firstVerification.profileId, recommendedProfile.profileId);
  assert.deepEqual(firstVerification.executionIdentity, firstPlanner.executionIdentity);
  assert.deepEqual(firstVerification.measurement, {
    plannedQueryCount: 2,
    lexQueryCount: 1,
    vecQueryCount: 1,
    hydeQueryCount: 0,
    planningMilliseconds: firstVerification.measurement.planningMilliseconds,
  });

  const adapterOnlyId = 'fixture-query-planning-v2';
  const adapterOnlyPlanner = {
    executionIdentity: createRecallQueryPlanningExecutionIdentity(
      recommendedProfile,
      adapterOnlyId,
      'adapter-only-planner-configuration',
      RecallInferenceBackend.CUSTOM,
      7_500,
    ),
    async planRecallQuery() {
      return [
        { type: 'lex' as const, query: 'Copper Finch records' },
        { type: 'vec' as const, query: 'how Finch connects recovery evidence' },
      ];
    },
  };
  const adapterOnlyService = createRecallConversationService(config, {
    embeddingProvider,
    queryPlanningProfile: recommendedProfile,
    queryPlanner: adapterOnlyPlanner,
    loadTokenizer: async () => TOKENIZER,
  });
  const adapterOnlyVerification = await adapterOnlyService.verifyQueryPlanningCapability();
  const adapterOnlySearch = await adapterOnlyService.search('source provenance', 1, {
    mode: 'hybrid',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(adapterOnlyVerification.profileId, recommendedProfile.profileId);
  assert.notEqual(
    adapterOnlyVerification.executionIdentity.cacheIdentity,
    firstVerification.executionIdentity.cacheIdentity,
  );
  assert.equal(adapterOnlySearch.results[0]?.entryId.value, 'planner-evidence');
  assert.equal(adapterOnlySearch.totalChunks, firstSearch.totalChunks);

  const replacementProfile: RecallQueryPlanningModelProfile = {
    ...recommendedProfile,
    profileId: 'replacement-query-planner-v2',
    model: 'replacement-query-planner',
  };
  let replacementPlannerCalls = 0;
  const replacementAdapterId = 'replacement-query-planning-v2';
  const replacementPlanner = {
    executionIdentity: createRecallQueryPlanningExecutionIdentity(
      replacementProfile,
      replacementAdapterId,
      'replacement-planner-configuration',
      RecallInferenceBackend.CUSTOM,
      8_000,
    ),
    async planRecallQuery() {
      replacementPlannerCalls += 1;
      return [
        { type: 'lex' as const, query: 'Copper Finch records' },
        { type: 'vec' as const, query: 'where Finch records connect to recovery evidence' },
      ];
    },
  };
  const replacementService = createRecallConversationService(config, {
    embeddingProvider,
    queryPlanningProfile: replacementProfile,
    queryPlanner: replacementPlanner,
    loadTokenizer: async () => TOKENIZER,
  });

  const replacementVerification = await replacementService.verifyQueryPlanningCapability();
  const replacementSearch = await replacementService.search('source provenance', 1, {
    mode: 'hybrid',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(replacementPlannerCalls, 1);
  assert.equal(replacementSearch.results[0]?.entryId.value, 'planner-evidence');
  assert.equal(replacementSearch.totalChunks, firstSearch.totalChunks);
  assert.equal(replacementVerification.profileId, replacementProfile.profileId);
  assert.equal(
    replacementVerification.executionIdentity.cacheIdentity,
    replacementPlanner.executionIdentity.cacheIdentity,
  );
  assert.equal(replacementSearch.searchPolicy.rankingMode, 'hybrid');
});

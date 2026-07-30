import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { EmbeddedInferenceDevicePolicy } from './enums.js';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { RecallDiagnosticsMode, RecallSearchScope } from './enums.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import {
  createQwenRerankingModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const QWEN_SERVICE_RERANKING_REQUEST_SCHEMA = Type.Object({
  query: Type.String(),
  documents: Type.Array(Type.String()),
});

const TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createQwenServiceTestConfig(directory: string, sessionsDirectory: string) {
  return {
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
}

function createQwenServiceTestEmbeddingProvider(rerankerFavorite: string) {
  return {
    async embedQuery(query: string) {
      return query === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0];
    },
    async embedDocuments(documents: readonly string[]) {
      return documents.map((document) =>
        document === rerankerFavorite ? [0.9, 0.1, 0] : [1, 0, 0],
      );
    },
  };
}

function applyKnownNodeLlamaCppExtraSigmoid(score: number): number {
  return 1 / (1 + Math.exp(-score));
}

void test('deep-rerank works end to end with built-in HTTP and embedded Qwen adapters', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-qwen-reranker-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const fusionFavorite = 'Exact fusion favorite evidence.';
  const rerankerFavorite = 'Semantically strongest recommended Qwen evidence.';
  await writeFile(
    join(sessionsDirectory, 'reranking.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'qwen-reranker-session',
        timestamp: '2026-07-27T10:00:00Z',
        cwd: '/qwen-reranker-project',
      },
      {
        type: 'message',
        id: 'fusion-favorite',
        parentId: null,
        timestamp: '2026-07-27T10:01:00Z',
        message: { role: 'assistant', content: fusionFavorite },
      },
      {
        type: 'message',
        id: 'reranker-favorite',
        parentId: 'fusion-favorite',
        timestamp: '2026-07-27T10:02:00Z',
        message: { role: 'assistant', content: rerankerFavorite },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const profile = createRecommendedQwenRerankingModelProfile();
  const config = createQwenServiceTestConfig(directory, sessionsDirectory);
  const embeddingProvider = createQwenServiceTestEmbeddingProvider(rerankerFavorite);
  const httpRequests: Array<{ query: string; documents: string[] }> = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const rawPayload: unknown = JSON.parse(body);
      const payload = Value.Parse(QWEN_SERVICE_RERANKING_REQUEST_SCHEMA, rawPayload);
      httpRequests.push(payload);
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: profile.model,
          object: 'list',
          usage: { prompt_tokens: 12, total_tokens: 12 },
          results: payload.documents.map((document, index) => ({
            index,
            relevance_score: document === rerankerFavorite ? 0.9 : 0.1,
          })),
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const httpProvider = createQwenHttpRerankingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  });
  const httpService = createRecallConversationService(config, {
    embeddingProvider,
    rerankingProfile: profile,
    reranker: httpProvider,
    loadTokenizer: async () => TOKENIZER,
  });
  const httpVerification = await httpService.verifyRerankingCapability({
    query: 'recommended evidence',
    documents: [fusionFavorite, rerankerFavorite],
    expectedScores: [0.1, 0.9],
  });
  const generationId = 'generation_qwen_reranker_service';
  await httpService.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [join(sessionsDirectory, 'reranking.jsonl')],
  });
  await httpService.activateValidatedRecallGeneration(generationId);

  assert.equal(httpVerification.profileId, profile.profileId);
  assert.deepEqual(httpVerification.executionIdentity, httpProvider.executionIdentity);
  assert.deepEqual(httpVerification.measurement, {
    queryCount: 1,
    documentCount: 2,
    rerankingMilliseconds: httpVerification.measurement.rerankingMilliseconds,
  });

  const httpSearch = await httpService.search('fusion favorite', 1, {
    mode: 'deep-rerank',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(httpSearch.results[0]?.entryId.value, 'reranker-favorite');
  assert.equal(httpSearch.results[0]?.rerankerScore, 0.9);
  assert.equal(httpRequests.length, 2);
  assert.deepEqual(httpSearch.searchPolicy.rerankerIdentity, {
    profileId: profile.profileId,
    adapterId: 'llama-cpp-http-reranking-v1',
    cacheIdentity: httpProvider.executionIdentity.cacheIdentity,
  });

  const embeddedProvider = createEmbeddedQwenRerankingProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    async verifyModelArtifact() {
      return '/models/qwen-reranker.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama() {
          return {
            gpu: false,
            async loadModel() {
              return {
                async createRankingContext() {
                  return {
                    async rankAll(query, documents) {
                      void query;
                      return documents.map((document) =>
                        applyKnownNodeLlamaCppExtraSigmoid(
                          document === rerankerFavorite ? 0.9 : 0.1,
                        ),
                      );
                    },
                    async dispose() {},
                  };
                },
                async dispose() {},
              };
            },
            async dispose() {},
          };
        },
      };
    },
  });
  t.after(() => embeddedProvider.dispose());
  const embeddedService = createRecallConversationService(config, {
    embeddingProvider,
    rerankingProfile: profile,
    reranker: embeddedProvider,
    loadTokenizer: async () => TOKENIZER,
  });

  const embeddedSearch = await embeddedService.search('fusion favorite', 1, {
    mode: 'deep-rerank',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(embeddedSearch.results[0]?.entryId.value, 'reranker-favorite');
  assert.ok(Math.abs((embeddedSearch.results[0]?.rerankerScore ?? 0) - 0.9) < 1e-12);
  assert.deepEqual(embeddedSearch.searchPolicy.rerankerIdentity, {
    profileId: profile.profileId,
    adapterId: 'node-llama-cpp-qwen-reranking-logit-recovery-v1',
    cacheIdentity: embeddedProvider.executionIdentity.cacheIdentity,
  });

  const replacementProfile = createQwenRerankingModelProfile('replacement-qwen-reranker');
  const replacementHttpProvider = createQwenHttpRerankingProvider(replacementProfile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  });
  const replacementService = createRecallConversationService(config, {
    embeddingProvider,
    rerankingProfile: replacementProfile,
    reranker: replacementHttpProvider,
    loadTokenizer: async () => TOKENIZER,
  });
  const replacementSearch = await replacementService.search('fusion favorite', 1, {
    mode: 'deep-rerank',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(replacementSearch.results[0]?.entryId.value, 'reranker-favorite');
  assert.equal(replacementSearch.totalChunks, httpSearch.totalChunks);
  assert.deepEqual(replacementSearch.searchPolicy.rerankerIdentity, {
    profileId: 'qwen-reranking:replacement-qwen-reranker',
    adapterId: 'llama-cpp-http-reranking-v1',
    cacheIdentity: replacementHttpProvider.executionIdentity.cacheIdentity,
  });

  const embeddingOnlyService = createRecallConversationService(config, {
    embeddingProvider,
    rerankingProfile: null,
    reranker: null,
    loadTokenizer: async () => TOKENIZER,
  });
  const embeddingOnlySearch = await embeddingOnlyService.search('fusion favorite', 1, {
    mode: 'hybrid',
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(embeddingOnlySearch.results[0]?.entryId.value, 'fusion-favorite');
  assert.throws(
    () =>
      embeddingOnlyService.search('fusion favorite', 1, {
        mode: 'deep-rerank',
        scope: RecallSearchScope.GLOBAL,
      }),
    /Recall reranking is not configured/u,
  );
  assert.throws(
    () =>
      embeddingOnlyService.search('fusion favorite', 1, {
        mode: 'query-planned',
        scope: RecallSearchScope.GLOBAL,
        plan: [{ type: 'vec', query: 'reranker favorite' }],
      }),
    /Recall reranking is not configured.*before query-planned search/u,
  );
});

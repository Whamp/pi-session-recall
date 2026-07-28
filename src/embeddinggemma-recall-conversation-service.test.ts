import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
} from './embedded-embeddinggemma-provider.js';
import { RecallDiagnosticsMode, RecallSearchScope } from './enums.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { readRecallIndexManifest } from './recall-index-manifest.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const EMBEDDING_HTTP_REQUEST_SCHEMA = Type.Object({
  model: Type.String(),
  input: Type.Array(Type.String()),
});

function createEmbeddingGemmaServiceConfig(
  directory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    embeddingCacheDirectory: join(directory, 'embedding-cache'),
    lockPath: join(directory, 'recall.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'legacy-config-must-not-select-profile',
    embeddingServedModelId: 'legacy-config-must-not-select-profile',
    embeddingArtifact: 'legacy-config-must-not-select-profile.gguf',
    embeddingQuantization: 'legacy',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'unused-reranker',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
}

function createSemanticFixtureVector(input: string): number[] {
  const vector = Array<number>(768).fill(0);
  const relevant =
    input.includes('atlas architecture rationale') || input.includes('retained atlas architecture');
  vector[relevant ? 1 : 0] = 1;
  return vector;
}

void test('recall service builds and searches one embedded-profile generation across embedded and HTTP execution', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-embeddinggemma-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const toolResult = 'ATLAS_TOOL_OUTPUT_MUST_REMAIN_LEXICAL_ONLY';
  await writeFile(
    join(sessionsDirectory, 'embeddinggemma.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'embeddinggemma-session',
        timestamp: '2026-08-01T10:00:00Z',
        cwd: '/embeddinggemma-project',
      },
      {
        type: 'message',
        id: 'embeddinggemma-assistant',
        parentId: null,
        timestamp: '2026-08-01T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'text',
              text: 'The retained atlas architecture keeps source occurrences attached to evidence.',
            },
            {
              type: 'toolCall',
              id: 'embeddinggemma-tool-call',
              name: 'read',
              arguments: { path: '/tmp/atlas.txt' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'embeddinggemma-tool-result',
        parentId: 'embeddinggemma-assistant',
        timestamp: '2026-08-01T10:02:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'embeddinggemma-tool-call',
          toolName: 'read',
          content: [{ type: 'text', text: toolResult }],
          isError: false,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const embeddedInputs: string[] = [];
  const embeddedProvider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: join(directory, 'models'),
    async verifyModelArtifact() {
      return join(directory, 'models', profile.source.artifact);
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
                embeddingVectorSize: 768,
                tokenize(text) {
                  return Array.from(text).map((character, index) => {
                    void character;
                    return index + 1;
                  });
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor(input) {
                      embeddedInputs.push(input);
                      return { vector: createSemanticFixtureVector(input) };
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
  const tokenizerIdentity = createEmbeddingGemmaTokenizerManifestIdentity(profile);
  const config = createEmbeddingGemmaServiceConfig(directory, sessionsDirectory);
  const embeddedService = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: embeddedProvider,
    tokenizerIdentity,
    loadTokenizer: () => embeddedProvider.loadConversationTokenizer(),
  });

  const indexed = await embeddedService.index();
  const embeddedSearch = await embeddedService.search('atlas architecture rationale', 2, {
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(indexed.totalChunks, 4);
  assert.equal(embeddedSearch.results[0]?.entryId.value, 'embeddinggemma-assistant');
  assert.ok(embeddedInputs.includes(`${profile.queryInputPrefix}${profile.canary.query}`));
  assert.ok(
    embeddedInputs.includes(
      `${profile.documentInputPrefix}The retained atlas architecture keeps source occurrences attached to evidence.`,
    ),
  );
  assert.ok(embeddedInputs.includes(`${profile.queryInputPrefix}atlas architecture rationale`));
  assert.ok(
    embeddedInputs.every(
      (input) => !input.startsWith(`${profile.documentInputPrefix}${toolResult}`),
    ),
  );
  assert.ok(embeddedInputs.every((input) => !input.includes('/tmp/atlas.txt')));

  const manifest = await readRecallIndexManifest(config.manifestPath);
  assert.equal(manifest?.embedding.dimensions, 768);
  assert.equal(manifest?.embedding.normalization, 'l2');
  assert.equal(manifest?.embedding.artifactSha256, profile.source.sha256);
  assert.deepEqual(manifest?.tokenizer, tokenizerIdentity);

  const httpRequests: Array<{ model: string; input: string[] }> = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const payload = Value.Parse(EMBEDDING_HTTP_REQUEST_SCHEMA, JSON.parse(body));
      httpRequests.push(payload);
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          data: payload.input.map((input, index) => ({
            index,
            embedding: createSemanticFixtureVector(input),
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
  const httpProvider = createLlamaCppHttpEmbeddingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    batchSize: 8,
  });
  const httpService = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: httpProvider,
    tokenizerIdentity,
    loadTokenizer: () => embeddedProvider.loadConversationTokenizer(),
  });

  const httpSearch = await httpService.search('atlas architecture rationale', 2, {
    scope: RecallSearchScope.GLOBAL,
  });

  assert.equal(httpSearch.results[0]?.entryId.value, 'embeddinggemma-assistant');
  assert.deepEqual(
    httpRequests.map((request) => request.input),
    [
      [`${profile.queryInputPrefix}${profile.canary.query}`],
      [`${profile.queryInputPrefix}${profile.canary.query}`],
      [`${profile.queryInputPrefix}atlas architecture rationale`],
    ],
  );
  assert.match(await readFile(config.manifestPath, 'utf8'), /"dimensions": 768/u);

  const octenService = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments() {
        return [[1, 0, 0]];
      },
    },
  });
  await assert.rejects(
    () =>
      octenService.search('atlas architecture rationale', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall index manifest incompatible.*embedding\.(?:requestModel|dimensions).*--rebuild/su,
  );
});

void test('recall service indexes with the same profile after automatic accelerator fallback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-embeddinggemma-fallback-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const requestedComputeBackends: unknown[] = [];
  const warnings: string[] = [];
  let modelLoadCount = 0;
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: join(directory, 'models'),
    onWarning(warning) {
      warnings.push(warning);
    },
    async verifyModelArtifact() {
      return join(directory, 'models', profile.source.artifact);
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlamaGpuTypes() {
          return ['cuda'];
        },
        async getLlama(options) {
          requestedComputeBackends.push(options.gpu);
          if (options.gpu === 'cuda') {
            throw new Error('CUDA fixture initialization failed');
          }
          return {
            gpu: false,
            async loadModel() {
              modelLoadCount += 1;
              return {
                embeddingVectorSize: 768,
                tokenize(text) {
                  return Array.from(text).map((character, index) => {
                    void character;
                    return index + 1;
                  });
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor() {
                      return { vector: createSemanticFixtureVector('retained atlas architecture') };
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
  t.after(() => provider.dispose());
  const service = createRecallConversationService(
    createEmbeddingGemmaServiceConfig(directory, sessionsDirectory),
    {
      embeddingProfile: profile,
      embeddingProvider: provider,
      tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
      loadTokenizer: () => provider.loadConversationTokenizer(),
    },
  );

  const indexed = await service.index();

  assert.equal(indexed.totalChunks, 0);
  assert.deepEqual(requestedComputeBackends, ['cuda', false]);
  assert.equal(modelLoadCount, 1);
  assert.equal(warnings.length, 1);
  assert.match(
    warnings[0] ?? '',
    /accelerator initialization failed for cuda; retrying the same profile embeddinggemma-300m-q8-0-v1 on CPU: CUDA fixture initialization failed/u,
  );
  assert.equal(provider.executionIdentity.computeBackend, 'cpu');
  assert.equal(provider.executionIdentity.fallbackFromComputeBackend, 'cuda');
  assert.equal(provider.executionIdentity.profileId, profile.profileId);
});

void test('recall service rejects a non-repeatable EmbeddingGemma canary before indexing', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-embeddinggemma-canary-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  let queryCount = 0;
  const service = createRecallConversationService(
    createEmbeddingGemmaServiceConfig(directory, sessionsDirectory),
    {
      embeddingProfile: profile,
      tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
      embeddingProvider: {
        async embedQuery() {
          const vector = Array<number>(768).fill(0);
          vector[queryCount % 2] = 1;
          queryCount += 1;
          return vector;
        },
        async embedDocuments() {
          assert.fail('documents must not be embedded after canary rejection');
        },
      },
      async loadTokenizer() {
        assert.fail('tokenizer must not load after canary rejection');
      },
    },
  );

  await assert.rejects(
    () => service.index(),
    /Recall embedding canary repeatability mismatch: expected cosine similarity at least 0\.9995, received 0/u,
  );
});

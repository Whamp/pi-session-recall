import assert from 'node:assert/strict';
import test from 'node:test';

import { createEmbeddedEmbeddingGemmaProvider } from './embedded-embeddinggemma-provider.js';
import { measureRecallEmbeddingProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

function createEmbeddingVector(first: number, second: number): number[] {
  return [first, second, ...Array<number>(766).fill(0)];
}

void test('embedded EmbeddingGemma provider passes deterministic CPU and tokenizer conformance', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const events: string[] = [];
  const embeddedInputs: string[] = [];
  const frozenTokenizerIds = new Map<string, number[]>([
    ['Durable source provenance.', [10, 11, 12, 13]],
    ['const evidenceId = "abc123";', [20, 21, 22, 23, 24, 25, 26]],
    ['Recall works 🔎✨', [30, 31, 32, 33, 34]],
    ['記録された決定 مرحبا بالعالم', [40, 41, 42, 43, 44, 45]],
  ]);
  let dynamicImportCount = 0;
  let artifactVerificationCount = 0;
  let getLlamaOptions: Record<string, unknown> | undefined;
  let loadModelOptions: Record<string, unknown> | undefined;
  let createContextOptions: Record<string, unknown> | undefined;

  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    contextSize: 2_048,
    threads: 3,
    async verifyModelArtifact() {
      artifactVerificationCount += 1;
      events.push('verify artifact');
      return '/models/embeddinggemma/profile/embeddinggemma-300M-Q8_0.gguf';
    },
    async loadNodeLlamaCpp() {
      dynamicImportCount += 1;
      events.push('dynamic import');
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama(options) {
          getLlamaOptions = options;
          events.push('load runtime');
          return {
            gpu: false,
            async loadModel(options) {
              loadModelOptions = options;
              events.push('load model');
              return {
                embeddingVectorSize: 768,
                tokenize(text, specialTokens) {
                  assert.equal(specialTokens, false);
                  const ids = frozenTokenizerIds.get(text);
                  if (!ids) {
                    throw new Error(`Unexpected tokenizer conformance input: ${text}`);
                  }
                  return ids;
                },
                async createEmbeddingContext(options) {
                  createContextOptions = options;
                  events.push('create context');
                  return {
                    async getEmbeddingFor(input) {
                      embeddedInputs.push(input);
                      if (input.startsWith(profile.queryInputPrefix)) {
                        return { vector: createEmbeddingVector(3, 4) };
                      }
                      return input.endsWith('first document')
                        ? { vector: createEmbeddingVector(0, 2) }
                        : { vector: createEmbeddingVector(-5, 0) };
                    },
                    async dispose() {
                      events.push('dispose context');
                    },
                  };
                },
                async dispose() {
                  events.push('dispose model');
                },
              };
            },
            async dispose() {
              events.push('dispose runtime');
            },
          };
        },
      };
    },
  });

  assert.equal(dynamicImportCount, 0);
  assert.equal(artifactVerificationCount, 0);

  const clockValues = [0, 7, 7, 18];
  const measurement = await measureRecallEmbeddingProviderConformance({
    provider,
    profile,
    query: 'find the decision',
    expectedQueryEmbedding: createEmbeddingVector(0.6, 0.8),
    documents: ['first document', 'second document'],
    expectedDocumentEmbeddings: [createEmbeddingVector(0, 1), createEmbeddingVector(-1, 0)],
    monotonicMilliseconds() {
      const value = clockValues.shift();
      assert.notEqual(value, undefined);
      return value ?? 0;
    },
  });
  const tokenizer = await provider.loadConversationTokenizer();

  for (const [text, expectedIds] of frozenTokenizerIds) {
    assert.deepEqual(tokenizer.encodeConversationText(text).ids, expectedIds);
  }
  assert.deepEqual(measurement, {
    queryCount: 1,
    documentCount: 2,
    queryMilliseconds: 7,
    documentMilliseconds: 11,
  });
  assert.deepEqual(embeddedInputs, [
    'task: search result | query: find the decision',
    'title: none | text: first document',
    'title: none | text: second document',
  ]);
  assert.deepEqual(getLlamaOptions, {
    build: 'never',
    gpu: false,
    logLevel: 'error',
    progressLogs: false,
    skipDownload: true,
  });
  assert.deepEqual(loadModelOptions, {
    modelPath: '/models/embeddinggemma/profile/embeddinggemma-300M-Q8_0.gguf',
    gpuLayers: 0,
  });
  assert.deepEqual(createContextOptions, { contextSize: 2_048, threads: 3 });
  assert.equal(dynamicImportCount, 1);
  assert.equal(artifactVerificationCount, 1);
  assert.deepEqual(provider.executionIdentity, {
    adapter: 'node-llama-cpp-embedded-v1',
    backend: 'embedded',
    device: 'cpu',
    nodeLlamaCppVersion: '3.18.1',
    profileId: 'embeddinggemma-300m-q8-0-v1',
  });

  await provider.dispose();
  assert.deepEqual(events.slice(-3), ['dispose context', 'dispose model', 'dispose runtime']);
});

void test('embedded EmbeddingGemma provider rejects an incompatible runtime dimension', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    async verifyModelArtifact() {
      return '/models/model.gguf';
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
                embeddingVectorSize: 384,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext() {
                  throw new Error('context must not load');
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

  await assert.rejects(
    () => provider.embedQuery('query'),
    /Recall embedded EmbeddingGemma model dimension mismatch: expected 768, received 384/u,
  );
});

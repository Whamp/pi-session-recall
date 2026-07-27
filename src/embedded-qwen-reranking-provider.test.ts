import assert from 'node:assert/strict';
import { setTimeout as sleep } from 'node:timers/promises';
import test from 'node:test';

import { EmbeddedInferenceDevicePolicy } from './enums.js';

import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { measureRecallRerankingProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedQwenRerankingModelProfile } from './recall-model-profiles.js';

function applyKnownNodeLlamaCppExtraSigmoid(score: number): number {
  return 1 / (1 + Math.exp(-score));
}

void test('embedded Qwen reranker restores llama.cpp score semantics and passes conformance', async (t) => {
  const profile = createRecommendedQwenRerankingModelProfile();
  const events: string[] = [];
  const rankInputs: Array<{ query: string; documents: string[] }> = [];
  const provider = createEmbeddedQwenRerankingProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    threads: 3,
    async verifyModelArtifact() {
      events.push('verify artifact');
      return '/models/qwen-reranker.gguf';
    },
    async loadNodeLlamaCpp() {
      events.push('dynamic import');
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama(options) {
          assert.equal(options.gpu, false);
          return {
            gpu: false,
            async loadModel(options) {
              assert.deepEqual(options, {
                modelPath: '/models/qwen-reranker.gguf',
                gpuLayers: 0,
              });
              return {
                async createRankingContext(options) {
                  assert.deepEqual(options, { contextSize: 4_096, threads: 3 });
                  return {
                    async rankAll(query, documents) {
                      rankInputs.push({ query, documents: [...documents] });
                      return [0.9, 0.1].map(applyKnownNodeLlamaCppExtraSigmoid);
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
  t.after(() => provider.dispose());

  const measurement = await measureRecallRerankingProviderConformance({
    provider,
    profile,
    query: 'source provenance',
    documents: ['Preserve exact source provenance.', 'The navigation bar is blue.'],
    expectedScores: [0.9, 0.1],
    maximumAbsoluteDifference: 1e-12,
  });

  assert.deepEqual(rankInputs, [
    {
      query: 'source provenance',
      documents: ['Preserve exact source provenance.', 'The navigation bar is blue.'],
    },
  ]);
  assert.equal(measurement.queryCount, 1);
  assert.equal(measurement.documentCount, 2);
  assert.deepEqual(provider.executionIdentity, {
    adapterId: 'node-llama-cpp-qwen-reranking-logit-recovery-v1',
    backend: 'embedded',
    cacheIdentity: 'qwen3-reranker-0.6b-q8-0-v1:node-llama-cpp-qwen-reranking-logit-recovery-v1',
    modelProfileId: 'qwen3-reranker-0.6b-q8-0-v1',
    computeBackend: 'cpu',
    deviceNames: ['CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: '3.18.1',
    parallelism: 1,
    probedComputeBackends: [],
  });
  assert.deepEqual(events.slice(0, 2), ['verify artifact', 'dynamic import']);
});

void test('embedded Qwen reranker reports automatic accelerator selection', async (t) => {
  const profile = createRecommendedQwenRerankingModelProfile();
  let loadModelOptions: Record<string, unknown> | undefined;
  const provider = createEmbeddedQwenRerankingProvider(profile, {
    modelCacheDirectory: '/models',
    async verifyModelArtifact() {
      return '/models/qwen-reranker.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlamaGpuTypes() {
          return ['cuda', 'vulkan'];
        },
        async getLlama(options) {
          assert.equal(options.gpu, 'cuda');
          return {
            gpu: 'cuda',
            async getGpuDeviceNames() {
              return ['NVIDIA Test Device'];
            },
            async loadModel(options) {
              loadModelOptions = options;
              return {
                async createRankingContext() {
                  return {
                    async rankAll() {
                      return [applyKnownNodeLlamaCppExtraSigmoid(0.75)];
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

  await provider.rerankDocuments('query', ['candidate']);

  assert.deepEqual(loadModelOptions, {
    modelPath: '/models/qwen-reranker.gguf',
    gpuLayers: {
      fitContext: { contextSize: 4_096, embeddingContext: true },
      max: 40,
    },
  });
  assert.equal(provider.executionIdentity.computeBackend, 'cuda');
  assert.deepEqual(provider.executionIdentity.deviceNames, ['NVIDIA Test Device']);
  assert.deepEqual(provider.executionIdentity.probedComputeBackends, ['cuda', 'vulkan']);
});

void test('embedded Qwen reranker retries automatic accelerator failure on CPU once', async (t) => {
  const profile = createRecommendedQwenRerankingModelProfile();
  const requestedBackends: unknown[] = [];
  const warnings: string[] = [];
  const provider = createEmbeddedQwenRerankingProvider(profile, {
    modelCacheDirectory: '/models',
    onWarning(warning) {
      warnings.push(warning);
    },
    async verifyModelArtifact() {
      return '/models/qwen-reranker.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlamaGpuTypes() {
          return ['cuda'];
        },
        async getLlama(options) {
          requestedBackends.push(options.gpu);
          if (options.gpu === 'cuda') {
            throw new Error('fixture CUDA initialization failed');
          }
          return {
            gpu: false,
            async loadModel() {
              return {
                async createRankingContext() {
                  return {
                    async rankAll() {
                      return [applyKnownNodeLlamaCppExtraSigmoid(0.75)];
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

  assert.deepEqual(await provider.rerankDocuments('query', ['candidate']), [0.75]);
  assert.deepEqual(requestedBackends, ['cuda', false]);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0] ?? '', /retrying the same profile qwen3-reranker-0\.6b-q8-0-v1 on CPU/u);
  assert.equal(provider.executionIdentity.computeBackend, 'cpu');
  assert.equal(provider.executionIdentity.fallbackFromComputeBackend, 'cuda');
});

void test('embedded Qwen reranker enforces timeout and caller cancellation', async (t) => {
  const profile = createRecommendedQwenRerankingModelProfile();
  const createProvider = (requestTimeoutMilliseconds: number, operationStarted?: () => void) =>
    createEmbeddedQwenRerankingProvider(profile, {
      modelCacheDirectory: '/models',
      device: EmbeddedInferenceDevicePolicy.CPU,
      requestTimeoutMilliseconds,
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
                      async rankAll() {
                        operationStarted?.();
                        await sleep(25);
                        return [applyKnownNodeLlamaCppExtraSigmoid(0.5)];
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

  const timeoutProvider = createProvider(5);
  t.after(() => timeoutProvider.dispose());
  await assert.rejects(
    () => timeoutProvider.rerankDocuments('query', ['candidate']),
    /Recall embedded Qwen reranker request timed out after 5 ms/u,
  );

  const started = Promise.withResolvers<void>();
  const cancellationProvider = createProvider(100, () => started.resolve());
  t.after(() => cancellationProvider.dispose());
  const controller = new AbortController();
  const cancellationReason = new Error('operator cancelled embedded reranking');
  const operation = cancellationProvider.rerankDocuments('query', ['candidate'], controller.signal);
  await started.promise;
  controller.abort(cancellationReason);
  await assert.rejects(
    () => operation,
    (error) => error === cancellationReason,
  );
});

void test('embedded Qwen reranker rejects uncorrected or differently transformed scores', async (t) => {
  const profile = createRecommendedQwenRerankingModelProfile();
  const provider = createEmbeddedQwenRerankingProvider(profile, {
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
                    async rankAll() {
                      return [0.9];
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

  await assert.rejects(
    () => provider.rerankDocuments('query', ['candidate']),
    /Recall embedded Qwen reranker score semantics mismatch at candidate index 0/u,
  );
});

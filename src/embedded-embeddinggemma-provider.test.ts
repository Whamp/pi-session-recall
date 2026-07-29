import assert from 'node:assert/strict';
import test from 'node:test';

import { EmbeddedInferenceDevicePolicy } from './enums.js';

import { createEmbeddedEmbeddingGemmaProvider } from './embedded-embeddinggemma-provider.js';
import { measureRecallEmbeddingProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

function createEmbeddingVector(first: number, second: number): number[] {
  return [first, second, ...Array<number>(766).fill(0)];
}

void test('embedded EmbeddingGemma provider reports automatic accelerator selection and bounded parallelism', async (t) => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const probedIncludes: string[] = [];
  const runtimeOptions: Array<Record<string, unknown>> = [];
  const modelOptions: Array<Record<string, unknown>> = [];
  const contextOptions: Array<Record<string, unknown>> = [];

  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    parallelism: 2,
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlamaGpuTypes(include) {
          probedIncludes.push(include);
          return ['cuda', 'vulkan'];
        },
        async getLlama(options) {
          runtimeOptions.push(options);
          return {
            gpu: 'cuda',
            async getGpuDeviceNames() {
              return ['NVIDIA Test Device'];
            },
            async loadModel(options) {
              modelOptions.push(options);
              return {
                embeddingVectorSize: 768,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext(options) {
                  contextOptions.push(options);
                  return {
                    async getEmbeddingFor() {
                      return { vector: createEmbeddingVector(3, 4) };
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

  await provider.embedQuery('accelerated recall');

  assert.deepEqual(probedIncludes, ['supported']);
  assert.equal(runtimeOptions.length, 1);
  assert.equal(runtimeOptions[0]?.gpu, 'cuda');
  assert.equal(runtimeOptions[0]?.progressLogs, false);
  assert.equal(runtimeOptions[0]?.debug, false);
  assert.equal(typeof runtimeOptions[0]?.logger, 'function');
  assert.deepEqual(modelOptions, [
    {
      modelPath: '/models/model.gguf',
      gpuLayers: {
        fitContext: { contextSize: 2_048, embeddingContext: true },
        max: 32,
      },
    },
  ]);
  assert.deepEqual(contextOptions, [{ contextSize: 2_048 }, { contextSize: 2_048 }]);
  assert.deepEqual(provider.executionIdentity, {
    adapter: 'node-llama-cpp-embedded-v2',
    backend: 'embedded',
    computeBackend: 'cuda',
    deviceNames: ['NVIDIA Test Device'],
    devicePolicy: 'auto',
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: '3.18.1',
    parallelism: 2,
    probedComputeBackends: ['cuda', 'vulkan'],
    profileId: 'embeddinggemma-300m-q8-0-v1',
  });
});

void test('embedded EmbeddingGemma provider rejects unbounded context-pool parallelism', () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();

  assert.throws(
    () =>
      createEmbeddedEmbeddingGemmaProvider(profile, {
        modelCacheDirectory: '/models',
        parallelism: 5,
      }),
    /parallelism invalid: expected an integer from 1 through 4, received 5/u,
  );
});

void test('embedded EmbeddingGemma provider passes deterministic explicit CPU and tokenizer conformance', async () => {
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
    device: EmbeddedInferenceDevicePolicy.CPU,
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
  assert.equal(typeof getLlamaOptions?.logger, 'function');
  assert.deepEqual(
    { ...getLlamaOptions, logger: undefined },
    {
      build: 'never',
      debug: false,
      gpu: false,
      logger: undefined,
      logLevel: 'error',
      progressLogs: false,
      skipDownload: true,
    },
  );
  assert.deepEqual(loadModelOptions, {
    modelPath: '/models/embeddinggemma/profile/embeddinggemma-300M-Q8_0.gguf',
    gpuLayers: 0,
  });
  assert.deepEqual(createContextOptions, { contextSize: 2_048, threads: 3 });
  assert.equal(dynamicImportCount, 1);
  assert.equal(artifactVerificationCount, 1);
  assert.deepEqual(provider.executionIdentity, {
    adapter: 'node-llama-cpp-embedded-v2',
    backend: 'embedded',
    computeBackend: 'cpu',
    deviceNames: ['CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: '3.18.1',
    parallelism: 1,
    probedComputeBackends: [],
    profileId: 'embeddinggemma-300m-q8-0-v1',
  });

  await provider.dispose();
  assert.deepEqual(events.slice(-3), ['dispose context', 'dispose model', 'dispose runtime']);
});

void test('embedded EmbeddingGemma disposal releases later resources after context disposal fails', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const events: string[] = [];
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama() {
          return {
            gpu: false as const,
            async loadModel() {
              return {
                embeddingVectorSize: 768,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor() {
                      return { vector: [1, ...Array<number>(767).fill(0)] };
                    },
                    async dispose() {
                      events.push('context');
                      throw new Error('fixture context disposal failed');
                    },
                  };
                },
                async dispose() {
                  events.push('model');
                },
              };
            },
            async dispose() {
              events.push('runtime');
            },
          };
        },
      };
    },
  });

  await provider.embedQuery('load resources');
  await assert.rejects(() => provider.dispose(), AggregateError);
  assert.deepEqual(events, ['context', 'model', 'runtime']);
});

void test('device enumeration failure disposes initialized EmbeddingGemma resources', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const events: string[] = [];
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.METAL,
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama() {
          return {
            gpu: 'metal' as const,
            async getGpuDeviceNames() {
              throw new Error('fixture device enumeration failed');
            },
            async loadModel() {
              return {
                embeddingVectorSize: 768,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor() {
                      return { vector: [1, ...Array<number>(767).fill(0)] };
                    },
                    async dispose() {
                      events.push('context');
                    },
                  };
                },
                async dispose() {
                  events.push('model');
                },
              };
            },
            async dispose() {
              events.push('runtime');
            },
          };
        },
      };
    },
  });

  await assert.rejects(() => provider.embedQuery('load resources'), /device enumeration failed/u);
  assert.deepEqual(events, ['context', 'model', 'runtime']);
});

void test('embedded EmbeddingGemma provider shares one model load across concurrent requests', async (t) => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  let runtimeLoadCount = 0;
  let modelLoadCount = 0;
  let contextCreationCount = 0;
  let activeContextCount = 0;
  let maximumActiveContextCount = 0;
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    parallelism: 2,
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama() {
          runtimeLoadCount += 1;
          return {
            gpu: false,
            async loadModel() {
              modelLoadCount += 1;
              return {
                embeddingVectorSize: 768,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext() {
                  contextCreationCount += 1;
                  return {
                    async getEmbeddingFor() {
                      activeContextCount += 1;
                      maximumActiveContextCount = Math.max(
                        maximumActiveContextCount,
                        activeContextCount,
                      );
                      await new Promise<void>((resolve) => {
                        setImmediate(resolve);
                      });
                      activeContextCount -= 1;
                      return { vector: createEmbeddingVector(3, 4) };
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

  const embeddings = await Promise.all([
    provider.embedQuery('one'),
    provider.embedQuery('two'),
    provider.embedQuery('three'),
    provider.embedQuery('four'),
  ]);

  assert.equal(runtimeLoadCount, 1);
  assert.equal(modelLoadCount, 1);
  assert.equal(contextCreationCount, 2);
  assert.equal(maximumActiveContextCount, 2);
  assert.deepEqual(
    embeddings,
    Array.from({ length: 4 }, () => createEmbeddingVector(0.6, 0.8)),
  );
});

void test('embedded EmbeddingGemma provider fails a broken explicit accelerator without fallback', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const requestedComputeBackends: unknown[] = [];
  const warnings: string[] = [];
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CUDA,
    onWarning(warning) {
      warnings.push(warning);
    },
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama(options) {
          requestedComputeBackends.push(options.gpu);
          throw new Error('explicit CUDA fixture failed');
        },
      };
    },
  });

  await assert.rejects(
    () => provider.embedQuery('must not change devices'),
    /explicit CUDA fixture failed/u,
  );
  assert.deepEqual(requestedComputeBackends, ['cuda']);
  assert.deepEqual(warnings, []);
  assert.equal(provider.executionIdentity.computeBackend, 'pending');
  assert.equal(provider.executionIdentity.profileId, profile.profileId);
});

void test('embedded EmbeddingGemma provider disposes idle resources and reloads on demand', async (t) => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const firstIdleDisposal = Promise.withResolvers<void>();
  let runtimeLoadCount = 0;
  let contextDisposalCount = 0;
  let modelDisposalCount = 0;
  let runtimeDisposalCount = 0;
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    idleTimeoutMilliseconds: 5,
    async verifyModelArtifact() {
      return '/models/model.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        async getLlama() {
          runtimeLoadCount += 1;
          return {
            gpu: false,
            async loadModel() {
              return {
                embeddingVectorSize: 768,
                tokenize() {
                  return [];
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor() {
                      return { vector: createEmbeddingVector(3, 4) };
                    },
                    async dispose() {
                      contextDisposalCount += 1;
                    },
                  };
                },
                async dispose() {
                  modelDisposalCount += 1;
                },
              };
            },
            async dispose() {
              runtimeDisposalCount += 1;
              firstIdleDisposal.resolve();
            },
          };
        },
      };
    },
  });
  t.after(() => provider.dispose());

  await provider.embedQuery('first load');
  await Promise.race([
    firstIdleDisposal.promise,
    new Promise<never>((resolve, reject) => {
      void resolve;
      setTimeout(() => reject(new Error('idle resources were not disposed')), 500);
    }),
  ]);

  assert.equal(contextDisposalCount, 1);
  assert.equal(modelDisposalCount, 1);
  assert.equal(runtimeDisposalCount, 1);

  await provider.embedQuery('reload after idle');
  assert.equal(runtimeLoadCount, 2);
});

void test('embedded EmbeddingGemma provider keeps a loaded tokenizer valid until disposal', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  let modelDisposed = false;
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    idleTimeoutMilliseconds: 5,
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
                embeddingVectorSize: 768,
                tokenize() {
                  if (modelDisposed) {
                    throw new Error('tokenizer model disposed');
                  }
                  return [17, 19, 23];
                },
                async createEmbeddingContext() {
                  return {
                    async getEmbeddingFor() {
                      return { vector: createEmbeddingVector(3, 4) };
                    },
                    async dispose() {},
                  };
                },
                async dispose() {
                  modelDisposed = true;
                },
              };
            },
            async dispose() {},
          };
        },
      };
    },
  });
  const tokenizer = await provider.loadConversationTokenizer();

  await new Promise<void>((resolve) => {
    setTimeout(resolve, 20);
  });

  assert.deepEqual(tokenizer.encodeConversationText('still valid').ids, [17, 19, 23]);
  assert.equal(modelDisposed, false);
  await provider.dispose();
  assert.equal(modelDisposed, true);
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

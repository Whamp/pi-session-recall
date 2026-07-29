import assert from 'node:assert/strict';
import test from 'node:test';

import { acquireSharedEmbeddedLlamaRuntime } from './acquireSharedEmbeddedLlamaRuntime.js';
import { createEmbeddedEmbeddingGemmaProvider } from './embedded-embeddinggemma-provider.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { EmbeddedInferenceDevicePolicy } from './enums.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

interface FixtureRuntime {
  id: string;
}

function isFixtureRuntime(value: unknown): value is FixtureRuntime {
  return typeof value === 'object' && value !== null && Reflect.get(value, 'id') === 'runtime-1';
}

void test('EmbeddingGemma and Qwen providers share one runtime without premature disposal', async () => {
  const runtimePoolIdentity = {};
  let runtimeLoadCount = 0;
  let runtimeDisposeCount = 0;
  const sharedRuntime = {
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
            async dispose() {},
          };
        },
        async createRankingContext() {
          return {
            async rankAll() {
              return [1 / (1 + Math.exp(-0.9))];
            },
            async dispose() {},
          };
        },
        async dispose() {},
      };
    },
    async dispose() {
      runtimeDisposeCount += 1;
    },
  };
  const getLlama = async () => {
    runtimeLoadCount += 1;
    await new Promise<void>((resolve) => {
      setImmediate(resolve);
    });
    return sharedRuntime;
  };
  const embeddingProvider = createEmbeddedEmbeddingGemmaProvider(
    createRecommendedEmbeddingGemmaModelProfile(),
    {
      modelCacheDirectory: '/models',
      device: EmbeddedInferenceDevicePolicy.CPU,
      async verifyModelArtifact() {
        return '/models/embedding.gguf';
      },
      async loadNodeLlamaCpp() {
        return {
          version: '3.18.1',
          runtimePoolIdentity,
          LlamaLogLevel: { error: 'error' },
          getLlama,
        };
      },
    },
  );
  const rerankingProvider = createEmbeddedQwenRerankingProvider(
    createRecommendedQwenRerankingModelProfile(),
    {
      modelCacheDirectory: '/models',
      device: EmbeddedInferenceDevicePolicy.CPU,
      async verifyModelArtifact() {
        return '/models/reranking.gguf';
      },
      async loadNodeLlamaCpp() {
        return {
          version: '3.18.1',
          runtimePoolIdentity,
          LlamaLogLevel: { error: 'error' },
          getLlama,
        };
      },
    },
  );

  await Promise.all([
    embeddingProvider.embedQuery('shared runtime'),
    rerankingProvider.rerankDocuments('shared runtime', ['candidate']),
  ]);
  assert.equal(runtimeLoadCount, 1);
  await embeddingProvider.dispose();
  assert.equal(runtimeDisposeCount, 0);
  await rerankingProvider.dispose();
  assert.equal(runtimeDisposeCount, 1);
});

void test('embedded capabilities share one in-flight runtime until every lease releases', async () => {
  const runtimePoolIdentity = {};
  let loadCount = 0;
  let disposeCount = 0;
  const loadRuntime = async () => {
    loadCount += 1;
    await new Promise<void>((resolve) => {
      setImmediate(resolve);
    });
    return { id: 'runtime-1' };
  };
  const disposeRuntime = async () => {
    disposeCount += 1;
  };

  const [embeddingLease, rerankingLease] = await Promise.all([
    acquireSharedEmbeddedLlamaRuntime(
      runtimePoolIdentity,
      'node-llama-cpp-3.18.1:cpu',
      loadRuntime,
      isFixtureRuntime,
      disposeRuntime,
    ),
    acquireSharedEmbeddedLlamaRuntime(
      runtimePoolIdentity,
      'node-llama-cpp-3.18.1:cpu',
      loadRuntime,
      isFixtureRuntime,
      disposeRuntime,
    ),
  ]);

  assert.equal(loadCount, 1);
  assert.equal(embeddingLease.runtime, rerankingLease.runtime);
  await embeddingLease.release();
  assert.equal(disposeCount, 0);
  await rerankingLease.release();
  assert.equal(disposeCount, 1);
});

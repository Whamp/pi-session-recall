import assert from 'node:assert/strict';
import test from 'node:test';

import { EmbeddedInferenceDevicePolicy } from './enums.js';

import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import { measureRecallQueryPlanningProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

interface InvalidEmbeddedQmdOutputCase {
  name: string;
  output: string;
}

const INVALID_EMBEDDED_QMD_OUTPUT_CASES: readonly InvalidEmbeddedQmdOutputCase[] = [
  {
    name: 'vector before the first lexical query',
    output: 'vec: Copper Finch semantic evidence\nlex: Copper Finch evidence',
  },
  {
    name: 'lexical query after a vector query',
    output:
      'lex: Copper Finch evidence\nvec: Copper Finch semantic evidence\nlex: retained Copper Finch records',
  },
  {
    name: 'non-final hypothetical-answer query',
    output:
      'lex: Copper Finch evidence\nhyde: Copper Finch appears in retained evidence.\nvec: Copper Finch semantic evidence',
  },
  {
    name: 'entry after a hypothetical-answer query',
    output:
      'lex: Copper Finch evidence\nvec: Copper Finch semantic evidence\nhyde: Copper Finch appears in retained evidence.\nvec: where Copper Finch was retained',
  },
  {
    name: '513-code-point query content',
    output: `lex: ${'x'.repeat(513)}\nvec: Copper Finch semantic evidence`,
  },
];

void test('embedded QMD query planner passes shared conformance with profile grammar and intent', async (t) => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const calls: Array<{ prompt: string; options: Record<string, unknown> }> = [];
  const disposals: string[] = [];
  let grammarText = '';
  const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    requestTimeoutMilliseconds: 4_321,
    async verifyModelArtifact() {
      return '/models/qmd-query-expansion-1.7B-q4_k_m.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        createChatSession() {
          return {
            async prompt(prompt: string, options: Record<string, unknown>) {
              calls.push({ prompt, options });
              return [
                'lex: Copper Finch recovery evidence',
                'vec: how Finch identifies original conversation evidence',
                'hyde: Copper Finch connects recalled recovery evidence to its original session.',
              ].join('\n');
            },
          };
        },
        async getLlama() {
          return {
            gpu: false,
            async createGrammar(options: { grammar: string }) {
              grammarText = options.grammar;
              return { grammar: options.grammar };
            },
            async loadModel(options: { modelPath: string; gpuLayers: unknown }) {
              assert.deepEqual(options, {
                modelPath: '/models/qmd-query-expansion-1.7B-q4_k_m.gguf',
                gpuLayers: 0,
              });
              return {
                async createContext(options: { contextSize: number }) {
                  assert.deepEqual(options, { contextSize: 2_048 });
                  return {
                    getSequence() {
                      return { id: 'fixture-sequence' };
                    },
                    async dispose() {
                      disposals.push('context');
                    },
                  };
                },
                async dispose() {
                  disposals.push('model');
                },
              };
            },
            async dispose() {
              disposals.push('runtime');
            },
          };
        },
      };
    },
  });
  t.after(() => provider.dispose());
  const pendingAdapterConfigurationIdentity =
    provider.executionIdentity.adapterConfigurationIdentity;

  const measurement = await measureRecallQueryPlanningProviderConformance({
    provider,
    profile,
    query: profile.conformanceCanary.query,
    recallIntent: profile.conformanceCanary.recallIntent,
    protectedTerms: profile.conformanceCanary.protectedTerms,
  });

  assert.equal(grammarText, profile.grammar);
  assert.equal(calls.length, 1);
  assert.equal(
    calls[0]?.prompt,
    '/no_think Expand this search query: Copper Finch\nQuery intent: Find Pi conversation evidence about the exact Copper Finch recovery entity.',
  );
  assert.deepEqual(calls[0]?.options, {
    grammar: { grammar: profile.grammar },
    maxTokens: 600,
    temperature: 0.7,
    topK: 20,
    topP: 0.8,
    repeatPenalty: { lastTokens: 64, presencePenalty: 0.5 },
    signal: calls[0]?.options.signal,
  });
  assert.ok(calls[0]?.options.signal instanceof AbortSignal);
  const { adapterConfigurationIdentity, cacheIdentity, ...executionIdentity } =
    provider.executionIdentity;
  assert.match(
    adapterConfigurationIdentity,
    /^node-llama-cpp-qmd-query-planning-config-v1:[a-f0-9]{64}$/u,
  );
  assert.notEqual(adapterConfigurationIdentity, pendingAdapterConfigurationIdentity);
  assert.match(cacheIdentity, /^recall-query-planning-execution-v1:[a-f0-9]{64}$/u);
  assert.deepEqual(executionIdentity, {
    adapterId: 'node-llama-cpp-qmd-query-planning-v1',
    adapterVersion: 'node-llama-cpp-qmd-query-planning-v1',
    backend: 'embedded',
    modelProfileId: profile.profileId,
    modelProfileIdentity: provider.executionIdentity.modelProfileIdentity,
    promptPolicy: profile.promptPolicy,
    grammarVersion: profile.grammarVersion,
    requestTimeoutMilliseconds: 4_321,
    computeBackend: 'cpu',
    deviceNames: ['CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    contextSize: 2_048,
    threads: null,
    nodeLlamaCppVersion: '3.18.1',
    physicalDeviceIdentity: ['cpu'],
    probedComputeBackends: [],
  });
  assert.deepEqual(measurement, {
    plannedQueryCount: 3,
    lexQueryCount: 1,
    vecQueryCount: 1,
    hydeQueryCount: 1,
    planningMilliseconds: measurement.planningMilliseconds,
  });
  assert.ok(measurement.planningMilliseconds >= 0);
  assert.deepEqual(disposals, ['context']);
});

void test('embedded QMD query planner rejects every invalid generated ordering and length', async (t) => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  let generatedOutput = '';
  const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    async verifyModelArtifact() {
      return '/models/qmd-query-expansion-1.7B-q4_k_m.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        createChatSession() {
          return {
            async prompt() {
              return generatedOutput;
            },
          };
        },
        async getLlama() {
          return {
            gpu: false,
            async createGrammar() {
              return {};
            },
            async loadModel() {
              return {
                async createContext() {
                  return { getSequence: () => ({}), async dispose() {} };
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

  for (const invalidCase of INVALID_EMBEDDED_QMD_OUTPUT_CASES) {
    generatedOutput = invalidCase.output;
    await assert.rejects(
      () => provider.planRecallQuery({ query: profile.conformanceCanary.query }),
      /Recall query planning output/u,
      invalidCase.name,
    );
  }
});

void test('embedded QMD query planner retries automatic accelerator failure on CPU once', async (t) => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const requestedBackends: unknown[] = [];
  const warnings: string[] = [];
  const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
    modelCacheDirectory: '/models',
    onWarning(warning) {
      warnings.push(warning);
    },
    async verifyModelArtifact() {
      return '/models/qmd-query-expansion-1.7B-q4_k_m.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        createChatSession() {
          return {
            async prompt() {
              return 'lex: Copper Finch records\nvec: Copper Finch evidence';
            },
          };
        },
        async getLlamaGpuTypes() {
          return ['metal'] as const;
        },
        async getLlama(options: Record<string, unknown>) {
          requestedBackends.push(options.gpu);
          if (options.gpu === 'metal') {
            throw new Error('fixture Metal initialization failed');
          }
          return {
            gpu: false,
            async createGrammar() {
              return {};
            },
            async loadModel() {
              return {
                async createContext() {
                  return { getSequence: () => ({}), async dispose() {} };
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
  const pendingAdapterConfigurationIdentity =
    provider.executionIdentity.adapterConfigurationIdentity;

  await measureRecallQueryPlanningProviderConformance({
    provider,
    profile,
    query: profile.conformanceCanary.query,
    recallIntent: profile.conformanceCanary.recallIntent,
    protectedTerms: profile.conformanceCanary.protectedTerms,
  });

  assert.deepEqual(requestedBackends, ['metal', false]);
  assert.equal(warnings.length, 1);
  assert.match(
    warnings[0] ?? '',
    /retrying the same profile qmd-query-expansion-1.7b-q4-k-m-v1 on CPU/u,
  );
  assert.equal(provider.executionIdentity.computeBackend, 'cpu');
  assert.equal(provider.executionIdentity.fallbackFromComputeBackend, 'metal');
  assert.notEqual(
    provider.executionIdentity.adapterConfigurationIdentity,
    pendingAdapterConfigurationIdentity,
  );
  assert.deepEqual(provider.executionIdentity.probedComputeBackends, ['metal']);
});

void test('embedded QMD query planner enforces timeout and caller cancellation', async (t) => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  function createPendingProvider(requestTimeoutMilliseconds: number) {
    return createEmbeddedQmdQueryPlanningProvider(profile, {
      modelCacheDirectory: '/models',
      device: EmbeddedInferenceDevicePolicy.CPU,
      requestTimeoutMilliseconds,
      async verifyModelArtifact() {
        return '/models/qmd-query-expansion-1.7B-q4_k_m.gguf';
      },
      async loadNodeLlamaCpp() {
        return {
          version: '3.18.1',
          LlamaLogLevel: { error: 'error' },
          createChatSession() {
            return {
              async prompt() {
                return new Promise<string>(() => {});
              },
            };
          },
          async getLlama() {
            return {
              gpu: false,
              async createGrammar() {
                return {};
              },
              async loadModel() {
                return {
                  async createContext() {
                    return { getSequence: () => ({}), async dispose() {} };
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
  }
  const timeoutProvider = createPendingProvider(5);
  t.after(() => timeoutProvider.dispose());
  await assert.rejects(
    () =>
      measureRecallQueryPlanningProviderConformance({
        provider: timeoutProvider,
        profile,
        query: profile.conformanceCanary.query,
        recallIntent: profile.conformanceCanary.recallIntent,
        protectedTerms: profile.conformanceCanary.protectedTerms,
      }),
    /Recall embedded QMD query planner request timed out after 5 ms/u,
  );

  const cancellationProvider = createPendingProvider(1_000);
  t.after(() => cancellationProvider.dispose());
  const cancellation = new AbortController();
  const cancellationReason = new Error('operator cancelled embedded planner verification');
  const cancelled = measureRecallQueryPlanningProviderConformance({
    provider: cancellationProvider,
    profile,
    query: profile.conformanceCanary.query,
    recallIntent: profile.conformanceCanary.recallIntent,
    protectedTerms: profile.conformanceCanary.protectedTerms,
    signal: cancellation.signal,
  });
  cancellation.abort(cancellationReason);
  await assert.rejects(() => cancelled, cancellationReason);
});

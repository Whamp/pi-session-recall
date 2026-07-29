import assert from 'node:assert/strict';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';

import { EmbeddedInferenceDevicePolicy } from './enums.js';

import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import { measureRecallQueryPlanningProviderConformance } from './recall-inference-conformance.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

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
                'hyde: Source provenance connects recalled evidence to its original session.',
                'lex: source provenance evidence',
                'vec: how source provenance identifies original conversation evidence',
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
    '/no_think Expand this search query: source provenance\nQuery intent: Find Pi conversation evidence about retained source provenance.',
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
  assert.deepEqual(provider.executionIdentity, {
    adapterId: 'node-llama-cpp-qmd-query-planning-v1',
    backend: 'embedded',
    cacheIdentity:
      'qmd-query-expansion-1.7b-q4-k-m-v1:node-llama-cpp-qmd-query-planning-v1:qmd-query-expansion-no-think-v1:qmd-typed-query-plan-v1',
    modelProfileId: profile.profileId,
    promptPolicy: profile.promptPolicy,
    grammarVersion: profile.grammarVersion,
    requestTimeoutMilliseconds: 4_321,
    computeBackend: 'cpu',
    deviceNames: ['CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: '3.18.1',
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

void test('embedded QMD disposal releases the runtime after model disposal fails', async () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const events: string[] = [];
  const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    async verifyModelArtifact() {
      return '/models/qmd-query-expansion.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        createChatSession() {
          return {
            async prompt() {
              return 'lex: fixture query\nvec: fixture semantic query';
            },
          };
        },
        async getLlama() {
          return {
            gpu: false as const,
            async createGrammar() {
              return {};
            },
            async loadModel() {
              return {
                async createContext() {
                  return { getSequence: () => ({}), async dispose() {} };
                },
                async dispose() {
                  events.push('model');
                  throw new Error('fixture model disposal failed');
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

  await provider.planRecallQuery({ query: 'load resources' });
  await assert.rejects(() => provider.dispose(), AggregateError);
  assert.deepEqual(events, ['model', 'runtime']);
});

void test('QMD context disposal failure preserves the operation error and schedules idle cleanup', async () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const events: string[] = [];
  const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
    modelCacheDirectory: '/models',
    device: EmbeddedInferenceDevicePolicy.CPU,
    idleTimeoutMilliseconds: 5,
    async verifyModelArtifact() {
      return '/models/qmd-query-expansion.gguf';
    },
    async loadNodeLlamaCpp() {
      return {
        version: '3.18.1',
        LlamaLogLevel: { error: 'error' },
        createChatSession() {
          return {
            async prompt() {
              throw new Error('fixture prompt failed');
            },
          };
        },
        async getLlama() {
          return {
            gpu: false as const,
            async createGrammar() {
              return {};
            },
            async loadModel() {
              return {
                async createContext() {
                  return {
                    getSequence: () => ({}),
                    async dispose() {
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

  await assert.rejects(
    () => provider.planRecallQuery({ query: 'failing operation' }),
    (error: unknown) => {
      assert.ok(error instanceof AggregateError);
      assert.deepEqual(
        error.errors.map((item) => (item instanceof Error ? item.message : String(item))),
        ['fixture prompt failed', 'fixture context disposal failed'],
      );
      return true;
    },
  );
  await sleep(20);
  assert.deepEqual(events, ['model', 'runtime']);
  await provider.dispose();
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
              return 'lex: source provenance records\nvec: source provenance evidence';
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

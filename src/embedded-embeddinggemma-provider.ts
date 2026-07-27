import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallTokenizerManifestIdentity } from './recall-index-manifest.js';
import { createRecallModelArtifactCache } from './recall-model-artifact-cache.js';
import type { RecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

/** Exact node-llama-cpp release adopted from QMD 2.6.3 for embedded inference. */
export const EMBEDDED_NODE_LLAMA_CPP_VERSION = '3.18.1';

/** Maximum context-pool parallelism accepted without relying on aggregate reported GPU memory. */
export const EMBEDDED_INFERENCE_MAX_PARALLELISM = 4;

/** Conservative upper bound for EmbeddingGemma layers considered for GPU offload. */
export const EMBEDDING_GEMMA_MAX_GPU_LAYERS = 32;

type EmbeddedInferenceDevicePolicy = 'auto' | 'cpu' | 'metal' | 'cuda' | 'vulkan';

type EmbeddedInferenceComputeBackend = 'cpu' | 'metal' | 'cuda' | 'vulkan';

type NodeLlamaCppGpuBackend = Exclude<EmbeddedInferenceComputeBackend, 'cpu'> | false;

interface EmbeddingGemmaLlamaEmbeddingContext {
  getEmbeddingFor(input: string): Promise<{ vector: readonly number[] }>;
  dispose(): Promise<void>;
}

interface EmbeddingGemmaLlamaModel {
  readonly embeddingVectorSize: number;
  tokenize(text: string, specialTokens?: boolean): readonly number[];
  createEmbeddingContext(options: {
    contextSize: number;
    threads?: number;
  }): Promise<EmbeddingGemmaLlamaEmbeddingContext>;
  dispose(): Promise<void>;
}

interface EmbeddingGemmaLlamaRuntime {
  readonly gpu: NodeLlamaCppGpuBackend;
  getGpuDeviceNames?(): Promise<string[]>;
  loadModel(options: {
    modelPath: string;
    gpuLayers:
      | number
      | {
          max: number;
          fitContext: { contextSize: number; embeddingContext: true };
        };
  }): Promise<EmbeddingGemmaLlamaModel>;
  dispose(): Promise<void>;
}

/** Minimal dynamically loaded node-llama-cpp surface required by embedded EmbeddingGemma. */
export interface EmbeddingGemmaNodeLlamaCppModule {
  version: string;
  LlamaLogLevel: { error: unknown };
  getLlamaGpuTypes?(include: 'supported'): Promise<readonly NodeLlamaCppGpuBackend[]>;
  getLlama(options: Record<string, unknown>): Promise<EmbeddingGemmaLlamaRuntime>;
}

/** Lazy embedded runtime settings and injectable native boundaries for deterministic conformance. */
export interface EmbeddedEmbeddingGemmaProviderOptions {
  modelCacheDirectory: string;
  contextSize?: number;
  threads?: number;
  device?: EmbeddedInferenceDevicePolicy;
  parallelism?: number;
  idleTimeoutMilliseconds?: number;
  onWarning?: (warning: string) => void;
  verifyModelArtifact?: () => Promise<string>;
  loadNodeLlamaCpp?: () => Promise<EmbeddingGemmaNodeLlamaCppModule>;
}

/** Search-policy evidence identifying one selected embedded EmbeddingGemma execution adapter. */
export interface EmbeddedEmbeddingGemmaExecutionIdentity {
  adapter: 'node-llama-cpp-embedded-v2';
  backend: 'embedded';
  computeBackend: EmbeddedInferenceComputeBackend | 'pending';
  deviceNames: readonly string[];
  devicePolicy: EmbeddedInferenceDevicePolicy;
  fallbackFromComputeBackend: Exclude<EmbeddedInferenceComputeBackend, 'cpu'> | null;
  nodeLlamaCppVersion: '3.18.1';
  parallelism: number;
  probedComputeBackends: readonly Exclude<EmbeddedInferenceComputeBackend, 'cpu'>[];
  profileId: RecommendedEmbeddingGemmaModelProfile['profileId'];
}

/** Embedded query/document provider plus its exact tokenizer and explicit native disposal. */
export interface EmbeddedEmbeddingGemmaProvider extends RecallEmbeddingProvider {
  readonly executionIdentity: Readonly<EmbeddedEmbeddingGemmaExecutionIdentity>;
  loadConversationTokenizer(): Promise<ConversationTextTokenizer>;
  dispose(): Promise<void>;
}

interface LoadedEmbeddingGemmaResources {
  runtime: EmbeddingGemmaLlamaRuntime;
  model: EmbeddingGemmaLlamaModel;
  contexts: readonly EmbeddingGemmaLlamaEmbeddingContext[];
  contextOperations: Promise<void>[];
  nextContextIndex: number;
}

async function loadInstalledNodeLlamaCpp(): Promise<EmbeddingGemmaNodeLlamaCppModule> {
  try {
    const loaded = await import('node-llama-cpp');
    return {
      version: EMBEDDED_NODE_LLAMA_CPP_VERSION,
      LlamaLogLevel: loaded.LlamaLogLevel,
      getLlamaGpuTypes: loaded.getLlamaGpuTypes,
      getLlama: loaded.getLlama,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall embedded EmbeddingGemma runtime unavailable: install optional node-llama-cpp@${EMBEDDED_NODE_LLAMA_CPP_VERSION}; ${message}`,
      { cause: error },
    );
  }
}

function throwIfEmbeddingAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error('Recall embedded EmbeddingGemma operation cancelled');
  }
}

function normalizeEmbeddingGemmaVector(
  vector: readonly number[],
  expectedDimensions: number,
): number[] {
  if (vector.length !== expectedDimensions) {
    throw new Error(
      `Recall embedded EmbeddingGemma vector dimension mismatch: expected ${expectedDimensions}, received ${vector.length}`,
    );
  }
  let squaredNorm = 0;
  for (const [index, value] of vector.entries()) {
    if (!Number.isFinite(value)) {
      throw new Error(
        `Recall embedded EmbeddingGemma vector invalid at dimension ${index}: expected a finite number`,
      );
    }
    squaredNorm += value * value;
  }
  if (squaredNorm === 0) {
    throw new Error('Recall embedded EmbeddingGemma vector invalid: norm must be positive');
  }
  const norm = Math.sqrt(squaredNorm);
  return vector.map((value) => value / norm);
}

function readEmbeddingErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Projects the GGUF-contained tokenizer into reproducible index-manifest identity. */
export function createEmbeddingGemmaTokenizerManifestIdentity(
  profile: RecommendedEmbeddingGemmaModelProfile,
): RecallTokenizerManifestIdentity {
  return {
    model: profile.tokenizer.model,
    revision: profile.source.revision,
    library: { name: 'node-llama-cpp', version: EMBEDDED_NODE_LLAMA_CPP_VERSION },
    encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
    assets: [
      {
        fileName: profile.source.artifact,
        sha256: profile.tokenizer.artifactSha256,
      },
    ],
  };
}

/** Creates a lazy accelerated EmbeddingGemma provider with explicit, bounded execution policy. */
export function createEmbeddedEmbeddingGemmaProvider(
  profile: RecommendedEmbeddingGemmaModelProfile,
  options: EmbeddedEmbeddingGemmaProviderOptions,
): EmbeddedEmbeddingGemmaProvider {
  const contextSize = options.contextSize ?? 2_048;
  if (!Number.isInteger(contextSize) || contextSize < 1) {
    throw new Error(
      `Recall embedded EmbeddingGemma context size invalid: expected a positive integer, received ${contextSize}`,
    );
  }
  if (
    options.threads !== undefined &&
    (!Number.isInteger(options.threads) || options.threads < 1)
  ) {
    throw new Error(
      `Recall embedded EmbeddingGemma thread count invalid: expected a positive integer, received ${options.threads}`,
    );
  }
  const parallelism = options.parallelism ?? 1;
  if (
    !Number.isInteger(parallelism) ||
    parallelism < 1 ||
    parallelism > EMBEDDED_INFERENCE_MAX_PARALLELISM
  ) {
    throw new Error(
      `Recall embedded EmbeddingGemma parallelism invalid: expected an integer from 1 through ${EMBEDDED_INFERENCE_MAX_PARALLELISM}, received ${parallelism}`,
    );
  }
  const idleTimeoutMilliseconds = options.idleTimeoutMilliseconds ?? 300_000;
  if (!Number.isInteger(idleTimeoutMilliseconds) || idleTimeoutMilliseconds < 0) {
    throw new Error(
      `Recall embedded EmbeddingGemma idle timeout invalid: expected a nonnegative integer in milliseconds, received ${idleTimeoutMilliseconds}`,
    );
  }
  const devicePolicy = options.device ?? 'auto';

  const verifyModelArtifact =
    options.verifyModelArtifact ??
    (async () => {
      const cache = createRecallModelArtifactCache({
        cacheDirectory: options.modelCacheDirectory,
        profile,
      });
      const status = await cache.verifyArtifact();
      if (status.state !== 'valid') {
        throw new Error(
          `Recall embedded EmbeddingGemma artifact unavailable: ${status.issue ?? `state is ${status.state}`}; ${status.repair ?? 'run model:embeddinggemma doctor'}`,
        );
      }
      return status.artifactPath;
    });
  const loadNodeLlamaCpp = options.loadNodeLlamaCpp ?? loadInstalledNodeLlamaCpp;
  const writeWarning =
    options.onWarning ?? ((warning: string) => process.stderr.write(`${warning}\n`));
  let resources: LoadedEmbeddingGemmaResources | undefined;
  let tokenizerResources: LoadedEmbeddingGemmaResources | undefined;
  let resourcesLoadPromise: Promise<LoadedEmbeddingGemmaResources> | undefined;
  let resourcesDisposalPromise: Promise<void> | undefined;
  let idleDisposalTimer: ReturnType<typeof setTimeout> | undefined;
  let activeOperationCount = 0;
  let disposed = false;
  let cpuFallbackWarningEmitted = false;
  let executionIdentity: Readonly<EmbeddedEmbeddingGemmaExecutionIdentity> = Object.freeze({
    adapter: 'node-llama-cpp-embedded-v2',
    backend: 'embedded',
    computeBackend: 'pending',
    deviceNames: [],
    devicePolicy,
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
    parallelism,
    probedComputeBackends: [],
    profileId: profile.profileId,
  });

  async function disposeLoadedResources(loaded: LoadedEmbeddingGemmaResources): Promise<void> {
    await Promise.all(loaded.contextOperations);
    for (const context of loaded.contexts) {
      await context.dispose();
    }
    await loaded.model.dispose();
    await loaded.runtime.dispose();
  }

  function clearIdleResourceDisposal(): void {
    if (idleDisposalTimer) {
      clearTimeout(idleDisposalTimer);
      idleDisposalTimer = undefined;
    }
  }

  function scheduleIdleResourceDisposal(loaded: LoadedEmbeddingGemmaResources): void {
    if (
      disposed ||
      idleTimeoutMilliseconds === 0 ||
      activeOperationCount !== 0 ||
      tokenizerResources === loaded
    ) {
      return;
    }
    clearIdleResourceDisposal();
    idleDisposalTimer = setTimeout(() => {
      idleDisposalTimer = undefined;
      if (disposed || activeOperationCount !== 0 || resources !== loaded) {
        return;
      }
      resources = undefined;
      resourcesLoadPromise = undefined;
      const disposal = disposeLoadedResources(loaded);
      resourcesDisposalPromise = disposal;
      void disposal
        .catch((error: unknown) => {
          writeWarning(
            `Recall embedded EmbeddingGemma idle disposal failed: ${readEmbeddingErrorMessage(error)}`,
          );
        })
        .finally(() => {
          if (resourcesDisposalPromise === disposal) {
            resourcesDisposalPromise = undefined;
          }
        });
    }, idleTimeoutMilliseconds);
    idleDisposalTimer.unref();
  }

  async function loadResourcesForBackend(
    nodeLlamaCpp: EmbeddingGemmaNodeLlamaCppModule,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ): Promise<LoadedEmbeddingGemmaResources> {
    let runtime: EmbeddingGemmaLlamaRuntime | undefined;
    let model: EmbeddingGemmaLlamaModel | undefined;
    const contexts: EmbeddingGemmaLlamaEmbeddingContext[] = [];
    try {
      const requestedGpu = computeBackend === 'cpu' ? false : computeBackend;
      runtime = await nodeLlamaCpp.getLlama({
        build: 'never',
        debug: false,
        gpu: requestedGpu,
        logger(_level: unknown, message: string) {
          process.stderr.write(`[Recall native inference] ${message.replace(/\n+$/u, '')}\n`);
        },
        logLevel: nodeLlamaCpp.LlamaLogLevel.error,
        progressLogs: false,
        skipDownload: true,
      });
      if (runtime.gpu !== requestedGpu) {
        throw new Error(
          `Recall embedded EmbeddingGemma compute backend mismatch: requested ${computeBackend}, received ${runtime.gpu === false ? 'cpu' : runtime.gpu}`,
        );
      }
      model = await runtime.loadModel({
        modelPath,
        gpuLayers:
          computeBackend === 'cpu'
            ? 0
            : {
                fitContext: { contextSize, embeddingContext: true },
                max: EMBEDDING_GEMMA_MAX_GPU_LAYERS,
              },
      });
      if (model.embeddingVectorSize !== profile.identity.dimensions) {
        throw new Error(
          `Recall embedded EmbeddingGemma model dimension mismatch: expected ${profile.identity.dimensions}, received ${model.embeddingVectorSize}`,
        );
      }
      for (let index = 0; index < parallelism; index += 1) {
        contexts.push(
          await model.createEmbeddingContext({
            contextSize,
            ...(options.threads === undefined ? {} : { threads: options.threads }),
          }),
        );
      }
      return {
        runtime,
        model,
        contexts,
        contextOperations: contexts.map(() => Promise.resolve()),
        nextContextIndex: 0,
      };
    } catch (error) {
      for (const context of contexts) {
        await context.dispose();
      }
      await model?.dispose();
      await runtime?.dispose();
      throw error;
    }
  }

  async function loadResources(): Promise<LoadedEmbeddingGemmaResources> {
    clearIdleResourceDisposal();
    if (disposed) {
      throw new Error('Recall embedded EmbeddingGemma provider disposed');
    }
    await resourcesDisposalPromise;
    if (disposed) {
      throw new Error('Recall embedded EmbeddingGemma provider disposed');
    }
    if (resources) {
      return resources;
    }
    resourcesLoadPromise ??= (async () => {
      const modelPath = await verifyModelArtifact();
      const nodeLlamaCpp = await loadNodeLlamaCpp();
      if (nodeLlamaCpp.version !== EMBEDDED_NODE_LLAMA_CPP_VERSION) {
        throw new Error(
          `Recall embedded EmbeddingGemma runtime version mismatch: expected ${EMBEDDED_NODE_LLAMA_CPP_VERSION}, received ${nodeLlamaCpp.version}`,
        );
      }
      const probedComputeBackends =
        devicePolicy === 'auto' && nodeLlamaCpp.getLlamaGpuTypes
          ? (await nodeLlamaCpp.getLlamaGpuTypes('supported')).filter(
              (backend): backend is Exclude<NodeLlamaCppGpuBackend, false> => backend !== false,
            )
          : [];
      const requestedComputeBackend: EmbeddedInferenceComputeBackend =
        devicePolicy === 'auto' ? (probedComputeBackends[0] ?? 'cpu') : devicePolicy;
      let selectedComputeBackend = requestedComputeBackend;
      let fallbackFromComputeBackend: Exclude<EmbeddedInferenceComputeBackend, 'cpu'> | null = null;
      try {
        resources = await loadResourcesForBackend(nodeLlamaCpp, modelPath, requestedComputeBackend);
      } catch (error) {
        if (devicePolicy !== 'auto' || requestedComputeBackend === 'cpu') {
          throw error;
        }
        fallbackFromComputeBackend = requestedComputeBackend;
        selectedComputeBackend = 'cpu';
        if (!cpuFallbackWarningEmitted) {
          cpuFallbackWarningEmitted = true;
          writeWarning(
            `Recall embedded EmbeddingGemma accelerator initialization failed for ${requestedComputeBackend}; retrying the same profile ${profile.profileId} on CPU: ${readEmbeddingErrorMessage(error)}`,
          );
        }
        resources = await loadResourcesForBackend(nodeLlamaCpp, modelPath, 'cpu');
      }
      const deviceNames =
        selectedComputeBackend === 'cpu'
          ? ['CPU']
          : ((await resources.runtime.getGpuDeviceNames?.()) ?? [selectedComputeBackend]);
      executionIdentity = Object.freeze({
        adapter: 'node-llama-cpp-embedded-v2',
        backend: 'embedded',
        computeBackend: selectedComputeBackend,
        deviceNames: Object.freeze([...deviceNames]),
        devicePolicy,
        fallbackFromComputeBackend,
        nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
        parallelism,
        probedComputeBackends: Object.freeze([...probedComputeBackends]),
        profileId: profile.profileId,
      });
      return resources;
    })();
    try {
      return await resourcesLoadPromise;
    } finally {
      if (!resources) {
        resourcesLoadPromise = undefined;
      }
    }
  }

  async function runWithEmbeddingContext<T>(
    loaded: LoadedEmbeddingGemmaResources,
    operation: (context: EmbeddingGemmaLlamaEmbeddingContext) => Promise<T>,
  ): Promise<T> {
    const contextIndex = loaded.nextContextIndex % loaded.contexts.length;
    loaded.nextContextIndex += 1;
    const previousOperation = loaded.contextOperations[contextIndex] ?? Promise.resolve();
    const completion = Promise.withResolvers<void>();
    loaded.contextOperations[contextIndex] = completion.promise;
    await previousOperation;
    try {
      const context = loaded.contexts[contextIndex];
      if (!context) {
        throw new Error('Recall embedded EmbeddingGemma context pool unavailable');
      }
      return await operation(context);
    } finally {
      completion.resolve();
    }
  }

  async function embedInput(input: string, signal?: AbortSignal): Promise<number[]> {
    activeOperationCount += 1;
    let loaded: LoadedEmbeddingGemmaResources | undefined;
    try {
      throwIfEmbeddingAborted(signal);
      loaded = await loadResources();
      return await runWithEmbeddingContext(loaded, async (context) => {
        throwIfEmbeddingAborted(signal);
        const embedding = await context.getEmbeddingFor(input);
        throwIfEmbeddingAborted(signal);
        return normalizeEmbeddingGemmaVector(embedding.vector, profile.identity.dimensions);
      });
    } finally {
      activeOperationCount -= 1;
      if (loaded) {
        scheduleIdleResourceDisposal(loaded);
      }
    }
  }

  return {
    get executionIdentity() {
      return executionIdentity;
    },
    embedQuery(query, signal) {
      return embedInput(`${profile.queryInputPrefix}${query}`, signal);
    },
    async embedDocuments(documents, signal) {
      return Promise.all(
        documents.map((document) =>
          embedInput(`${profile.documentInputPrefix}${document}`, signal),
        ),
      );
    },
    async loadConversationTokenizer() {
      activeOperationCount += 1;
      let loaded: LoadedEmbeddingGemmaResources | undefined;
      try {
        loaded = await loadResources();
        tokenizerResources = loaded;
        const tokenizerModel = loaded.model;
        return {
          encodeConversationText(text) {
            return { ids: [...tokenizerModel.tokenize(text, false)] };
          },
        };
      } finally {
        activeOperationCount -= 1;
        if (loaded) {
          scheduleIdleResourceDisposal(loaded);
        }
      }
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      clearIdleResourceDisposal();
      await resourcesDisposalPromise;
      const loaded = resources ?? (await resourcesLoadPromise);
      resources = undefined;
      tokenizerResources = undefined;
      resourcesLoadPromise = undefined;
      if (loaded) {
        await disposeLoadedResources(loaded);
      }
    },
  };
}

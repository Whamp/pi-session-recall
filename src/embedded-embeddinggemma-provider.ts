import {
  acquireEmbeddedLlamaRuntimeForBackend,
  initializeEmbeddedProviderResources,
  resolveEmbeddedModelGpuLayers,
  type NodeLlamaCppGpuBackend,
} from './acquireSharedEmbeddedLlamaRuntime.js';
import {
  createEmbeddedProviderResourceLifecycle,
  disposeEmbeddedProviderResourceLayers,
} from './embedded-provider-resource-lifecycle.js';
import { EmbeddedInferenceDevicePolicy, type EmbeddedInferenceComputeBackend } from './enums.js';
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
  runtimePoolIdentity?: object;
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
  fallbackFromComputeBackend:
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
    | null;
  nodeLlamaCppVersion: '3.18.1';
  parallelism: number;
  probedComputeBackends: readonly (
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
  )[];
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
  releaseRuntime(): Promise<void>;
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
      runtimePoolIdentity: loaded,
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

function isEmbeddingGemmaLlamaRuntime(value: unknown): value is EmbeddingGemmaLlamaRuntime {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const gpu: unknown = Reflect.get(value, 'gpu');
  return (
    (gpu === false || gpu === 'metal' || gpu === 'cuda' || gpu === 'vulkan') &&
    typeof Reflect.get(value, 'loadModel') === 'function' &&
    typeof Reflect.get(value, 'dispose') === 'function'
  );
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
  const devicePolicy = options.device ?? EmbeddedInferenceDevicePolicy.AUTO;

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
  let tokenizerResources: LoadedEmbeddingGemmaResources | undefined;
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

  async function disposeEmbeddingGemmaResourceLayers(options: {
    contextOperations: readonly Promise<void>[];
    contexts: readonly EmbeddingGemmaLlamaEmbeddingContext[];
    model: EmbeddingGemmaLlamaModel | undefined;
    releaseRuntime: (() => Promise<void>) | undefined;
  }): Promise<void> {
    await disposeEmbeddedProviderResourceLayers({
      failureMessage: 'Recall embedded EmbeddingGemma resource disposal failed',
      layers: [
        options.contextOperations.map((operation) => async () => {
          await operation;
        }),
        options.contexts.map((context) => () => context.dispose()),
        [
          async () => {
            await options.model?.dispose();
          },
        ],
        [
          async () => {
            await options.releaseRuntime?.();
          },
        ],
      ],
    });
  }

  async function disposeLoadedResources(loaded: LoadedEmbeddingGemmaResources): Promise<void> {
    await disposeEmbeddingGemmaResourceLayers(loaded);
  }

  async function loadResourcesForBackend(
    nodeLlamaCpp: EmbeddingGemmaNodeLlamaCppModule,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ): Promise<LoadedEmbeddingGemmaResources> {
    let releaseRuntime: (() => Promise<void>) | undefined;
    let model: EmbeddingGemmaLlamaModel | undefined;
    const contexts: EmbeddingGemmaLlamaEmbeddingContext[] = [];
    const initializeResources = async (): Promise<LoadedEmbeddingGemmaResources> => {
      const acquired = await acquireEmbeddedLlamaRuntimeForBackend({
        capabilityLabel: 'EmbeddingGemma',
        computeBackend,
        nodeLlamaCpp,
        isRuntime: (value): value is EmbeddingGemmaLlamaRuntime =>
          isEmbeddingGemmaLlamaRuntime(value),
      });
      const { runtime } = acquired;
      releaseRuntime = () => acquired.releaseRuntime();
      model = await runtime.loadModel({
        modelPath,
        gpuLayers: resolveEmbeddedModelGpuLayers({
          computeBackend,
          maxGpuLayers: EMBEDDING_GEMMA_MAX_GPU_LAYERS,
          fitContext: { contextSize, embeddingContext: true },
        }),
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
        releaseRuntime,
        model,
        contexts,
        contextOperations: contexts.map(() => Promise.resolve()),
        nextContextIndex: 0,
      };
    };
    return initializeResources().then(
      (loaded) => loaded,
      async (error: unknown) => {
        const [disposalResult] = await Promise.allSettled([
          disposeEmbeddingGemmaResourceLayers({
            contextOperations: [],
            contexts,
            model,
            releaseRuntime,
          }),
        ]);
        if (disposalResult?.status === 'rejected') {
          throw new AggregateError(
            [error, disposalResult.reason],
            'Recall embedded EmbeddingGemma initialization cleanup failed',
          );
        }
        throw error;
      },
    );
  }

  const resourceLifecycle = createEmbeddedProviderResourceLifecycle<LoadedEmbeddingGemmaResources>({
    disposedErrorMessage: 'Recall embedded EmbeddingGemma provider disposed',
    idleTimeoutMilliseconds,
    async loadResources(fallbackWarningAlreadyEmitted) {
      const initialized = await initializeEmbeddedProviderResources({
        capabilityLabel: 'EmbeddingGemma',
        devicePolicy,
        expectedNodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
        fallbackWarningAlreadyEmitted,
        verifyModelArtifact,
        loadNodeLlamaCpp,
        initializeForBackend: loadResourcesForBackend,
        disposeResources: disposeLoadedResources,
        writeFallbackWarning: (fallbackFromComputeBackend, error) => {
          writeWarning(
            `Recall embedded EmbeddingGemma accelerator initialization failed for ${fallbackFromComputeBackend}; retrying the same profile ${profile.profileId} on CPU: ${readEmbeddingErrorMessage(error)}`,
          );
        },
      });
      executionIdentity = Object.freeze({
        adapter: 'node-llama-cpp-embedded-v2',
        backend: 'embedded',
        computeBackend: initialized.selectedComputeBackend,
        deviceNames: Object.freeze([...initialized.deviceNames]),
        devicePolicy,
        fallbackFromComputeBackend: initialized.fallbackFromComputeBackend,
        nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
        parallelism,
        probedComputeBackends: Object.freeze([...initialized.probedComputeBackends]),
        profileId: profile.profileId,
      });
      return initialized;
    },
    canDisposeResourcesWhenIdle: (loaded) => tokenizerResources !== loaded,
    writeIdleDisposalWarning(error) {
      writeWarning(
        `Recall embedded EmbeddingGemma idle disposal failed: ${readEmbeddingErrorMessage(error)}`,
      );
    },
  });

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
    throwIfEmbeddingAborted(signal);
    return resourceLifecycle.runWithResources(async (loaded) => {
      throwIfEmbeddingAborted(signal);
      return runWithEmbeddingContext(loaded, async (context) => {
        throwIfEmbeddingAborted(signal);
        const embedding = await context.getEmbeddingFor(input);
        throwIfEmbeddingAborted(signal);
        return normalizeEmbeddingGemmaVector(embedding.vector, profile.identity.dimensions);
      });
    });
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
      return resourceLifecycle.runWithResources(async (loaded) => {
        tokenizerResources = loaded;
        const tokenizerModel = loaded.model;
        return {
          encodeConversationText(text) {
            return { ids: [...tokenizerModel.tokenize(text, false)] };
          },
        };
      });
    },
    async dispose() {
      tokenizerResources = undefined;
      await resourceLifecycle.dispose();
    },
  };
}

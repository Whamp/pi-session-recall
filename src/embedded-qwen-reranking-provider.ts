import { createCanonicalIdentity } from './create-canonical-identity.js';
import {
  acquireSharedEmbeddedLlamaRuntime,
  mapNodeLlamaCppComputeBackend,
  writeEmbeddedLlamaLog,
  type NodeLlamaCppGpuBackend,
} from './acquireSharedEmbeddedLlamaRuntime.js';
import {
  EMBEDDED_INFERENCE_MAX_PARALLELISM,
  EMBEDDED_NODE_LLAMA_CPP_VERSION,
} from './embedded-embeddinggemma-provider.js';
import {
  EmbeddedInferenceComputeBackend,
  EmbeddedInferenceDevicePolicy,
  RecallInferenceBackend,
} from './enums.js';
import {
  createRecallRerankingExecutionIdentity,
  normalizeRecallPhysicalDeviceIdentity,
  resolveRecallCpuHardwareIdentity,
  type RecallIdentifiedRerankingProvider,
  type RecallRerankingExecutionIdentity,
} from './recall-inference-capabilities.js';
import { createRecallModelArtifactCache } from './recall-model-artifact-cache.js';
import type { RecommendedQwenRerankingModelProfile } from './recall-model-profiles.js';

/** Conservative upper bound for Qwen reranker layers considered for GPU offload. */
export const QWEN_RERANKER_MAX_GPU_LAYERS = 40;

const NODE_LLAMA_CPP_RERANKING_ADAPTER_ID = 'node-llama-cpp-qwen-reranking-logit-recovery-v1';
const NODE_LLAMA_CPP_EXTRA_SIGMOID_MINIMUM = 0.5;
const NODE_LLAMA_CPP_EXTRA_SIGMOID_MAXIMUM = 1 / (1 + Math.exp(-1));

interface QwenLlamaRankingContext {
  rankAll(query: string, documents: string[]): Promise<number[]>;
  dispose(): Promise<void>;
}

interface QwenLlamaModel {
  createRankingContext(options: {
    contextSize: number;
    threads?: number;
  }): Promise<QwenLlamaRankingContext>;
  dispose(): Promise<void>;
}

interface QwenLlamaRuntime {
  readonly gpu: NodeLlamaCppGpuBackend;
  getGpuDeviceNames?(): Promise<string[]>;
  loadModel(options: {
    modelPath: string;
    gpuLayers:
      | number
      | { max: number; fitContext: { contextSize: number; embeddingContext: true } };
  }): Promise<QwenLlamaModel>;
  dispose(): Promise<void>;
}

/** Minimal dynamically loaded node-llama-cpp surface required by embedded Qwen reranking. */
export interface QwenRerankingNodeLlamaCppModule {
  version: string;
  runtimePoolIdentity?: object;
  LlamaLogLevel: { error: unknown };
  getLlamaGpuTypes?(include: 'supported'): Promise<readonly NodeLlamaCppGpuBackend[]>;
  getLlama(options: Record<string, unknown>): Promise<QwenLlamaRuntime>;
}

/** Lazy embedded Qwen reranker settings and injectable native boundaries for conformance. */
export interface EmbeddedQwenRerankingProviderOptions {
  modelCacheDirectory: string;
  contextSize?: number;
  threads?: number;
  device?: EmbeddedInferenceDevicePolicy;
  parallelism?: number;
  requestTimeoutMilliseconds?: number;
  idleTimeoutMilliseconds?: number;
  onWarning?: (warning: string) => void;
  verifyModelArtifact?: () => Promise<string>;
  loadNodeLlamaCpp?: () => Promise<QwenRerankingNodeLlamaCppModule>;
}

/** Search and cache evidence for one selected embedded Qwen reranking adapter. */
export interface EmbeddedQwenRerankingExecutionIdentity extends RecallRerankingExecutionIdentity {
  adapterId: typeof NODE_LLAMA_CPP_RERANKING_ADAPTER_ID;
  backend: RecallInferenceBackend.EMBEDDED;
  computeBackend: EmbeddedInferenceComputeBackend | 'pending';
  deviceNames: readonly string[];
  devicePolicy: EmbeddedInferenceDevicePolicy;
  fallbackFromComputeBackend:
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
    | null;
  contextSize: number;
  threads: number | null;
  nodeLlamaCppVersion: typeof EMBEDDED_NODE_LLAMA_CPP_VERSION;
  parallelism: number;
  physicalDeviceIdentity: readonly string[];
  probedComputeBackends: readonly (
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
  )[];
}

/** Embedded reranking provider with corrected llama.cpp scores and explicit native disposal. */
export interface EmbeddedQwenRerankingProvider extends RecallIdentifiedRerankingProvider {
  readonly executionIdentity: Readonly<EmbeddedQwenRerankingExecutionIdentity>;
  /** Loads native resources far enough to bind resolved compute and physical hardware identity. */
  resolveExecutionIdentity(): Promise<Readonly<EmbeddedQwenRerankingExecutionIdentity>>;
  dispose(): Promise<void>;
}

interface LoadedQwenRerankingResources {
  runtime: QwenLlamaRuntime;
  releaseRuntime(): Promise<void>;
  model: QwenLlamaModel;
  contexts: readonly QwenLlamaRankingContext[];
  contextOperations: Promise<void>[];
  nextContextIndex: number;
}

async function loadInstalledNodeLlamaCpp(): Promise<QwenRerankingNodeLlamaCppModule> {
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
      `Recall embedded Qwen reranker runtime unavailable: install optional node-llama-cpp@${EMBEDDED_NODE_LLAMA_CPP_VERSION}; ${message}`,
      { cause: error },
    );
  }
}

function isQwenLlamaRuntime(value: unknown): value is QwenLlamaRuntime {
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

function readQwenRerankingErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function throwIfQwenRerankingAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error('Recall embedded Qwen reranking operation cancelled');
  }
}

function recoverLlamaCppQwenRerankingScore(score: number, candidateIndex: number): number {
  if (
    !Number.isFinite(score) ||
    score < NODE_LLAMA_CPP_EXTRA_SIGMOID_MINIMUM ||
    score > NODE_LLAMA_CPP_EXTRA_SIGMOID_MAXIMUM
  ) {
    throw new Error(
      `Recall embedded Qwen reranker score semantics mismatch at candidate index ${candidateIndex}: node-llama-cpp@${EMBEDDED_NODE_LLAMA_CPP_VERSION} must return its known extra-sigmoid value from ${NODE_LLAMA_CPP_EXTRA_SIGMOID_MINIMUM} through ${NODE_LLAMA_CPP_EXTRA_SIGMOID_MAXIMUM}, received ${score}`,
    );
  }
  const recoveredScore = Math.log(score / (1 - score));
  if (recoveredScore < -1e-12 || recoveredScore > 1 + 1e-12) {
    throw new Error(
      `Recall embedded Qwen reranker recovered score outside llama.cpp range at candidate index ${candidateIndex}: ${recoveredScore}`,
    );
  }
  return Math.min(Math.max(recoveredScore, 0), 1);
}

function waitForQwenRerankingOperation<T>(
  operation: Promise<T>,
  requestTimeoutMilliseconds: number,
  signal?: AbortSignal,
): Promise<T> {
  const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
  const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
  return new Promise<T>((resolve, reject) => {
    const rejectForAbort = () => {
      if (timeoutSignal.aborted && !signal?.aborted) {
        reject(
          new Error(
            `Recall embedded Qwen reranker request timed out after ${requestTimeoutMilliseconds} ms`,
          ),
        );
        return;
      }
      reject(
        signal?.reason instanceof Error
          ? signal.reason
          : new Error('Recall embedded Qwen reranking operation cancelled'),
      );
    };
    if (requestSignal.aborted) {
      rejectForAbort();
      return;
    }
    requestSignal.addEventListener('abort', rejectForAbort, { once: true });
    operation.then(
      (value) => {
        requestSignal.removeEventListener('abort', rejectForAbort);
        resolve(value);
      },
      (error: unknown) => {
        requestSignal.removeEventListener('abort', rejectForAbort);
        reject(error);
      },
    );
  });
}

/** Creates a lazy accelerated Qwen reranker that restores llama.cpp probability semantics. */
export function createEmbeddedQwenRerankingProvider(
  profile: RecommendedQwenRerankingModelProfile,
  options: EmbeddedQwenRerankingProviderOptions,
): EmbeddedQwenRerankingProvider {
  const contextSize = options.contextSize ?? 4_096;
  if (!Number.isInteger(contextSize) || contextSize < 1) {
    throw new Error(
      `Recall embedded Qwen reranker context size invalid: expected a positive integer, received ${contextSize}`,
    );
  }
  if (
    options.threads !== undefined &&
    (!Number.isInteger(options.threads) || options.threads < 1)
  ) {
    throw new Error(
      `Recall embedded Qwen reranker thread count invalid: expected a positive integer, received ${options.threads}`,
    );
  }
  const parallelism = options.parallelism ?? 1;
  if (
    !Number.isInteger(parallelism) ||
    parallelism < 1 ||
    parallelism > EMBEDDED_INFERENCE_MAX_PARALLELISM
  ) {
    throw new Error(
      `Recall embedded Qwen reranker parallelism invalid: expected an integer from 1 through ${EMBEDDED_INFERENCE_MAX_PARALLELISM}, received ${parallelism}`,
    );
  }
  const requestTimeoutMilliseconds = options.requestTimeoutMilliseconds ?? 60_000;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall embedded Qwen reranker request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }
  const idleTimeoutMilliseconds = options.idleTimeoutMilliseconds ?? 300_000;
  if (!Number.isInteger(idleTimeoutMilliseconds) || idleTimeoutMilliseconds < 0) {
    throw new Error(
      `Recall embedded Qwen reranker idle timeout invalid: expected a nonnegative integer in milliseconds, received ${idleTimeoutMilliseconds}`,
    );
  }
  const devicePolicy = options.device ?? EmbeddedInferenceDevicePolicy.AUTO;
  const verifyModelArtifact =
    options.verifyModelArtifact ??
    (async () => {
      const status = await createRecallModelArtifactCache({
        cacheDirectory: options.modelCacheDirectory,
        profile,
      }).verifyArtifact();
      if (status.state !== 'valid') {
        throw new Error(
          `Recall embedded Qwen reranker artifact unavailable: ${status.issue ?? `state is ${status.state}`}; ${status.repair}`,
        );
      }
      return status.artifactPath;
    });
  const loadNodeLlamaCpp = options.loadNodeLlamaCpp ?? loadInstalledNodeLlamaCpp;
  const writeWarning =
    options.onWarning ?? ((warning: string) => process.stderr.write(`${warning}\n`));

  let resources: LoadedQwenRerankingResources | undefined;
  let resourcesLoadPromise: Promise<LoadedQwenRerankingResources> | undefined;
  let resourcesDisposalPromise: Promise<void> | undefined;
  let idleDisposalTimer: ReturnType<typeof setTimeout> | undefined;
  let activeOperationCount = 0;
  let disposed = false;
  let cpuFallbackWarningEmitted = false;
  function createBaseExecutionIdentity(
    computeBackend: EmbeddedInferenceComputeBackend | 'pending',
    fallbackFromComputeBackend:
      | EmbeddedInferenceComputeBackend.METAL
      | EmbeddedInferenceComputeBackend.CUDA
      | EmbeddedInferenceComputeBackend.VULKAN
      | null,
    physicalDeviceIdentity: readonly string[],
  ): Readonly<RecallRerankingExecutionIdentity> {
    return createRecallRerankingExecutionIdentity(
      profile,
      NODE_LLAMA_CPP_RERANKING_ADAPTER_ID,
      createCanonicalIdentity('node-llama-cpp-qwen-reranking-config-v1', {
        computeBackend,
        contextSize,
        devicePolicy,
        fallbackFromComputeBackend,
        nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
        parallelism,
        physicalDeviceIdentity,
        requestTimeoutMilliseconds,
        threads: options.threads ?? null,
      }),
      RecallInferenceBackend.EMBEDDED,
      requestTimeoutMilliseconds,
    );
  }
  let executionIdentity: Readonly<EmbeddedQwenRerankingExecutionIdentity> = Object.freeze({
    ...createBaseExecutionIdentity('pending', null, []),
    adapterId: NODE_LLAMA_CPP_RERANKING_ADAPTER_ID,
    backend: RecallInferenceBackend.EMBEDDED,
    computeBackend: 'pending',
    deviceNames: [],
    devicePolicy,
    fallbackFromComputeBackend: null,
    contextSize,
    threads: options.threads ?? null,
    nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
    parallelism,
    physicalDeviceIdentity: [],
    probedComputeBackends: [],
  });

  async function disposeQwenRerankingResources(
    loaded: LoadedQwenRerankingResources,
  ): Promise<void> {
    await Promise.all(loaded.contextOperations);
    for (const context of loaded.contexts) {
      await context.dispose();
    }
    await loaded.model.dispose();
    await loaded.releaseRuntime();
  }

  function clearQwenRerankingIdleDisposal(): void {
    if (idleDisposalTimer) {
      clearTimeout(idleDisposalTimer);
      idleDisposalTimer = undefined;
    }
  }

  function scheduleQwenRerankingIdleDisposal(loaded: LoadedQwenRerankingResources): void {
    if (disposed || idleTimeoutMilliseconds === 0 || activeOperationCount !== 0) {
      return;
    }
    clearQwenRerankingIdleDisposal();
    idleDisposalTimer = setTimeout(() => {
      idleDisposalTimer = undefined;
      if (disposed || activeOperationCount !== 0 || resources !== loaded) {
        return;
      }
      resources = undefined;
      resourcesLoadPromise = undefined;
      const disposal = disposeQwenRerankingResources(loaded);
      resourcesDisposalPromise = disposal;
      void disposal
        .catch((error: unknown) => {
          writeWarning(
            `Recall embedded Qwen reranker idle disposal failed: ${readQwenRerankingErrorMessage(error)}`,
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

  async function loadQwenRerankingResourcesForBackend(
    nodeLlamaCpp: QwenRerankingNodeLlamaCppModule,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ): Promise<LoadedQwenRerankingResources> {
    let runtime: QwenLlamaRuntime | undefined;
    let releaseRuntime: (() => Promise<void>) | undefined;
    let model: QwenLlamaModel | undefined;
    const contexts: QwenLlamaRankingContext[] = [];
    try {
      const requestedGpu =
        computeBackend === EmbeddedInferenceComputeBackend.CPU ? false : computeBackend;
      const loadRuntime = () =>
        nodeLlamaCpp.getLlama({
          build: 'never',
          debug: false,
          gpu: requestedGpu,
          logger: writeEmbeddedLlamaLog,
          logLevel: nodeLlamaCpp.LlamaLogLevel.error,
          progressLogs: false,
          skipDownload: true,
        });
      if (nodeLlamaCpp.runtimePoolIdentity) {
        const lease = await acquireSharedEmbeddedLlamaRuntime(
          nodeLlamaCpp.runtimePoolIdentity,
          `${nodeLlamaCpp.version}:${computeBackend}`,
          loadRuntime,
          isQwenLlamaRuntime,
          (sharedRuntime) => sharedRuntime.dispose(),
        );
        runtime = lease.runtime;
        releaseRuntime = () => lease.release();
      } else {
        runtime = await loadRuntime();
        releaseRuntime = () => runtime?.dispose() ?? Promise.resolve();
      }
      if (runtime.gpu !== requestedGpu) {
        throw new Error(
          `Recall embedded Qwen reranker compute backend mismatch: requested ${computeBackend}, received ${runtime.gpu === false ? 'cpu' : runtime.gpu}`,
        );
      }
      model = await runtime.loadModel({
        modelPath,
        gpuLayers:
          computeBackend === EmbeddedInferenceComputeBackend.CPU
            ? 0
            : {
                fitContext: { contextSize, embeddingContext: true },
                max: QWEN_RERANKER_MAX_GPU_LAYERS,
              },
      });
      for (let index = 0; index < parallelism; index += 1) {
        contexts.push(
          await model.createRankingContext({
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
    } catch (error) {
      for (const context of contexts) {
        await context.dispose();
      }
      await model?.dispose();
      await releaseRuntime?.();
      throw error;
    }
  }

  async function loadQwenRerankingResources(): Promise<LoadedQwenRerankingResources> {
    clearQwenRerankingIdleDisposal();
    if (disposed) {
      throw new Error('Recall embedded Qwen reranking provider disposed');
    }
    await resourcesDisposalPromise;
    if (resources) {
      return resources;
    }
    resourcesLoadPromise ??= (async () => {
      const modelPath = await verifyModelArtifact();
      const nodeLlamaCpp = await loadNodeLlamaCpp();
      if (nodeLlamaCpp.version !== EMBEDDED_NODE_LLAMA_CPP_VERSION) {
        throw new Error(
          `Recall embedded Qwen reranker runtime version mismatch: expected ${EMBEDDED_NODE_LLAMA_CPP_VERSION}, received ${nodeLlamaCpp.version}`,
        );
      }
      const probedComputeBackends =
        devicePolicy === EmbeddedInferenceDevicePolicy.AUTO && nodeLlamaCpp.getLlamaGpuTypes
          ? (await nodeLlamaCpp.getLlamaGpuTypes('supported'))
              .filter(
                (backend): backend is Exclude<NodeLlamaCppGpuBackend, false> => backend !== false,
              )
              .map(mapNodeLlamaCppComputeBackend)
          : [];
      const requestedComputeBackend: EmbeddedInferenceComputeBackend =
        devicePolicy === EmbeddedInferenceDevicePolicy.AUTO
          ? (probedComputeBackends[0] ?? EmbeddedInferenceComputeBackend.CPU)
          : devicePolicy === EmbeddedInferenceDevicePolicy.CPU
            ? EmbeddedInferenceComputeBackend.CPU
            : devicePolicy === EmbeddedInferenceDevicePolicy.METAL
              ? EmbeddedInferenceComputeBackend.METAL
              : devicePolicy === EmbeddedInferenceDevicePolicy.CUDA
                ? EmbeddedInferenceComputeBackend.CUDA
                : EmbeddedInferenceComputeBackend.VULKAN;
      let selectedComputeBackend = requestedComputeBackend;
      let fallbackFromComputeBackend:
        | EmbeddedInferenceComputeBackend.METAL
        | EmbeddedInferenceComputeBackend.CUDA
        | EmbeddedInferenceComputeBackend.VULKAN
        | null = null;
      try {
        resources = await loadQwenRerankingResourcesForBackend(
          nodeLlamaCpp,
          modelPath,
          requestedComputeBackend,
        );
      } catch (error) {
        if (
          devicePolicy !== EmbeddedInferenceDevicePolicy.AUTO ||
          requestedComputeBackend === EmbeddedInferenceComputeBackend.CPU
        ) {
          throw error;
        }
        fallbackFromComputeBackend = requestedComputeBackend;
        selectedComputeBackend = EmbeddedInferenceComputeBackend.CPU;
        if (!cpuFallbackWarningEmitted) {
          cpuFallbackWarningEmitted = true;
          writeWarning(
            `Recall embedded Qwen reranker accelerator initialization failed for ${requestedComputeBackend}; retrying the same profile ${profile.profileId} on CPU: ${readQwenRerankingErrorMessage(error)}`,
          );
        }
        resources = await loadQwenRerankingResourcesForBackend(
          nodeLlamaCpp,
          modelPath,
          EmbeddedInferenceComputeBackend.CPU,
        );
      }
      const cpuHardwareIdentity =
        selectedComputeBackend === EmbeddedInferenceComputeBackend.CPU
          ? resolveRecallCpuHardwareIdentity()
          : null;
      const deviceNames = cpuHardwareIdentity?.deviceNames ??
        (await resources.runtime.getGpuDeviceNames?.()) ?? [selectedComputeBackend];
      const physicalDeviceIdentity =
        cpuHardwareIdentity?.physicalDeviceIdentity ??
        normalizeRecallPhysicalDeviceIdentity(deviceNames);
      executionIdentity = Object.freeze({
        ...createBaseExecutionIdentity(
          selectedComputeBackend,
          fallbackFromComputeBackend,
          physicalDeviceIdentity,
        ),
        adapterId: NODE_LLAMA_CPP_RERANKING_ADAPTER_ID,
        backend: RecallInferenceBackend.EMBEDDED,
        computeBackend: selectedComputeBackend,
        deviceNames: Object.freeze([...deviceNames]),
        devicePolicy,
        fallbackFromComputeBackend,
        contextSize,
        threads: options.threads ?? null,
        nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
        parallelism,
        physicalDeviceIdentity,
        probedComputeBackends: Object.freeze([...probedComputeBackends]),
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

  function startQwenRerankingContextOperation(
    loaded: LoadedQwenRerankingResources,
    query: string,
    documents: readonly string[],
  ): Promise<number[]> {
    const contextIndex = loaded.nextContextIndex % loaded.contexts.length;
    loaded.nextContextIndex += 1;
    const context = loaded.contexts[contextIndex];
    if (!context) {
      throw new Error('Recall embedded Qwen reranker context pool unavailable');
    }
    const previousOperation = loaded.contextOperations[contextIndex] ?? Promise.resolve();
    const operation = previousOperation.then(() => context.rankAll(query, [...documents]));
    loaded.contextOperations[contextIndex] = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }

  return {
    get executionIdentity() {
      return executionIdentity;
    },
    async resolveExecutionIdentity() {
      await loadQwenRerankingResources();
      return executionIdentity;
    },
    async rerankDocuments(query, documents, signal) {
      if (documents.length === 0) {
        return [];
      }
      activeOperationCount += 1;
      let loaded: LoadedQwenRerankingResources | undefined;
      try {
        throwIfQwenRerankingAborted(signal);
        loaded = await loadQwenRerankingResources();
        throwIfQwenRerankingAborted(signal);
        const scores = await waitForQwenRerankingOperation(
          startQwenRerankingContextOperation(loaded, query, documents),
          requestTimeoutMilliseconds,
          signal,
        );
        if (scores.length !== documents.length) {
          throw new Error(
            `Recall embedded Qwen reranker score count mismatch: expected ${documents.length}, received ${scores.length}`,
          );
        }
        return scores.map(recoverLlamaCppQwenRerankingScore);
      } finally {
        activeOperationCount -= 1;
        if (loaded) {
          scheduleQwenRerankingIdleDisposal(loaded);
        }
      }
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      clearQwenRerankingIdleDisposal();
      await resourcesDisposalPromise;
      const loaded = resources ?? (await resourcesLoadPromise);
      resources = undefined;
      resourcesLoadPromise = undefined;
      if (loaded) {
        await disposeQwenRerankingResources(loaded);
      }
    },
  };
}

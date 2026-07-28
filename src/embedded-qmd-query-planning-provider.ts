import {
  acquireSharedEmbeddedLlamaRuntime,
  mapNodeLlamaCppComputeBackend,
  writeEmbeddedLlamaLog,
  type NodeLlamaCppGpuBackend,
} from './acquireSharedEmbeddedLlamaRuntime.js';
import { EMBEDDED_NODE_LLAMA_CPP_VERSION } from './embedded-embeddinggemma-provider.js';
import {
  EmbeddedInferenceComputeBackend,
  EmbeddedInferenceDevicePolicy,
  RecallInferenceBackend,
} from './enums.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  type RecallIdentifiedQueryPlanningProvider,
  type RecallQueryPlanningExecutionIdentity,
} from './recall-inference-capabilities.js';
import { createRecallModelArtifactCache } from './recall-model-artifact-cache.js';
import type { RecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';
import {
  formatQmdQueryPlanningPrompt,
  parseQmdQueryPlanningOutput,
} from './recall-query-planning-policy.js';

/** Conservative upper bound for QMD query planner layers considered for GPU offload. */
export const QMD_QUERY_PLANNER_MAX_GPU_LAYERS = 40;

const NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID = 'node-llama-cpp-qmd-query-planning-v1';

type QmdQueryPlanningGrammar = object;

interface QmdQueryPlanningContext {
  getSequence(): unknown;
  dispose(): Promise<void>;
}

interface QmdQueryPlanningModel {
  createContext(options: {
    contextSize: number;
    threads?: number;
  }): Promise<QmdQueryPlanningContext>;
  dispose(): Promise<void>;
}

interface QmdQueryPlanningRuntime {
  readonly gpu: NodeLlamaCppGpuBackend;
  getGpuDeviceNames?(): Promise<string[]>;
  createGrammar(options: { grammar: string }): Promise<QmdQueryPlanningGrammar>;
  loadModel(options: {
    modelPath: string;
    gpuLayers: number | { max: number; fitContext: { contextSize: number } };
  }): Promise<QmdQueryPlanningModel>;
  dispose(): Promise<void>;
}

interface QmdQueryPlanningChatSession {
  prompt(prompt: string, options: Record<string, unknown>): Promise<string>;
}

/** Minimal dynamically loaded node-llama-cpp surface required by embedded QMD planning. */
export interface QmdQueryPlanningNodeLlamaCppModule {
  version: string;
  runtimePoolIdentity?: object;
  LlamaLogLevel: { error: unknown };
  createChatSession(contextSequence: unknown): QmdQueryPlanningChatSession;
  getLlamaGpuTypes?(include: 'supported'): Promise<readonly NodeLlamaCppGpuBackend[]>;
  getLlama(options: Record<string, unknown>): Promise<QmdQueryPlanningRuntime>;
}

/** Lazy embedded QMD planner settings and injectable native boundaries for conformance. */
export interface EmbeddedQmdQueryPlanningProviderOptions {
  modelCacheDirectory: string;
  threads?: number;
  device?: EmbeddedInferenceDevicePolicy;
  requestTimeoutMilliseconds?: number;
  idleTimeoutMilliseconds?: number;
  onWarning?: (warning: string) => void;
  verifyModelArtifact?: () => Promise<string>;
  loadNodeLlamaCpp?: () => Promise<QmdQueryPlanningNodeLlamaCppModule>;
}

/** Query planner execution identity including selected native compute backend and device. */
export interface EmbeddedQmdQueryPlanningExecutionIdentity extends RecallQueryPlanningExecutionIdentity {
  adapterId: typeof NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID;
  backend: RecallInferenceBackend.EMBEDDED;
  computeBackend: EmbeddedInferenceComputeBackend | 'pending';
  deviceNames: readonly string[];
  devicePolicy: EmbeddedInferenceDevicePolicy;
  fallbackFromComputeBackend:
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
    | null;
  nodeLlamaCppVersion: typeof EMBEDDED_NODE_LLAMA_CPP_VERSION;
  probedComputeBackends: readonly (
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
  )[];
}

/** Embedded QMD planner with inspectable execution identity and explicit native disposal. */
export interface EmbeddedQmdQueryPlanningProvider extends RecallIdentifiedQueryPlanningProvider {
  readonly executionIdentity: Readonly<EmbeddedQmdQueryPlanningExecutionIdentity>;
  dispose(): Promise<void>;
}

interface LoadedQmdQueryPlanningResources {
  nodeLlamaCpp: QmdQueryPlanningNodeLlamaCppModule;
  runtime: QmdQueryPlanningRuntime;
  releaseRuntime(): Promise<void>;
  model: QmdQueryPlanningModel;
  grammar: QmdQueryPlanningGrammar;
}

async function loadInstalledNodeLlamaCpp(): Promise<QmdQueryPlanningNodeLlamaCppModule> {
  try {
    const loaded = await import('node-llama-cpp');
    return {
      version: EMBEDDED_NODE_LLAMA_CPP_VERSION,
      runtimePoolIdentity: loaded,
      LlamaLogLevel: loaded.LlamaLogLevel,
      createChatSession(contextSequence) {
        const nativeSession: unknown = Reflect.construct(loaded.LlamaChatSession, [
          { contextSequence },
        ]);
        if (typeof nativeSession !== 'object' || nativeSession === null) {
          throw new Error('Recall embedded QMD query planner chat session invalid');
        }
        const nativePrompt: unknown = Reflect.get(nativeSession, 'prompt');
        if (typeof nativePrompt !== 'function') {
          throw new Error('Recall embedded QMD query planner chat prompt unavailable');
        }
        return {
          async prompt(prompt, options) {
            const result: unknown = Reflect.apply(nativePrompt, nativeSession, [prompt, options]);
            const output: unknown = await Promise.resolve(result);
            if (typeof output !== 'string') {
              throw new Error('Recall embedded QMD query planner chat output invalid');
            }
            return output;
          },
        };
      },
      getLlamaGpuTypes: loaded.getLlamaGpuTypes,
      getLlama: loaded.getLlama,
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall embedded QMD query planner runtime unavailable: install optional node-llama-cpp@${EMBEDDED_NODE_LLAMA_CPP_VERSION}; ${message}`,
      { cause: error },
    );
  }
}

function isQmdQueryPlanningRuntime(value: unknown): value is QmdQueryPlanningRuntime {
  if (typeof value !== 'object' || value === null) {
    return false;
  }
  const gpu: unknown = Reflect.get(value, 'gpu');
  return (
    (gpu === false || gpu === 'metal' || gpu === 'cuda' || gpu === 'vulkan') &&
    typeof Reflect.get(value, 'createGrammar') === 'function' &&
    typeof Reflect.get(value, 'loadModel') === 'function' &&
    typeof Reflect.get(value, 'dispose') === 'function'
  );
}

function readQmdQueryPlanningErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function throwIfQmdQueryPlanningAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error('Recall embedded QMD query planning operation cancelled');
  }
}

function waitForQmdQueryPlanningOperation<T>(
  operation: Promise<T>,
  requestSignal: AbortSignal,
  timeoutSignal: AbortSignal,
  requestTimeoutMilliseconds: number,
  callerSignal?: AbortSignal,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const rejectForAbort = () => {
      if (timeoutSignal.aborted && !callerSignal?.aborted) {
        reject(
          new Error(
            `Recall embedded QMD query planner request timed out after ${requestTimeoutMilliseconds} ms`,
          ),
        );
        return;
      }
      reject(
        callerSignal?.reason instanceof Error
          ? callerSignal.reason
          : new Error('Recall embedded QMD query planning operation cancelled'),
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

/** Creates a lazy accelerated QMD query planner with same-profile automatic CPU fallback. */
export function createEmbeddedQmdQueryPlanningProvider(
  profile: RecommendedQmdQueryPlanningModelProfile,
  options: EmbeddedQmdQueryPlanningProviderOptions,
): EmbeddedQmdQueryPlanningProvider {
  if (
    options.threads !== undefined &&
    (!Number.isInteger(options.threads) || options.threads < 1)
  ) {
    throw new Error(
      `Recall embedded QMD query planner thread count invalid: expected a positive integer, received ${options.threads}`,
    );
  }
  const requestTimeoutMilliseconds = options.requestTimeoutMilliseconds ?? 60_000;
  if (!Number.isInteger(requestTimeoutMilliseconds) || requestTimeoutMilliseconds < 1) {
    throw new Error(
      `Recall embedded QMD query planner request timeout invalid: expected a positive integer, received ${requestTimeoutMilliseconds}`,
    );
  }
  const idleTimeoutMilliseconds = options.idleTimeoutMilliseconds ?? 300_000;
  if (!Number.isInteger(idleTimeoutMilliseconds) || idleTimeoutMilliseconds < 0) {
    throw new Error(
      `Recall embedded QMD query planner idle timeout invalid: expected a nonnegative integer in milliseconds, received ${idleTimeoutMilliseconds}`,
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
          `Recall embedded QMD query planner artifact unavailable: ${status.issue ?? `state is ${status.state}`}; ${status.repair}`,
        );
      }
      return status.artifactPath;
    });
  const loadNodeLlamaCpp = options.loadNodeLlamaCpp ?? loadInstalledNodeLlamaCpp;
  const writeWarning =
    options.onWarning ?? ((warning: string) => process.stderr.write(`${warning}\n`));

  let resources: LoadedQmdQueryPlanningResources | undefined;
  let resourcesLoadPromise: Promise<LoadedQmdQueryPlanningResources> | undefined;
  let resourcesDisposalPromise: Promise<void> | undefined;
  let idleDisposalTimer: ReturnType<typeof setTimeout> | undefined;
  let activeOperationCount = 0;
  let disposed = false;
  let cpuFallbackWarningEmitted = false;
  const baseExecutionIdentity = createRecallQueryPlanningExecutionIdentity(
    profile,
    NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
    RecallInferenceBackend.EMBEDDED,
    requestTimeoutMilliseconds,
  );
  let executionIdentity: Readonly<EmbeddedQmdQueryPlanningExecutionIdentity> = Object.freeze({
    ...baseExecutionIdentity,
    adapterId: NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
    backend: RecallInferenceBackend.EMBEDDED,
    computeBackend: 'pending',
    deviceNames: [],
    devicePolicy,
    fallbackFromComputeBackend: null,
    nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
    probedComputeBackends: [],
  });

  async function disposeQmdQueryPlanningResources(
    loaded: LoadedQmdQueryPlanningResources,
  ): Promise<void> {
    await loaded.model.dispose();
    await loaded.releaseRuntime();
  }

  function clearQmdQueryPlanningIdleDisposal(): void {
    if (idleDisposalTimer) {
      clearTimeout(idleDisposalTimer);
      idleDisposalTimer = undefined;
    }
  }

  function scheduleQmdQueryPlanningIdleDisposal(loaded: LoadedQmdQueryPlanningResources): void {
    if (disposed || idleTimeoutMilliseconds === 0 || activeOperationCount !== 0) {
      return;
    }
    clearQmdQueryPlanningIdleDisposal();
    idleDisposalTimer = setTimeout(() => {
      idleDisposalTimer = undefined;
      if (disposed || activeOperationCount !== 0 || resources !== loaded) {
        return;
      }
      resources = undefined;
      resourcesLoadPromise = undefined;
      const disposal = disposeQmdQueryPlanningResources(loaded);
      resourcesDisposalPromise = disposal;
      void disposal
        .catch((error: unknown) => {
          writeWarning(
            `Recall embedded QMD query planner idle disposal failed: ${readQmdQueryPlanningErrorMessage(error)}`,
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

  async function loadQmdQueryPlanningResourcesForBackend(
    nodeLlamaCpp: QmdQueryPlanningNodeLlamaCppModule,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ): Promise<LoadedQmdQueryPlanningResources> {
    let runtime: QmdQueryPlanningRuntime | undefined;
    let releaseRuntime: (() => Promise<void>) | undefined;
    let model: QmdQueryPlanningModel | undefined;
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
          isQmdQueryPlanningRuntime,
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
          `Recall embedded QMD query planner compute backend mismatch: requested ${computeBackend}, received ${runtime.gpu === false ? 'cpu' : runtime.gpu}`,
        );
      }
      model = await runtime.loadModel({
        modelPath,
        gpuLayers:
          computeBackend === EmbeddedInferenceComputeBackend.CPU
            ? 0
            : {
                fitContext: { contextSize: profile.generationPolicy.contextSize },
                max: QMD_QUERY_PLANNER_MAX_GPU_LAYERS,
              },
      });
      const grammar = await runtime.createGrammar({ grammar: profile.grammar });
      return { nodeLlamaCpp, runtime, releaseRuntime, model, grammar };
    } catch (error) {
      await model?.dispose();
      await releaseRuntime?.();
      throw error;
    }
  }

  async function loadQmdQueryPlanningResources(): Promise<LoadedQmdQueryPlanningResources> {
    clearQmdQueryPlanningIdleDisposal();
    if (disposed) {
      throw new Error('Recall embedded QMD query planning provider disposed');
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
          `Recall embedded QMD query planner runtime version mismatch: expected ${EMBEDDED_NODE_LLAMA_CPP_VERSION}, received ${nodeLlamaCpp.version}`,
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
        resources = await loadQmdQueryPlanningResourcesForBackend(
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
            `Recall embedded QMD query planner accelerator initialization failed for ${requestedComputeBackend}; retrying the same profile ${profile.profileId} on CPU: ${readQmdQueryPlanningErrorMessage(error)}`,
          );
        }
        resources = await loadQmdQueryPlanningResourcesForBackend(
          nodeLlamaCpp,
          modelPath,
          EmbeddedInferenceComputeBackend.CPU,
        );
      }
      const deviceNames =
        selectedComputeBackend === EmbeddedInferenceComputeBackend.CPU
          ? ['CPU']
          : ((await resources.runtime.getGpuDeviceNames?.()) ?? [selectedComputeBackend]);
      executionIdentity = Object.freeze({
        ...baseExecutionIdentity,
        adapterId: NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
        backend: RecallInferenceBackend.EMBEDDED,
        computeBackend: selectedComputeBackend,
        deviceNames: Object.freeze([...deviceNames]),
        devicePolicy,
        fallbackFromComputeBackend,
        nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
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

  return {
    get executionIdentity() {
      return executionIdentity;
    },
    async planRecallQuery(request, signal) {
      activeOperationCount += 1;
      let loaded: LoadedQmdQueryPlanningResources | undefined;
      let context: QmdQueryPlanningContext | undefined;
      try {
        throwIfQmdQueryPlanningAborted(signal);
        loaded = await loadQmdQueryPlanningResources();
        throwIfQmdQueryPlanningAborted(signal);
        context = await loaded.model.createContext({
          contextSize: profile.generationPolicy.contextSize,
          ...(options.threads === undefined ? {} : { threads: options.threads }),
        });
        const session = loaded.nodeLlamaCpp.createChatSession(context.getSequence());
        const timeoutSignal = AbortSignal.timeout(requestTimeoutMilliseconds);
        const requestSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
        const output = await waitForQmdQueryPlanningOperation(
          session.prompt(formatQmdQueryPlanningPrompt(request.query, request.recallIntent), {
            grammar: loaded.grammar,
            maxTokens: profile.generationPolicy.maximumOutputTokens,
            temperature: profile.generationPolicy.temperature,
            topK: profile.generationPolicy.topK,
            topP: profile.generationPolicy.topP,
            repeatPenalty: {
              lastTokens: profile.generationPolicy.repeatPenaltyLastTokens,
              presencePenalty: profile.generationPolicy.presencePenalty,
            },
            signal: requestSignal,
          }),
          requestSignal,
          timeoutSignal,
          requestTimeoutMilliseconds,
          signal,
        );
        return parseQmdQueryPlanningOutput(output, profile);
      } finally {
        await context?.dispose();
        activeOperationCount -= 1;
        if (loaded) {
          scheduleQmdQueryPlanningIdleDisposal(loaded);
        }
      }
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      clearQmdQueryPlanningIdleDisposal();
      await resourcesDisposalPromise;
      const loaded = resources ?? (await resourcesLoadPromise);
      resources = undefined;
      resourcesLoadPromise = undefined;
      if (loaded) {
        await disposeQmdQueryPlanningResources(loaded);
      }
    },
  };
}

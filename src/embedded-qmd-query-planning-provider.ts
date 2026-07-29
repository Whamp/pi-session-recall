import { createCanonicalIdentity } from './create-canonical-identity.js';
import {
  acquireEmbeddedLlamaRuntimeForBackend,
  initializeEmbeddedProviderResources,
  resolveEmbeddedModelGpuLayers,
  type NodeLlamaCppGpuBackend,
} from './acquireSharedEmbeddedLlamaRuntime.js';
import { EMBEDDED_NODE_LLAMA_CPP_VERSION } from './embedded-embeddinggemma-provider.js';
import {
  createEmbeddedProviderResourceLifecycle,
  disposeEmbeddedProviderResourceLayers,
} from './embedded-provider-resource-lifecycle.js';
import {
  EmbeddedInferenceComputeBackend,
  EmbeddedInferenceDevicePolicy,
  RecallInferenceBackend,
} from './enums.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  resolveRecallPhysicalDeviceIdentity,
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
const NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_VERSION = '1';

function createEmbeddedQmdQueryPlanningAdapterConfigurationIdentity(configuration: {
  devicePolicy: EmbeddedInferenceDevicePolicy;
  computeBackend: EmbeddedInferenceComputeBackend | 'pending';
  fallbackFromComputeBackend:
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
    | null;
  physicalDeviceIdentity: readonly string[];
  threads: number | null;
  requestTimeoutMilliseconds: number;
  idleTimeoutMilliseconds: number;
}): string {
  return createCanonicalIdentity('node-llama-cpp-qmd-query-planning-config-v1', {
    ...configuration,
    nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
  });
}

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
  contextSize: number;
  threads: number | null;
  nodeLlamaCppVersion: typeof EMBEDDED_NODE_LLAMA_CPP_VERSION;
  idleTimeoutMilliseconds: number;
  physicalDeviceIdentity: readonly string[];
  probedComputeBackends: readonly (
    | EmbeddedInferenceComputeBackend.METAL
    | EmbeddedInferenceComputeBackend.CUDA
    | EmbeddedInferenceComputeBackend.VULKAN
  )[];
}

/** Embedded QMD planner with inspectable execution identity and explicit native disposal. */
export interface EmbeddedQmdQueryPlanningProvider extends RecallIdentifiedQueryPlanningProvider {
  readonly executionIdentity: Readonly<EmbeddedQmdQueryPlanningExecutionIdentity>;
  /** Loads native resources far enough to bind resolved compute and physical device identity. */
  resolveExecutionIdentity(): Promise<Readonly<EmbeddedQmdQueryPlanningExecutionIdentity>>;
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

  function createBaseExecutionIdentity(
    computeBackend: EmbeddedInferenceComputeBackend | 'pending',
    fallbackFromComputeBackend:
      | EmbeddedInferenceComputeBackend.METAL
      | EmbeddedInferenceComputeBackend.CUDA
      | EmbeddedInferenceComputeBackend.VULKAN
      | null,
    physicalDeviceIdentity: readonly string[],
  ): Readonly<RecallQueryPlanningExecutionIdentity> {
    return createRecallQueryPlanningExecutionIdentity(
      profile,
      NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
      createEmbeddedQmdQueryPlanningAdapterConfigurationIdentity({
        devicePolicy,
        computeBackend,
        fallbackFromComputeBackend,
        physicalDeviceIdentity,
        threads: options.threads ?? null,
        requestTimeoutMilliseconds,
        idleTimeoutMilliseconds,
      }),
      RecallInferenceBackend.EMBEDDED,
      requestTimeoutMilliseconds,
      NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_VERSION,
    );
  }
  const configuredComputeBackend: EmbeddedInferenceComputeBackend | 'pending' =
    devicePolicy === EmbeddedInferenceDevicePolicy.AUTO
      ? 'pending'
      : devicePolicy === EmbeddedInferenceDevicePolicy.CPU
        ? EmbeddedInferenceComputeBackend.CPU
        : devicePolicy === EmbeddedInferenceDevicePolicy.METAL
          ? EmbeddedInferenceComputeBackend.METAL
          : devicePolicy === EmbeddedInferenceDevicePolicy.CUDA
            ? EmbeddedInferenceComputeBackend.CUDA
            : EmbeddedInferenceComputeBackend.VULKAN;
  let executionIdentity: Readonly<EmbeddedQmdQueryPlanningExecutionIdentity> = Object.freeze({
    ...createBaseExecutionIdentity(configuredComputeBackend, null, []),
    adapterId: NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
    backend: RecallInferenceBackend.EMBEDDED,
    computeBackend: configuredComputeBackend,
    deviceNames: [],
    devicePolicy,
    fallbackFromComputeBackend: null,
    contextSize: profile.generationPolicy.contextSize,
    threads: options.threads ?? null,
    nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
    idleTimeoutMilliseconds,
    physicalDeviceIdentity: [],
    probedComputeBackends: [],
  });

  async function disposeQmdQueryPlanningResourceLayers(options: {
    model: QmdQueryPlanningModel | undefined;
    releaseRuntime: (() => Promise<void>) | undefined;
  }): Promise<void> {
    await disposeEmbeddedProviderResourceLayers({
      failureMessage: 'Recall embedded QMD query planning resource disposal failed',
      layers: [
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

  async function disposeQmdQueryPlanningResources(
    loaded: LoadedQmdQueryPlanningResources,
  ): Promise<void> {
    await disposeQmdQueryPlanningResourceLayers(loaded);
  }

  async function loadQmdQueryPlanningResourcesForBackend(
    nodeLlamaCpp: QmdQueryPlanningNodeLlamaCppModule,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ): Promise<LoadedQmdQueryPlanningResources> {
    let releaseRuntime: (() => Promise<void>) | undefined;
    let model: QmdQueryPlanningModel | undefined;
    const initializeResources = async (): Promise<LoadedQmdQueryPlanningResources> => {
      const acquired = await acquireEmbeddedLlamaRuntimeForBackend({
        capabilityLabel: 'QMD query planner',
        computeBackend,
        nodeLlamaCpp,
        isRuntime: (value): value is QmdQueryPlanningRuntime => isQmdQueryPlanningRuntime(value),
      });
      const { runtime } = acquired;
      releaseRuntime = () => acquired.releaseRuntime();
      model = await runtime.loadModel({
        modelPath,
        gpuLayers: resolveEmbeddedModelGpuLayers({
          computeBackend,
          maxGpuLayers: QMD_QUERY_PLANNER_MAX_GPU_LAYERS,
          fitContext: { contextSize: profile.generationPolicy.contextSize },
        }),
      });
      const grammar = await runtime.createGrammar({ grammar: profile.grammar });
      return { nodeLlamaCpp, runtime, releaseRuntime, model, grammar };
    };
    return initializeResources().then(
      (loaded) => loaded,
      async (error: unknown) => {
        const [disposalResult] = await Promise.allSettled([
          disposeQmdQueryPlanningResourceLayers({ model, releaseRuntime }),
        ]);
        if (disposalResult?.status === 'rejected') {
          throw new AggregateError(
            [error, disposalResult.reason],
            'Recall embedded QMD query planning initialization cleanup failed',
          );
        }
        throw error;
      },
    );
  }

  const resourceLifecycle =
    createEmbeddedProviderResourceLifecycle<LoadedQmdQueryPlanningResources>({
      disposedErrorMessage: 'Recall embedded QMD query planning provider disposed',
      idleTimeoutMilliseconds,
      async loadResources(fallbackWarningAlreadyEmitted) {
        const initialized = await initializeEmbeddedProviderResources({
          capabilityLabel: 'QMD query planner',
          devicePolicy,
          expectedNodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
          fallbackWarningAlreadyEmitted,
          verifyModelArtifact,
          loadNodeLlamaCpp,
          initializeForBackend: loadQmdQueryPlanningResourcesForBackend,
          disposeResources: disposeQmdQueryPlanningResources,
          writeFallbackWarning: (fallbackFromComputeBackend, error) => {
            writeWarning(
              `Recall embedded QMD query planner accelerator initialization failed for ${fallbackFromComputeBackend}; retrying the same profile ${profile.profileId} on CPU: ${readQmdQueryPlanningErrorMessage(error)}`,
            );
          },
        });
        const { deviceNames, physicalDeviceIdentity } = resolveRecallPhysicalDeviceIdentity(
          initialized.selectedComputeBackend,
          initialized.deviceNames,
        );
        executionIdentity = Object.freeze({
          ...createBaseExecutionIdentity(
            initialized.selectedComputeBackend,
            initialized.fallbackFromComputeBackend,
            physicalDeviceIdentity,
          ),
          adapterId: NODE_LLAMA_CPP_QUERY_PLANNING_ADAPTER_ID,
          backend: RecallInferenceBackend.EMBEDDED,
          computeBackend: initialized.selectedComputeBackend,
          deviceNames: Object.freeze([...deviceNames]),
          devicePolicy,
          fallbackFromComputeBackend: initialized.fallbackFromComputeBackend,
          contextSize: profile.generationPolicy.contextSize,
          threads: options.threads ?? null,
          nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
          idleTimeoutMilliseconds,
          physicalDeviceIdentity,
          probedComputeBackends: Object.freeze([...initialized.probedComputeBackends]),
        });
        return initialized;
      },
      writeIdleDisposalWarning(error) {
        writeWarning(
          `Recall embedded QMD query planner idle disposal failed: ${readQmdQueryPlanningErrorMessage(error)}`,
        );
      },
    });

  return {
    get executionIdentity() {
      return executionIdentity;
    },
    async resolveExecutionIdentity() {
      await resourceLifecycle.runWithResources(async () => undefined);
      return executionIdentity;
    },
    async planRecallQuery(request, signal) {
      throwIfQmdQueryPlanningAborted(signal);
      return resourceLifecycle.runWithResources(async (loaded) => {
        let context: QmdQueryPlanningContext | undefined;
        let operationFailed = false;
        let operationError: unknown;
        let plannedQuery: ReturnType<typeof parseQmdQueryPlanningOutput> | undefined;
        try {
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
          plannedQuery = parseQmdQueryPlanningOutput(output, profile);
        } catch (error) {
          operationFailed = true;
          operationError = error;
        }
        let contextDisposalFailed = false;
        let contextDisposalError: unknown;
        try {
          await context?.dispose();
        } catch (error) {
          contextDisposalFailed = true;
          contextDisposalError = error;
        }
        if (operationFailed && contextDisposalFailed) {
          throw new AggregateError(
            [operationError, contextDisposalError],
            'Recall embedded QMD query planning operation and context disposal failed',
          );
        }
        if (operationFailed) {
          throw operationError;
        }
        if (contextDisposalFailed) {
          throw contextDisposalError;
        }
        if (!plannedQuery) {
          throw new Error('Recall embedded QMD query planning operation produced no query plan');
        }
        return plannedQuery;
      });
    },
    dispose: () => resourceLifecycle.dispose(),
  };
}

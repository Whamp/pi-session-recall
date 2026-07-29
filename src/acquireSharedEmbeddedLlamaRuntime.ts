import { EmbeddedInferenceComputeBackend, EmbeddedInferenceDevicePolicy } from './enums.js';

/** node-llama-cpp GPU selector, where false requests CPU execution. */
export type NodeLlamaCppGpuBackend = 'metal' | 'cuda' | 'vulkan' | false;

/** Accelerator backends that may be requested before an automatic CPU fallback. */
export type EmbeddedInferenceAcceleratorBackend =
  | EmbeddedInferenceComputeBackend.METAL
  | EmbeddedInferenceComputeBackend.CUDA
  | EmbeddedInferenceComputeBackend.VULKAN;

interface SharedEmbeddedLlamaRuntimeEntry {
  referenceCount: number;
  runtimePromise: Promise<unknown>;
  disposeRuntime(runtime: unknown): Promise<void>;
}

/** One reference-counted lease on a shared embedded node-llama-cpp runtime. */
export interface SharedEmbeddedLlamaRuntimeLease<Runtime> {
  runtime: Runtime;
  release(): Promise<void>;
}

/** Result of selecting and initializing one embedded backend under auto/explicit device policy. */
export interface EmbeddedInferenceBackendInitialization<Resources> {
  resources: Resources;
  selectedComputeBackend: EmbeddedInferenceComputeBackend;
  fallbackFromComputeBackend: EmbeddedInferenceAcceleratorBackend | null;
  fallbackWarningEmitted: boolean;
}

const SHARED_EMBEDDED_LLAMA_RUNTIME_POOLS = new WeakMap<
  object,
  Map<string, SharedEmbeddedLlamaRuntimeEntry>
>();

/** Converts a native GPU selector to the persisted embedded compute backend. */
export function mapNodeLlamaCppComputeBackend(
  backend: Exclude<NodeLlamaCppGpuBackend, false>,
): EmbeddedInferenceAcceleratorBackend {
  if (backend === 'metal') {
    return EmbeddedInferenceComputeBackend.METAL;
  }
  if (backend === 'cuda') {
    return EmbeddedInferenceComputeBackend.CUDA;
  }
  return EmbeddedInferenceComputeBackend.VULKAN;
}

/** Probes supported accelerator backends only when device policy is automatic. */
export async function probeSupportedEmbeddedComputeBackends(
  devicePolicy: EmbeddedInferenceDevicePolicy,
  getLlamaGpuTypes?: (include: 'supported') => Promise<readonly NodeLlamaCppGpuBackend[]>,
): Promise<readonly EmbeddedInferenceAcceleratorBackend[]> {
  return devicePolicy === EmbeddedInferenceDevicePolicy.AUTO && getLlamaGpuTypes
    ? (await getLlamaGpuTypes('supported'))
        .filter((backend): backend is Exclude<NodeLlamaCppGpuBackend, false> => backend !== false)
        .map(mapNodeLlamaCppComputeBackend)
    : [];
}

/** Resolves the first automatic accelerator or an explicit device-policy backend. */
export function resolveEmbeddedInferenceComputeBackend(
  devicePolicy: EmbeddedInferenceDevicePolicy,
  probedComputeBackends: readonly EmbeddedInferenceAcceleratorBackend[],
): EmbeddedInferenceComputeBackend {
  return devicePolicy === EmbeddedInferenceDevicePolicy.AUTO
    ? (probedComputeBackends[0] ?? EmbeddedInferenceComputeBackend.CPU)
    : devicePolicy === EmbeddedInferenceDevicePolicy.CPU
      ? EmbeddedInferenceComputeBackend.CPU
      : devicePolicy === EmbeddedInferenceDevicePolicy.METAL
        ? EmbeddedInferenceComputeBackend.METAL
        : devicePolicy === EmbeddedInferenceDevicePolicy.CUDA
          ? EmbeddedInferenceComputeBackend.CUDA
          : EmbeddedInferenceComputeBackend.VULKAN;
}

/**
 * Initializes resources for the requested backend, retrying once on CPU only under AUTO.
 * Explicit device policies remain fail-closed and never fall back.
 */
export async function initializeEmbeddedInferenceWithAutoCpuFallback<Resources>(options: {
  devicePolicy: EmbeddedInferenceDevicePolicy;
  requestedComputeBackend: EmbeddedInferenceComputeBackend;
  initialize: (backend: EmbeddedInferenceComputeBackend) => Promise<Resources>;
  fallbackWarningAlreadyEmitted: boolean;
  writeFallbackWarning: (
    fallbackFromComputeBackend: EmbeddedInferenceAcceleratorBackend,
    error: unknown,
  ) => void;
}): Promise<EmbeddedInferenceBackendInitialization<Resources>> {
  try {
    return {
      resources: await options.initialize(options.requestedComputeBackend),
      selectedComputeBackend: options.requestedComputeBackend,
      fallbackFromComputeBackend: null,
      fallbackWarningEmitted: options.fallbackWarningAlreadyEmitted,
    };
  } catch (error) {
    if (
      options.devicePolicy !== EmbeddedInferenceDevicePolicy.AUTO ||
      options.requestedComputeBackend === EmbeddedInferenceComputeBackend.CPU
    ) {
      throw error;
    }
    const fallbackFromComputeBackend = options.requestedComputeBackend;
    let fallbackWarningEmitted = options.fallbackWarningAlreadyEmitted;
    if (!fallbackWarningEmitted) {
      fallbackWarningEmitted = true;
      options.writeFallbackWarning(fallbackFromComputeBackend, error);
    }
    return {
      resources: await options.initialize(EmbeddedInferenceComputeBackend.CPU),
      selectedComputeBackend: EmbeddedInferenceComputeBackend.CPU,
      fallbackFromComputeBackend,
      fallbackWarningEmitted,
    };
  }
}

/** Reads CPU or accelerator device names for one initialized embedded runtime. */
export async function readEmbeddedInferenceDeviceNames(
  selectedComputeBackend: EmbeddedInferenceComputeBackend,
  runtime: { getGpuDeviceNames?: () => Promise<readonly string[]> | readonly string[] },
): Promise<readonly string[]> {
  return selectedComputeBackend === EmbeddedInferenceComputeBackend.CPU
    ? ['CPU']
    : ((await runtime.getGpuDeviceNames?.()) ?? [selectedComputeBackend]);
}

/** Maps a compute backend to the node-llama-cpp GPU selector (false for CPU). */
export function resolveNodeLlamaCppGpuBackend(
  computeBackend: EmbeddedInferenceComputeBackend,
): NodeLlamaCppGpuBackend {
  if (computeBackend === EmbeddedInferenceComputeBackend.CPU) {
    return false;
  }
  return computeBackend;
}

/** Chooses CPU-offload or accelerator fitContext gpuLayers for one model load. */
export function resolveEmbeddedModelGpuLayers<FitContext extends { contextSize: number }>(options: {
  computeBackend: EmbeddedInferenceComputeBackend;
  maxGpuLayers: number;
  fitContext: FitContext;
}): number | { max: number; fitContext: FitContext } {
  if (options.computeBackend === EmbeddedInferenceComputeBackend.CPU) {
    return 0;
  }
  return {
    fitContext: options.fitContext,
    max: options.maxGpuLayers,
  };
}

/** Shared node-llama-cpp module surface required to acquire a compute-backend runtime. */
export interface EmbeddedLlamaModuleForRuntimeAcquisition<Runtime> {
  version: string;
  runtimePoolIdentity?: object;
  LlamaLogLevel: { error: unknown };
  getLlama(options: Record<string, unknown>): Promise<Runtime>;
}

/**
 * Acquires a shared or private node-llama-cpp runtime for one compute backend.
 * Throws on GPU mismatch without releasing; callers dispose on failure.
 */
export async function acquireEmbeddedLlamaRuntimeForBackend<
  Runtime extends { gpu: NodeLlamaCppGpuBackend; dispose(): Promise<void> },
>(options: {
  capabilityLabel: string;
  computeBackend: EmbeddedInferenceComputeBackend;
  nodeLlamaCpp: EmbeddedLlamaModuleForRuntimeAcquisition<Runtime>;
  isRuntime: (value: unknown) => value is Runtime;
}): Promise<{ runtime: Runtime; releaseRuntime(): Promise<void> }> {
  const requestedGpu = resolveNodeLlamaCppGpuBackend(options.computeBackend);
  const loadRuntime = () =>
    options.nodeLlamaCpp.getLlama({
      build: 'never',
      debug: false,
      gpu: requestedGpu,
      logger: writeEmbeddedLlamaLog,
      logLevel: options.nodeLlamaCpp.LlamaLogLevel.error,
      progressLogs: false,
      skipDownload: true,
    });
  let runtime: Runtime;
  let releaseRuntime: () => Promise<void>;
  if (options.nodeLlamaCpp.runtimePoolIdentity) {
    const lease = await acquireSharedEmbeddedLlamaRuntime(
      options.nodeLlamaCpp.runtimePoolIdentity,
      `${options.nodeLlamaCpp.version}:${options.computeBackend}`,
      loadRuntime,
      options.isRuntime,
      (sharedRuntime) => sharedRuntime.dispose(),
    );
    runtime = lease.runtime;
    releaseRuntime = () => lease.release();
  } else {
    runtime = await loadRuntime();
    releaseRuntime = () => runtime.dispose();
  }
  if (runtime.gpu !== requestedGpu) {
    throw new Error(
      `Recall embedded ${options.capabilityLabel} compute backend mismatch: requested ${options.computeBackend}, received ${runtime.gpu === false ? 'cpu' : runtime.gpu}`,
    );
  }
  return { runtime, releaseRuntime };
}

/** Result of the shared embedded provider load orchestration (before capability-specific identity). */
export interface EmbeddedProviderResourceInitialization<Resources> {
  resources: Resources;
  selectedComputeBackend: EmbeddedInferenceComputeBackend;
  fallbackFromComputeBackend: EmbeddedInferenceAcceleratorBackend | null;
  fallbackWarningEmitted: boolean;
  probedComputeBackends: readonly EmbeddedInferenceAcceleratorBackend[];
  deviceNames: readonly string[];
}

/**
 * Verifies the model artifact, pins the node-llama-cpp version, probes/resolves the backend,
 * initializes with AUTO CPU fallback, and reads device names for execution identity.
 */
export async function initializeEmbeddedProviderResources<
  Module extends {
    version: string;
    getLlamaGpuTypes?(include: 'supported'): Promise<readonly NodeLlamaCppGpuBackend[]>;
  },
  Resources extends {
    runtime: { getGpuDeviceNames?: () => Promise<readonly string[]> | readonly string[] };
  },
>(options: {
  capabilityLabel: string;
  devicePolicy: EmbeddedInferenceDevicePolicy;
  expectedNodeLlamaCppVersion: string;
  fallbackWarningAlreadyEmitted: boolean;
  verifyModelArtifact: () => Promise<string>;
  loadNodeLlamaCpp: () => Promise<Module>;
  initializeForBackend: (
    nodeLlamaCpp: Module,
    modelPath: string,
    computeBackend: EmbeddedInferenceComputeBackend,
  ) => Promise<Resources>;
  writeFallbackWarning: (
    fallbackFromComputeBackend: EmbeddedInferenceAcceleratorBackend,
    error: unknown,
  ) => void;
}): Promise<EmbeddedProviderResourceInitialization<Resources>> {
  const modelPath = await options.verifyModelArtifact();
  const nodeLlamaCpp = await options.loadNodeLlamaCpp();
  if (nodeLlamaCpp.version !== options.expectedNodeLlamaCppVersion) {
    throw new Error(
      `Recall embedded ${options.capabilityLabel} runtime version mismatch: expected ${options.expectedNodeLlamaCppVersion}, received ${nodeLlamaCpp.version}`,
    );
  }
  const probedComputeBackends = await probeSupportedEmbeddedComputeBackends(
    options.devicePolicy,
    nodeLlamaCpp.getLlamaGpuTypes
      ? (include) => nodeLlamaCpp.getLlamaGpuTypes!(include)
      : undefined,
  );
  const requestedComputeBackend = resolveEmbeddedInferenceComputeBackend(
    options.devicePolicy,
    probedComputeBackends,
  );
  const initialized = await initializeEmbeddedInferenceWithAutoCpuFallback({
    devicePolicy: options.devicePolicy,
    requestedComputeBackend,
    fallbackWarningAlreadyEmitted: options.fallbackWarningAlreadyEmitted,
    initialize: (computeBackend) =>
      options.initializeForBackend(nodeLlamaCpp, modelPath, computeBackend),
    writeFallbackWarning: options.writeFallbackWarning,
  });
  const deviceNames = await readEmbeddedInferenceDeviceNames(
    initialized.selectedComputeBackend,
    initialized.resources.runtime,
  );
  return {
    resources: initialized.resources,
    selectedComputeBackend: initialized.selectedComputeBackend,
    fallbackFromComputeBackend: initialized.fallbackFromComputeBackend,
    fallbackWarningEmitted: initialized.fallbackWarningEmitted,
    probedComputeBackends,
    deviceNames,
  };
}

/** Routes node-llama-cpp native logs to stderr without corrupting JSON stdout. */
export function writeEmbeddedLlamaLog(level: unknown, message: string): void {
  void level;
  process.stderr.write(`[Recall native inference] ${message.replace(/\n+$/u, '')}\n`);
}

/** Shares one in-flight runtime load across compatible embedded capabilities. */
export async function acquireSharedEmbeddedLlamaRuntime<Runtime>(
  runtimePoolIdentity: object,
  runtimeKey: string,
  loadRuntime: () => Promise<Runtime>,
  isRuntime: (value: unknown) => value is Runtime,
  disposeRuntime: (runtime: Runtime) => Promise<void>,
): Promise<SharedEmbeddedLlamaRuntimeLease<Runtime>> {
  let runtimePool = SHARED_EMBEDDED_LLAMA_RUNTIME_POOLS.get(runtimePoolIdentity);
  if (!runtimePool) {
    runtimePool = new Map();
    SHARED_EMBEDDED_LLAMA_RUNTIME_POOLS.set(runtimePoolIdentity, runtimePool);
  }
  let entry = runtimePool.get(runtimeKey);
  if (!entry) {
    entry = {
      referenceCount: 0,
      runtimePromise: loadRuntime(),
      async disposeRuntime(runtime) {
        if (!isRuntime(runtime)) {
          throw new Error(`Recall shared embedded runtime invalid for ${runtimeKey}`);
        }
        await disposeRuntime(runtime);
      },
    };
    runtimePool.set(runtimeKey, entry);
  }
  entry.referenceCount += 1;
  let released = false;
  const release = async (): Promise<void> => {
    if (released) {
      return;
    }
    released = true;
    entry.referenceCount -= 1;
    if (entry.referenceCount !== 0 || runtimePool.get(runtimeKey) !== entry) {
      return;
    }
    runtimePool.delete(runtimeKey);
    const runtime = await entry.runtimePromise;
    await entry.disposeRuntime(runtime);
  };

  try {
    const runtime = await entry.runtimePromise;
    if (!isRuntime(runtime)) {
      await release();
      throw new Error(`Recall shared embedded runtime incompatible for ${runtimeKey}`);
    }
    return { runtime, release };
  } catch (error) {
    await release();
    throw error;
  }
}

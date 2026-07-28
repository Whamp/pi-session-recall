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

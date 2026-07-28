import { EmbeddedInferenceComputeBackend } from './enums.js';

/** node-llama-cpp GPU selector, where false requests CPU execution. */
export type NodeLlamaCppGpuBackend = 'metal' | 'cuda' | 'vulkan' | false;

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

const SHARED_EMBEDDED_LLAMA_RUNTIME_POOLS = new WeakMap<
  object,
  Map<string, SharedEmbeddedLlamaRuntimeEntry>
>();

/** Converts a native GPU selector to the persisted embedded compute backend. */
export function mapNodeLlamaCppComputeBackend(
  backend: Exclude<NodeLlamaCppGpuBackend, false>,
):
  | EmbeddedInferenceComputeBackend.METAL
  | EmbeddedInferenceComputeBackend.CUDA
  | EmbeddedInferenceComputeBackend.VULKAN {
  if (backend === 'metal') {
    return EmbeddedInferenceComputeBackend.METAL;
  }
  if (backend === 'cuda') {
    return EmbeddedInferenceComputeBackend.CUDA;
  }
  return EmbeddedInferenceComputeBackend.VULKAN;
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

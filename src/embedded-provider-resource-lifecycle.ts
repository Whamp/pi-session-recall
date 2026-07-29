import type { EmbeddedProviderResourceInitialization } from './acquireSharedEmbeddedLlamaRuntime.js';

/** Runs embedded inference operations against one single-flight, idle-disposed resource set. */
export interface EmbeddedProviderResourceLifecycle<Resources> {
  runWithResources<Result>(operation: (resources: Resources) => Promise<Result>): Promise<Result>;
  dispose(): Promise<void>;
}

/** Disposes ordered resource layers, continuing after failures and aggregating every error. */
export async function disposeEmbeddedProviderResourceLayers(options: {
  failureMessage: string;
  layers: readonly (readonly (() => Promise<void>)[])[];
}): Promise<void> {
  const errors: unknown[] = [];
  for (const layer of options.layers) {
    const results = await Promise.allSettled(
      layer.map(async (disposeResource) => disposeResource()),
    );
    for (const result of results) {
      if (result.status === 'rejected') {
        const reason: unknown = result.reason;
        errors.push(reason);
      }
    }
  }
  if (errors.length > 0) {
    throw new AggregateError(errors, options.failureMessage);
  }
}

/** Creates one embedded provider resource lifecycle with single-flight load and idle disposal. */
export function createEmbeddedProviderResourceLifecycle<Resources>(options: {
  disposedErrorMessage: string;
  idleTimeoutMilliseconds: number;
  loadResources(
    fallbackWarningAlreadyEmitted: boolean,
  ): Promise<EmbeddedProviderResourceInitialization<Resources>>;
  canDisposeResourcesWhenIdle?: (resources: Resources) => boolean;
  writeIdleDisposalWarning(error: unknown): void;
}): EmbeddedProviderResourceLifecycle<Resources> {
  let loadedResources: EmbeddedProviderResourceInitialization<Resources> | undefined;
  let resourcesLoadPromise: Promise<EmbeddedProviderResourceInitialization<Resources>> | undefined;
  let resourcesDisposalPromise: Promise<void> | undefined;
  let idleDisposalTimer: ReturnType<typeof setTimeout> | undefined;
  let activeOperationCount = 0;
  let disposed = false;
  let fallbackWarningEmitted = false;

  function clearIdleResourceDisposal(): void {
    if (idleDisposalTimer) {
      clearTimeout(idleDisposalTimer);
      idleDisposalTimer = undefined;
    }
  }

  function scheduleIdleResourceDisposal(
    loaded: EmbeddedProviderResourceInitialization<Resources>,
  ): void {
    if (
      disposed ||
      options.idleTimeoutMilliseconds === 0 ||
      activeOperationCount !== 0 ||
      loadedResources !== loaded ||
      options.canDisposeResourcesWhenIdle?.(loaded.resources) === false
    ) {
      return;
    }
    clearIdleResourceDisposal();
    idleDisposalTimer = setTimeout(() => {
      idleDisposalTimer = undefined;
      if (disposed || activeOperationCount !== 0 || loadedResources !== loaded) {
        return;
      }
      loadedResources = undefined;
      resourcesLoadPromise = undefined;
      const disposal = loaded.disposeResources();
      resourcesDisposalPromise = disposal;
      void disposal
        .catch((error: unknown) => {
          options.writeIdleDisposalWarning(error);
        })
        .finally(() => {
          if (resourcesDisposalPromise === disposal) {
            resourcesDisposalPromise = undefined;
          }
        });
    }, options.idleTimeoutMilliseconds);
    idleDisposalTimer.unref();
  }

  async function loadProviderResources(): Promise<
    EmbeddedProviderResourceInitialization<Resources>
  > {
    clearIdleResourceDisposal();
    if (disposed) {
      throw new Error(options.disposedErrorMessage);
    }
    await resourcesDisposalPromise;
    if (disposed) {
      throw new Error(options.disposedErrorMessage);
    }
    if (loadedResources) {
      return loadedResources;
    }
    resourcesLoadPromise ??= (async () => {
      const loaded = await options.loadResources(fallbackWarningEmitted);
      loadedResources = loaded;
      fallbackWarningEmitted = loaded.fallbackWarningEmitted;
      return loaded;
    })();
    try {
      return await resourcesLoadPromise;
    } finally {
      if (!loadedResources) {
        resourcesLoadPromise = undefined;
      }
    }
  }

  async function runWithResources<Result>(
    operation: (resources: Resources) => Promise<Result>,
  ): Promise<Result> {
    activeOperationCount += 1;
    let loaded: EmbeddedProviderResourceInitialization<Resources> | undefined;
    try {
      loaded = await loadProviderResources();
      return await operation(loaded.resources);
    } finally {
      activeOperationCount -= 1;
      if (loaded) {
        scheduleIdleResourceDisposal(loaded);
      }
    }
  }

  async function disposeProviderResources(): Promise<void> {
    if (disposed) {
      return;
    }
    disposed = true;
    clearIdleResourceDisposal();
    await resourcesDisposalPromise;
    const loaded = loadedResources ?? (await resourcesLoadPromise);
    loadedResources = undefined;
    resourcesLoadPromise = undefined;
    if (loaded) {
      await loaded.disposeResources();
    }
  }

  return { runWithResources, dispose: disposeProviderResources };
}

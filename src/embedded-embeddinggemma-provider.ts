import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallTokenizerManifestIdentity } from './recall-index-manifest.js';
import { createRecallModelArtifactCache } from './recall-model-artifact-cache.js';
import type { RecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

/** Exact node-llama-cpp release adopted from QMD 2.6.3 for embedded inference. */
export const EMBEDDED_NODE_LLAMA_CPP_VERSION = '3.18.1';

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
  readonly gpu: string | false;
  loadModel(options: { modelPath: string; gpuLayers: number }): Promise<EmbeddingGemmaLlamaModel>;
  dispose(): Promise<void>;
}

/** Minimal dynamically loaded node-llama-cpp surface required by embedded EmbeddingGemma. */
export interface EmbeddingGemmaNodeLlamaCppModule {
  version: string;
  LlamaLogLevel: { error: unknown };
  getLlama(options: Record<string, unknown>): Promise<EmbeddingGemmaLlamaRuntime>;
}

/** Lazy CPU runtime settings and injectable native boundaries for deterministic conformance. */
export interface EmbeddedEmbeddingGemmaProviderOptions {
  modelCacheDirectory: string;
  contextSize?: number;
  threads?: number;
  verifyModelArtifact?: () => Promise<string>;
  loadNodeLlamaCpp?: () => Promise<EmbeddingGemmaNodeLlamaCppModule>;
}

/** Search-policy evidence identifying one embedded EmbeddingGemma execution adapter. */
export interface EmbeddedEmbeddingGemmaExecutionIdentity {
  adapter: 'node-llama-cpp-embedded-v1';
  backend: 'embedded';
  device: 'cpu';
  nodeLlamaCppVersion: '3.18.1';
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
  context: EmbeddingGemmaLlamaEmbeddingContext;
}

async function loadInstalledNodeLlamaCpp(): Promise<EmbeddingGemmaNodeLlamaCppModule> {
  try {
    const loaded = await import('node-llama-cpp');
    return {
      version: EMBEDDED_NODE_LLAMA_CPP_VERSION,
      LlamaLogLevel: loaded.LlamaLogLevel,
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

/** Creates a lazy, CPU-only EmbeddingGemma provider from the checksum-verified model cache. */
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
  let resources: LoadedEmbeddingGemmaResources | undefined;
  let resourcesLoadPromise: Promise<LoadedEmbeddingGemmaResources> | undefined;
  let serializedOperation: Promise<void> = Promise.resolve();
  let disposed = false;

  async function loadResources(): Promise<LoadedEmbeddingGemmaResources> {
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
      let runtime: EmbeddingGemmaLlamaRuntime | undefined;
      let model: EmbeddingGemmaLlamaModel | undefined;
      try {
        runtime = await nodeLlamaCpp.getLlama({
          build: 'never',
          gpu: false,
          logLevel: nodeLlamaCpp.LlamaLogLevel.error,
          progressLogs: false,
          skipDownload: true,
        });
        model = await runtime.loadModel({ modelPath, gpuLayers: 0 });
        if (model.embeddingVectorSize !== profile.identity.dimensions) {
          throw new Error(
            `Recall embedded EmbeddingGemma model dimension mismatch: expected ${profile.identity.dimensions}, received ${model.embeddingVectorSize}`,
          );
        }
        const context = await model.createEmbeddingContext({
          contextSize,
          ...(options.threads === undefined ? {} : { threads: options.threads }),
        });
        resources = { runtime, model, context };
        return resources;
      } catch (error) {
        await model?.dispose();
        await runtime?.dispose();
        throw error;
      }
    })();
    try {
      return await resourcesLoadPromise;
    } finally {
      if (!resources) {
        resourcesLoadPromise = undefined;
      }
    }
  }

  async function runSerialized<T>(operation: () => Promise<T>): Promise<T> {
    const previousOperation = serializedOperation;
    const completion = Promise.withResolvers<void>();
    serializedOperation = completion.promise;
    await previousOperation;
    try {
      return await operation();
    } finally {
      completion.resolve();
    }
  }

  async function embedInput(input: string, signal?: AbortSignal): Promise<number[]> {
    return runSerialized(async () => {
      throwIfEmbeddingAborted(signal);
      const loaded = await loadResources();
      throwIfEmbeddingAborted(signal);
      const embedding = await loaded.context.getEmbeddingFor(input);
      throwIfEmbeddingAborted(signal);
      return normalizeEmbeddingGemmaVector(embedding.vector, profile.identity.dimensions);
    });
  }

  return {
    executionIdentity: Object.freeze({
      adapter: 'node-llama-cpp-embedded-v1',
      backend: 'embedded',
      device: 'cpu',
      nodeLlamaCppVersion: EMBEDDED_NODE_LLAMA_CPP_VERSION,
      profileId: profile.profileId,
    }),
    embedQuery(query, signal) {
      return embedInput(`${profile.queryInputPrefix}${query}`, signal);
    },
    async embedDocuments(documents, signal) {
      const embeddings: number[][] = [];
      for (const document of documents) {
        embeddings.push(await embedInput(`${profile.documentInputPrefix}${document}`, signal));
      }
      return embeddings;
    },
    async loadConversationTokenizer() {
      const loaded = await loadResources();
      return {
        encodeConversationText(text) {
          return { ids: [...loaded.model.tokenize(text, false)] };
        },
      };
    },
    async dispose() {
      if (disposed) {
        return;
      }
      disposed = true;
      await serializedOperation;
      const loaded = resources ?? (await resourcesLoadPromise);
      if (!loaded) {
        return;
      }
      await loaded.context.dispose();
      await loaded.model.dispose();
      await loaded.runtime.dispose();
      resources = undefined;
      resourcesLoadPromise = undefined;
    },
  };
}

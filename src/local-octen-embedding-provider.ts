import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { Tokenizer } from '@huggingface/tokenizers';
import type * as OnnxRuntimeModule from 'onnxruntime-node';
import type * as OnnxRuntimeWebModule from 'onnxruntime-web';

import { LocalOctenRuntimeBackend } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { createStoredRecallEmbedding } from './recall-stored-embedding.js';

/** Final token added by the certified artifact's tokenizer post-processor. */
export const LOCAL_OCTEN_END_TOKEN_ID = 151_643;

/** Token IDs and attention values for one batch-one ONNX operation. */
export interface LocalOctenInferenceInput {
  inputIds: readonly number[];
  attentionMask: readonly number[];
}

/** Raw last-hidden-state tensor returned by one ONNX operation. */
export interface LocalOctenInferenceOutput {
  dimensions: readonly number[];
  data: Float32Array;
}

/** Minimal releasable ONNX session boundary used by the local provider. */
export interface LocalOctenInferenceSession {
  run(input: LocalOctenInferenceInput): Promise<LocalOctenInferenceOutput>;
  release(): Promise<void>;
}

/** Minimal tokenizer boundary used by the local provider. */
export interface LocalOctenEmbeddingTokenizer {
  encode(text: string): number[];
}

/** Runtime and artifact paths for one local Octen embedding provider. */
export interface LocalOctenEmbeddingProviderOptions {
  modelDirectory: string;
  nativeDimensions: number;
  parallelism?: number;
  intraOperationThreads?: number;
  runtimeBackend?: LocalOctenRuntimeBackend;
  loadTokenizer?: (modelDirectory: string) => Promise<LocalOctenEmbeddingTokenizer>;
  loadSession?: (
    modelDirectory: string,
    intraOperationThreads: number,
    runtimeBackend: LocalOctenRuntimeBackend,
  ) => Promise<LocalOctenInferenceSession>;
}

/** Local provider with an explicit native or WASM runtime cleanup boundary. */
export interface LocalOctenEmbeddingProvider extends RecallEmbeddingProvider {
  close(): Promise<void>;
}

/** Resolves the certified native or WASM backend for one supported platform. */
export function resolveLocalOctenRuntimeBackend(
  platform: NodeJS.Platform = process.platform,
  architecture: string = process.arch,
): LocalOctenRuntimeBackend {
  if (platform === 'darwin' && architecture === 'arm64') {
    return LocalOctenRuntimeBackend.NATIVE;
  }
  if (
    (platform === 'linux' && architecture === 'x64') ||
    (platform === 'darwin' && architecture === 'x64')
  ) {
    return LocalOctenRuntimeBackend.WASM;
  }
  throw new Error(
    `Local Octen embedding runtime is unsupported on ${platform}/${architecture}; configure the external Octen HTTP profile instead`,
  );
}

/** Rejects unsupported local setup before a model download starts. */
export function assertLocalOctenPlatformSupported(): void {
  resolveLocalOctenRuntimeBackend();
}

function parseTokenizerAsset(content: string, path: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Local Octen tokenizer invalid at ${path}: ${message}`, { cause: error });
  }
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Local Octen tokenizer invalid at ${path}: expected an object`);
  }
  return parsed;
}

async function loadLocalOctenTokenizer(
  modelDirectory: string,
): Promise<LocalOctenEmbeddingTokenizer> {
  const tokenizerPath = join(modelDirectory, 'tokenizer.json');
  const tokenizerConfigPath = join(modelDirectory, 'tokenizer_config.json');
  const [tokenizerContent, tokenizerConfigContent] = await Promise.all([
    readFile(tokenizerPath, 'utf8'),
    readFile(tokenizerConfigPath, 'utf8'),
  ]);
  const tokenizer = new Tokenizer(
    parseTokenizerAsset(tokenizerContent, tokenizerPath),
    parseTokenizerAsset(tokenizerConfigContent, tokenizerConfigPath),
  );
  return {
    encode(text) {
      const encoding = tokenizer.encode(text, {
        'add_special_tokens': true,
        'return_token_type_ids': false,
      });
      return [...encoding.ids];
    },
  };
}

async function loadNativeLocalOctenInferenceSession(
  modelDirectory: string,
  intraOperationThreads: number,
): Promise<LocalOctenInferenceSession> {
  let runtime: typeof OnnxRuntimeModule;
  try {
    runtime = await import('onnxruntime-node');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Local Octen native runtime failed to load on ${process.platform}/${process.arch}: ${message}; reinstall dependencies or configure the external Octen HTTP profile`,
      { cause: error },
    );
  }
  const session = await runtime.InferenceSession.create(join(modelDirectory, 'model.int8.onnx'), {
    executionProviders: ['cpu'],
    executionMode: 'sequential',
    graphOptimizationLevel: 'all',
    intraOpNumThreads: intraOperationThreads,
    interOpNumThreads: 1,
  });
  return {
    async run(input) {
      const tokenCount = input.inputIds.length;
      const inputIds = new runtime.Tensor(
        'int64',
        BigInt64Array.from(input.inputIds, (value) => BigInt(value)),
        [1, tokenCount],
      );
      const attentionMask = new runtime.Tensor(
        'int64',
        BigInt64Array.from(input.attentionMask, (value) => BigInt(value)),
        [1, tokenCount],
      );
      const outputs = await session.run({ input_ids: inputIds, attention_mask: attentionMask });
      const hiddenState = outputs.last_hidden_state;
      if (!hiddenState || !(hiddenState.data instanceof Float32Array)) {
        throw new Error('Local Octen ONNX output last_hidden_state is missing or not FP32');
      }
      return { dimensions: hiddenState.dims, data: hiddenState.data };
    },
    async release() {
      await session.release();
    },
  };
}

async function loadWasmLocalOctenInferenceSession(
  modelDirectory: string,
): Promise<LocalOctenInferenceSession> {
  let runtime: typeof OnnxRuntimeWebModule;
  try {
    runtime = await import('onnxruntime-web');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Local Octen WASM runtime failed to load on ${process.platform}/${process.arch}: ${message}; reinstall dependencies or configure the external Octen HTTP profile`,
      { cause: error },
    );
  }
  runtime.env.wasm.numThreads = 1;
  runtime.env.wasm.simd = true;
  const [model, weights] = await Promise.all([
    readFile(join(modelDirectory, 'model.int8.onnx')),
    readFile(join(modelDirectory, 'model.int8.onnx.data')),
  ]);
  const session = await runtime.InferenceSession.create(model, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
    externalData: [{ path: 'model.int8.onnx.data', data: weights }],
  });
  return {
    async run(input) {
      const tokenCount = input.inputIds.length;
      const inputIds = new runtime.Tensor(
        'int64',
        BigInt64Array.from(input.inputIds, (value) => BigInt(value)),
        [1, tokenCount],
      );
      const attentionMask = new runtime.Tensor(
        'int64',
        BigInt64Array.from(input.attentionMask, (value) => BigInt(value)),
        [1, tokenCount],
      );
      const outputs = await session.run({ input_ids: inputIds, attention_mask: attentionMask });
      const hiddenState = outputs.last_hidden_state;
      if (!hiddenState || !(hiddenState.data instanceof Float32Array)) {
        throw new Error('Local Octen ONNX output last_hidden_state is missing or not FP32');
      }
      return { dimensions: hiddenState.dims, data: hiddenState.data };
    },
    async release() {
      await session.release();
    },
  };
}

async function loadLocalOctenInferenceSession(
  modelDirectory: string,
  intraOperationThreads: number,
  runtimeBackend: LocalOctenRuntimeBackend,
): Promise<LocalOctenInferenceSession> {
  return runtimeBackend === LocalOctenRuntimeBackend.NATIVE
    ? loadNativeLocalOctenInferenceSession(modelDirectory, intraOperationThreads)
    : loadWasmLocalOctenInferenceSession(modelDirectory);
}

function assertPositiveInteger(value: number, label: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${label} must be a positive integer`);
  }
}

/** Creates a lazy local Octen provider with ordered, bounded batch-one inference. */
export function createLocalOctenEmbeddingProvider(
  options: LocalOctenEmbeddingProviderOptions,
): LocalOctenEmbeddingProvider {
  assertPositiveInteger(options.nativeDimensions, 'Local Octen native dimensions');
  const configuredParallelism = options.parallelism ?? 4;
  const runtimeBackend = options.runtimeBackend ?? resolveLocalOctenRuntimeBackend();
  const parallelism = runtimeBackend === LocalOctenRuntimeBackend.WASM ? 1 : configuredParallelism;
  const intraOperationThreads = options.intraOperationThreads ?? 4;
  assertPositiveInteger(configuredParallelism, 'Local Octen parallelism');
  assertPositiveInteger(intraOperationThreads, 'Local Octen intra-operation threads');
  const loadTokenizer = options.loadTokenizer ?? loadLocalOctenTokenizer;
  const loadSession = options.loadSession ?? loadLocalOctenInferenceSession;
  let resourcesPromise:
    | Promise<{
        tokenizer: LocalOctenEmbeddingTokenizer;
        session: LocalOctenInferenceSession;
      }>
    | undefined;
  let closed = false;
  let releasePromise: Promise<void> | undefined;

  async function getResources() {
    if (closed) {
      throw new Error('Local Octen embedding provider is closed');
    }
    resourcesPromise ??= (async () => {
      const tokenizer = await loadTokenizer(options.modelDirectory);
      const session = await loadSession(
        options.modelDirectory,
        intraOperationThreads,
        runtimeBackend,
      );
      return { tokenizer, session };
    })();
    return resourcesPromise;
  }

  async function embedText(text: string, signal?: AbortSignal): Promise<number[]> {
    signal?.throwIfAborted();
    const { tokenizer, session } = await getResources();
    signal?.throwIfAborted();
    const inputIds = tokenizer.encode(text);
    for (const tokenId of inputIds) {
      if (!Number.isInteger(tokenId) || tokenId < 0) {
        throw new Error(`Local Octen tokenizer returned invalid token ID: ${tokenId}`);
      }
    }
    if (inputIds.at(-1) !== LOCAL_OCTEN_END_TOKEN_ID) {
      throw new Error(
        `Local Octen tokenizer contract invalid: expected final token ${LOCAL_OCTEN_END_TOKEN_ID}, received ${inputIds.at(-1) ?? 'none'}`,
      );
    }
    const output = await session.run({
      inputIds,
      attentionMask: inputIds.map(() => 1),
    });
    signal?.throwIfAborted();
    const expectedDimensions = [1, inputIds.length, options.nativeDimensions];
    if (
      output.dimensions.length !== expectedDimensions.length ||
      output.dimensions.some((value, index) => value !== expectedDimensions[index]) ||
      output.data.length !== inputIds.length * options.nativeDimensions
    ) {
      throw new Error(
        `Local Octen ONNX output shape invalid: expected [${expectedDimensions.join(', ')}], received [${output.dimensions.join(', ')}] with ${output.data.length} values`,
      );
    }
    const finalTokenOffset = (inputIds.length - 1) * options.nativeDimensions;
    const nativeEmbedding = Array.from(
      output.data.subarray(finalTokenOffset, finalTokenOffset + options.nativeDimensions),
    );
    for (const [index, value] of nativeEmbedding.entries()) {
      if (!Number.isFinite(value)) {
        throw new Error(`Local Octen ONNX output contains non-finite value at dimension ${index}`);
      }
    }
    return createStoredRecallEmbedding(
      nativeEmbedding,
      options.nativeDimensions,
      options.nativeDimensions,
    );
  }

  return {
    embedQuery: embedText,
    async embedDocuments(documents, signal) {
      const results = new Array<number[]>(documents.length);
      let nextIndex = 0;
      async function worker(): Promise<void> {
        while (true) {
          signal?.throwIfAborted();
          const index = nextIndex;
          nextIndex += 1;
          if (index >= documents.length) {
            return;
          }
          results[index] = await embedText(documents[index]!, signal);
        }
      }
      await Promise.all(
        Array.from({ length: Math.min(parallelism, documents.length) }, () => worker()),
      );
      return results;
    },
    async close() {
      if (releasePromise) {
        return releasePromise;
      }
      closed = true;
      releasePromise = (async () => {
        if (!resourcesPromise) {
          return;
        }
        try {
          const { session } = await resourcesPromise;
          await session.release();
        } catch {
          // Resource acquisition failed before a reachable session existed.
        }
      })();
      return releasePromise;
    },
  };
}

/** Runs one deterministic normalized embedding and releases the selected runtime session. */
export async function probeLocalOctenEmbeddingRuntime(
  modelDirectory: string,
  nativeDimensions: number,
): Promise<{ dimensions: number; norm: number }> {
  const provider = createLocalOctenEmbeddingProvider({ modelDirectory, nativeDimensions });
  try {
    const embedding = await provider.embedQuery('Pi Session Recall local embedding health check');
    const norm = Math.sqrt(embedding.reduce((sum, value) => sum + value * value, 0));
    return { dimensions: embedding.length, norm };
  } finally {
    await provider.close();
  }
}

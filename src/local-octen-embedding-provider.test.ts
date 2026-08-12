import assert from 'node:assert/strict';
import { setTimeout as sleep } from 'node:timers/promises';
import test from 'node:test';

import { LocalOctenRuntimeBackend } from './enums.js';
import {
  createLocalOctenEmbeddingProvider,
  LOCAL_OCTEN_END_TOKEN_ID,
  resolveLocalOctenRuntimeBackend,
  type LocalOctenInferenceInput,
  type LocalOctenInferenceOutput,
} from './local-octen-embedding-provider.js';

void test('local Octen runtime backend is deterministic for each supported platform', () => {
  assert.equal(resolveLocalOctenRuntimeBackend('linux', 'x64'), LocalOctenRuntimeBackend.WASM);
  assert.equal(resolveLocalOctenRuntimeBackend('darwin', 'arm64'), LocalOctenRuntimeBackend.NATIVE);
  assert.equal(resolveLocalOctenRuntimeBackend('darwin', 'x64'), LocalOctenRuntimeBackend.WASM);
  assert.throws(
    () => resolveLocalOctenRuntimeBackend('win32', 'x64'),
    /unsupported on win32\/x64/u,
  );
});

void test('local Octen provider uses the tokenizer final token, pools it, and L2-normalizes', async () => {
  const inputs: LocalOctenInferenceInput[] = [];
  let released = 0;
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 4,
    parallelism: 2,
    runtimeBackend: LocalOctenRuntimeBackend.NATIVE,
    async loadTokenizer() {
      return {
        encode(text) {
          assert.equal(text, 'query text');
          return [10, 11, LOCAL_OCTEN_END_TOKEN_ID];
        },
      };
    },
    async loadSession() {
      return {
        async run(input) {
          inputs.push(input);
          return {
            dimensions: [1, 3, 4],
            data: new Float32Array([9, 9, 9, 9, 8, 8, 8, 8, 3, 4, 0, 0]),
          };
        },
        async release() {
          released += 1;
        },
      };
    },
  });

  assert.deepEqual(await provider.embedQuery('query text'), [
    Math.fround(0.6),
    Math.fround(0.8),
    0,
    0,
  ]);
  assert.deepEqual(inputs, [
    {
      inputIds: [10, 11, LOCAL_OCTEN_END_TOKEN_ID],
      attentionMask: [1, 1, 1],
    },
  ]);
  await provider.close();
  await provider.close();
  assert.equal(released, 1);
  await assert.rejects(provider.embedQuery('query text'), /provider is closed/u);
});

void test('local Octen provider preserves document order under bounded concurrency', async () => {
  let active = 0;
  let maximumActive = 0;
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 2,
    parallelism: 2,
    runtimeBackend: LocalOctenRuntimeBackend.NATIVE,
    async loadTokenizer() {
      return {
        encode(text) {
          return [Number(text), LOCAL_OCTEN_END_TOKEN_ID];
        },
      };
    },
    async loadSession() {
      return {
        async run(input): Promise<LocalOctenInferenceOutput> {
          active += 1;
          maximumActive = Math.max(maximumActive, active);
          const value = input.inputIds[0]!;
          await sleep((4 - value) * 5);
          active -= 1;
          return {
            dimensions: [1, 2, 2],
            data: new Float32Array([0, 0, value, 1]),
          };
        },
        async release() {},
      };
    },
  });

  const embeddings = await provider.embedDocuments(['1', '2', '3']);

  assert.equal(maximumActive, 2);
  assert.equal(embeddings.length, 3);
  assert.ok(Math.abs(embeddings[0]![0]! - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(embeddings[0]![1]! - Math.SQRT1_2) < 1e-6);
  assert.ok(embeddings[1]![0]! > embeddings[0]![0]!);
  assert.ok(embeddings[2]![0]! > embeddings[1]![0]!);
});

void test('local Octen WASM provider serializes batch-one operations', async () => {
  let active = 0;
  let maximumActive = 0;
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 2,
    parallelism: 4,
    runtimeBackend: LocalOctenRuntimeBackend.WASM,
    async loadTokenizer() {
      return { encode: () => [1, LOCAL_OCTEN_END_TOKEN_ID] };
    },
    async loadSession() {
      return {
        async run() {
          active += 1;
          maximumActive = Math.max(maximumActive, active);
          await sleep(2);
          active -= 1;
          return {
            dimensions: [1, 2, 2],
            data: new Float32Array([0, 0, 1, 1]),
          };
        },
        async release() {},
      };
    },
  });

  await provider.embedDocuments(['one', 'two', 'three']);
  assert.equal(maximumActive, 1);
});

void test('local Octen provider rejects invalid output without returning a vector', async () => {
  const outputs: LocalOctenInferenceOutput[] = [
    { dimensions: [1, 2, 3], data: new Float32Array(6) },
    {
      dimensions: [1, 2, 2],
      data: new Float32Array([0, 0, Number.NaN, 1]),
    },
  ];
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 2,
    runtimeBackend: LocalOctenRuntimeBackend.NATIVE,
    async loadTokenizer() {
      return { encode: () => [1, LOCAL_OCTEN_END_TOKEN_ID] };
    },
    async loadSession() {
      return {
        async run() {
          return outputs.shift()!;
        },
        async release() {},
      };
    },
  });

  await assert.rejects(provider.embedQuery('wrong shape'), /expected \[1, 2, 2\]/u);
  await assert.rejects(provider.embedQuery('non-finite'), /non-finite value/u);
});

void test('local Octen provider does not create a native session after tokenizer load failure', async () => {
  let sessionLoaded = false;
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 2,
    runtimeBackend: LocalOctenRuntimeBackend.NATIVE,
    async loadTokenizer() {
      throw new Error('tokenizer corrupt');
    },
    async loadSession() {
      sessionLoaded = true;
      throw new Error('must not load');
    },
  });

  await assert.rejects(provider.embedQuery('broken'), /tokenizer corrupt/u);
  assert.equal(sessionLoaded, false);
  await assert.doesNotReject(provider.close());
});

void test('local Octen provider honors cancellation before loading native runtime', async () => {
  const controller = new AbortController();
  controller.abort(new Error('stop'));
  let loaded = false;
  const provider = createLocalOctenEmbeddingProvider({
    modelDirectory: '/models/local-octen',
    nativeDimensions: 2,
    runtimeBackend: LocalOctenRuntimeBackend.NATIVE,
    async loadTokenizer() {
      loaded = true;
      return { encode: () => [1, LOCAL_OCTEN_END_TOKEN_ID] };
    },
    async loadSession() {
      loaded = true;
      throw new Error('must not load');
    },
  });

  await assert.rejects(provider.embedQuery('cancelled', controller.signal), /stop/u);
  assert.equal(loaded, false);
});

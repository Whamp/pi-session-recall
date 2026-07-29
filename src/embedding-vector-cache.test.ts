import assert from 'node:assert/strict';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
  normalizeConversationTextForEmbedding,
} from './embedding-vector-cache.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallDiagnosticsClock } from './recall-operation-diagnostics.js';
import { createRecallIndexManifest } from './recall-index-manifest.js';

function createTestCacheIdentity(options: { dimensions?: number; pooling?: string } = {}) {
  const dimensions = options.dimensions ?? 3;
  const canaryEmbedding: number[] = [];
  for (let index = 0; index < dimensions; index += 1) {
    canaryEmbedding.push(index === dimensions - 1 ? 1 : 0);
  }
  return createEmbeddingVectorCacheIdentity(
    createRecallIndexManifest({
      embeddingIdentity: {
        requestModel: 'test-request-model',
        servedModelId: 'test-served-model',
        artifact: 'test-model.fp32',
        dimensions,
        quantization: 'fp32',
        pooling: options.pooling ?? 'last',
      },
      canaryEmbedding,
    }),
  );
}

void test('embedding cache keys normalized text and full geometry identity while preserving order', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'embedding-vector-cache-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const embeddedInputs: string[][] = [];
  const embeddingProvider: RecallEmbeddingProvider = {
    async embedQuery(query, signal) {
      const vectors = await this.embedDocuments([query], signal);
      const vector = vectors[0];
      if (!vector) {
        throw new Error('missing query vector');
      }
      return vector;
    },
    async embedDocuments(texts) {
      embeddedInputs.push([...texts]);
      return texts.map((text) => [text.length / 3, text.includes('changed') ? 2 : 1, 0]);
    },
  };
  const cache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity(),
    embeddingRequestBatchSize: 1,
    embeddingProvider,
  });

  const first = await cache.resolveEmbeddingVectors(['cafe\u0301', 'second']);
  assert.equal(first.cacheHits, 0);
  assert.equal(first.newlyEmbeddedChunks, 2);
  assert.equal(first.embeddingRequestCount, 2);
  assert.deepEqual(embeddedInputs, [['café'], ['second']]);
  assert.deepEqual(first.vectors, [
    [Math.fround('café'.length / 3), 1, 0],
    [2, 1, 0],
  ]);

  const second = await cache.resolveEmbeddingVectors(['café', 'changed text']);
  assert.equal(second.cacheHits, 1);
  assert.equal(second.newlyEmbeddedChunks, 1);
  assert.equal(second.embeddingRequestCount, 1);
  assert.deepEqual(second.vectors, [
    first.vectors[0],
    [Math.fround('changed text'.length / 3), 2, 0],
  ]);
  assert.deepEqual(embeddedInputs.at(-1), ['changed text']);

  const changedModelCache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity({ pooling: 'mean' }),
    embeddingRequestBatchSize: 8,
    embeddingProvider,
  });
  const changedModel = await changedModelCache.resolveEmbeddingVectors(['café']);
  assert.equal(changedModel.cacheHits, 0);
  assert.equal(changedModel.newlyEmbeddedChunks, 1);
  assert.equal(changedModel.embeddingRequestCount, 1);
});

void test('embedding cache writes one atomic checksummed file and rejects corruption', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'embedding-vector-cache-corrupt-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const cache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity(),
    embeddingRequestBatchSize: 8,
    embeddingProvider: {
      async embedQuery(query, signal) {
        const vectors = await this.embedDocuments([query], signal);
        const vector = vectors[0];
        if (!vector) {
          throw new Error('missing query vector');
        }
        return vector;
      },
      async embedDocuments() {
        return [[1, 2, 3]];
      },
    },
  });

  await cache.resolveEmbeddingVectors(['durable vector']);
  const relativePaths = await readdir(directory, { recursive: true });
  const cacheFiles = relativePaths.filter((path) => path.endsWith('.fp32'));
  assert.equal(cacheFiles.length, 1);
  assert.ok(relativePaths.every((path) => !path.endsWith('.tmp')));
  const relativeCachePath = cacheFiles[0];
  assert.ok(relativeCachePath);
  const cachePath = join(directory, relativeCachePath);
  const corrupt = await readFile(cachePath);
  corrupt[corrupt.length - 1] = (corrupt.at(-1) ?? 0) ^ 0xff;
  await writeFile(cachePath, corrupt);

  await assert.rejects(
    () => cache.resolveEmbeddingVectors(['durable vector']),
    /Recall embedding cache invalid.*checksum mismatch/s,
  );
});

void test('embedding cache rejects wrong dimensions and non-finite FP32 values before reuse', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'embedding-vector-cache-shape-'));
  t.after(() => rm(directory, { recursive: true, force: true }));

  for (const [content, vector, expectedError] of [
    ['wrong dimensions', [1, 2], /dimension mismatch/],
    ['nan vector', [1, Number.NaN, 3], /value 1 is not finite/],
    ['fp32 overflow', [1, Number.MAX_VALUE, 3], /value 1 is not finite/],
  ] as const) {
    const embeddingProvider: RecallEmbeddingProvider = {
      async embedQuery(query, signal) {
        const vectors = await this.embedDocuments([query], signal);
        const vector = vectors[0];
        if (!vector) {
          throw new Error('missing query vector');
        }
        return vector;
      },
      async embedDocuments() {
        return [[...vector]];
      },
    };
    const cache = createEmbeddingVectorCache({
      cacheDirectory: directory,
      identity: createTestCacheIdentity(),
      embeddingRequestBatchSize: 8,
      embeddingProvider,
    });
    await assert.rejects(() => cache.resolveEmbeddingVectors([content]), expectedError);
  }

  let validEmbeddingRequests = 0;
  const validCache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity(),
    embeddingRequestBatchSize: 8,
    embeddingProvider: {
      async embedQuery(query, signal) {
        const vectors = await this.embedDocuments([query], signal);
        const vector = vectors[0];
        if (!vector) {
          throw new Error('missing query vector');
        }
        return vector;
      },
      async embedDocuments(texts) {
        validEmbeddingRequests += 1;
        return texts.map(() => [1, 2, 3]);
      },
    },
  });
  const recovered = await validCache.resolveEmbeddingVectors([
    'wrong dimensions',
    'nan vector',
    'fp32 overflow',
  ]);
  assert.equal(recovered.cacheHits, 0);
  assert.equal(recovered.newlyEmbeddedChunks, 3);
  assert.equal(validEmbeddingRequests, 1);
});

void test('embedding cache reports exclusive local cache and embedding-server milliseconds', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'embedding-vector-cache-timing-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const timestamps = [0, 10, 30, 40];
  let timestampIndex = 0;
  const diagnosticsClock: RecallDiagnosticsClock = {
    monotonicMilliseconds() {
      const timestamp = timestamps[timestampIndex];
      timestampIndex += 1;
      if (timestamp === undefined) {
        throw new Error('Embedding cache timing test exhausted fake clock values');
      }
      return timestamp;
    },
    wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
  };
  const cache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity(),
    embeddingRequestBatchSize: 8,
    diagnosticsClock,
    embeddingProvider: {
      async embedQuery(query, signal) {
        const vectors = await this.embedDocuments([query], signal);
        const vector = vectors[0];
        if (!vector) {
          throw new Error('missing query vector');
        }
        return vector;
      },
      async embedDocuments() {
        return [[1, 2, 3]];
      },
    },
  });

  const result = await cache.resolveEmbeddingVectors(['timed miss']);

  assert.equal(result.embeddingCacheResolutionMilliseconds, 20);
  assert.equal(result.embeddingServerRequestMilliseconds, 20);
  assert.equal(timestampIndex, 4);
});

void test('embedding cache reports server time when an embedding request fails', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'embedding-vector-cache-failed-timing-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const timestamps = [0, 10, 30];
  let timestampIndex = 0;
  const serverRequestMilliseconds: number[] = [];
  const cache = createEmbeddingVectorCache({
    cacheDirectory: directory,
    identity: createTestCacheIdentity(),
    embeddingRequestBatchSize: 8,
    diagnosticsClock: {
      monotonicMilliseconds() {
        const timestamp = timestamps[timestampIndex];
        timestampIndex += 1;
        if (timestamp === undefined) {
          throw new Error('Failed embedding timing test exhausted fake clock values');
        }
        return timestamp;
      },
      wallClockIsoTimestamp: () => '2026-07-27T10:00:00.000Z',
    },
    embeddingProvider: {
      async embedQuery(query, signal) {
        const vectors = await this.embedDocuments([query], signal);
        const vector = vectors[0];
        if (!vector) {
          throw new Error('missing query vector');
        }
        return vector;
      },
      async embedDocuments() {
        throw new Error('embedding server unavailable');
      },
    },
  });

  await assert.rejects(
    () =>
      cache.resolveEmbeddingVectors(['failed timed miss'], undefined, {
        recordEmbeddingServerRequest(milliseconds) {
          serverRequestMilliseconds.push(milliseconds);
        },
      }),
    /embedding server unavailable/u,
  );
  assert.deepEqual(serverRequestMilliseconds, [20]);
});

void test('embedding text normalization is idempotent and rejects mislabeled cache identity', () => {
  const decomposed = 'Cafe\u0301';
  const normalized = normalizeConversationTextForEmbedding(decomposed);
  assert.equal(normalized, 'Café');
  assert.equal(normalizeConversationTextForEmbedding(normalized), normalized);

  const identity = createTestCacheIdentity();
  identity.chunkPolicy.normalization = 'different-normalization-v1';
  assert.throws(
    () =>
      createEmbeddingVectorCache({
        cacheDirectory: '/unused',
        identity,
        embeddingRequestBatchSize: 8,
        embeddingProvider: {
          async embedQuery(query, signal) {
            const vectors = await this.embedDocuments([query], signal);
            const vector = vectors[0];
            if (!vector) {
              throw new Error('missing query vector');
            }
            return vector;
          },
          async embedDocuments() {
            return [];
          },
        },
      }),
    /Recall embedding cache normalization mismatch/,
  );
});

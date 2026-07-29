import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { RecallInferenceBackend } from './enums.js';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import {
  measureRecallEmbeddingProviderConformance,
  measureRecallRerankingProviderConformance,
} from './recall-inference-conformance.js';
import {
  createOctenEmbeddingModelProfile,
  createQwenRerankingModelProfile,
  createRecommendedEmbeddingGemmaModelProfile,
} from './recall-model-profiles.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createLlamaCppHttpEmbeddingProvider as createOctenHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';

const EMBEDDING_REQUEST_SCHEMA = Type.Object({
  input: Type.Array(Type.String()),
  model: Type.String(),
});

const RERANKING_REQUEST_SCHEMA = Type.Object({
  model: Type.String(),
  query: Type.String(),
  documents: Type.Array(Type.String()),
  'top_n': Type.Integer(),
});

void test('Octen HTTP embedding provider passes shared query and document conformance', async (t) => {
  const requests: Array<ReturnType<typeof Value.Parse<typeof EMBEDDING_REQUEST_SCHEMA>>> = [];
  const vectorsByInput = new Map<string, number[]>([
    ['Where is source provenance handled?', [1, 0, 0]],
    ['Source provenance is retained.', [0, 1, 0]],
    ['The navigation bar is blue.', [0, 0, 1]],
  ]);
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const payload = Value.Parse(EMBEDDING_REQUEST_SCHEMA, JSON.parse(body));
      requests.push(payload);
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          data: payload.input.map((input, index) => ({
            index,
            embedding: vectorsByInput.get(input),
          })),
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const profile = createOctenEmbeddingModelProfile({
    requestModel: 'octen-embed',
    servedModelId: 'Octen/Octen-Embedding-4B',
    artifact: 'Octen-Embedding-4B.Q8_0.gguf',
    dimensions: 3,
    quantization: 'Q8_0',
    pooling: 'last',
  });
  const provider = createOctenHttpEmbeddingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    batchSize: 8,
  });
  const clockValues = [0, 7, 7, 18];

  const measurement = await measureRecallEmbeddingProviderConformance({
    provider,
    profile,
    query: 'Where is source provenance handled?',
    expectedQueryEmbedding: [1, 0, 0],
    documents: ['Source provenance is retained.', 'The navigation bar is blue.'],
    expectedDocumentEmbeddings: [
      [0, 1, 0],
      [0, 0, 1],
    ],
    monotonicMilliseconds() {
      const value = clockValues.shift();
      assert.notEqual(value, undefined);
      return value ?? 0;
    },
  });

  assert.deepEqual(requests, [
    {
      model: 'octen-embed',
      input: ['Where is source provenance handled?'],
    },
    {
      model: 'octen-embed',
      input: ['Source provenance is retained.', 'The navigation bar is blue.'],
    },
  ]);
  assert.deepEqual(measurement, {
    queryCount: 1,
    documentCount: 2,
    queryMilliseconds: 7,
    documentMilliseconds: 11,
  });
});

void test('llama.cpp HTTP embedding provider preserves EmbeddingGemma asymmetric semantics', async (t) => {
  const requests: Array<ReturnType<typeof Value.Parse<typeof EMBEDDING_REQUEST_SCHEMA>>> = [];
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const queryVector = [1, ...Array<number>(767).fill(0)];
  const firstDocumentVector = [0, 1, ...Array<number>(766).fill(0)];
  const secondDocumentVector = [0, 0, 1, ...Array<number>(765).fill(0)];
  const vectorsByInput = new Map<string, number[]>([
    [`${profile.queryInputPrefix}source provenance`, queryVector],
    [`${profile.documentInputPrefix}Source provenance is retained.`, firstDocumentVector],
    [`${profile.documentInputPrefix}The navigation bar is blue.`, secondDocumentVector],
  ]);
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const payload = Value.Parse(EMBEDDING_REQUEST_SCHEMA, JSON.parse(body));
      requests.push(payload);
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          data: payload.input.map((input, index) => ({
            index,
            embedding: vectorsByInput.get(input),
          })),
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const provider = createLlamaCppHttpEmbeddingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    batchSize: 8,
  });

  await measureRecallEmbeddingProviderConformance({
    provider,
    profile,
    query: 'source provenance',
    expectedQueryEmbedding: queryVector,
    documents: ['Source provenance is retained.', 'The navigation bar is blue.'],
    expectedDocumentEmbeddings: [firstDocumentVector, secondDocumentVector],
  });

  assert.deepEqual(requests, [
    {
      model: 'embeddinggemma-300M-Q8_0',
      input: ['task: search result | query: source provenance'],
    },
    {
      model: 'embeddinggemma-300M-Q8_0',
      input: [
        'title: none | text: Source provenance is retained.',
        'title: none | text: The navigation bar is blue.',
      ],
    },
  ]);
});

void test('Qwen HTTP reranking provider passes shared ordered-score conformance', async (t) => {
  const requests: Array<ReturnType<typeof Value.Parse<typeof RERANKING_REQUEST_SCHEMA>>> = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      requests.push(Value.Parse(RERANKING_REQUEST_SCHEMA, JSON.parse(body)));
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: 'qwen3-rerank',
          object: 'list',
          usage: { 'prompt_tokens': 42, 'total_tokens': 42 },
          results: [
            { index: 1, 'relevance_score': 0.125 },
            { index: 0, 'relevance_score': 0.875 },
          ],
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const profile = createQwenRerankingModelProfile('qwen3-rerank');
  const provider = createQwenHttpRerankingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  });
  const clockValues = [0, 13];

  assert.deepEqual(provider.executionIdentity, {
    adapterId: 'llama-cpp-http-reranking-v1',
    backend: 'llama-cpp-http',
    cacheIdentity: 'qwen-reranking:qwen3-rerank:llama-cpp-http-reranking-v1',
    modelProfileId: 'qwen-reranking:qwen3-rerank',
  });
  const measurement = await measureRecallRerankingProviderConformance({
    provider,
    profile,
    query: 'source provenance',
    documents: ['Preserve exact source provenance.', 'The navigation bar is blue.'],
    expectedScores: [0.875, 0.125],
    monotonicMilliseconds() {
      const value = clockValues.shift();
      assert.notEqual(value, undefined);
      return value ?? 0;
    },
  });

  assert.deepEqual(requests, [
    {
      model: 'qwen3-rerank',
      query: 'source provenance',
      documents: ['Preserve exact source provenance.', 'The navigation bar is blue.'],
      'top_n': 2,
    },
  ]);
  assert.deepEqual(measurement, {
    queryCount: 1,
    documentCount: 2,
    rerankingMilliseconds: 13,
  });
});

void test('Qwen HTTP reranking rejects out-of-range scores, timeout, and cancellation', async (t) => {
  let responseMode: 'out-of-range' | 'pending' = 'out-of-range';
  const server = createServer((request, response) => {
    request.resume();
    request.on('end', () => {
      if (responseMode === 'pending') {
        return;
      }
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: 'qwen3-rerank',
          object: 'list',
          usage: { prompt_tokens: 1, total_tokens: 1 },
          results: [{ index: 0, relevance_score: 1.01 }],
        }),
      );
    });
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const profile = createQwenRerankingModelProfile('qwen3-rerank');

  const rangeProvider = createQwenHttpRerankingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  });
  await assert.rejects(
    () =>
      measureRecallRerankingProviderConformance({
        provider: rangeProvider,
        profile,
        query: 'query',
        documents: ['candidate'],
        expectedScores: [1],
      }),
    /Recall Qwen HTTP reranker score outside profile range at candidate index 0/u,
  );

  responseMode = 'pending';
  const timeoutProvider = createQwenHttpRerankingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    requestTimeoutMilliseconds: 5,
  });
  await assert.rejects(
    () =>
      measureRecallRerankingProviderConformance({
        provider: timeoutProvider,
        profile,
        query: 'query',
        documents: ['candidate'],
        expectedScores: [0.5],
      }),
    /Recall reranker request timed out after 5 ms/u,
  );

  const cancellation = new AbortController();
  const cancellationReason = new Error('operator cancelled reranker conformance');
  const cancelled = measureRecallRerankingProviderConformance({
    provider: rangeProvider,
    profile,
    query: 'query',
    documents: ['candidate'],
    expectedScores: [0.5],
    signal: cancellation.signal,
  });
  cancellation.abort(cancellationReason);
  await assert.rejects(
    () => cancelled,
    /Recall reranker request failed .*operator cancelled reranker conformance/u,
  );
});

void test('embedding conformance rejects non-normalized vectors for an L2 profile', async () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const nonNormalizedVector = [2, ...Array<number>(767).fill(0)];

  await assert.rejects(
    () =>
      measureRecallEmbeddingProviderConformance({
        profile,
        provider: {
          async embedQuery() {
            return nonNormalizedVector;
          },
          async embedDocuments() {
            return [nonNormalizedVector];
          },
        },
        query: 'query',
        expectedQueryEmbedding: nonNormalizedVector,
        documents: ['document'],
        expectedDocumentEmbeddings: [nonNormalizedVector],
      }),
    /Recall query embedding conformance normalization mismatch: expected L2 norm 1/u,
  );
});

void test('embedding conformance rejects document vectors returned out of order', async () => {
  const profile = createOctenEmbeddingModelProfile({
    requestModel: 'fixture-embed',
    servedModelId: 'fixture-embed',
    artifact: 'fixture.gguf',
    dimensions: 2,
    quantization: 'fixture',
    pooling: 'last',
  });

  await assert.rejects(
    () =>
      measureRecallEmbeddingProviderConformance({
        profile,
        provider: {
          async embedQuery() {
            return [1, 0];
          },
          async embedDocuments() {
            return [
              [0, 1],
              [1, 0],
            ];
          },
        },
        query: 'query',
        expectedQueryEmbedding: [1, 0],
        documents: ['first', 'second'],
        expectedDocumentEmbeddings: [
          [1, 0],
          [0, 1],
        ],
      }),
    /Recall document embedding index 0 conformance vector mismatch at dimension 0/u,
  );
});

void test('reranking conformance rejects a non-finite relevance score', async () => {
  const profile = createQwenRerankingModelProfile('fixture-reranker');

  await assert.rejects(
    () =>
      measureRecallRerankingProviderConformance({
        profile,
        provider: {
          executionIdentity: {
            adapterId: 'fixture-reranking-v1',
            backend: RecallInferenceBackend.CUSTOM,
            cacheIdentity: `${profile.profileId}:fixture-reranking-v1`,
            modelProfileId: profile.profileId,
          },
          async rerankDocuments() {
            return [Number.NaN];
          },
        },
        query: 'query',
        documents: ['candidate'],
        expectedScores: [0.5],
      }),
    /Recall reranking conformance score invalid at candidate index 0/u,
  );
});

void test('reranking conformance rejects double-sigmoid fixture scores', async () => {
  const profile = createQwenRerankingModelProfile('fixture-reranker');
  const expectedLlamaCppScores = [0.9, 0.1];
  const doubleSigmoidScores = expectedLlamaCppScores.map((score) => 1 / (1 + Math.exp(-score)));

  await assert.rejects(
    () =>
      measureRecallRerankingProviderConformance({
        profile,
        provider: {
          executionIdentity: {
            adapterId: 'known-double-sigmoid-v1',
            backend: RecallInferenceBackend.CUSTOM,
            cacheIdentity: `${profile.profileId}:known-double-sigmoid-v1`,
            modelProfileId: profile.profileId,
          },
          async rerankDocuments() {
            return doubleSigmoidScores;
          },
        },
        query: 'source provenance',
        documents: ['relevant evidence', 'unrelated evidence'],
        expectedScores: expectedLlamaCppScores,
        maximumAbsoluteDifference: 1e-12,
      }),
    /Recall reranking conformance score mismatch at candidate index 0/u,
  );
});

void test('reranking conformance rejects scores outside the profile range', async () => {
  const profile = createQwenRerankingModelProfile('fixture-reranker');

  await assert.rejects(
    () =>
      measureRecallRerankingProviderConformance({
        profile,
        provider: {
          executionIdentity: {
            adapterId: 'fixture-reranking-v1',
            backend: RecallInferenceBackend.CUSTOM,
            cacheIdentity: `${profile.profileId}:fixture-reranking-v1`,
            modelProfileId: profile.profileId,
          },
          async rerankDocuments() {
            return [1.01];
          },
        },
        query: 'query',
        documents: ['candidate'],
        expectedScores: [1.01],
      }),
    /Recall reranking conformance score outside profile range at candidate index 0/u,
  );
});

import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createLocalEmbeddingClient } from './local-embedding-client.js';

const localEmbeddingRequestSchema = Type.Object({
  input: Type.Array(Type.String()),
  model: Type.String(),
});

void test('local embedding client batches OpenAI-compatible requests and preserves input order', async (t) => {
  const requests: unknown[] = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      const payload: unknown = JSON.parse(body);
      const { input, model } = Value.Parse(localEmbeddingRequestSchema, payload);
      requests.push(payload);
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          data: input.map((text, index) => ({ index, embedding: [text.length, index, 1] })),
          model,
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

  const client = createLocalEmbeddingClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'local-test',
    dimensions: 3,
    batchSize: 2,
  });
  const vectors = await client.embedTexts(['a', 'bb', 'ccc']);

  assert.deepEqual(vectors, [
    [1, 0, 1],
    [2, 1, 1],
    [3, 0, 1],
  ]);
  assert.equal(requests.length, 2);
});

void test('local embedding client retries a transport failure before receiving a response', async (t) => {
  let requestCount = 0;
  const server = createServer((request, response) => {
    request.resume();
    requestCount += 1;
    if (requestCount === 1) {
      request.socket.destroy();
      return;
    }
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ data: [{ index: 0, embedding: [1, 2, 3] }] }));
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');

  const client = createLocalEmbeddingClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'local-test',
    dimensions: 3,
  });

  assert.deepEqual(await client.embedTexts(['retry transport failure']), [[1, 2, 3]]);
  assert.equal(requestCount, 2);
});

void test('local embedding client times out a request without relying on caller cancellation', async (t) => {
  const server = createServer((request) => {
    request.resume();
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const client = createLocalEmbeddingClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'local-test',
    dimensions: 3,
    requestTimeoutMilliseconds: 25,
  });

  await assert.rejects(
    () => client.embedTexts(['timeout'], AbortSignal.timeout(250)),
    /Recall embedding request timed out after 25 ms at http:\/\/127\.0\.0\.1:\d+\/v1\/embeddings/,
  );
});

void test('local embedding client rejects a non-finite vector value', async (t) => {
  const server = createServer((request, response) => {
    void request;
    response.setHeader('content-type', 'application/json');
    response.end('{"data":[{"index":0,"embedding":[1,2,1e400]}]}');
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');

  const client = createLocalEmbeddingClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'local-test',
    dimensions: 3,
  });

  await assert.rejects(
    () => client.embedTexts(['non-finite']),
    /Recall embedding response invalid/,
  );
});

void test('local embedding client rejects vectors with the wrong dimensions', async (t) => {
  const server = createServer((request, response) => {
    void request;
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ data: [{ index: 0, embedding: [1, 2] }] }));
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');

  const client = createLocalEmbeddingClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'local-test',
    dimensions: 3,
    batchSize: 2,
  });
  await assert.rejects(
    () => client.embedTexts(['wrong width']),
    /Recall embedding dimension mismatch/,
  );
});

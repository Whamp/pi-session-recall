import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createLocalRerankerClient } from './local-reranker-client.js';

const LOCAL_RERANKER_REQUEST_SCHEMA = Type.Object({
  model: Type.String(),
  query: Type.String(),
  documents: Type.Array(Type.String()),
  top_n: Type.Integer(),
});

void test('local reranker rejects an invalid base URL before sending a request', () => {
  assert.throws(
    () => createLocalRerankerClient({ baseUrl: 'not a URL', model: 'qwen3-rerank' }),
    /Recall reranker base URL invalid: not a URL/,
  );
});

void test('local reranker rejects a blank model before sending a request', () => {
  assert.throws(
    () => createLocalRerankerClient({ baseUrl: 'http://reranker.test/v1', model: '   ' }),
    /Recall reranker model invalid: expected a non-blank model name/,
  );
});

void test('local reranker maps relevance scores by returned candidate index', async (t) => {
  const requests: unknown[] = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', (chunk) => {
      body += chunk;
    });
    request.on('end', () => {
      requests.push(JSON.parse(body));
      response.setHeader('content-type', 'application/json');
      response.end(
        JSON.stringify({
          model: 'qwen3-rerank',
          object: 'list',
          usage: { prompt_tokens: 42, total_tokens: 42 },
          results: [
            { index: 1, relevance_score: 0.125 },
            { index: 0, relevance_score: 0.875 },
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

  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });
  const scores = await client.rerankDocuments('source provenance', [
    'Preserve exact source provenance.',
    'The navigation bar is blue.',
  ]);

  assert.deepEqual(scores, [0.875, 0.125]);
  assert.equal(requests.length, 1);
  assert.deepEqual(Value.Parse(LOCAL_RERANKER_REQUEST_SCHEMA, requests[0]), {
    model: 'qwen3-rerank',
    query: 'source provenance',
    documents: ['Preserve exact source provenance.', 'The navigation bar is blue.'],
    top_n: 2,
  });
});

void test('local reranker rejects duplicate candidate indexes', async (t) => {
  const server = createServer((request, response) => {
    request.resume();
    response.setHeader('content-type', 'application/json');
    response.end(
      JSON.stringify({
        model: 'qwen3-rerank',
        object: 'list',
        usage: { prompt_tokens: 42, total_tokens: 42 },
        results: [
          { index: 0, relevance_score: 0.875 },
          { index: 0, relevance_score: 0.125 },
        ],
      }),
    );
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });

  await assert.rejects(
    () => client.rerankDocuments('source provenance', ['first', 'second']),
    /Recall reranker response duplicate candidate index 0/,
  );
});

void test('local reranker rejects an out-of-range candidate index', async (t) => {
  const server = createServer((request, response) => {
    request.resume();
    response.setHeader('content-type', 'application/json');
    response.end(
      JSON.stringify({
        model: 'qwen3-rerank',
        object: 'list',
        usage: { prompt_tokens: 42, total_tokens: 42 },
        results: [
          { index: 0, relevance_score: 0.875 },
          { index: 2, relevance_score: 0.125 },
        ],
      }),
    );
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });

  await assert.rejects(
    () => client.rerankDocuments('source provenance', ['first', 'second']),
    /Recall reranker response candidate index out of range: 2 for 2 documents/,
  );
});

void test('local reranker rejects incomplete candidate coverage', async (t) => {
  const server = createServer((request, response) => {
    request.resume();
    response.setHeader('content-type', 'application/json');
    response.end(
      JSON.stringify({
        model: 'qwen3-rerank',
        object: 'list',
        usage: { prompt_tokens: 21, total_tokens: 21 },
        results: [{ index: 0, relevance_score: 0.875 }],
      }),
    );
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });

  await assert.rejects(
    () => client.rerankDocuments('source provenance', ['first', 'second']),
    /Recall reranker response count mismatch: expected 2, received 1/,
  );
});

void test('local reranker rejects a non-finite relevance score', async (t) => {
  const server = createServer((request, response) => {
    request.resume();
    response.setHeader('content-type', 'application/json');
    response.end(
      '{"model":"qwen3-rerank","object":"list","usage":{"prompt_tokens":21,"total_tokens":21},"results":[{"index":0,"relevance_score":1e400}]}',
    );
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });

  await assert.rejects(
    () => client.rerankDocuments('source provenance', ['first']),
    /Recall reranker response invalid/,
  );
});

void test('local reranker reports transport unavailability with endpoint context', async () => {
  const server = createServer();
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  await new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    });
  });
  const client = createLocalRerankerClient({
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
    model: 'qwen3-rerank',
  });

  await assert.rejects(
    () => client.rerankDocuments('source provenance', ['first']),
    /^Error: Recall reranker request failed at http:\/\/127\.0\.0\.1:\d+\/v1\/rerank:/,
  );
});

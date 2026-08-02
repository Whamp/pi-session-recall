import assert from 'node:assert/strict';
import test from 'node:test';

import { isUnknownRecord } from './is-unknown-record.js';
import { createOctenHttpEmbeddingProvider } from './octen-http-embedding-provider.js';

void test('Octen provider transforms native query and document vectors through one stored-prefix path', async (t) => {
  const requests: unknown[] = [];
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (_input, init) => {
    if (typeof init?.body !== 'string') {
      throw new Error('Octen test request body missing');
    }
    const body: unknown = JSON.parse(init.body);
    if (
      !isUnknownRecord(body) ||
      !Array.isArray(body.input) ||
      !body.input.every((value) => typeof value === 'string')
    ) {
      throw new Error('Octen test request inputs invalid');
    }
    requests.push(body);
    const inputs: string[] = body.input;
    return new Response(
      JSON.stringify({
        data: inputs.map((input, index) => ({
          index,
          embedding: input.includes('query') ? [3, 4, 9, 9] : [4, 3, 8, 8],
        })),
      }),
      { status: 200, headers: { 'content-type': 'application/json' } },
    );
  };
  const provider = createOctenHttpEmbeddingProvider({
    baseUrl: 'http://127.0.0.1:8090/v1',
    model: 'octen-embed',
    nativeDimensions: 4,
    storedDimensions: 2,
    batchSize: 8,
  });

  assert.deepEqual(await provider.embedQuery('query text'), [Math.fround(0.6), Math.fround(0.8)]);
  assert.deepEqual(await provider.embedDocuments(['document text']), [
    [Math.fround(0.8), Math.fround(0.6)],
  ]);
  assert.deepEqual(requests, [
    { model: 'octen-embed', input: ['query text'] },
    { model: 'octen-embed', input: ['document text'] },
  ]);
});

import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';

import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

void test('llama.cpp HTTP embedding provider rejects a non-HTTP endpoint', () => {
  assert.throws(
    () =>
      createLlamaCppHttpEmbeddingProvider(createRecommendedEmbeddingGemmaModelProfile(), {
        baseUrl: 'file:///models',
      }),
    /llama\.cpp HTTP embedding base URL invalid protocol/u,
  );
});

void test('llama.cpp HTTP embedding provider retries a dropped connection', async (t) => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const expectedVector = [1, ...Array<number>(profile.identity.dimensions - 1).fill(0)];
  let requestCount = 0;
  const server = createServer((request, response) => {
    requestCount += 1;
    if (requestCount === 1) {
      request.socket.destroy();
      return;
    }
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify({ data: [{ index: 0, embedding: expectedVector }] }));
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(() => server.close());
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const provider = createLlamaCppHttpEmbeddingProvider(profile, {
    baseUrl: `http://127.0.0.1:${address.port}/v1`,
  });

  assert.deepEqual(await provider.embedDocuments(['transient connection recovery']), [
    expectedVector,
  ]);
  assert.equal(requestCount, 2);
});

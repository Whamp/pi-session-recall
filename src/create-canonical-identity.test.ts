import assert from 'node:assert/strict';
import test from 'node:test';

import { createCanonicalIdentity } from './create-canonical-identity.js';

void test('canonical identity ignores object key order but preserves result-affecting values', () => {
  const first = createCanonicalIdentity('fixture-v1', {
    endpoint: 'http://reranker.test/v1/rerank',
    timeout: 1_000,
  });
  const equivalent = createCanonicalIdentity('fixture-v1', {
    timeout: 1_000,
    endpoint: 'http://reranker.test/v1/rerank',
  });
  const changed = createCanonicalIdentity('fixture-v1', {
    endpoint: 'http://reranker.test/v1/rerank',
    timeout: 2_000,
  });

  assert.equal(first, equivalent);
  assert.notEqual(first, changed);
});

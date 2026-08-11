import assert from 'node:assert/strict';
import test from 'node:test';

import { LEGACY_V6_ZVEC_IDENTITY } from './legacy-v6-zvec-identity.js';

void test('legacy-v6 rollback keeps the exact frozen production Zvec identity', () => {
  assert.deepEqual(LEGACY_V6_ZVEC_IDENTITY, {
    schemaVersion: 8,
    ftsConfigurationVersion: 2,
    vectorQuantization: 'fp32',
    metric: 'inner-product',
    hnswM: 50,
    hnswEfConstruction: 500,
    hnswEfSearch: 300,
  });
  assert.equal(Object.isFrozen(LEGACY_V6_ZVEC_IDENTITY), true);
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
} from '@zvec/zvec';

import { openLegacyV7FlatZvecCertificationControl } from './legacy-v7-flat-zvec-certification-control.js';
import { parseProjectIdentity } from './resolve-project-identity.js';

void test('legacy-v7 certification control searches its dense-only flat schema', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'legacy-v7-flat-control-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const databasePath = join(root, 'zvec');
  const projectIdentity = parseProjectIdentity('git-origin:github.com/Whamp/example');
  const projectIdentityDigest = createHash('sha256').update(projectIdentity).digest('hex');
  const writer = ZVecCreateAndOpen(
    databasePath,
    new ZVecCollectionSchema({
      name: 'legacy_v7_flat_control',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: 3,
        indexParams: { indexType: ZVecIndexType.FLAT, metricType: ZVecMetricType.IP },
      },
      fields: [{ name: 'projectIdentityDigest', dataType: ZVecDataType.STRING }],
    }),
  );
  writer.upsertSync([
    {
      id: 'matching-project',
      vectors: { embedding: [1, 0, 0] },
      fields: { projectIdentityDigest },
    },
    {
      id: 'other-project',
      vectors: { embedding: [0, 1, 0] },
      fields: { projectIdentityDigest: createHash('sha256').update('other').digest('hex') },
    },
  ]);
  writer.closeSync();

  const control = openLegacyV7FlatZvecCertificationControl({ databasePath, dimensions: 3 });
  t.after(() => control.close());

  assert.deepEqual(control.searchDocumentIds([1, 0, 0], 2), ['matching-project', 'other-project']);
  assert.deepEqual(control.searchDocumentIds([1, 0, 0], 2, projectIdentity), ['matching-project']);
});

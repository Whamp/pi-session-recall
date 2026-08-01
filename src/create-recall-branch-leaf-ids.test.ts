import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecallBranchLeafIdsByEntryId } from './create-recall-branch-leaf-ids.js';

void test('recall branch leaves preserve branching ancestry', () => {
  const result = createRecallBranchLeafIdsByEntryId({
    entryIds: ['root', 'left', 'right', 'left-leaf'],
    parentEntryIds: [null, 'root', 'root', 'left'],
  });

  assert.deepEqual(result.get('root'), ['left-leaf', 'right']);
  assert.deepEqual(result.get('left'), ['left-leaf']);
  assert.deepEqual(result.get('right'), ['right']);
  assert.deepEqual(result.get('left-leaf'), ['left-leaf']);
});

void test(
  'recall branch leaves remain bounded for a five-thousand-entry chain',
  { timeout: 2_000 },
  () => {
    const entryIds = Array.from({ length: 5_000 }, (_, index) => `entry-${index}`);
    const parentEntryIds = entryIds.map((_, index) =>
      index === 0 ? null : (entryIds[index - 1] ?? null),
    );

    const result = createRecallBranchLeafIdsByEntryId({ entryIds, parentEntryIds });

    assert.equal(result.size, entryIds.length);
    assert.deepEqual(result.get('entry-0'), ['entry-4999']);
    assert.deepEqual(result.get('entry-2500'), ['entry-4999']);
    assert.deepEqual(result.get('entry-4999'), ['entry-4999']);
  },
);

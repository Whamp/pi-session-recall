import assert from 'node:assert/strict';
import test from 'node:test';

import { isUnknownRecord } from './is-unknown-record.js';

void test('unknown record narrowing excludes null arrays and primitives', () => {
  assert.equal(isUnknownRecord({ key: 'value' }), true);
  assert.equal(isUnknownRecord(null), false);
  assert.equal(isUnknownRecord([]), false);
  assert.equal(isUnknownRecord('value'), false);
});

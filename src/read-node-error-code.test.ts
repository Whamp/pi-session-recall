import assert from 'node:assert/strict';
import test from 'node:test';

import { readNodeErrorCode } from './read-node-error-code.js';

void test('Node error codes are narrowed without type assertions', () => {
  assert.equal(
    readNodeErrorCode(Object.assign(new Error('missing'), { code: 'ENOENT' })),
    'ENOENT',
  );
  assert.equal(readNodeErrorCode({ code: 404 }), undefined);
  assert.equal(readNodeErrorCode('ENOENT'), undefined);
});

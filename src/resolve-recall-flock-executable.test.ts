import assert from 'node:assert/strict';
import { access, constants } from 'node:fs/promises';
import test from 'node:test';

import { resolveRecallFlockExecutable } from './resolve-recall-flock-executable.js';

void test('resolveRecallFlockExecutable returns an existing executable path', async () => {
  const flockPath = resolveRecallFlockExecutable();
  assert.ok(typeof flockPath === 'string' && flockPath.length > 0, 'returns a non-empty string');
  await access(flockPath, constants.X_OK);
});

void test('resolveRecallFlockExecutable returns the same path on repeated calls', () => {
  const first = resolveRecallFlockExecutable();
  const second = resolveRecallFlockExecutable();
  assert.equal(first, second);
});

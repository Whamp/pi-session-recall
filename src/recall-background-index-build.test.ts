import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { readRecallBackgroundIndexStatusRecord } from './recall-background-index-build.js';

void test('missing background index status means no build has started', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-background-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  assert.equal(await readRecallBackgroundIndexStatusRecord(join(root, 'missing.json')), null);
});

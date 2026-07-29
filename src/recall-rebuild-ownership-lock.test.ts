import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { tryAcquireRecallRebuildOwnershipLock } from './recall-rebuild-ownership-lock.js';

void test('rebuild ownership admits one process and becomes available after release', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-rebuild-ownership-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lockPath = join(directory, 'rebuild.lock');

  const owner = await tryAcquireRecallRebuildOwnershipLock(lockPath);
  assert.ok(owner);
  assert.equal(await tryAcquireRecallRebuildOwnershipLock(lockPath), null);

  await owner.release();
  await owner.release();
  const successor = await tryAcquireRecallRebuildOwnershipLock(lockPath);
  assert.ok(successor);
  await successor.release();
});

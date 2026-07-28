import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { syncRecallDirectory } from './sync-recall-directory.js';

void test('recall directory sync requires an existing directory', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sync-recall-directory-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const existingDirectory = join(directory, 'existing');
  await mkdir(existingDirectory);

  await syncRecallDirectory(existingDirectory);
  await assert.rejects(() => syncRecallDirectory(join(directory, 'missing')), { code: 'ENOENT' });
});

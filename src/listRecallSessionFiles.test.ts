import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { listRecallSessionFiles } from './listRecallSessionFiles.js';

void test('recall session discovery returns sorted nested JSONL files and tolerates a missing root', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-session-files-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const nestedDirectory = join(root, 'nested');
  await mkdir(nestedDirectory);
  await writeFile(join(root, 'z.jsonl'), '{}\n');
  await writeFile(join(nestedDirectory, 'a.jsonl'), '{}\n');
  await writeFile(join(root, 'not-a-session.txt'), '{}\n');

  assert.deepEqual(await listRecallSessionFiles(root), [
    join(nestedDirectory, 'a.jsonl'),
    join(root, 'z.jsonl'),
  ]);
  assert.deepEqual(await listRecallSessionFiles(join(root, 'missing')), []);
});

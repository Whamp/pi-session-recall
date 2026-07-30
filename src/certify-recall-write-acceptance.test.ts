import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { isUnknownRecord } from './is-unknown-record.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const TSX_IMPORT = import.meta.resolve('tsx');
const CERTIFIER_PATH = new URL('./certify-recall-write-acceptance.ts', import.meta.url).pathname;

void test('standalone write certifier refuses evidence without a clean immutable Git candidate', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-write-certifier-owner-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await assert.rejects(
    () =>
      EXEC_FILE_ASYNC(process.execPath, ['--import', TSX_IMPORT, CERTIFIER_PATH], { cwd: root }),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.equal(error.code, 1);
      assert.match(String(error.stderr), /not a git repository/u);
      return true;
    },
  );
});

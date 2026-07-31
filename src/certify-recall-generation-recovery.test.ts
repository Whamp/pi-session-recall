import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import { runRecoveryCertificationCommand } from './certify-recall-generation-recovery.js';
import { isUnknownRecord } from './is-unknown-record.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const TSX_IMPORT = import.meta.resolve('tsx');
const CERTIFIER_PATH = new URL('./certify-recall-generation-recovery.ts', import.meta.url).pathname;

void test('generation recovery certifier rejects a focused test pattern that matches no test', async () => {
  await assert.rejects(
    runRecoveryCertificationCommand({
      name: 'no-match regression',
      executable: process.execPath,
      argumentsList: [
        '--import',
        'tsx',
        '--test',
        '--test-reporter=tap',
        '--test-name-pattern=THIS_PATTERN_MATCHES_NO_TEST_122_REVIEW',
        'src/build-recall-fixed-snapshot-generation.test.ts',
      ],
      projectDirectory: process.cwd(),
      expectedTestNamePattern: /THIS_PATTERN_MATCHES_NO_TEST_122_REVIEW/u,
    }),
    /did not run an expected named test/u,
  );
});

void test('generation recovery certifier refuses evidence without a clean immutable Git candidate', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generation-recovery-certifier-'));
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

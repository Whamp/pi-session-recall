import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import {
  runRecoveryCertificationCommand,
  runRecoveryCertificationGitWhitespaceCheck,
} from './certify-recall-generation-recovery.js';
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
      expectedPassingTestNamePatterns: [/THIS_PATTERN_MATCHES_NO_TEST_122_REVIEW/u],
    }),
    /did not pass an expected named test/u,
  );
});

for (const directive of ['skip', 'todo'] as const) {
  void test(`generation recovery certifier rejects a matching ${directive} focused test`, async (t) => {
    const root = await mkdtemp(join(tmpdir(), `recall-recovery-${directive}-`));
    t.after(() => rm(root, { recursive: true, force: true }));
    const testPath = join(root, `${directive}.test.mjs`);
    const declaration =
      directive === 'skip'
        ? "test.skip('required recovery check', () => {});"
        : "test.todo('required recovery check');";
    await writeFile(testPath, `import test from 'node:test';\n${declaration}\n`);

    await assert.rejects(
      runRecoveryCertificationCommand({
        name: `${directive} regression`,
        executable: process.execPath,
        argumentsList: ['--test', '--test-reporter=tap', testPath],
        projectDirectory: root,
        expectedPassingTestNamePatterns: [/^required recovery check$/u],
      }),
      /did not pass an expected named test/u,
    );
  });
}

void test('generation recovery Git whitespace check covers committed candidate changes', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-recovery-git-diff-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await EXEC_FILE_ASYNC('git', ['init', '--quiet'], { cwd: root });
  await writeFile(join(root, 'evidence.txt'), 'clean evidence\n');
  await EXEC_FILE_ASYNC('git', ['add', 'evidence.txt'], { cwd: root });
  await EXEC_FILE_ASYNC(
    'git',
    [
      '-c',
      'user.name=Recall Test',
      '-c',
      'user.email=recall@example.invalid',
      'commit',
      '-qm',
      'base',
    ],
    { cwd: root },
  );
  const { stdout: baseCommitOutput } = await EXEC_FILE_ASYNC('git', ['rev-parse', 'HEAD'], {
    cwd: root,
  });
  await writeFile(join(root, 'evidence.txt'), 'committed trailing whitespace \n');
  await EXEC_FILE_ASYNC('git', ['add', 'evidence.txt'], { cwd: root });
  await EXEC_FILE_ASYNC(
    'git',
    [
      '-c',
      'user.name=Recall Test',
      '-c',
      'user.email=recall@example.invalid',
      'commit',
      '-qm',
      'candidate',
    ],
    { cwd: root },
  );
  const { stdout: candidateCommitOutput } = await EXEC_FILE_ASYNC('git', ['rev-parse', 'HEAD'], {
    cwd: root,
  });

  await assert.rejects(
    runRecoveryCertificationGitWhitespaceCheck({
      projectDirectory: root,
      baseCommit: baseCommitOutput.trim(),
      candidateCommit: candidateCommitOutput.trim(),
    }),
    /Git whitespace check/u,
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

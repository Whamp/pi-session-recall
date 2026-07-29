import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { removeStaleRecallEvaluationTemporaryFiles } from './recall-evaluation-file-system.js';
import {
  assertRecallEvaluationGitRevisionCurrent,
  readCleanRecallEvaluationGitRevision,
} from './recall-evaluation-git-revision.js';

void test('recall evaluation revision rejects source changes not represented by HEAD', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-evaluation-git-revision-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  execFileSync('git', ['init', '--quiet'], { cwd: directory });
  execFileSync('git', ['config', 'user.email', 'recall-evaluation@example.test'], {
    cwd: directory,
  });
  execFileSync('git', ['config', 'user.name', 'Recall Evaluation'], { cwd: directory });
  await writeFile(join(directory, 'source.ts'), 'export const value = 1;\n');
  execFileSync('git', ['add', 'source.ts'], { cwd: directory });
  execFileSync('git', ['commit', '--quiet', '-m', 'fixture'], { cwd: directory });
  const revision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: directory,
    encoding: 'utf8',
  }).trim();

  assert.equal(readCleanRecallEvaluationGitRevision(directory), revision);

  await writeFile(join(directory, 'source.ts'), 'export const value = 2;\n');
  assert.throws(
    () => readCleanRecallEvaluationGitRevision(directory),
    /Recall evaluation source revision unavailable: worktree has uncommitted changes/u,
  );
});

void test('recall evaluation publication rejects HEAD and tracked-source drift while retaining prior evidence', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-evaluation-current-revision-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  execFileSync('git', ['init', '--quiet'], { cwd: directory });
  execFileSync('git', ['config', 'user.email', 'recall-evaluation@example.test'], {
    cwd: directory,
  });
  execFileSync('git', ['config', 'user.name', 'Recall Evaluation'], { cwd: directory });
  const sourcePath = join(directory, 'source.ts');
  const evidencePath = join(directory, 'evidence.json');
  await writeFile(sourcePath, 'export const value = 1;\n');
  await writeFile(evidencePath, '{"previous":true}\n');
  execFileSync('git', ['add', 'source.ts', 'evidence.json'], { cwd: directory });
  execFileSync('git', ['commit', '--quiet', '-m', 'fixture'], { cwd: directory });
  const capturedRevision = readCleanRecallEvaluationGitRevision(directory);

  await writeFile(sourcePath, 'export const value = 2;\n');
  execFileSync('git', ['add', 'source.ts'], { cwd: directory });
  execFileSync('git', ['commit', '--quiet', '-m', 'drift HEAD'], { cwd: directory });
  assert.throws(
    () => assertRecallEvaluationGitRevisionCurrent(directory, capturedRevision),
    /source revision changed during execution/u,
  );
  assert.equal(await readFile(evidencePath, 'utf8'), '{"previous":true}\n');

  execFileSync('git', ['reset', '--hard', '--quiet', capturedRevision], { cwd: directory });
  await writeFile(sourcePath, 'export const value = 3;\n');
  assert.throws(
    () => assertRecallEvaluationGitRevisionCurrent(directory, capturedRevision),
    /worktree changed during execution/u,
  );
  assert.equal(await readFile(evidencePath, 'utf8'), '{"previous":true}\n');
});

void test('stale publication temp recovery is exact and does not weaken the Git cleanliness gate', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-evaluation-temp-recovery-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  execFileSync('git', ['init', '--quiet'], { cwd: directory });
  execFileSync('git', ['config', 'user.email', 'recall-evaluation@example.test'], {
    cwd: directory,
  });
  execFileSync('git', ['config', 'user.name', 'Recall Evaluation'], { cwd: directory });
  const evidencePath = join(directory, 'evidence.json');
  await writeFile(evidencePath, '{"previous":true}\n');
  execFileSync('git', ['add', 'evidence.json'], { cwd: directory });
  execFileSync('git', ['commit', '--quiet', '-m', 'fixture'], { cwd: directory });
  const staleTemporaryPath = `${evidencePath}.321.123e4567-e89b-42d3-a456-426614174000.tmp`;
  const unrelatedTemporaryPath = join(
    directory,
    'unrelated.321.123e4567-e89b-42d3-a456-426614174000.tmp',
  );
  const unrelatedWorkPath = join(directory, 'notes.txt');
  await writeFile(staleTemporaryPath, 'interrupted publication');
  await writeFile(unrelatedTemporaryPath, 'unrelated temp');
  await writeFile(unrelatedWorkPath, 'untracked work');

  await removeStaleRecallEvaluationTemporaryFiles([evidencePath]);

  await assert.rejects(() => access(staleTemporaryPath));
  await access(unrelatedTemporaryPath);
  assert.throws(
    () => readCleanRecallEvaluationGitRevision(directory),
    /worktree has uncommitted changes/u,
  );

  await rm(unrelatedTemporaryPath);
  await rm(unrelatedWorkPath);
  assert.match(readCleanRecallEvaluationGitRevision(directory), /^[a-f0-9]{40}$/u);
});

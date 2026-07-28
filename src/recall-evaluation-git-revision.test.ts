import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { readCleanRecallEvaluationGitRevision } from './recall-evaluation-git-revision.js';

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

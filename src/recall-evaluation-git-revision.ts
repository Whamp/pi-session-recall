import { execFileSync } from 'node:child_process';

/** Returns the exact clean Git revision that recall evaluation evidence may claim as its source. */
export function readCleanRecallEvaluationGitRevision(projectDirectory: string): string {
  const worktreeStatus = execFileSync('git', ['status', '--porcelain', '--untracked-files=all'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
  if (worktreeStatus) {
    throw new Error(
      'Recall evaluation source revision unavailable: worktree has uncommitted changes; commit or remove them before generating evidence',
    );
  }
  return execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
}

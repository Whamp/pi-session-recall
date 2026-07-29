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

/** Revalidates HEAD and a fully clean worktree; output files receive no pre-publication exception. */
export function assertRecallEvaluationGitRevisionCurrent(
  projectDirectory: string,
  expectedRevision: string,
): void {
  const currentRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
  if (currentRevision !== expectedRevision) {
    throw new Error(
      'Recall evaluation publication rejected: source revision changed during execution',
    );
  }
  const worktreeStatus = execFileSync('git', ['status', '--porcelain', '--untracked-files=all'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
  if (worktreeStatus) {
    throw new Error('Recall evaluation publication rejected: worktree changed during execution');
  }
}

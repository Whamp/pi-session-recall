import { execFileSync } from 'node:child_process';

function readRecallEvaluationWorktreeStatus(projectDirectory: string): string {
  return execFileSync('git', ['status', '--porcelain', '--untracked-files=all'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
}

function readRecallEvaluationHead(projectDirectory: string): string {
  return execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: projectDirectory,
    encoding: 'utf8',
  }).trim();
}

/** Returns the exact clean Git revision that recall evaluation evidence may claim as its source. */
export function readCleanRecallEvaluationGitRevision(projectDirectory: string): string {
  if (readRecallEvaluationWorktreeStatus(projectDirectory)) {
    throw new Error(
      'Recall evaluation source revision unavailable: worktree has uncommitted changes; commit or remove them before generating evidence',
    );
  }
  return readRecallEvaluationHead(projectDirectory);
}

/** Revalidates HEAD and a fully clean worktree; output files receive no pre-publication exception. */
export function assertRecallEvaluationGitRevisionCurrent(
  projectDirectory: string,
  expectedRevision: string,
): void {
  if (readRecallEvaluationHead(projectDirectory) !== expectedRevision) {
    throw new Error(
      'Recall evaluation publication rejected: source revision changed during execution',
    );
  }
  if (readRecallEvaluationWorktreeStatus(projectDirectory)) {
    throw new Error('Recall evaluation publication rejected: worktree changed during execution');
  }
}

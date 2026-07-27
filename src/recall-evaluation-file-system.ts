import { randomUUID } from 'node:crypto';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, relative } from 'node:path';

/** Reports whether a candidate path is the evaluation root or one of its descendants. */
export function isPathInsideRecallEvaluationArea(
  evaluationRootPath: string,
  candidatePath: string,
): boolean {
  const pathFromRoot = relative(evaluationRootPath, candidatePath);
  return pathFromRoot === '' || (!pathFromRoot.startsWith('..') && !isAbsolute(pathFromRoot));
}

/** Atomically replaces one publishable evaluation evidence file without partial writes. */
export async function writeAtomicRecallEvaluationFile(
  path: string,
  content: string,
): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, content, { encoding: 'utf8', flag: 'wx' });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

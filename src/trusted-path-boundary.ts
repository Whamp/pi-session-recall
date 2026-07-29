import { realpath } from 'node:fs/promises';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';

import { readNodeErrorCode } from './read-node-error-code.js';

/** Resolves symlinks through the nearest existing ancestor while retaining missing descendants. */
export async function resolveCanonicalPathBoundary(candidatePath: string): Promise<string> {
  const missingSegments: string[] = [];
  let existingAncestorPath = candidatePath;
  while (true) {
    try {
      return resolve(await realpath(existingAncestorPath), ...missingSegments);
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ENOENT') {
        throw error;
      }
      const parentPath = dirname(existingAncestorPath);
      if (parentPath === existingAncestorPath) {
        throw error;
      }
      missingSegments.unshift(basename(existingAncestorPath));
      existingAncestorPath = parentPath;
    }
  }
}

/** Tests canonical path containment using path components rather than string prefixes. */
export function isCanonicalPathWithinBoundary(candidatePath: string, rootPath: string): boolean {
  const rootRelativePath = relative(rootPath, candidatePath);
  return (
    rootRelativePath !== '' &&
    rootRelativePath !== '..' &&
    !rootRelativePath.startsWith(`..${sep}`) &&
    !isAbsolute(rootRelativePath)
  );
}

import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, normalize, resolve } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const physicalSessionIgnoreStateSchema = Type.Object(
  {
    version: Type.Literal(1),
    ignoredPhysicalSessionPaths: Type.Array(Type.String()),
  },
  { additionalProperties: false },
);

interface PhysicalSessionIgnoreState {
  version: 1;
  ignoredPhysicalSessionPaths: string[];
}

function assertCanonicalIgnoredPhysicalSessionPaths(paths: readonly string[]): void {
  for (const physicalSessionPath of paths) {
    if (
      !isAbsolute(physicalSessionPath) ||
      normalize(physicalSessionPath) !== physicalSessionPath
    ) {
      throw new Error(
        `Physical session ignore state contains noncanonical path: ${physicalSessionPath}`,
      );
    }
  }
  const sortedUniquePaths = [...new Set(paths)].sort();
  if (
    sortedUniquePaths.length !== paths.length ||
    sortedUniquePaths.some((physicalSessionPath, index) => physicalSessionPath !== paths[index])
  ) {
    throw new Error('Physical session ignore state paths must be unique and sorted');
  }
}

async function readPhysicalSessionIgnoreState(
  physicalSessionIgnorePath: string,
): Promise<PhysicalSessionIgnoreState> {
  try {
    const parsed: unknown = JSON.parse(await readFile(physicalSessionIgnorePath, 'utf8'));
    const state = Value.Parse(physicalSessionIgnoreStateSchema, parsed);
    assertCanonicalIgnoredPhysicalSessionPaths(state.ignoredPhysicalSessionPaths);
    return state;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return { version: 1, ignoredPhysicalSessionPaths: [] };
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Physical session ignore state invalid at ${physicalSessionIgnorePath}: ${message}`,
      { cause: error },
    );
  }
}

async function writePhysicalSessionIgnoreState(
  physicalSessionIgnorePath: string,
  ignoredPhysicalSessionPaths: readonly string[],
): Promise<void> {
  await mkdir(dirname(physicalSessionIgnorePath), { recursive: true });
  const temporaryPath = `${physicalSessionIgnorePath}.${process.pid}.tmp`;
  const state: PhysicalSessionIgnoreState = {
    version: 1,
    ignoredPhysicalSessionPaths: [...ignoredPhysicalSessionPaths],
  };
  await writeFile(temporaryPath, `${JSON.stringify(state)}\n`, 'utf8');
  await rename(temporaryPath, physicalSessionIgnorePath);
}

/** Resolves one exact physical session path lexically against the supplied base directory. */
export function normalizePhysicalSessionPath(baseDirectory: string, inputPath: string): string {
  return resolve(baseDirectory, inputPath);
}

/** Lists the validated, sorted exact physical session paths excluded from index maintenance. */
export async function listIgnoredPhysicalSessionPaths(
  physicalSessionIgnorePath: string,
): Promise<readonly string[]> {
  const state = await readPhysicalSessionIgnoreState(physicalSessionIgnorePath);
  return [...state.ignoredPhysicalSessionPaths];
}

/** Persists one canonical exact physical session path; returns false when it was already ignored. */
export async function addIgnoredPhysicalSessionPath(
  physicalSessionIgnorePath: string,
  normalizedSessionPath: string,
): Promise<boolean> {
  assertCanonicalIgnoredPhysicalSessionPaths([normalizedSessionPath]);
  const state = await readPhysicalSessionIgnoreState(physicalSessionIgnorePath);
  if (state.ignoredPhysicalSessionPaths.includes(normalizedSessionPath)) {
    return false;
  }
  const paths = [...state.ignoredPhysicalSessionPaths, normalizedSessionPath].sort();
  await writePhysicalSessionIgnoreState(physicalSessionIgnorePath, paths);
  return true;
}

/** Removes one canonical exact physical session path; returns false when it was not ignored. */
export async function removeIgnoredPhysicalSessionPath(
  physicalSessionIgnorePath: string,
  normalizedSessionPath: string,
): Promise<boolean> {
  assertCanonicalIgnoredPhysicalSessionPaths([normalizedSessionPath]);
  const state = await readPhysicalSessionIgnoreState(physicalSessionIgnorePath);
  if (!state.ignoredPhysicalSessionPaths.includes(normalizedSessionPath)) {
    return false;
  }
  await writePhysicalSessionIgnoreState(
    physicalSessionIgnorePath,
    state.ignoredPhysicalSessionPaths.filter(
      (physicalSessionPath) => physicalSessionPath !== normalizedSessionPath,
    ),
  );
  return true;
}

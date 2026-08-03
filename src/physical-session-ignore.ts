import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, normalize, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const PHYSICAL_SESSION_IGNORE_LOCK_RETRY_MS = 10;
const PHYSICAL_SESSION_IGNORE_STATE_SCHEMA = Type.Object(
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

async function acquirePhysicalSessionIgnoreStateLock(
  physicalSessionIgnoreStatePath: string,
): Promise<() => Promise<void>> {
  await mkdir(dirname(physicalSessionIgnoreStatePath), { recursive: true });
  const lockPath = `${physicalSessionIgnoreStatePath}.lock`;
  while (true) {
    try {
      await mkdir(lockPath);
      return async () => rm(lockPath, { recursive: true, force: true });
    } catch (error) {
      if (readNodeErrorCode(error) !== 'EEXIST') {
        throw error;
      }
      await sleep(PHYSICAL_SESSION_IGNORE_LOCK_RETRY_MS);
    }
  }
}

async function readPhysicalSessionIgnoreState(
  physicalSessionIgnoreStatePath: string,
): Promise<PhysicalSessionIgnoreState> {
  try {
    const parsed: unknown = JSON.parse(await readFile(physicalSessionIgnoreStatePath, 'utf8'));
    const state = Value.Parse(PHYSICAL_SESSION_IGNORE_STATE_SCHEMA, parsed);
    assertCanonicalIgnoredPhysicalSessionPaths(state.ignoredPhysicalSessionPaths);
    return state;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return { version: 1, ignoredPhysicalSessionPaths: [] };
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Physical session ignore state invalid at ${physicalSessionIgnoreStatePath}: ${message}`,
      { cause: error },
    );
  }
}

async function writePhysicalSessionIgnoreState(
  physicalSessionIgnoreStatePath: string,
  ignoredPhysicalSessionPaths: readonly string[],
): Promise<void> {
  await mkdir(dirname(physicalSessionIgnoreStatePath), { recursive: true });
  const temporaryPath = `${physicalSessionIgnoreStatePath}.${process.pid}.${randomUUID()}.tmp`;
  const state: PhysicalSessionIgnoreState = {
    version: 1,
    ignoredPhysicalSessionPaths: [...ignoredPhysicalSessionPaths],
  };
  try {
    await writeFile(temporaryPath, `${JSON.stringify(state)}\n`, 'utf8');
    await rename(temporaryPath, physicalSessionIgnoreStatePath);
  } finally {
    await rm(temporaryPath, { force: true });
  }
}

/** Resolves one exact physical session path lexically against the supplied base directory. */
export function normalizePhysicalSessionPath(baseDirectory: string, inputPath: string): string {
  return resolve(baseDirectory, inputPath);
}

/** Lists the validated, sorted exact physical session paths excluded from index maintenance. */
export async function listIgnoredPhysicalSessionPaths(
  physicalSessionIgnoreStatePath: string,
): Promise<readonly string[]> {
  const state = await readPhysicalSessionIgnoreState(physicalSessionIgnoreStatePath);
  return [...state.ignoredPhysicalSessionPaths];
}

/** Persists one canonical exact physical session path; returns false when it was already ignored. */
export async function addIgnoredPhysicalSessionPath(
  physicalSessionIgnoreStatePath: string,
  normalizedSessionPath: string,
): Promise<boolean> {
  assertCanonicalIgnoredPhysicalSessionPaths([normalizedSessionPath]);
  const releaseLock = await acquirePhysicalSessionIgnoreStateLock(physicalSessionIgnoreStatePath);
  try {
    const state = await readPhysicalSessionIgnoreState(physicalSessionIgnoreStatePath);
    if (state.ignoredPhysicalSessionPaths.includes(normalizedSessionPath)) {
      return false;
    }
    const paths = [...state.ignoredPhysicalSessionPaths, normalizedSessionPath].sort();
    await writePhysicalSessionIgnoreState(physicalSessionIgnoreStatePath, paths);
    return true;
  } finally {
    await releaseLock();
  }
}

/** Removes one canonical exact physical session path; returns false when it was not ignored. */
export async function removeIgnoredPhysicalSessionPath(
  physicalSessionIgnoreStatePath: string,
  normalizedSessionPath: string,
): Promise<boolean> {
  assertCanonicalIgnoredPhysicalSessionPaths([normalizedSessionPath]);
  const releaseLock = await acquirePhysicalSessionIgnoreStateLock(physicalSessionIgnoreStatePath);
  try {
    const state = await readPhysicalSessionIgnoreState(physicalSessionIgnoreStatePath);
    if (!state.ignoredPhysicalSessionPaths.includes(normalizedSessionPath)) {
      return false;
    }
    await writePhysicalSessionIgnoreState(
      physicalSessionIgnoreStatePath,
      state.ignoredPhysicalSessionPaths.filter(
        (physicalSessionPath) => physicalSessionPath !== normalizedSessionPath,
      ),
    );
    return true;
  } finally {
    await releaseLock();
  }
}

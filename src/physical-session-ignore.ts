import { randomUUID } from 'node:crypto';
import { link, lstat, mkdir, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, normalize, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { readNodeErrorCode } from './read-node-error-code.js';

const PHYSICAL_SESSION_IGNORE_LOCK_RETRY_MS = 250;
const PHYSICAL_SESSION_IGNORE_LOCK_TOKEN_PATTERN = /^[0-9a-f-]{36}$/u;
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

interface PhysicalSessionIgnoreLockOwner {
  processId: number;
  token?: string;
}

interface PhysicalSessionIgnoreLockObservation {
  ownerValue?: string;
  owner?: PhysicalSessionIgnoreLockOwner;
}

function readPhysicalSessionIgnoreLockToken(value: string): string | undefined {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || !('token' in parsed)) {
      return undefined;
    }
    const token = parsed.token;
    return typeof token === 'string' && PHYSICAL_SESSION_IGNORE_LOCK_TOKEN_PATTERN.test(token)
      ? token
      : undefined;
  } catch {
    return undefined;
  }
}

function readPhysicalSessionIgnoreLockOwner(
  value: string,
): PhysicalSessionIgnoreLockOwner | undefined {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || !('pid' in parsed)) {
      return undefined;
    }
    const processId = parsed.pid;
    if (typeof processId !== 'number' || !Number.isSafeInteger(processId) || processId <= 0) {
      return undefined;
    }
    if (!('token' in parsed)) {
      return { processId };
    }
    const token = parsed.token;
    if (typeof token !== 'string' || !PHYSICAL_SESSION_IGNORE_LOCK_TOKEN_PATTERN.test(token)) {
      return undefined;
    }
    return { processId, token };
  } catch {
    return undefined;
  }
}

function isPhysicalSessionIgnoreLockOwnerAlive(processId: number): boolean {
  try {
    process.kill(processId, 0);
    return true;
  } catch (error) {
    const errorCode = readNodeErrorCode(error);
    if (errorCode === 'ESRCH') {
      return false;
    }
    if (errorCode === 'EPERM') {
      return true;
    }
    throw error;
  }
}

async function observePhysicalSessionIgnoreLock(
  lockPath: string,
): Promise<PhysicalSessionIgnoreLockObservation | undefined> {
  try {
    const lockStats = await lstat(lockPath);
    if (!lockStats.isDirectory()) {
      throw new Error(`Physical session ignore lock is not a directory at ${lockPath}`);
    }
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return undefined;
    }
    throw error;
  }

  try {
    const ownerValue = await readFile(`${lockPath}/owner.json`, 'utf8');
    const owner = readPhysicalSessionIgnoreLockOwner(ownerValue);
    return owner === undefined ? { ownerValue } : { ownerValue, owner };
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return {};
    }
    throw error;
  }
}

async function readPhysicalSessionIgnoreDirectoryRecoveryToken(
  recoveryPath: string,
): Promise<string | undefined> {
  let recoveryValue: string;
  try {
    recoveryValue = await readFile(recoveryPath, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return undefined;
    }
    throw error;
  }
  const token = readPhysicalSessionIgnoreLockToken(recoveryValue);
  if (token === undefined) {
    throw new Error(`Physical session ignore lock recovery metadata invalid at ${recoveryPath}`);
  }
  return token;
}

async function readOrCreatePhysicalSessionIgnoreDirectoryRecoveryToken(
  lockPath: string,
): Promise<string | undefined> {
  const recoveryPath = `${lockPath}/recovery.json`;
  const existingToken = await readPhysicalSessionIgnoreDirectoryRecoveryToken(recoveryPath);
  if (existingToken !== undefined) {
    return existingToken;
  }

  const token = randomUUID();
  const candidatePath = `${lockPath}/recovery.${process.pid}.${token}.candidate`;
  try {
    await writeFile(candidatePath, `${JSON.stringify({ token })}\n`, {
      encoding: 'utf8',
      flag: 'wx',
    });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return undefined;
    }
    throw error;
  }
  try {
    try {
      await link(candidatePath, recoveryPath);
      return token;
    } catch (error) {
      const errorCode = readNodeErrorCode(error);
      if (errorCode === 'ENOENT') {
        return undefined;
      }
      if (errorCode !== 'EEXIST') {
        throw error;
      }
      return readPhysicalSessionIgnoreDirectoryRecoveryToken(recoveryPath);
    }
  } finally {
    await rm(candidatePath, { force: true });
  }
}

async function recoverStalePhysicalSessionIgnoreLock(
  lockPath: string,
  staleObservation: PhysicalSessionIgnoreLockObservation,
): Promise<boolean> {
  const recoveryToken =
    staleObservation.owner?.token ??
    (await readOrCreatePhysicalSessionIgnoreDirectoryRecoveryToken(lockPath));
  if (recoveryToken === undefined) {
    return false;
  }

  const currentObservation = await observePhysicalSessionIgnoreLock(lockPath);
  if (
    currentObservation === undefined ||
    currentObservation.ownerValue !== staleObservation.ownerValue ||
    (currentObservation.owner !== undefined &&
      isPhysicalSessionIgnoreLockOwnerAlive(currentObservation.owner.processId))
  ) {
    return false;
  }

  const recoveredPath = `${lockPath}.recovered.${recoveryToken}`;
  try {
    // Retain the nonempty recovered generation so delayed reclaimers cannot rename a successor.
    await rename(lockPath, recoveredPath);
    return true;
  } catch (error) {
    const errorCode = readNodeErrorCode(error);
    if (errorCode === 'EEXIST' || errorCode === 'ENOENT' || errorCode === 'ENOTEMPTY') {
      return false;
    }
    throw error;
  }
}

async function releasePhysicalSessionIgnoreStateLock(
  lockPath: string,
  ownerValue: string,
  ownerToken: string,
): Promise<void> {
  const observation = await observePhysicalSessionIgnoreLock(lockPath);
  if (observation?.ownerValue !== ownerValue) {
    throw new Error(`Physical session ignore lock ownership changed before release at ${lockPath}`);
  }
  const releasedPath = `${lockPath}.released.${ownerToken}`;
  await rename(lockPath, releasedPath);
  await rm(releasedPath, { recursive: true, force: true });
}

async function acquirePhysicalSessionIgnoreStateLock(
  physicalSessionIgnoreStatePath: string,
): Promise<() => Promise<void>> {
  await mkdir(dirname(physicalSessionIgnoreStatePath), { recursive: true });
  const lockPath = `${physicalSessionIgnoreStatePath}.lock`;
  const ownerToken = randomUUID();
  const ownerValue = `${JSON.stringify({ pid: process.pid, token: ownerToken })}\n`;
  const candidatePath = `${lockPath}.${process.pid}.${ownerToken}.candidate`;
  await mkdir(candidatePath);
  await writeFile(`${candidatePath}/owner.json`, ownerValue, 'utf8');
  let unreadableOwnerCount = 0;
  try {
    while (true) {
      const observation = await observePhysicalSessionIgnoreLock(lockPath);
      if (observation === undefined) {
        try {
          await rename(candidatePath, lockPath);
          return async () =>
            releasePhysicalSessionIgnoreStateLock(lockPath, ownerValue, ownerToken);
        } catch (error) {
          const errorCode = readNodeErrorCode(error);
          if (errorCode !== 'EEXIST' && errorCode !== 'ENOTEMPTY') {
            throw error;
          }
          continue;
        }
      }
      if (observation.owner === undefined) {
        unreadableOwnerCount += 1;
        if (unreadableOwnerCount < 4) {
          await sleep(PHYSICAL_SESSION_IGNORE_LOCK_RETRY_MS);
          continue;
        }
      } else if (isPhysicalSessionIgnoreLockOwnerAlive(observation.owner.processId)) {
        unreadableOwnerCount = 0;
        await sleep(PHYSICAL_SESSION_IGNORE_LOCK_RETRY_MS);
        continue;
      }

      await recoverStalePhysicalSessionIgnoreLock(lockPath, observation);
      unreadableOwnerCount = 0;
    }
  } finally {
    await rm(candidatePath, { recursive: true, force: true });
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

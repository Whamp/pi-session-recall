import { randomUUID } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readdir, readlink, rename, rm, symlink } from 'node:fs/promises';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';

import { assertCurrentRecallIndexManifestLayout } from './recall-index-manifest.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const ACTIVE_DATABASE_LINK_NAME = 'active';
const CANDIDATE_DATABASE_PREFIX = 'candidate-';
const DATABASE_GENERATION_PREFIX = 'generation-';

/** Current-format generation paths that move together during atomic activation. */
export interface RecallDatabasePaths {
  sqliteDatabasePath: string;
  manifestPath: string;
  indexMaintenanceStatusPath: string;
}

/** Configuration needed to resolve current-format Recall database generations. */
export interface RecallDatabaseGenerationConfig extends RecallDatabasePaths {
  databaseGenerationRootPath?: string;
}

/** One isolated candidate database prepared beside the active Recall database. */
export interface RecallDatabaseCandidate {
  directoryPath: string;
  paths: RecallDatabasePaths;
  staleCandidatesRemoved: number;
}

/** One complete Recall database generation waiting for explicit activation. */
export interface StagedRecallDatabase {
  databaseTarget: string;
  directoryPath: string;
  paths: RecallDatabasePaths;
}

function getRecallDatabaseDataDirectory(config: RecallDatabaseGenerationConfig): string {
  if (!config.databaseGenerationRootPath) {
    throw new Error('Recall database generations are not configured');
  }
  return dirname(config.databaseGenerationRootPath);
}

function getRootRecallDatabasePaths(config: RecallDatabaseGenerationConfig): RecallDatabasePaths {
  return {
    sqliteDatabasePath: config.sqliteDatabasePath,
    manifestPath: config.manifestPath,
    indexMaintenanceStatusPath: config.indexMaintenanceStatusPath,
  };
}

function createRecallDatabasePaths(
  config: RecallDatabaseGenerationConfig,
  directoryPath: string,
): RecallDatabasePaths {
  return {
    sqliteDatabasePath: join(directoryPath, basename(config.sqliteDatabasePath)),
    manifestPath: join(directoryPath, basename(config.manifestPath)),
    indexMaintenanceStatusPath: join(directoryPath, basename(config.indexMaintenanceStatusPath)),
  };
}

async function readActiveDatabaseTarget(
  config: RecallDatabaseGenerationConfig,
): Promise<string | null> {
  const activeLinkPath = join(getRecallDatabaseDataDirectory(config), ACTIVE_DATABASE_LINK_NAME);
  try {
    return await readlink(activeLinkPath);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw new Error(`Recall active database pointer invalid at ${activeLinkPath}`, {
      cause: error,
    });
  }
}

function resolveDatabaseTargetDirectory(
  config: RecallDatabaseGenerationConfig,
  target: string,
): string {
  const dataDirectory = getRecallDatabaseDataDirectory(config);
  const targetDirectory = resolve(dataDirectory, target);
  const generationRoot = resolve(config.databaseGenerationRootPath ?? '');
  const relativeGenerationPath = relative(generationRoot, targetDirectory);
  if (
    targetDirectory !== dataDirectory &&
    (relativeGenerationPath.startsWith(`..${sep}`) || relativeGenerationPath === '..')
  ) {
    throw new Error(`Recall database pointer target escapes the managed data directory: ${target}`);
  }
  return targetDirectory;
}

async function replaceActiveDatabaseTarget(
  config: RecallDatabaseGenerationConfig,
  target: string,
): Promise<void> {
  const dataDirectory = getRecallDatabaseDataDirectory(config);
  const activeLinkPath = join(dataDirectory, ACTIVE_DATABASE_LINK_NAME);
  const temporaryLinkPath = join(dataDirectory, `.active-${randomUUID()}.tmp`);
  await mkdir(dataDirectory, { recursive: true });
  try {
    await symlink(target, temporaryLinkPath, 'dir');
    await rename(temporaryLinkPath, activeLinkPath);
  } finally {
    await rm(temporaryLinkPath, { force: true });
  }
}

function assertRequiredRecallDatabaseFiles(
  paths: RecallDatabasePaths,
  directoryPath: string,
  databaseKind: 'candidate' | 'staged',
): void {
  const requiredPaths = [
    paths.sqliteDatabasePath,
    paths.manifestPath,
    paths.indexMaintenanceStatusPath,
  ];
  const missingFileNames = requiredPaths
    .filter((requiredPath) => !existsSync(requiredPath))
    .map((requiredPath) => basename(requiredPath));
  if (missingFileNames.length > 0) {
    throw new Error(
      `Recall ${databaseKind} database incomplete at ${directoryPath}: missing ${missingFileNames.join(', ')}; rebuild with psr index --rebuild.`,
    );
  }
}

async function assertCompleteRecallDatabase(
  paths: RecallDatabasePaths,
  directoryPath: string,
  databaseKind: 'candidate' | 'staged',
): Promise<void> {
  assertRequiredRecallDatabaseFiles(paths, directoryPath, databaseKind);
  await assertCurrentRecallIndexManifestLayout(paths.manifestPath);
}

function resolveStagedRecallDatabase(
  config: RecallDatabaseGenerationConfig,
  databaseTarget: string,
): StagedRecallDatabase {
  const directoryPath = resolveDatabaseTargetDirectory(config, databaseTarget);
  const generationRootPath = resolve(config.databaseGenerationRootPath ?? '');
  if (
    dirname(directoryPath) !== generationRootPath ||
    !basename(directoryPath).startsWith(DATABASE_GENERATION_PREFIX) ||
    relative(getRecallDatabaseDataDirectory(config), directoryPath) !== databaseTarget
  ) {
    throw new Error(`Recall staged database target invalid: ${databaseTarget}`);
  }
  return {
    databaseTarget,
    directoryPath,
    paths: createRecallDatabasePaths(config, directoryPath),
  };
}

/** Resolves the current-format database used by normal indexing and search. */
export async function resolveActiveRecallDatabasePaths(
  config: RecallDatabaseGenerationConfig,
): Promise<RecallDatabasePaths> {
  if (!config.databaseGenerationRootPath) {
    return getRootRecallDatabasePaths(config);
  }
  const activeTarget = await readActiveDatabaseTarget(config);
  if (activeTarget === null) {
    return getRootRecallDatabasePaths(config);
  }
  return createRecallDatabasePaths(config, resolveDatabaseTargetDirectory(config, activeTarget));
}

/** Removes stale candidates and creates one empty candidate beside the active database. */
export async function createRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
): Promise<RecallDatabaseCandidate> {
  if (!config.databaseGenerationRootPath) {
    throw new Error('Recall database generations are not configured');
  }
  await mkdir(config.databaseGenerationRootPath, { recursive: true });
  let staleCandidatesRemoved = 0;
  for (const entry of await readdir(config.databaseGenerationRootPath, {
    withFileTypes: true,
  })) {
    if (entry.isDirectory() && entry.name.startsWith(CANDIDATE_DATABASE_PREFIX)) {
      await rm(join(config.databaseGenerationRootPath, entry.name), {
        recursive: true,
        force: true,
      });
      staleCandidatesRemoved += 1;
    }
  }
  const directoryPath = join(
    config.databaseGenerationRootPath,
    `${CANDIDATE_DATABASE_PREFIX}${randomUUID()}`,
  );
  await mkdir(directoryPath);
  return {
    directoryPath,
    paths: createRecallDatabasePaths(config, directoryPath),
    staleCandidatesRemoved,
  };
}

/** Reopens the sole interrupted candidate so a staged rebuild can continue. */
export async function resumeRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
): Promise<RecallDatabaseCandidate> {
  if (!config.databaseGenerationRootPath) {
    throw new Error('Recall database generations are not configured');
  }
  await mkdir(config.databaseGenerationRootPath, { recursive: true });
  const candidateNames = (await readdir(config.databaseGenerationRootPath, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && entry.name.startsWith(CANDIDATE_DATABASE_PREFIX))
    .map((entry) => entry.name);
  if (candidateNames.length !== 1) {
    throw new Error(
      `Recall candidate resume requires exactly one interrupted candidate; found ${candidateNames.length}`,
    );
  }
  const directoryPath = join(config.databaseGenerationRootPath, candidateNames[0] ?? '');
  return {
    directoryPath,
    paths: createRecallDatabasePaths(config, directoryPath),
    staleCandidatesRemoved: 0,
  };
}

/** Promotes a complete candidate to a durable generation without changing the active pointer. */
export async function stageRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
  candidate: RecallDatabaseCandidate,
): Promise<StagedRecallDatabase> {
  await assertCompleteRecallDatabase(candidate.paths, candidate.directoryPath, 'candidate');
  const directoryPath = join(
    config.databaseGenerationRootPath ?? '',
    `${DATABASE_GENERATION_PREFIX}${basename(candidate.directoryPath).slice(CANDIDATE_DATABASE_PREFIX.length)}`,
  );
  await rename(candidate.directoryPath, directoryPath);
  const databaseTarget = relative(getRecallDatabaseDataDirectory(config), directoryPath);
  return {
    databaseTarget,
    directoryPath,
    paths: createRecallDatabasePaths(config, directoryPath),
  };
}

/** Atomically activates one exact staged current-format generation. */
export async function activateStagedRecallDatabase(
  config: RecallDatabaseGenerationConfig,
  databaseTarget: string,
): Promise<void> {
  const staged = resolveStagedRecallDatabase(config, databaseTarget);
  await assertCompleteRecallDatabase(staged.paths, staged.directoryPath, 'staged');
  const activeTarget = await readActiveDatabaseTarget(config);
  if (activeTarget === staged.databaseTarget) {
    throw new Error(`Recall staged database is already active: ${staged.databaseTarget}`);
  }
  await replaceActiveDatabaseTarget(config, staged.databaseTarget);
}

/** Stages and atomically activates a completed candidate for ordinary rebuilds. */
export async function activateRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
  candidate: RecallDatabaseCandidate,
): Promise<void> {
  const staged = await stageRecallDatabaseCandidate(config, candidate);
  await activateStagedRecallDatabase(config, staged.databaseTarget);
}

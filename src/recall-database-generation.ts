import { existsSync } from 'node:fs';
import {
  mkdir,
  readFile,
  readdir,
  readlink,
  rename,
  rm,
  symlink,
  writeFile,
} from 'node:fs/promises';
import { basename, dirname, join, relative, resolve, sep } from 'node:path';
import { randomUUID } from 'node:crypto';

import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const ACTIVE_DATABASE_LINK_NAME = 'active';
const CANDIDATE_DATABASE_PREFIX = 'candidate-';
const DATABASE_GENERATION_PREFIX = 'generation-';
const PREVIOUS_DATABASE_FILE_NAME = '.previous-database.json';

/** Generation-owned paths that move together when a recall database is activated. */
export interface RecallDatabasePaths {
  databasePath: string;
  catalogPath: string;
  statePath: string;
  manifestPath: string;
  indexMaintenanceStatusPath: string;
}

/** Configuration needed to resolve legacy and generation-owned recall database paths. */
export interface RecallDatabaseGenerationConfig extends RecallDatabasePaths {
  databaseGenerationRootPath?: string;
}

/** One isolated candidate database prepared beside the active recall database. */
export interface RecallDatabaseCandidate {
  directoryPath: string;
  paths: RecallDatabasePaths;
  staleCandidatesRemoved: number;
}

/** Result of atomically making a completed candidate the active recall database. */
export interface RecallDatabaseActivation {
  previousAvailable: boolean;
}

function getRecallDatabaseDataDirectory(config: RecallDatabaseGenerationConfig): string {
  if (!config.databaseGenerationRootPath) {
    throw new Error('Recall database generations are not configured');
  }
  return dirname(config.databaseGenerationRootPath);
}

function getLegacyRecallDatabasePaths(config: RecallDatabaseGenerationConfig): RecallDatabasePaths {
  return {
    databasePath: config.databasePath,
    catalogPath: config.catalogPath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    indexMaintenanceStatusPath: config.indexMaintenanceStatusPath,
  };
}

function createRecallDatabasePaths(
  config: RecallDatabaseGenerationConfig,
  directoryPath: string,
): RecallDatabasePaths {
  return {
    databasePath: join(directoryPath, basename(config.databasePath)),
    catalogPath: join(directoryPath, basename(config.catalogPath)),
    statePath: join(directoryPath, basename(config.statePath)),
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

function hasLegacyRecallDatabase(config: RecallDatabaseGenerationConfig): boolean {
  return (
    existsSync(config.databasePath) &&
    (existsSync(config.catalogPath) || existsSync(config.statePath)) &&
    existsSync(config.manifestPath)
  );
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

function readPreviousDatabaseTarget(value: string, path: string): string {
  const parsed: unknown = JSON.parse(value);
  if (
    !isUnknownRecord(parsed) ||
    parsed.version !== 1 ||
    typeof parsed.target !== 'string' ||
    !parsed.target
  ) {
    throw new Error(`Recall previous database record invalid at ${path}`);
  }
  return parsed.target;
}

async function assertDatabaseTargetExists(
  config: RecallDatabaseGenerationConfig,
  target: string,
): Promise<void> {
  const targetDirectory = resolveDatabaseTargetDirectory(config, target);
  const paths = createRecallDatabasePaths(config, targetDirectory);
  if (!existsSync(paths.databasePath) || !existsSync(paths.manifestPath)) {
    throw new Error('Previous recall database is missing and cannot be restored');
  }
}

/** Resolves the database used by normal indexing and search, falling back to the legacy layout. */
export async function resolveActiveRecallDatabasePaths(
  config: RecallDatabaseGenerationConfig,
): Promise<RecallDatabasePaths> {
  if (!config.databaseGenerationRootPath) {
    return getLegacyRecallDatabasePaths(config);
  }
  const activeTarget = await readActiveDatabaseTarget(config);
  if (activeTarget === null) {
    return getLegacyRecallDatabasePaths(config);
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

/** Atomically activates a completed candidate while recording its immediately previous database. */
export async function activateRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
  candidate: RecallDatabaseCandidate,
): Promise<RecallDatabaseActivation> {
  const activeTarget = await readActiveDatabaseTarget(config);
  const previousTarget = activeTarget ?? (hasLegacyRecallDatabase(config) ? '.' : null);
  if (
    !existsSync(candidate.paths.databasePath) ||
    !existsSync(candidate.paths.catalogPath) ||
    !existsSync(candidate.paths.manifestPath) ||
    !existsSync(candidate.paths.indexMaintenanceStatusPath)
  ) {
    throw new Error(`Recall candidate database incomplete at ${candidate.directoryPath}`);
  }
  if (previousTarget !== null) {
    await writeFile(
      join(candidate.directoryPath, PREVIOUS_DATABASE_FILE_NAME),
      `${JSON.stringify({ version: 1, target: previousTarget }, null, 2)}\n`,
      'utf8',
    );
  }
  const generationDirectory = join(
    config.databaseGenerationRootPath ?? '',
    `${DATABASE_GENERATION_PREFIX}${basename(candidate.directoryPath).slice(CANDIDATE_DATABASE_PREFIX.length)}`,
  );
  await rename(candidate.directoryPath, generationDirectory);
  const generationTarget = relative(getRecallDatabaseDataDirectory(config), generationDirectory);
  await replaceActiveDatabaseTarget(config, generationTarget);
  return { previousAvailable: previousTarget !== null };
}

/** Atomically restores the database recorded as previous by the active generation. */
export async function restorePreviousRecallDatabase(
  config: RecallDatabaseGenerationConfig,
): Promise<void> {
  if (!config.databaseGenerationRootPath) {
    throw new Error('No previous recall database is available to restore');
  }
  const activeTarget = await readActiveDatabaseTarget(config);
  if (activeTarget === null) {
    throw new Error('No previous recall database is available to restore');
  }
  const activeDirectory = resolveDatabaseTargetDirectory(config, activeTarget);
  const previousRecordPath = join(activeDirectory, PREVIOUS_DATABASE_FILE_NAME);
  let previousTarget: string;
  try {
    previousTarget = readPreviousDatabaseTarget(
      await readFile(previousRecordPath, 'utf8'),
      previousRecordPath,
    );
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      throw new Error('No previous recall database is available to restore');
    }
    throw error;
  }
  await assertDatabaseTargetExists(config, previousTarget);
  await replaceActiveDatabaseTarget(config, previousTarget);
}

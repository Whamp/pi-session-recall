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

import { RecallIndexManifestLayout } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { detectRecallIndexManifestLayout } from './recall-index-manifest.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const ACTIVE_DATABASE_LINK_NAME = 'active';
const CANDIDATE_DATABASE_PREFIX = 'candidate-';
const DATABASE_GENERATION_PREFIX = 'generation-';
const PREVIOUS_DATABASE_FILE_NAME = '.previous-database.json';

/** Generation-owned paths that move together when a recall database is activated. */
export interface RecallDatabasePaths {
  /** Unified version 8 SQLite Recall database path. */
  sqliteDatabasePath: string;
  /** Temporary version 6 Zvec path retained for rollback detection. */
  legacyV6ZvecDatabasePath: string;
  /** Temporary version 6 index-state path retained for rollback detection. */
  legacyV6StatePath: string;
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

/** One completed recall database generation waiting for explicit activation. */
export interface StagedRecallDatabase {
  databaseTarget: string;
  directoryPath: string;
  paths: RecallDatabasePaths;
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
    sqliteDatabasePath: config.sqliteDatabasePath,
    legacyV6ZvecDatabasePath: config.legacyV6ZvecDatabasePath,
    legacyV6StatePath: config.legacyV6StatePath,
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
    legacyV6ZvecDatabasePath: join(directoryPath, basename(config.legacyV6ZvecDatabasePath)),
    legacyV6StatePath: join(directoryPath, basename(config.legacyV6StatePath)),
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

async function hasLegacyVersion6RecallDatabase(
  config: RecallDatabaseGenerationConfig,
): Promise<boolean> {
  if (
    !existsSync(config.legacyV6ZvecDatabasePath) ||
    !existsSync(config.legacyV6StatePath) ||
    !existsSync(config.manifestPath)
  ) {
    return false;
  }
  return (
    (await detectRecallIndexManifestLayout(config.manifestPath)) ===
    RecallIndexManifestLayout.LEGACY_V6
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

function assertRequiredRecallDatabaseFiles(
  requiredPaths: readonly string[],
  directoryPath: string,
  databaseKind: 'candidate' | 'staged' | 'Previous',
): void {
  const missingFileNames = requiredPaths
    .filter((requiredPath) => !existsSync(requiredPath))
    .map((requiredPath) => basename(requiredPath));
  if (missingFileNames.length > 0) {
    const databaseLabel =
      databaseKind === 'Previous' ? 'Previous recall database' : `Recall ${databaseKind} database`;
    throw new Error(
      `${databaseLabel} incomplete at ${directoryPath}: missing ${missingFileNames.join(', ')}; rebuild with psr index --rebuild.`,
    );
  }
}

async function assertCompleteVersion8RecallDatabase(
  paths: RecallDatabasePaths,
  directoryPath: string,
  databaseKind: 'candidate' | 'staged',
): Promise<void> {
  assertRequiredRecallDatabaseFiles(
    [paths.sqliteDatabasePath, paths.manifestPath, paths.indexMaintenanceStatusPath],
    directoryPath,
    databaseKind,
  );
  const layout = await detectRecallIndexManifestLayout(paths.manifestPath);
  if (layout !== RecallIndexManifestLayout.UNIFIED_SQLITE_V8) {
    throw new Error(
      `Recall ${databaseKind} database layout ${layout} at ${directoryPath} is incompatible; rebuild with psr index --rebuild.`,
    );
  }
}

async function assertPreviousDatabaseTargetComplete(
  config: RecallDatabaseGenerationConfig,
  target: string,
): Promise<void> {
  const targetDirectory = resolveDatabaseTargetDirectory(config, target);
  const paths = createRecallDatabasePaths(config, targetDirectory);
  if (!existsSync(paths.manifestPath)) {
    assertRequiredRecallDatabaseFiles([paths.manifestPath], targetDirectory, 'Previous');
  }
  const layout = await detectRecallIndexManifestLayout(paths.manifestPath);
  if (layout === RecallIndexManifestLayout.LEGACY_V6) {
    assertRequiredRecallDatabaseFiles(
      [paths.legacyV6ZvecDatabasePath, paths.legacyV6StatePath, paths.manifestPath],
      targetDirectory,
      'Previous',
    );
    return;
  }
  assertRequiredRecallDatabaseFiles(
    [paths.sqliteDatabasePath, paths.manifestPath, paths.indexMaintenanceStatusPath],
    targetDirectory,
    'Previous',
  );
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

/** Reopens the sole interrupted candidate so a staged production rebuild can continue. */
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
  await assertCompleteVersion8RecallDatabase(candidate.paths, candidate.directoryPath, 'candidate');
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

/** Atomically activates one exact staged generation while recording the current database. */
export async function activateStagedRecallDatabase(
  config: RecallDatabaseGenerationConfig,
  databaseTarget: string,
): Promise<RecallDatabaseActivation> {
  const staged = resolveStagedRecallDatabase(config, databaseTarget);
  await assertCompleteVersion8RecallDatabase(staged.paths, staged.directoryPath, 'staged');
  const activeTarget = await readActiveDatabaseTarget(config);
  if (activeTarget === staged.databaseTarget) {
    throw new Error(`Recall staged database is already active: ${staged.databaseTarget}`);
  }
  const previousTarget =
    activeTarget ?? ((await hasLegacyVersion6RecallDatabase(config)) ? '.' : null);
  if (previousTarget !== null) {
    await writeFile(
      join(staged.directoryPath, PREVIOUS_DATABASE_FILE_NAME),
      `${JSON.stringify({ version: 1, target: previousTarget }, null, 2)}\n`,
      'utf8',
    );
  }
  await replaceActiveDatabaseTarget(config, staged.databaseTarget);
  return { previousAvailable: previousTarget !== null };
}

/** Stages and atomically activates a completed candidate for ordinary rebuilds. */
export async function activateRecallDatabaseCandidate(
  config: RecallDatabaseGenerationConfig,
  candidate: RecallDatabaseCandidate,
): Promise<RecallDatabaseActivation> {
  const staged = await stageRecallDatabaseCandidate(config, candidate);
  return activateStagedRecallDatabase(config, staged.databaseTarget);
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
  await assertPreviousDatabaseTargetComplete(config, previousTarget);
  await replaceActiveDatabaseTarget(config, previousTarget);
}

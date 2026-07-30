import { randomUUID } from 'node:crypto';
import { lstatSync, realpathSync } from 'node:fs';
import { mkdir, open, readdir, rename, rm } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

import { RecallDiagnosticsMode } from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type {
  RecallConversationConfig,
  RecallSearchCandidateLimits,
} from './recall-conversation-config.js';

/** Reports whether a candidate path is the evaluation root or one of its descendants. */
export function isPathInsideRecallEvaluationArea(
  evaluationRootPath: string,
  candidatePath: string,
): boolean {
  const pathFromRoot = relative(evaluationRootPath, candidatePath);
  return pathFromRoot === '' || (!pathFromRoot.startsWith('..') && !isAbsolute(pathFromRoot));
}

/** Safe scalar overrides for deterministic private-evaluation embedding identity. */
export interface PrivateRecallEvaluationEmbeddingIdentity {
  model: string;
  servedModelId: string;
  artifact: string;
  quantization: string;
  pooling: string;
  dimensions: number;
}

/** Inputs for a service config whose writable paths are owned only by one evaluation work area. */
export interface PrivateRecallEvaluationConfigOptions {
  baseConfig: RecallConversationConfig;
  evaluationRootDirectory: string;
  workDirectory: string;
  sessionsDirectory: string;
  immutableInputPaths: readonly string[];
  candidateLimits: RecallSearchCandidateLimits;
  embeddingIdentity?: PrivateRecallEvaluationEmbeddingIdentity;
}

function assertPrivateRecallEvaluationPathHasNoSymlinkAncestor(
  evaluationRootDirectory: string,
  workDirectory: string,
): void {
  const physicalEvaluationRoot = realpathSync(evaluationRootDirectory);
  if (physicalEvaluationRoot !== evaluationRootDirectory) {
    throw new Error('Private recall evaluation root contains a symbolic link');
  }
  let candidatePath = evaluationRootDirectory;
  const workPathFromRoot = relative(evaluationRootDirectory, workDirectory);
  for (const pathSegment of workPathFromRoot.split(sep)) {
    candidatePath = join(candidatePath, pathSegment);
    try {
      if (lstatSync(candidatePath).isSymbolicLink()) {
        throw new Error('Private recall evaluation work area contains a symbolic link');
      }
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        break;
      }
      throw error;
    }
  }
}

function resolvePhysicalPathFromExistingAncestor(path: string): string {
  let existingAncestor = resolve(path);
  const missingPathSegments: string[] = [];
  while (true) {
    try {
      return resolve(realpathSync(existingAncestor), ...missingPathSegments);
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ENOENT') {
        throw error;
      }
      const parentDirectory = dirname(existingAncestor);
      if (parentDirectory === existingAncestor) {
        throw error;
      }
      missingPathSegments.unshift(basename(existingAncestor));
      existingAncestor = parentDirectory;
    }
  }
}

function readProductionRecallProtectedPaths(
  config: RecallConversationConfig,
): Readonly<Record<string, string>> {
  return {
    sessionsDirectory: config.sessionsDirectory,
    dataDirectory: config.dataDirectory,
    databasePath: config.databasePath,
    projectionDatabasePath: config.projectionDatabasePath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    tokenizerCacheDirectory: config.tokenizerCacheDirectory,
    lockPath: config.lockPath,
    diagnosticLogPath: config.diagnosticLogPath,
    retainedDiagnosticLogPath: config.retainedDiagnosticLogPath,
    markerSpoolDirectory: config.markerSpoolDirectory,
    markerQuarantineDirectory: config.markerQuarantineDirectory,
    markerControlDirectory: config.markerControlDirectory,
    workerOwnershipLockPath: config.workerOwnershipLockPath,
    generationRootDirectory: config.generationRootDirectory,
    activeGenerationPointerPath: config.activeGenerationPointerPath,
    generationRegistryPath: config.generationRegistryPath,
    backlogSummaryPath: config.backlogSummaryPath,
    incrementalDiagnosticLogPath: config.incrementalDiagnosticLogPath,
    backgroundIndexStatusPath:
      config.backgroundIndexStatusPath ??
      join(config.dataDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath:
      config.backgroundIndexRequestPath ??
      join(config.dataDirectory, 'background-index-request.json'),
  };
}

function assertPrivateRecallEvaluationWritablePaths(
  workDirectory: string,
  immutableInputPaths: readonly string[],
  writablePaths: Readonly<Record<string, string>>,
): void {
  for (const [pathName, writablePath] of Object.entries(writablePaths)) {
    if (!isPathInsideRecallEvaluationArea(workDirectory, writablePath)) {
      throw new Error(`Private recall evaluation writable path escaped its work area: ${pathName}`);
    }
    for (const immutableInputPath of immutableInputPaths) {
      if (
        isPathInsideRecallEvaluationArea(immutableInputPath, writablePath) ||
        isPathInsideRecallEvaluationArea(writablePath, immutableInputPath)
      ) {
        throw new Error(
          `Private recall evaluation writable path overlaps an immutable input: ${pathName}`,
        );
      }
    }
  }
}

/** Derives and validates every private-evaluation writable and selector path from its work area. */
export function createPrivateRecallEvaluationConfig(
  options: PrivateRecallEvaluationConfigOptions,
): RecallConversationConfig {
  const evaluationRootDirectory = resolve(options.evaluationRootDirectory);
  const workDirectory = resolve(options.workDirectory);
  if (
    workDirectory === evaluationRootDirectory ||
    !isPathInsideRecallEvaluationArea(evaluationRootDirectory, workDirectory)
  ) {
    throw new Error('Private recall evaluation work area escaped its validated evaluation root');
  }
  assertPrivateRecallEvaluationPathHasNoSymlinkAncestor(evaluationRootDirectory, workDirectory);
  const physicalWorkDirectory = resolvePhysicalPathFromExistingAncestor(workDirectory);
  for (const [pathName, productionPath] of Object.entries(
    readProductionRecallProtectedPaths(options.baseConfig),
  )) {
    const physicalProductionPath = resolvePhysicalPathFromExistingAncestor(productionPath);
    if (
      isPathInsideRecallEvaluationArea(physicalProductionPath, physicalWorkDirectory) ||
      isPathInsideRecallEvaluationArea(physicalWorkDirectory, physicalProductionPath)
    ) {
      throw new Error(
        `Private recall evaluation work area overlaps a production path: ${pathName}`,
      );
    }
  }
  const immutableInputPaths = options.immutableInputPaths.map((path) => resolve(path));
  for (const immutableInputPath of immutableInputPaths) {
    if (
      isPathInsideRecallEvaluationArea(immutableInputPath, workDirectory) ||
      isPathInsideRecallEvaluationArea(workDirectory, immutableInputPath)
    ) {
      throw new Error('Private recall evaluation work area overlaps an immutable input');
    }
  }
  const writablePaths = {
    dataDirectory: workDirectory,
    databasePath: resolve(workDirectory, 'zvec'),
    projectionDatabasePath: resolve(workDirectory, 'session-projections'),
    statePath: resolve(workDirectory, 'index-state.json'),
    manifestPath: resolve(workDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: resolve(workDirectory, 'tokenizers'),
    lockPath: resolve(workDirectory, 'operation.lock'),
    diagnosticLogPath: resolve(workDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: resolve(workDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: resolve(workDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: resolve(workDirectory, 'markers', 'quarantine'),
    markerControlDirectory: resolve(workDirectory, 'markers', 'control'),
    workerOwnershipLockPath: resolve(workDirectory, 'incremental-worker.lock'),
    generationRootDirectory: resolve(workDirectory, 'generations'),
    activeGenerationPointerPath: resolve(workDirectory, 'active-generation.json'),
    generationRegistryPath: resolve(workDirectory, 'generation-registry.json'),
    backlogSummaryPath: resolve(workDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: resolve(workDirectory, 'incremental-diagnostics.jsonl'),
    backgroundIndexStatusPath: resolve(workDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath: resolve(workDirectory, 'background-index-request.json'),
  };
  assertPrivateRecallEvaluationWritablePaths(workDirectory, immutableInputPaths, writablePaths);
  const fusedPoolLimit =
    options.candidateLimits.dense +
    options.candidateLimits.lexical +
    options.candidateLimits.identifier;
  const embeddingIdentity = options.embeddingIdentity;
  return {
    ...options.baseConfig,
    sessionsDirectory: resolve(options.sessionsDirectory),
    ...writablePaths,
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    embeddingBaseUrl: options.baseConfig.embeddingBaseUrl,
    embeddingModel: embeddingIdentity?.model ?? options.baseConfig.embeddingModel,
    embeddingServedModelId:
      embeddingIdentity?.servedModelId ?? options.baseConfig.embeddingServedModelId,
    embeddingArtifact: embeddingIdentity?.artifact ?? options.baseConfig.embeddingArtifact,
    embeddingQuantization:
      embeddingIdentity?.quantization ?? options.baseConfig.embeddingQuantization,
    embeddingPooling: embeddingIdentity?.pooling ?? options.baseConfig.embeddingPooling,
    embeddingDimensions: embeddingIdentity?.dimensions ?? options.baseConfig.embeddingDimensions,
    embeddingBatchSize: options.baseConfig.embeddingBatchSize,
    rerankerBaseUrl: options.baseConfig.rerankerBaseUrl,
    rerankerModel: options.baseConfig.rerankerModel,
    ...(options.baseConfig.queryPlannerBaseUrl
      ? { queryPlannerBaseUrl: options.baseConfig.queryPlannerBaseUrl }
      : {}),
    projectLineages: options.baseConfig.projectLineages,
    searchCandidateLimits: { ...options.candidateLimits },
    fusedPoolLimit,
    rerankPoolLimit: fusedPoolLimit,
    ...(options.baseConfig.chunkPolicy
      ? { chunkPolicy: { ...options.baseConfig.chunkPolicy } }
      : {}),
  };
}

/** Minimal file handle contract used to observe durable evaluation publication ordering. */
export interface RecallEvaluationFileHandle {
  writeFile(content: string): Promise<void>;
  sync(): Promise<void>;
  close(): Promise<void>;
}

/** File operations required by durable evaluation publication and interrupted-run recovery. */
export interface RecallEvaluationFileSystem {
  mkdir(path: string): Promise<string | undefined>;
  open(path: string, flags: 'r' | 'wx'): Promise<RecallEvaluationFileHandle>;
  rename(from: string, to: string): Promise<void>;
  rm(path: string): Promise<void>;
  readdir(path: string): Promise<string[]>;
}

const NODE_RECALL_EVALUATION_FILE_SYSTEM: RecallEvaluationFileSystem = {
  async mkdir(path) {
    return mkdir(path, { recursive: true });
  },
  open,
  rename,
  async rm(path) {
    await rm(path, { force: true });
  },
  readdir,
};

function escapeRegularExpression(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/gu, '\\$&');
}

function createRecallEvaluationTemporaryFilePattern(destinationPath: string): RegExp {
  const destinationName = escapeRegularExpression(basename(destinationPath));
  return new RegExp(
    `^${destinationName}\\.\\d+\\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\\.tmp$`,
    'u',
  );
}

function normalizeRecallEvaluationFileError(error: unknown): Error {
  return error instanceof Error
    ? error
    : new Error('Recall evaluation file operation failed with a non-Error value', { cause: error });
}

async function syncRecallEvaluationDirectory(
  directoryPath: string,
  fileSystem: RecallEvaluationFileSystem,
): Promise<void> {
  const handle = await fileSystem.open(directoryPath, 'r');
  let syncError: Error | undefined;
  try {
    await handle.sync();
  } catch (error) {
    syncError = normalizeRecallEvaluationFileError(error);
  }
  try {
    await handle.close();
  } catch (error) {
    if (!syncError) {
      syncError = normalizeRecallEvaluationFileError(error);
    }
  }
  if (syncError) {
    throw syncError;
  }
}

async function persistCreatedRecallEvaluationDirectories(
  destinationDirectory: string,
  firstCreatedDirectory: string | undefined,
  fileSystem: RecallEvaluationFileSystem,
): Promise<void> {
  if (!firstCreatedDirectory) {
    return;
  }
  const firstCreatedPath = resolve(firstCreatedDirectory);
  const destinationPath = resolve(destinationDirectory);
  const remainingPath = relative(firstCreatedPath, destinationPath);
  if (remainingPath.startsWith('..') || isAbsolute(remainingPath)) {
    throw new Error('Recall evaluation directory creation escaped the destination hierarchy');
  }
  await syncRecallEvaluationDirectory(dirname(firstCreatedPath), fileSystem);
  let directoryPath = firstCreatedPath;
  await syncRecallEvaluationDirectory(directoryPath, fileSystem);
  for (const pathSegment of remainingPath.split(sep).filter(Boolean)) {
    directoryPath = join(directoryPath, pathSegment);
    await syncRecallEvaluationDirectory(directoryPath, fileSystem);
  }
}

async function writeAndSyncRecallEvaluationTemporaryFile(
  temporaryPath: string,
  content: string,
  fileSystem: RecallEvaluationFileSystem,
): Promise<void> {
  const handle = await fileSystem.open(temporaryPath, 'wx');
  let writeError: Error | undefined;
  try {
    await handle.writeFile(content);
    await handle.sync();
  } catch (error) {
    writeError = normalizeRecallEvaluationFileError(error);
  }
  try {
    await handle.close();
  } catch (error) {
    if (!writeError) {
      writeError = normalizeRecallEvaluationFileError(error);
    }
  }
  if (writeError) {
    throw writeError;
  }
}

/** Removes only complete writer artifacts for the exact publication destinations supplied. */
export async function removeStaleRecallEvaluationTemporaryFiles(
  destinationPaths: readonly string[],
  fileSystem: RecallEvaluationFileSystem = NODE_RECALL_EVALUATION_FILE_SYSTEM,
): Promise<void> {
  for (const destinationPath of new Set(destinationPaths.map((path) => resolve(path)))) {
    const destinationDirectory = dirname(destinationPath);
    let names: string[];
    try {
      names = await fileSystem.readdir(destinationDirectory);
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        continue;
      }
      throw error;
    }
    const temporaryFilePattern = createRecallEvaluationTemporaryFilePattern(destinationPath);
    for (const name of names) {
      if (temporaryFilePattern.test(name)) {
        await fileSystem.rm(join(destinationDirectory, name));
      }
    }
  }
}

/** Durably replaces one publishable file after syncing its temp and containing directory. */
export async function writeAtomicRecallEvaluationFile(
  path: string,
  content: string,
  fileSystem: RecallEvaluationFileSystem = NODE_RECALL_EVALUATION_FILE_SYSTEM,
): Promise<void> {
  const destinationPath = resolve(path);
  const destinationDirectory = dirname(destinationPath);
  const firstCreatedDirectory = await fileSystem.mkdir(destinationDirectory);
  await persistCreatedRecallEvaluationDirectories(
    destinationDirectory,
    firstCreatedDirectory,
    fileSystem,
  );
  const temporaryPath = `${destinationPath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeAndSyncRecallEvaluationTemporaryFile(temporaryPath, content, fileSystem);
    await fileSystem.rename(temporaryPath, destinationPath);
    await syncRecallEvaluationDirectory(destinationDirectory, fileSystem);
  } catch (error) {
    const publicationError = normalizeRecallEvaluationFileError(error);
    await Promise.allSettled([fileSystem.rm(temporaryPath)]);
    throw publicationError;
  }
}

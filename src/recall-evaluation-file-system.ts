import { randomUUID } from 'node:crypto';
import { lstatSync, realpathSync } from 'node:fs';
import { mkdir, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

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

function readProductionRecallProtectedPaths(
  config: RecallConversationConfig,
): Readonly<Record<string, string>> {
  const generationDataDirectory = dirname(config.manifestPath);
  return {
    sessionsDirectory: config.sessionsDirectory,
    databasePath: config.databasePath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    tokenizerCacheDirectory: config.tokenizerCacheDirectory,
    embeddingCacheDirectory: config.embeddingCacheDirectory,
    lockPath: config.lockPath,
    generationsDirectory:
      config.generationsDirectory ?? join(generationDataDirectory, 'index-generations'),
    activeGenerationPath:
      config.activeGenerationPath ?? join(generationDataDirectory, 'active-generation.json'),
    stagingGenerationPath:
      config.stagingGenerationPath ?? join(generationDataDirectory, 'staging-generation.json'),
    backgroundIndexStatusPath:
      config.backgroundIndexStatusPath ??
      join(generationDataDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath:
      config.backgroundIndexRequestPath ??
      join(generationDataDirectory, 'background-index-request.json'),
    diagnosticLogPath: config.diagnosticLogPath,
    retainedDiagnosticLogPath: config.retainedDiagnosticLogPath,
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
  for (const [pathName, productionPath] of Object.entries(
    readProductionRecallProtectedPaths(options.baseConfig),
  )) {
    const resolvedProductionPath = resolve(productionPath);
    if (
      isPathInsideRecallEvaluationArea(resolvedProductionPath, workDirectory) ||
      isPathInsideRecallEvaluationArea(workDirectory, resolvedProductionPath)
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
    databasePath: resolve(workDirectory, 'zvec'),
    statePath: resolve(workDirectory, 'index-state.json'),
    manifestPath: resolve(workDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: resolve(workDirectory, 'tokenizers'),
    embeddingCacheDirectory: resolve(workDirectory, 'embedding-cache'),
    lockPath: resolve(workDirectory, 'operation.lock'),
    generationsDirectory: resolve(workDirectory, 'index-generations'),
    activeGenerationPath: resolve(workDirectory, 'active-generation.json'),
    stagingGenerationPath: resolve(workDirectory, 'staging-generation.json'),
    backgroundIndexStatusPath: resolve(workDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath: resolve(workDirectory, 'background-index-request.json'),
    diagnosticLogPath: resolve(workDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: resolve(workDirectory, 'diagnostics.previous.jsonl'),
  };
  assertPrivateRecallEvaluationWritablePaths(workDirectory, immutableInputPaths, writablePaths);
  const fusedPoolLimit =
    options.candidateLimits.dense +
    options.candidateLimits.lexical +
    options.candidateLimits.identifier;
  const embeddingIdentity = options.embeddingIdentity;
  return {
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

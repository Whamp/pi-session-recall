import { randomUUID } from 'node:crypto';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline/promises';

import {
  LocalOctenModelDownloadProgressKind,
  LocalOctenModelStatusKind,
  RecallEmbeddingProfile,
} from './enums.js';
import {
  createLocalOctenModelManager,
  type LocalOctenModelManager,
} from './local-octen-model-manager.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { confirmTerminalAction } from './terminal-confirm.js';

const PSR_SETUP_USAGE = [
  'psr setup [--local|--external] [--yes] [--index]',
  '          [--config <path>] [--model-root <path>]',
  '          [--parallelism <N>] [--intra-op-threads <N>]',
  '          [--base-url <url> --model <name> --served-model-id <id>]',
  '          [--native-dimensions <N> --batch-size <N>]',
].join('\n');
const DEFAULT_EXTERNAL_MODEL = 'octen-embed';
const DEFAULT_EXTERNAL_SERVED_MODEL_ID = 'Octen/Octen-Embedding-4B';
const DEFAULT_EXTERNAL_NATIVE_DIMENSIONS = 2_560;
const DEFAULT_EXTERNAL_BATCH_SIZE = 16;
const STORED_DIMENSIONS = 1_024;
const EMBEDDING_SETTING_KEYS = [
  'embeddingProfile',
  'embeddingBaseUrl',
  'embeddingModel',
  'embeddingServedModelId',
  'embeddingNativeDimensions',
  'embeddingStoredDimensions',
  'embeddingBatchSize',
  'localModelRootDirectory',
  'localEmbeddingParallelism',
  'localEmbeddingIntraOperationThreads',
] as const;

/** Result of setup before the outer CLI optionally starts a full rebuild. */
export interface PsrSetupCliResult {
  exitCode: number;
  runInitialIndex: boolean;
}

/** Replaceable terminal and artifact boundaries for setup behavior. */
export interface PsrSetupCliDependencies {
  getHomeDirectory: () => string;
  selectProfile: () => Promise<RecallEmbeddingProfile>;
  confirm: (question: string) => Promise<boolean>;
  createModelManager: (modelRootDirectory: string) => LocalOctenModelManager;
  writeOutput: (text: string) => void;
  writeProgress: (text: string) => void;
}

async function selectTerminalEmbeddingProfile(): Promise<RecallEmbeddingProfile> {
  if (!process.stdin.isTTY || !process.stderr.isTTY) {
    throw new Error('Setup requires --local or --external outside an interactive terminal');
  }
  const interface_ = createInterface({ input: process.stdin, output: process.stderr });
  try {
    const answer = await interface_.question(
      'Embedding profile: local Octen (default) or external HTTP? [local/external] ',
    );
    const normalized = answer.trim().toLowerCase();
    if (normalized === '' || normalized === 'local' || normalized === 'l') {
      return RecallEmbeddingProfile.LOCAL_OCTEN;
    }
    if (normalized === 'external' || normalized === 'e' || normalized === 'http') {
      return RecallEmbeddingProfile.OCTEN_HTTP;
    }
    throw new Error('Setup embedding profile must be local or external');
  } finally {
    interface_.close();
  }
}

const DEFAULT_PSR_SETUP_CLI_DEPENDENCIES: PsrSetupCliDependencies = {
  getHomeDirectory: homedir,
  selectProfile: selectTerminalEmbeddingProfile,
  confirm: confirmTerminalAction,
  createModelManager(modelRootDirectory) {
    return createLocalOctenModelManager({ modelRootDirectory });
  },
  writeOutput(text) {
    process.stdout.write(text);
  },
  writeProgress(text) {
    process.stderr.write(text);
  },
};

interface ParsedSetupArguments {
  local: boolean;
  external: boolean;
  approved: boolean;
  runInitialIndex: boolean;
  configPath?: string;
  modelRootDirectory?: string;
  baseUrl?: string;
  model?: string;
  servedModelId?: string;
  nativeDimensions?: number;
  batchSize?: number;
  localParallelism?: number;
  localIntraOperationThreads?: number;
}

function readPositiveInteger(value: string | undefined, flag: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${flag} requires a positive integer\n${PSR_SETUP_USAGE}`);
  }
  return parsed;
}

function parseSetupArguments(argumentsList: readonly string[]): ParsedSetupArguments {
  const parsed: ParsedSetupArguments = {
    local: false,
    external: false,
    approved: false,
    runInitialIndex: false,
  };
  const seen = new Set<string>();
  for (let index = 0; index < argumentsList.length; index += 1) {
    const flag = argumentsList[index];
    if (!flag || seen.has(flag)) {
      throw new Error(PSR_SETUP_USAGE);
    }
    seen.add(flag);
    switch (flag) {
      case '--local':
        parsed.local = true;
        break;
      case '--external':
        parsed.external = true;
        break;
      case '--yes':
        parsed.approved = true;
        break;
      case '--index':
        parsed.runInitialIndex = true;
        break;
      case '--config':
      case '--model-root':
      case '--base-url':
      case '--model':
      case '--served-model-id': {
        const value = argumentsList[index + 1];
        if (!value || value.startsWith('--')) {
          throw new Error(PSR_SETUP_USAGE);
        }
        if (flag === '--config') parsed.configPath = value;
        if (flag === '--model-root') parsed.modelRootDirectory = value;
        if (flag === '--base-url') parsed.baseUrl = value;
        if (flag === '--model') parsed.model = value;
        if (flag === '--served-model-id') parsed.servedModelId = value;
        index += 1;
        break;
      }
      case '--native-dimensions':
      case '--batch-size':
      case '--parallelism':
      case '--intra-op-threads': {
        const value = readPositiveInteger(argumentsList[index + 1], flag);
        if (flag === '--native-dimensions') parsed.nativeDimensions = value;
        if (flag === '--batch-size') parsed.batchSize = value;
        if (flag === '--parallelism') parsed.localParallelism = value;
        if (flag === '--intra-op-threads') parsed.localIntraOperationThreads = value;
        index += 1;
        break;
      }
      default:
        throw new Error(PSR_SETUP_USAGE);
    }
  }
  if (parsed.local && parsed.external) {
    throw new Error(PSR_SETUP_USAGE);
  }
  return parsed;
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

async function readExistingConfig(path: string): Promise<Record<string, unknown>> {
  let content: string;
  try {
    content = await readFile(path, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return {};
    }
    throw error;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall setup cannot read invalid configuration at ${path}: ${message}`, {
      cause: error,
    });
  }
  if (!isUnknownRecord(parsed)) {
    throw new Error(`Recall setup configuration at ${path} must contain one JSON object`);
  }
  return { ...parsed };
}

async function writeConfigAtomically(path: string, config: Record<string, unknown>): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(config, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
      mode: 0o600,
    });
    await rename(temporaryPath, path);
  } catch (error) {
    await rm(temporaryPath, { force: true });
    throw error;
  }
}

function createEmbeddingConfig(
  existing: Record<string, unknown>,
  profile: RecallEmbeddingProfile,
  parsed: ParsedSetupArguments,
  modelRootDirectory: string,
): Record<string, unknown> {
  const config = { ...existing };
  for (const key of EMBEDDING_SETTING_KEYS) {
    delete config[key];
  }
  if (profile === RecallEmbeddingProfile.LOCAL_OCTEN) {
    config.embeddingProfile = RecallEmbeddingProfile.LOCAL_OCTEN;
    config.localModelRootDirectory = modelRootDirectory;
    if (parsed.localParallelism !== undefined) {
      config.localEmbeddingParallelism = parsed.localParallelism;
    }
    if (parsed.localIntraOperationThreads !== undefined) {
      config.localEmbeddingIntraOperationThreads = parsed.localIntraOperationThreads;
    }
    return config;
  }

  if (!parsed.baseUrl) {
    throw new Error(`psr setup --external requires --base-url\n${PSR_SETUP_USAGE}`);
  }
  let baseUrl: URL;
  try {
    baseUrl = new URL(parsed.baseUrl);
  } catch (error) {
    throw new Error(`psr setup --base-url is invalid: ${parsed.baseUrl}`, { cause: error });
  }
  if (baseUrl.protocol !== 'http:' && baseUrl.protocol !== 'https:') {
    throw new Error('psr setup --base-url must use http or https');
  }
  const nativeDimensions = parsed.nativeDimensions ?? DEFAULT_EXTERNAL_NATIVE_DIMENSIONS;
  if (nativeDimensions < STORED_DIMENSIONS) {
    throw new Error(
      `psr setup external native dimensions ${nativeDimensions} cannot store ${STORED_DIMENSIONS} dimensions`,
    );
  }
  config.embeddingProfile = RecallEmbeddingProfile.OCTEN_HTTP;
  config.embeddingBaseUrl = baseUrl.toString().replace(/\/$/u, '');
  config.embeddingModel = parsed.model ?? DEFAULT_EXTERNAL_MODEL;
  config.embeddingServedModelId = parsed.servedModelId ?? DEFAULT_EXTERNAL_SERVED_MODEL_ID;
  config.embeddingNativeDimensions = nativeDimensions;
  config.embeddingStoredDimensions = STORED_DIMENSIONS;
  config.embeddingBatchSize = parsed.batchSize ?? DEFAULT_EXTERNAL_BATCH_SIZE;
  return config;
}

/** Configures one fresh local or external embedding profile without indexing by itself. */
export async function runPsrSetupCli(
  argumentsList: readonly string[],
  dependencies: PsrSetupCliDependencies = DEFAULT_PSR_SETUP_CLI_DEPENDENCIES,
): Promise<PsrSetupCliResult> {
  const parsed = parseSetupArguments(argumentsList);
  const homeDirectory = dependencies.getHomeDirectory();
  const configPath = resolve(
    parsed.configPath ?? join(homeDirectory, '.pi', 'agent', 'recall.json'),
  );
  const modelRootDirectory = resolve(
    parsed.modelRootDirectory ?? join(homeDirectory, '.pi', 'agent', 'recall-models'),
  );
  const profile = parsed.local
    ? RecallEmbeddingProfile.LOCAL_OCTEN
    : parsed.external
      ? RecallEmbeddingProfile.OCTEN_HTTP
      : await dependencies.selectProfile();
  const configExists = await pathExists(configPath);
  if (
    configExists &&
    !parsed.approved &&
    !(await dependencies.confirm(`Replace embedding settings in ${configPath}?`))
  ) {
    dependencies.writeOutput('Recall setup cancelled; existing configuration unchanged.\n');
    return { exitCode: 1, runInitialIndex: false };
  }
  const existing = await readExistingConfig(configPath);
  const nextConfig = createEmbeddingConfig(existing, profile, parsed, modelRootDirectory);

  if (profile === RecallEmbeddingProfile.LOCAL_OCTEN) {
    if (
      parsed.baseUrl !== undefined ||
      parsed.model !== undefined ||
      parsed.servedModelId !== undefined ||
      parsed.nativeDimensions !== undefined ||
      parsed.batchSize !== undefined
    ) {
      throw new Error(`Local setup cannot use external HTTP flags\n${PSR_SETUP_USAGE}`);
    }
    const manager = dependencies.createModelManager(modelRootDirectory);
    const status = await manager.status();
    const modelApproved =
      status.kind === LocalOctenModelStatusKind.READY ||
      parsed.approved ||
      (await dependencies.confirm(
        `Download ${(status.totalBytes / 1_073_741_824).toFixed(2)} GiB local Octen model?`,
      ));
    if (!modelApproved) {
      dependencies.writeOutput('Recall setup cancelled; local model was not downloaded.\n');
      return { exitCode: 1, runInitialIndex: false };
    }
    await manager.download({
      approved: true,
      onProgress(event) {
        if (event.kind === LocalOctenModelDownloadProgressKind.DOWNLOADING_FILE) {
          dependencies.writeProgress(`Downloading ${event.fileName ?? 'model artifact'}...\n`);
        } else if (event.kind === LocalOctenModelDownloadProgressKind.FILE_VERIFIED) {
          dependencies.writeProgress(`Verified ${event.fileName ?? 'model artifact'}.\n`);
        }
      },
    });
  } else if (
    parsed.modelRootDirectory !== undefined ||
    parsed.localParallelism !== undefined ||
    parsed.localIntraOperationThreads !== undefined
  ) {
    throw new Error(`External setup cannot use local model flags\n${PSR_SETUP_USAGE}`);
  }

  await writeConfigAtomically(configPath, nextConfig);
  dependencies.writeOutput(
    profile === RecallEmbeddingProfile.LOCAL_OCTEN
      ? `Local Octen embeddings configured in ${configPath}. Run psr index --rebuild to build recall.\n`
      : `External Octen HTTP embeddings configured in ${configPath}. Run psr index --rebuild to build recall.\n`,
  );
  return { exitCode: 0, runInitialIndex: parsed.runInitialIndex };
}

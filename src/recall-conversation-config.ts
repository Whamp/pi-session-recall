import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { isAbsolute, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallConversationConfig } from './recall-conversation-service.js';
import { RecallDiagnosticsMode } from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import {
  isCanonicalPathWithinBoundary,
  resolveCanonicalPathBoundary,
} from './trusted-path-boundary.js';

const DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT = 40;
const MAX_RECALL_CHANNEL_CANDIDATE_LIMIT = 200;
const DEFAULT_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS = 500;
const MAX_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS = 500;

const recallConfigFileSchema = Type.Object(
  {
    sessionsDirectory: Type.Optional(Type.String({ minLength: 1 })),
    dataDirectory: Type.Optional(Type.String({ minLength: 1 })),
    diagnostics: Type.Optional(Type.Enum(RecallDiagnosticsMode)),
    embeddingBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    embeddingModel: Type.Optional(Type.String({ minLength: 1 })),
    embeddingServedModelId: Type.Optional(Type.String({ minLength: 1 })),
    embeddingArtifact: Type.Optional(Type.String({ minLength: 1 })),
    embeddingQuantization: Type.Optional(Type.String({ minLength: 1 })),
    embeddingPooling: Type.Optional(Type.String({ minLength: 1 })),
    embeddingDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingBatchSize: Type.Optional(Type.Integer({ minimum: 1 })),
    rerankerBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    rerankerModel: Type.Optional(Type.String({ minLength: 1 })),
    denseCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
    ),
    lexicalCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
    ),
    identifierCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
    ),
    searchWriteWindowWaitMilliseconds: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS }),
    ),
    projectLineages: Type.Optional(
      Type.Record(
        Type.String({ minLength: 1 }),
        Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
      ),
    ),
  },
  { additionalProperties: false },
);

/** Inputs used to locate and override recall configuration, primarily for tests and embedding migrations. */
export interface RecallConversationConfigLoadOptions {
  homeDirectory?: string;
  configPath?: string;
  environment?: Record<string, string | undefined>;
}

function parsePositiveInteger(value: string, settingName: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`Recall configuration invalid integer for ${settingName}: ${value}`);
  }
  return parsed;
}

function parseRecallCandidateLimit(value: string, settingName: string): number {
  const parsed = parsePositiveInteger(value, settingName);
  if (parsed > MAX_RECALL_CHANNEL_CANDIDATE_LIMIT) {
    throw new Error(
      `Recall configuration candidate limit for ${settingName} exceeds ${MAX_RECALL_CHANNEL_CANDIDATE_LIMIT}: ${value}`,
    );
  }
  return parsed;
}

function parseRecallSearchWriteWindowWait(value: string): number {
  const parsed = parsePositiveInteger(value, 'PI_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS');
  if (parsed > MAX_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS) {
    throw new Error(
      `Recall configuration search write-window wait exceeds ${MAX_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS}: ${value}`,
    );
  }
  return parsed;
}

function resolveRecallCandidateLimit(
  settingName: string,
  environmentValue?: string,
  fileValue?: number,
): number {
  return environmentValue === undefined
    ? (fileValue ?? DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT)
    : parseRecallCandidateLimit(environmentValue, settingName);
}

async function assertRecallDataDirectoryIsolated(
  dataDirectory: string,
  sessionsDirectory: string,
): Promise<void> {
  if (!isAbsolute(dataDirectory)) {
    throw new Error(`Recall configuration data directory must be absolute: ${dataDirectory}`);
  }
  if (!isAbsolute(sessionsDirectory)) {
    throw new Error(
      `Recall configuration session directory must be absolute: ${sessionsDirectory}`,
    );
  }
  const [canonicalDataDirectory, canonicalSessionsDirectory] = await Promise.all([
    resolveCanonicalPathBoundary(dataDirectory),
    resolveCanonicalPathBoundary(sessionsDirectory),
  ]);
  if (
    canonicalDataDirectory === canonicalSessionsDirectory ||
    isCanonicalPathWithinBoundary(canonicalDataDirectory, canonicalSessionsDirectory) ||
    isCanonicalPathWithinBoundary(canonicalSessionsDirectory, canonicalDataDirectory)
  ) {
    throw new Error(
      `Recall configuration data directory must not overlap the session directory: ${dataDirectory}`,
    );
  }
}

async function readRecallConfigFile(
  configPath: string,
): Promise<ReturnType<typeof Value.Parse<typeof recallConfigFileSchema>>> {
  try {
    const raw: unknown = JSON.parse(await readFile(configPath, 'utf8'));
    return Value.Parse(recallConfigFileSchema, raw);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return {};
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall configuration invalid at ${configPath}: ${message}`, { cause: error });
  }
}

/** Loads validated conversation recall paths plus local embedding and Qwen reranker settings. */
export async function loadRecallConversationConfig(
  options: RecallConversationConfigLoadOptions = {},
): Promise<RecallConversationConfig> {
  const homeDirectory = options.homeDirectory ?? homedir();
  const environment = options.environment ?? process.env;
  const configPath =
    options.configPath ??
    environment.PI_RECALL_CONFIG ??
    join(homeDirectory, '.pi', 'agent', 'recall.json');
  const file = await readRecallConfigFile(configPath);
  const dataDirectory =
    environment.PI_RECALL_DATA_DIRECTORY ??
    file.dataDirectory ??
    join(homeDirectory, '.pi', 'agent', 'recall');
  const sessionsDirectory =
    environment.PI_RECALL_SESSIONS_DIRECTORY ??
    file.sessionsDirectory ??
    join(homeDirectory, '.pi', 'agent', 'sessions');
  await assertRecallDataDirectoryIsolated(dataDirectory, sessionsDirectory);
  const projectLineages = normalizeRecallProjectLineages(file.projectLineages ?? {});

  return {
    sessionsDirectory,
    dataDirectory,
    databasePath: join(dataDirectory, 'zvec'),
    projectionDatabasePath: join(dataDirectory, 'session-projections'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'embedding-cache'),
    lockPath: join(dataDirectory, 'operation.lock'),
    diagnosticsMode: file.diagnostics ?? RecallDiagnosticsMode.SLOW,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(dataDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(dataDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(dataDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(dataDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(dataDirectory, 'generations'),
    activeGenerationPointerPath: join(dataDirectory, 'active-generation.json'),
    generationRegistryPath: join(dataDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(dataDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(dataDirectory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl:
      environment.PI_RECALL_EMBEDDING_BASE_URL ??
      file.embeddingBaseUrl ??
      'http://192.168.0.67:8090/v1',
    embeddingModel: environment.PI_RECALL_EMBEDDING_MODEL ?? file.embeddingModel ?? 'octen-embed',
    embeddingServedModelId:
      environment.PI_RECALL_EMBEDDING_SERVED_MODEL_ID ??
      file.embeddingServedModelId ??
      'Octen/Octen-Embedding-4B',
    embeddingArtifact:
      environment.PI_RECALL_EMBEDDING_ARTIFACT ??
      file.embeddingArtifact ??
      'Octen-Embedding-4B.Q8_0.gguf',
    embeddingQuantization:
      environment.PI_RECALL_EMBEDDING_QUANTIZATION ?? file.embeddingQuantization ?? 'Q8_0',
    embeddingPooling: environment.PI_RECALL_EMBEDDING_POOLING ?? file.embeddingPooling ?? 'last',
    embeddingDimensions: environment.PI_RECALL_EMBEDDING_DIMENSIONS
      ? parsePositiveInteger(
          environment.PI_RECALL_EMBEDDING_DIMENSIONS,
          'PI_RECALL_EMBEDDING_DIMENSIONS',
        )
      : (file.embeddingDimensions ?? 2560),
    embeddingBatchSize: environment.PI_RECALL_EMBEDDING_BATCH_SIZE
      ? parsePositiveInteger(
          environment.PI_RECALL_EMBEDDING_BATCH_SIZE,
          'PI_RECALL_EMBEDDING_BATCH_SIZE',
        )
      : (file.embeddingBatchSize ?? 16),
    rerankerBaseUrl:
      environment.PI_RECALL_RERANKER_BASE_URL ??
      file.rerankerBaseUrl ??
      'http://192.168.0.67:8091/v1',
    rerankerModel: environment.PI_RECALL_RERANKER_MODEL ?? file.rerankerModel ?? 'qwen3-rerank',
    projectLineages,
    searchWriteWindowWaitMilliseconds: environment.PI_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS
      ? parseRecallSearchWriteWindowWait(
          environment.PI_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS,
        )
      : (file.searchWriteWindowWaitMilliseconds ??
        DEFAULT_RECALL_SEARCH_WRITE_WINDOW_WAIT_MILLISECONDS),
    searchCandidateLimits: {
      dense: resolveRecallCandidateLimit(
        'PI_RECALL_DENSE_CANDIDATE_LIMIT',
        environment.PI_RECALL_DENSE_CANDIDATE_LIMIT,
        file.denseCandidateLimit,
      ),
      lexical: resolveRecallCandidateLimit(
        'PI_RECALL_LEXICAL_CANDIDATE_LIMIT',
        environment.PI_RECALL_LEXICAL_CANDIDATE_LIMIT,
        file.lexicalCandidateLimit,
      ),
      identifier: resolveRecallCandidateLimit(
        'PI_RECALL_IDENTIFIER_CANDIDATE_LIMIT',
        environment.PI_RECALL_IDENTIFIER_CANDIDATE_LIMIT,
        file.identifierCandidateLimit,
      ),
    },
  };
}

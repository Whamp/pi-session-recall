import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import { RecallDiagnosticsMode } from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  normalizeRecallProjectLineages,
  type RecallProjectLineages,
} from './resolve-project-identity.js';

const DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT = 40;
const MAX_RECALL_CHANNEL_CANDIDATE_LIMIT = 200;

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
    queryPlannerBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    denseCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
    ),
    lexicalCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
    ),
    identifierCandidateLimit: Type.Optional(
      Type.Integer({ minimum: 1, maximum: MAX_RECALL_CHANNEL_CANDIDATE_LIMIT }),
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

/** Per-channel candidate caps applied before recall rank fusion. */
export interface RecallSearchCandidateLimits {
  dense: number;
  lexical: number;
  identifier: number;
}

/** Runtime paths, bounded retrieval channels, and local embedding plus reranker identity. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  databasePath: string;
  statePath: string;
  manifestPath: string;
  tokenizerCacheDirectory: string;
  embeddingCacheDirectory: string;
  lockPath: string;
  /** Managed index generation directories; defaults beside the legacy manifest. */
  generationsDirectory?: string;
  /** Atomic active-generation selection file; defaults beside the legacy manifest. */
  activeGenerationPath?: string;
  /** Resumable staging-generation selection file; defaults beside the legacy manifest. */
  stagingGenerationPath?: string;
  /** One bounded detached-build status record; defaults beside the legacy manifest. */
  backgroundIndexStatusPath?: string;
  /** Ephemeral detached-worker request; defaults beside the legacy manifest. */
  backgroundIndexRequestPath?: string;
  diagnosticsMode: RecallDiagnosticsMode;
  diagnosticLogPath: string;
  retainedDiagnosticLogPath: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingServedModelId: string;
  embeddingArtifact: string;
  embeddingQuantization: string;
  embeddingPooling: string;
  embeddingDimensions: number;
  embeddingBatchSize: number;
  rerankerBaseUrl: string;
  rerankerModel: string;
  queryPlannerBaseUrl?: string;
  projectLineages: RecallProjectLineages;
  searchCandidateLimits: RecallSearchCandidateLimits;
  chunkPolicy?: RecallChunkPolicy;
}

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

function resolveRecallCandidateLimit(
  settingName: string,
  environmentValue?: string,
  fileValue?: number,
): number {
  return environmentValue === undefined
    ? (fileValue ?? DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT)
    : parseRecallCandidateLimit(environmentValue, settingName);
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
  const projectLineages = normalizeRecallProjectLineages(file.projectLineages ?? {});

  return {
    sessionsDirectory:
      environment.PI_RECALL_SESSIONS_DIRECTORY ??
      file.sessionsDirectory ??
      join(homeDirectory, '.pi', 'agent', 'sessions'),
    databasePath: join(dataDirectory, 'zvec'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'embedding-cache'),
    lockPath: join(dataDirectory, 'operation.lock'),
    generationsDirectory: join(dataDirectory, 'index-generations'),
    activeGenerationPath: join(dataDirectory, 'active-generation.json'),
    stagingGenerationPath: join(dataDirectory, 'staging-generation.json'),
    backgroundIndexStatusPath: join(dataDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath: join(dataDirectory, 'background-index-request.json'),
    diagnosticsMode: file.diagnostics ?? RecallDiagnosticsMode.SLOW,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
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
    queryPlannerBaseUrl:
      environment.PI_RECALL_QUERY_PLANNER_BASE_URL ??
      file.queryPlannerBaseUrl ??
      'http://192.168.0.67:8092/v1',
    projectLineages,
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

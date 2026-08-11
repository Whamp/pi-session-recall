import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallEmbeddingProfile } from './enums.js';
import { LOCAL_OCTEN_ARTIFACT_IDENTITY } from './local-octen-model-manager.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import { DEFAULT_RECALL_CHUNK_POLICY } from './recall-index-manifest.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT = 8;
const DEFAULT_OCTEN_NATIVE_DIMENSIONS = 2_560;
const DEFAULT_OCTEN_STORED_DIMENSIONS = 1_024;
const DEFAULT_LOCAL_OCTEN_PARALLELISM = 4;
const DEFAULT_LOCAL_OCTEN_INTRA_OPERATION_THREADS = 4;

const recallConfigFileSchema = Type.Object(
  {
    sessionsDirectory: Type.Optional(Type.String({ minLength: 1 })),
    dataDirectory: Type.Optional(Type.String({ minLength: 1 })),
    embeddingProfile: Type.Optional(Type.Enum(RecallEmbeddingProfile)),
    embeddingBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    embeddingModel: Type.Optional(Type.String({ minLength: 1 })),
    embeddingServedModelId: Type.Optional(Type.String({ minLength: 1 })),
    embeddingNativeDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingStoredDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingBatchSize: Type.Optional(Type.Integer({ minimum: 1 })),
    localModelRootDirectory: Type.Optional(Type.String({ minLength: 1 })),
    localEmbeddingParallelism: Type.Optional(Type.Integer({ minimum: 1 })),
    localEmbeddingIntraOperationThreads: Type.Optional(Type.Integer({ minimum: 1 })),
    projectLineages: Type.Optional(
      Type.Record(
        Type.String({ minLength: 1 }),
        Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
      ),
    ),
  },
  { additionalProperties: false },
);

/** Inputs used to locate and override the standalone recall configuration. */
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

function readEmbeddingProfile(value: string | undefined): RecallEmbeddingProfile {
  if (value === undefined || value === RecallEmbeddingProfile.OCTEN_HTTP) {
    return RecallEmbeddingProfile.OCTEN_HTTP;
  }
  if (value === RecallEmbeddingProfile.LOCAL_OCTEN) {
    return RecallEmbeddingProfile.LOCAL_OCTEN;
  }
  throw new Error(`Recall configuration embedding profile is unsupported: ${value}`);
}

/** Loads paths and one explicit local or direct HTTP Octen embedding profile. */
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
  const embeddingProfile = readEmbeddingProfile(
    environment.PI_RECALL_EMBEDDING_PROFILE ?? file.embeddingProfile,
  );
  const localProfileSelected = embeddingProfile === RecallEmbeddingProfile.LOCAL_OCTEN;
  const httpOverrides = [
    environment.PI_RECALL_EMBEDDING_BASE_URL,
    environment.PI_RECALL_EMBEDDING_MODEL,
    environment.PI_RECALL_EMBEDDING_SERVED_MODEL_ID,
    environment.PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS,
    environment.PI_RECALL_EMBEDDING_STORED_DIMENSIONS,
    environment.PI_RECALL_EMBEDDING_BATCH_SIZE,
    file.embeddingBaseUrl,
    file.embeddingModel,
    file.embeddingServedModelId,
    file.embeddingNativeDimensions,
    file.embeddingStoredDimensions,
    file.embeddingBatchSize,
  ];
  if (localProfileSelected && httpOverrides.some((value) => value !== undefined)) {
    throw new Error('Recall local Octen profile cannot use HTTP embedding settings');
  }
  const embeddingNativeDimensions = localProfileSelected
    ? LOCAL_OCTEN_ARTIFACT_IDENTITY.nativeDimensions
    : environment.PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS
      ? parsePositiveInteger(
          environment.PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS,
          'PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS',
        )
      : (file.embeddingNativeDimensions ?? DEFAULT_OCTEN_NATIVE_DIMENSIONS);
  const embeddingStoredDimensions = localProfileSelected
    ? LOCAL_OCTEN_ARTIFACT_IDENTITY.nativeDimensions
    : environment.PI_RECALL_EMBEDDING_STORED_DIMENSIONS
      ? parsePositiveInteger(
          environment.PI_RECALL_EMBEDDING_STORED_DIMENSIONS,
          'PI_RECALL_EMBEDDING_STORED_DIMENSIONS',
        )
      : (file.embeddingStoredDimensions ?? DEFAULT_OCTEN_STORED_DIMENSIONS);
  if (embeddingStoredDimensions > embeddingNativeDimensions) {
    throw new Error(
      `Recall configuration stored dimensions ${embeddingStoredDimensions} exceed native dimensions ${embeddingNativeDimensions}`,
    );
  }
  if (embeddingStoredDimensions !== DEFAULT_OCTEN_STORED_DIMENSIONS) {
    throw new Error(
      `Recall configuration stored dimensions ${embeddingStoredDimensions} do not match the manifest version 8 FP32 vector width ${DEFAULT_OCTEN_STORED_DIMENSIONS}`,
    );
  }

  const localEmbeddingParallelism = environment.PI_RECALL_LOCAL_EMBEDDING_PARALLELISM
    ? parsePositiveInteger(
        environment.PI_RECALL_LOCAL_EMBEDDING_PARALLELISM,
        'PI_RECALL_LOCAL_EMBEDDING_PARALLELISM',
      )
    : (file.localEmbeddingParallelism ?? DEFAULT_LOCAL_OCTEN_PARALLELISM);
  const localEmbeddingIntraOperationThreads =
    environment.PI_RECALL_LOCAL_EMBEDDING_INTRA_OPERATION_THREADS
      ? parsePositiveInteger(
          environment.PI_RECALL_LOCAL_EMBEDDING_INTRA_OPERATION_THREADS,
          'PI_RECALL_LOCAL_EMBEDDING_INTRA_OPERATION_THREADS',
        )
      : (file.localEmbeddingIntraOperationThreads ??
        DEFAULT_LOCAL_OCTEN_INTRA_OPERATION_THREADS);

  return {
    sessionsDirectory:
      environment.PI_RECALL_SESSIONS_DIRECTORY ??
      file.sessionsDirectory ??
      join(homeDirectory, '.pi', 'agent', 'sessions'),
    sqliteDatabasePath: join(dataDirectory, 'recall.sqlite'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(dataDirectory, 'physical-session-ignore.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    lockPath: join(dataDirectory, 'operation.lock'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
    embeddingProfile,
    embeddingBaseUrl: localProfileSelected
      ? 'local://octen-embedding-0.6b'
      : (environment.PI_RECALL_EMBEDDING_BASE_URL ??
        file.embeddingBaseUrl ??
        'http://192.168.0.67:8090/v1'),
    embeddingModel: localProfileSelected
      ? LOCAL_OCTEN_ARTIFACT_IDENTITY.artifactId
      : (environment.PI_RECALL_EMBEDDING_MODEL ?? file.embeddingModel ?? 'octen-embed'),
    embeddingServedModelId: localProfileSelected
      ? 'Octen/Octen-Embedding-0.6B'
      : (environment.PI_RECALL_EMBEDDING_SERVED_MODEL_ID ??
        file.embeddingServedModelId ??
        'Octen/Octen-Embedding-4B'),
    embeddingNativeDimensions,
    embeddingStoredDimensions,
    embeddingBatchSize: localProfileSelected
      ? 1
      : environment.PI_RECALL_EMBEDDING_BATCH_SIZE
        ? parsePositiveInteger(
            environment.PI_RECALL_EMBEDDING_BATCH_SIZE,
            'PI_RECALL_EMBEDDING_BATCH_SIZE',
          )
        : (file.embeddingBatchSize ?? 16),
    localModelRootDirectory:
      environment.PI_RECALL_LOCAL_MODEL_ROOT_DIRECTORY ??
      file.localModelRootDirectory ??
      join(homeDirectory, '.pi', 'agent', 'recall-models'),
    localEmbeddingParallelism,
    localEmbeddingIntraOperationThreads,
    projectLineages: normalizeRecallProjectLineages(file.projectLineages ?? {}),
    searchCandidateLimits: {
      dense: DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT,
      invocation: DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT,
    },
    chunkPolicy: { ...DEFAULT_RECALL_CHUNK_POLICY },
  };
}

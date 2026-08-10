import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallConversationConfig } from './recall-conversation-service.js';
import { DEFAULT_RECALL_CHUNK_POLICY } from './recall-index-manifest.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT = 8;
const DEFAULT_OCTEN_NATIVE_DIMENSIONS = 2_560;
const DEFAULT_OCTEN_STORED_DIMENSIONS = 1_024;

const recallConfigFileSchema = Type.Object(
  {
    sessionsDirectory: Type.Optional(Type.String({ minLength: 1 })),
    dataDirectory: Type.Optional(Type.String({ minLength: 1 })),
    embeddingBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    embeddingModel: Type.Optional(Type.String({ minLength: 1 })),
    embeddingServedModelId: Type.Optional(Type.String({ minLength: 1 })),
    embeddingNativeDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingStoredDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingBatchSize: Type.Optional(Type.Integer({ minimum: 1 })),
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

/** Loads paths and one direct Octen HTTP embedding profile for `psr` and read-only search. */
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
  const embeddingNativeDimensions = environment.PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS
    ? parsePositiveInteger(
        environment.PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS,
        'PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS',
      )
    : (file.embeddingNativeDimensions ?? DEFAULT_OCTEN_NATIVE_DIMENSIONS);
  const embeddingStoredDimensions = environment.PI_RECALL_EMBEDDING_STORED_DIMENSIONS
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
      `Recall configuration stored dimensions ${embeddingStoredDimensions} do not match the compact dense store width ${DEFAULT_OCTEN_STORED_DIMENSIONS}`,
    );
  }

  return {
    sessionsDirectory:
      environment.PI_RECALL_SESSIONS_DIRECTORY ??
      file.sessionsDirectory ??
      join(homeDirectory, '.pi', 'agent', 'sessions'),
    databasePath: join(dataDirectory, 'zvec'),
    catalogPath: join(dataDirectory, 'recall-catalog.sqlite'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    physicalSessionIgnoreStatePath: join(dataDirectory, 'physical-session-ignore.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    lockPath: join(dataDirectory, 'operation.lock'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
    embeddingBaseUrl:
      environment.PI_RECALL_EMBEDDING_BASE_URL ??
      file.embeddingBaseUrl ??
      'http://192.168.0.67:8090/v1',
    embeddingModel: environment.PI_RECALL_EMBEDDING_MODEL ?? file.embeddingModel ?? 'octen-embed',
    embeddingServedModelId:
      environment.PI_RECALL_EMBEDDING_SERVED_MODEL_ID ??
      file.embeddingServedModelId ??
      'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions,
    embeddingStoredDimensions,
    embeddingBatchSize: environment.PI_RECALL_EMBEDDING_BATCH_SIZE
      ? parsePositiveInteger(
          environment.PI_RECALL_EMBEDDING_BATCH_SIZE,
          'PI_RECALL_EMBEDDING_BATCH_SIZE',
        )
      : (file.embeddingBatchSize ?? 16),
    projectLineages: normalizeRecallProjectLineages(file.projectLineages ?? {}),
    searchCandidateLimits: {
      dense: DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT,
      invocation: DEFAULT_RECALL_CHANNEL_CANDIDATE_LIMIT,
    },
    chunkPolicy: { ...DEFAULT_RECALL_CHUNK_POLICY },
  };
}

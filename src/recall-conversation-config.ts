import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallConversationConfig } from './recall-conversation-service.js';
import { readNodeErrorCode } from './read-node-error-code.js';

const recallConfigFileSchema = Type.Object(
  {
    sessionsDirectory: Type.Optional(Type.String({ minLength: 1 })),
    dataDirectory: Type.Optional(Type.String({ minLength: 1 })),
    embeddingBaseUrl: Type.Optional(Type.String({ minLength: 1 })),
    embeddingModel: Type.Optional(Type.String({ minLength: 1 })),
    embeddingDimensions: Type.Optional(Type.Integer({ minimum: 1 })),
    embeddingBatchSize: Type.Optional(Type.Integer({ minimum: 1 })),
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

/** Loads validated conversation recall paths and local embedding settings. */
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

  return {
    sessionsDirectory:
      environment.PI_RECALL_SESSIONS_DIRECTORY ??
      file.sessionsDirectory ??
      join(homeDirectory, '.pi', 'agent', 'sessions'),
    databasePath: join(dataDirectory, 'zvec'),
    statePath: join(dataDirectory, 'index-state.json'),
    lockPath: join(dataDirectory, 'operation.lock'),
    embeddingBaseUrl:
      environment.PI_RECALL_EMBEDDING_BASE_URL ??
      file.embeddingBaseUrl ??
      'http://192.168.0.67:8090/v1',
    embeddingModel: environment.PI_RECALL_EMBEDDING_MODEL ?? file.embeddingModel ?? 'octen-embed',
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
  };
}

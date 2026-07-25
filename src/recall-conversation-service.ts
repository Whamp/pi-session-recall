import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  indexChangedConversationSessions,
  type ConversationIndexProgress,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { createLocalEmbeddingClient, type LocalEmbeddingClient } from './local-embedding-client.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  openZvecConversationStore,
  type RecallSearchResult,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';

/** Runtime paths and local embedding settings for conversation recall. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  databasePath: string;
  statePath: string;
  lockPath: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingDimensions: number;
  embeddingBatchSize: number;
}

/** One completed recall query, including incremental-index work performed first. */
export interface RecallConversationSearch {
  results: RecallSearchResult[];
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
}

/** Search and maintenance operations exposed by the recall extension. */
export interface RecallConversationService {
  search(
    query: string,
    limit: number,
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
  ): Promise<RecallConversationSearch>;
  index(
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
    optimize?: boolean,
  ): Promise<{ indexSummary: ConversationIndexSummary; totalChunks: number }>;
}

interface RecallConversationDependencies {
  embeddings?: LocalEmbeddingClient;
  openStore?: () => ZvecConversationStore;
}

function readLockOwnerProcessId(value: string): number | undefined {
  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== 'object' || !('pid' in parsed)) {
      return undefined;
    }
    const processId = Reflect.get(parsed, 'pid');
    return typeof processId === 'number' && Number.isInteger(processId) ? processId : undefined;
  } catch {
    return undefined;
  }
}

function isProcessAlive(processId: number): boolean {
  try {
    process.kill(processId, 0);
    return true;
  } catch (error) {
    return readNodeErrorCode(error) === 'EPERM';
  }
}

async function acquireRecallConversationLock(
  lockPath: string,
  signal?: AbortSignal,
): Promise<() => Promise<void>> {
  await mkdir(dirname(lockPath), { recursive: true });
  let unreadableOwnerCount = 0;
  while (true) {
    if (signal?.aborted) {
      throw new Error('Recall conversation operation cancelled', { cause: signal.reason });
    }
    try {
      await mkdir(lockPath);
      await writeFile(
        `${lockPath}/owner.json`,
        `${JSON.stringify({ pid: process.pid })}\n`,
        'utf8',
      );
      return async () => rm(lockPath, { recursive: true, force: true });
    } catch (error) {
      if (readNodeErrorCode(error) !== 'EEXIST') {
        throw error;
      }
      let ownerProcessId: number | undefined;
      try {
        ownerProcessId = readLockOwnerProcessId(await readFile(`${lockPath}/owner.json`, 'utf8'));
      } catch (readError) {
        if (readNodeErrorCode(readError) !== 'ENOENT') {
          throw readError;
        }
      }
      if (ownerProcessId === undefined) {
        unreadableOwnerCount += 1;
        if (unreadableOwnerCount >= 4) {
          await rm(lockPath, { recursive: true, force: true });
          unreadableOwnerCount = 0;
          continue;
        }
      } else {
        unreadableOwnerCount = 0;
        if (!isProcessAlive(ownerProcessId)) {
          await rm(lockPath, { recursive: true, force: true });
          continue;
        }
      }
      await sleep(250, undefined, signal ? { signal } : undefined);
    }
  }
}

/** Creates the incremental indexing and semantic search service used by the Pi recall tool. */
export function createRecallConversationService(
  config: RecallConversationConfig,
  dependencies: RecallConversationDependencies = {},
): RecallConversationService {
  const embeddings =
    dependencies.embeddings ??
    createLocalEmbeddingClient({
      baseUrl: config.embeddingBaseUrl,
      model: config.embeddingModel,
      dimensions: config.embeddingDimensions,
      batchSize: config.embeddingBatchSize,
    });
  const openStore =
    dependencies.openStore ??
    (() =>
      openZvecConversationStore({
        databasePath: config.databasePath,
        dimensions: config.embeddingDimensions,
      }));
  let activeOperation: Promise<void> | undefined;

  async function runSerialized<T>(operation: () => Promise<T>): Promise<T> {
    while (activeOperation) {
      await activeOperation;
    }
    const operationFinished = Promise.withResolvers<void>();
    activeOperation = operationFinished.promise;
    try {
      return await operation();
    } finally {
      activeOperation = undefined;
      operationFinished.resolve();
    }
  }

  async function withLockedStore<T>(
    signal: AbortSignal | undefined,
    operation: (store: ZvecConversationStore) => Promise<T>,
  ): Promise<T> {
    return runSerialized(async () => {
      const releaseLock = await acquireRecallConversationLock(config.lockPath, signal);
      let store: ZvecConversationStore | undefined;
      try {
        store = openStore();
        return await operation(store);
      } finally {
        store?.close();
        await releaseLock();
      }
    });
  }

  async function updateConversationIndex(
    store: ZvecConversationStore,
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
  ): Promise<ConversationIndexSummary> {
    return indexChangedConversationSessions({
      sessionsDirectory: config.sessionsDirectory,
      statePath: config.statePath,
      store,
      embeddings,
      ...(signal ? { signal } : {}),
      ...(onProgress ? { onProgress } : {}),
    });
  }

  return {
    search(query, limit, signal, onProgress) {
      return withLockedStore(signal, async (store) => {
        const indexSummary = await updateConversationIndex(store, signal, onProgress);
        const queryEmbedding = (await embeddings.embedTexts([query], signal))[0];
        if (!queryEmbedding) {
          throw new Error('Recall embedding response missing query vector');
        }
        return {
          results: store.search(queryEmbedding, limit),
          indexSummary,
          totalChunks: store.count(),
        };
      });
    },
    index(signal, onProgress, optimize = false) {
      return withLockedStore(signal, async (store) => {
        const indexSummary = await updateConversationIndex(store, signal, onProgress);
        if (optimize && (indexSummary.embeddedChunks > 0 || indexSummary.deletedChunks > 0)) {
          await store.optimize();
        }
        return { indexSummary, totalChunks: store.count() };
      });
    },
  };
}

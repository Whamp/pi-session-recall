import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import {
  fuseRecallSearchCandidates,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import {
  indexChangedConversationSessions,
  type ConversationIndexProgress,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { createLocalEmbeddingClient, type LocalEmbeddingClient } from './local-embedding-client.js';
import { createLocalRerankerClient, type LocalRerankerClient } from './local-reranker-client.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import {
  assertRecallIndexManifestCompatible,
  calculateRecallEmbeddingCanaryCosineSimilarity,
  createRecallIndexManifest,
  readRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
  type RecallIndexManifest,
} from './recall-index-manifest.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  rerankRecallSearchResults,
  RECALL_ACTIVE_BRANCH_PRIOR,
  RECALL_RERANK_POLICY_VERSION,
  type RerankedRecallSearchResult,
} from './rerank-recall-search-results.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';

/** Runtime paths, bounded retrieval channels, and local embedding plus reranker identity. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  databasePath: string;
  statePath: string;
  manifestPath: string;
  tokenizerCacheDirectory: string;
  embeddingCacheDirectory: string;
  lockPath: string;
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
  searchCandidateLimits: RecallSearchCandidateLimits;
  chunkPolicy?: RecallChunkPolicy;
}

/** Per-channel candidate caps applied before recall rank fusion. */
export interface RecallSearchCandidateLimits {
  dense: number;
  lexical: number;
  identifier: number;
}

/** Exact fusion, Qwen reranking, branch-prior, and neighbor policy for one search. */
export interface RecallSearchPolicy {
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  rerankPolicyVersion: number;
  rerankerModel: string;
  activeBranchPrior: number;
  candidateLimits: RecallSearchCandidateLimits;
}

/** One read-only hybrid query against a previously built compatible index. */
export interface RecallConversationSearch {
  results: RerankedRecallSearchResult[];
  totalChunks: number;
  searchPolicy: RecallSearchPolicy;
}

/** Cancellation, progress, optimization, and generation replacement for explicit indexing. */
export interface RecallConversationIndexOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ConversationIndexProgress) => void;
  optimize?: boolean;
  rebuild?: boolean;
}

/** Read-only search and explicit index-maintenance operations exposed by the extension. */
export interface RecallConversationService {
  search(query: string, limit: number, signal?: AbortSignal): Promise<RecallConversationSearch>;
  index(
    options?: RecallConversationIndexOptions,
  ): Promise<{ indexSummary: ConversationIndexSummary; totalChunks: number }>;
}

/** Injectable local model, tokenizer, and zvec boundaries used by tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddings?: LocalEmbeddingClient;
  reranker?: LocalRerankerClient;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write') => ZvecConversationStore;
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

async function assertRecallIndexUnlockedForSearch(lockPath: string): Promise<void> {
  try {
    await stat(lockPath);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return;
    }
    throw error;
  }
  let ownerDescription = 'held by an unreadable owner';
  try {
    const processId = readLockOwnerProcessId(await readFile(`${lockPath}/owner.json`, 'utf8'));
    if (processId !== undefined) {
      ownerDescription = isProcessAlive(processId)
        ? `held by process ${processId}`
        : `a stale lock from dead process ${processId}; run /pi-session-recall-index after the quality gate passes to recover`;
    }
  } catch (error) {
    if (readNodeErrorCode(error) !== 'ENOENT') {
      throw error;
    }
  }
  throw new Error(
    `Recall index write lock at ${lockPath} is ${ownerDescription}; read-only search did not remove the lock`,
  );
}

function createEmbeddingModelIdentity(
  config: RecallConversationConfig,
): RecallEmbeddingModelIdentity {
  return {
    requestModel: config.embeddingModel,
    servedModelId: config.embeddingServedModelId,
    artifact: config.embeddingArtifact,
    dimensions: config.embeddingDimensions,
    quantization: config.embeddingQuantization,
    pooling: config.embeddingPooling,
  };
}

/** Creates the explicit indexing and read-only search service used by the Pi recall extension. */
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
  const reranker =
    dependencies.reranker ??
    createLocalRerankerClient({
      baseUrl: config.rerankerBaseUrl,
      model: config.rerankerModel,
    });
  const loadTokenizer =
    dependencies.loadTokenizer ??
    (() => loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory }));
  const openStore =
    dependencies.openStore ??
    ((mode) =>
      openZvecConversationStore({
        databasePath: config.databasePath,
        dimensions: config.embeddingDimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      }));
  let activeOperation: Promise<void> | undefined;
  let conversationTokenizer: ConversationTextTokenizer | undefined;

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

  async function getConversationTokenizer(): Promise<ConversationTextTokenizer> {
    if (!conversationTokenizer) {
      conversationTokenizer = await loadTokenizer();
    }
    return conversationTokenizer;
  }

  async function readCurrentEmbeddingCanary(signal?: AbortSignal): Promise<number[]> {
    const embedding = (await embeddings.embedTexts([RECALL_EMBEDDING_CANARY_TEXT], signal))[0];
    if (!embedding) {
      throw new Error('Recall embedding response missing canary vector');
    }
    return [...embedding];
  }

  function createExpectedManifest(canaryEmbedding: readonly number[]): RecallIndexManifest {
    return createRecallIndexManifest({
      embeddingIdentity: createEmbeddingModelIdentity(config),
      canaryEmbedding,
      ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
    });
  }

  async function readCanonicalRebuildCanary(signal?: AbortSignal): Promise<number[]> {
    const currentManifest = createExpectedManifest(await readCurrentEmbeddingCanary(signal));
    const currentCanary = currentManifest.embedding.canaryVector;
    let actualManifest: RecallIndexManifest | null;
    try {
      actualManifest = await readRecallIndexManifest(config.manifestPath);
    } catch (error) {
      if (
        error instanceof Error &&
        error.message.startsWith(`Recall index manifest invalid at ${config.manifestPath}:`)
      ) {
        return currentCanary;
      }
      throw error;
    }
    if (!actualManifest || actualManifest.embedding.dimensions !== config.embeddingDimensions) {
      return currentCanary;
    }
    const cosineSimilarity = calculateRecallEmbeddingCanaryCosineSimilarity(
      actualManifest.embedding.canaryVector,
      currentCanary,
      config.embeddingDimensions,
    );
    return cosineSimilarity >= actualManifest.embedding.canaryMinimumCosineSimilarity
      ? [...actualManifest.embedding.canaryVector]
      : currentCanary;
  }

  async function readRequiredManifest(): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(config.manifestPath);
    if (!actual) {
      throw new Error(
        `Recall index manifest missing at ${config.manifestPath}; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    return actual;
  }

  async function prepareIndexForWrite(
    signal?: AbortSignal,
    preflightedCanary?: readonly number[],
  ): Promise<{
    tokenizer: ConversationTextTokenizer;
    manifest: RecallIndexManifest;
    embeddingModelPreflighted: boolean;
  }> {
    const actual = await readRecallIndexManifest(config.manifestPath);
    if (!actual && (existsSync(config.databasePath) || existsSync(config.statePath))) {
      throw new Error(
        `Recall index manifest missing at ${config.manifestPath} for existing index data; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    const tokenizer = await getConversationTokenizer();
    if (actual) {
      const expected = createExpectedManifest(actual.embedding.canaryVector);
      assertRecallIndexManifestCompatible(actual, expected, config.manifestPath);
      return { tokenizer, manifest: actual, embeddingModelPreflighted: false };
    }
    const expected = createExpectedManifest(
      preflightedCanary ?? (await readCurrentEmbeddingCanary(signal)),
    );
    await writeRecallIndexManifest(config.manifestPath, expected);
    return { tokenizer, manifest: expected, embeddingModelPreflighted: true };
  }

  async function removeRecallIndexGeneration(): Promise<void> {
    await rm(config.databasePath, { recursive: true, force: true });
    await rm(config.statePath, { force: true });
    await rm(config.manifestPath, { force: true });
  }

  async function updateConversationIndex(
    store: ZvecConversationStore,
    tokenizer: ConversationTextTokenizer,
    manifest: RecallIndexManifest,
    embeddingModelPreflighted: boolean,
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
  ): Promise<ConversationIndexSummary> {
    let modelPreflighted = embeddingModelPreflighted;
    async function embedTextsAfterModelPreflight(
      texts: string[],
      embeddingSignal?: AbortSignal,
    ): Promise<number[][]> {
      if (!modelPreflighted) {
        const expected = createExpectedManifest(await readCurrentEmbeddingCanary(embeddingSignal));
        assertRecallIndexManifestCompatible(manifest, expected, config.manifestPath);
        modelPreflighted = true;
      }
      return embeddings.embedTexts(texts, embeddingSignal);
    }
    const preflightedEmbeddings: LocalEmbeddingClient = {
      embedTexts: embedTextsAfterModelPreflight,
    };
    const embeddingCache = createEmbeddingVectorCache({
      cacheDirectory: config.embeddingCacheDirectory,
      identity: createEmbeddingVectorCacheIdentity(manifest),
      embeddingRequestBatchSize: config.embeddingBatchSize,
      embeddings: preflightedEmbeddings,
    });
    return indexChangedConversationSessions({
      sessionsDirectory: config.sessionsDirectory,
      statePath: config.statePath,
      store,
      embeddingCache,
      tokenizer,
      chunkPolicy: {
        maxTokens: manifest.chunkPolicy.maxTokens,
        overlapTokens: manifest.chunkPolicy.overlapTokens,
      },
      ...(signal ? { signal } : {}),
      ...(onProgress ? { onProgress } : {}),
    });
  }

  return {
    search(query, limit, signal) {
      return runSerialized(async () => {
        const searchQuery = query.trim();
        if (!searchQuery) {
          throw new Error('Recall query must not be blank');
        }
        const actualManifest = await readRequiredManifest();
        await assertRecallIndexUnlockedForSearch(config.lockPath);
        const expectedManifest = createExpectedManifest(await readCurrentEmbeddingCanary(signal));
        assertRecallIndexManifestCompatible(actualManifest, expectedManifest, config.manifestPath);
        const store = openStore('read');
        try {
          const queryEmbedding = (await embeddings.embedTexts([searchQuery], signal))[0];
          if (!queryEmbedding) {
            throw new Error('Recall embedding response missing query vector');
          }
          const fusedCandidates = fuseRecallSearchCandidates(
            {
              denseCandidates: store.searchDenseCandidates(
                queryEmbedding,
                config.searchCandidateLimits.dense,
              ),
              lexicalCandidates: store.searchLexicalCandidates(
                searchQuery,
                config.searchCandidateLimits.lexical,
              ),
              identifierCandidates: store.searchIdentifierCandidates(
                searchQuery,
                config.searchCandidateLimits.identifier,
              ),
            },
            config.searchCandidateLimits.dense +
              config.searchCandidateLimits.lexical +
              config.searchCandidateLimits.identifier,
          );
          const results = await rerankRecallSearchResults({
            query: searchQuery,
            candidates: fusedCandidates,
            resultLimit: limit,
            reranker,
            fetchConversationChunks: store.fetchConversationChunks,
            ...(signal ? { signal } : {}),
          });
          return {
            results,
            totalChunks: store.count(),
            searchPolicy: {
              rankFusionVersion: RECALL_RANK_FUSION_VERSION,
              reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
              rerankPolicyVersion: RECALL_RERANK_POLICY_VERSION,
              rerankerModel: config.rerankerModel,
              activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
              candidateLimits: { ...config.searchCandidateLimits },
            },
          };
        } finally {
          store.close();
        }
      });
    },
    index(options = {}) {
      return runSerialized(async () => {
        const releaseLock = await acquireRecallConversationLock(config.lockPath, options.signal);
        let store: ZvecConversationStore | undefined;
        try {
          let rebuildCanary: number[] | undefined;
          if (options.rebuild) {
            await getConversationTokenizer();
            rebuildCanary = await readCanonicalRebuildCanary(options.signal);
            await removeRecallIndexGeneration();
          }
          const preparedIndex = await prepareIndexForWrite(options.signal, rebuildCanary);
          store = openStore('write');
          const indexSummary = await updateConversationIndex(
            store,
            preparedIndex.tokenizer,
            preparedIndex.manifest,
            preparedIndex.embeddingModelPreflighted,
            options.signal,
            options.onProgress,
          );
          if (
            options.optimize &&
            (indexSummary.cacheHits > 0 ||
              indexSummary.newlyEmbeddedChunks > 0 ||
              indexSummary.deletedChunks > 0)
          ) {
            await store.optimize();
          }
          return { indexSummary, totalChunks: store.count() };
        } finally {
          store?.close();
          await releaseLock();
        }
      });
    },
  };
}

import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  fuseRecallSearchCandidates,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import {
  indexChangedConversationSession,
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
  createLineageResolver,
  resolveProjectIdentity,
  type ProjectIdentity,
  type RecallProjectLineages,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import {
  rankFusedRecallSearchResults,
  rerankRecallSearchResults,
  RECALL_ACTIVE_BRANCH_PRIOR,
  RECALL_RERANK_POLICY_VERSION,
  type RankedRecallSearchResult,
} from './rank-recall-search-results.js';
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
  projectLineages: RecallProjectLineages;
  searchCandidateLimits: RecallSearchCandidateLimits;
  chunkPolicy?: RecallChunkPolicy;
}

/** Per-channel candidate caps applied before recall rank fusion. */
export interface RecallSearchCandidateLimits {
  dense: number;
  lexical: number;
  identifier: number;
}

/** User-selected ranking depth for hybrid-only or local Qwen recall search. */
export type RecallSearchMode = 'hybrid' | 'deep-rerank';

/** Trusted invocation context, cancellation, and ranking depth for one recall search. */
export interface RecallConversationSearchOptions {
  mode?: RecallSearchMode;
  scope?: RecallSearchScope;
  invocationDirectory?: string;
  activeSessionPath?: string;
  signal?: AbortSignal;
}

/** Exact fusion, optional Qwen reranking, branch-prior, and neighbor policy for one search. */
export interface RecallSearchPolicy {
  scope: RecallSearchScope;
  invocationProjectIdentity: ProjectIdentity | null;
  rankingMode: RecallSearchMode;
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  rerankPolicyVersion: number | null;
  rerankerModel: string | null;
  activeBranchPrior: number;
  candidateLimits: RecallSearchCandidateLimits;
}

/** One ranked recall result labeled by its explicit relationship to the invocation project. */
export interface RecallConversationSearchResult extends RankedRecallSearchResult {
  evidenceRelation: RecallEvidenceRelation;
}

/** One read-only hybrid query against a previously built compatible index. */
export interface RecallConversationSearch {
  results: RecallConversationSearchResult[];
  totalChunks: number;
  searchPolicy: RecallSearchPolicy;
}

/** Cancellation, lock-wait milliseconds, and generation controls for conversation indexing. */
export interface RecallConversationIndexOptions {
  signal?: AbortSignal;
  lockWaitMilliseconds?: number;
  requireExistingGeneration?: boolean;
  onProgress?: (progress: ConversationIndexProgress) => void;
  optimize?: boolean;
  rebuild?: boolean;
}

/** Counts from one completed full or targeted conversation index update. */
export interface RecallConversationIndexResult {
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
}

/** Search plus full and targeted index-maintenance operations exposed by the extension. */
export interface RecallConversationService {
  search(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<RecallConversationSearch>;
  index(options?: RecallConversationIndexOptions): Promise<RecallConversationIndexResult>;
  reconcileSession(
    sessionPath: string,
    options?: Pick<RecallConversationIndexOptions, 'signal' | 'lockWaitMilliseconds'>,
  ): Promise<RecallConversationIndexResult>;
}

/** Injectable local model, tokenizer, and zvec boundaries used by tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddings?: LocalEmbeddingClient;
  reranker?: LocalRerankerClient;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write') => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
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
      try {
        await sleep(250, undefined, signal ? { signal } : undefined);
      } catch (error) {
        if (signal?.aborted) {
          throw new Error('Recall conversation operation cancelled', { cause: signal.reason });
        }
        throw error;
      }
    }
  }
}

function createRecallLockAcquisitionSignal(
  signal: AbortSignal | undefined,
  lockWaitMilliseconds: number | undefined,
): AbortSignal | undefined {
  if (lockWaitMilliseconds === undefined) {
    return signal;
  }
  const timeoutSignal = AbortSignal.timeout(lockWaitMilliseconds);
  return signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
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
  const resolveSearchProjectIdentity = createLineageResolver(
    config.projectLineages,
    dependencies.resolveProjectIdentity ?? resolveProjectIdentity,
  );
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
      projectLineages: config.projectLineages,
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
    requireExistingGeneration = false,
  ): Promise<{
    tokenizer: ConversationTextTokenizer;
    manifest: RecallIndexManifest;
    embeddingModelPreflighted: boolean;
  }> {
    const actual = await readRecallIndexManifest(config.manifestPath);
    if (!actual && requireExistingGeneration) {
      throw new Error(
        `Recall automatic session ingestion requires an existing index generation at ${config.manifestPath}; initialize it with /pi-session-recall-index --rebuild`,
      );
    }
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
    sessionPath?: string,
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
    const indexerOptions = {
      statePath: config.statePath,
      store,
      embeddingCache,
      tokenizer,
      chunkPolicy: {
        maxTokens: manifest.chunkPolicy.maxTokens,
        overlapTokens: manifest.chunkPolicy.overlapTokens,
      },
      resolveProjectIdentity: resolveSearchProjectIdentity,
      ...(signal ? { signal } : {}),
    };
    return sessionPath
      ? indexChangedConversationSession({ ...indexerOptions, sessionPath })
      : indexChangedConversationSessions({
          ...indexerOptions,
          sessionsDirectory: config.sessionsDirectory,
          ...(onProgress ? { onProgress } : {}),
        });
  }

  async function reconcileActiveConversationSession(
    sessionPath: string,
    signal?: AbortSignal,
    lockWaitMilliseconds?: number,
  ): Promise<RecallConversationIndexResult> {
    const lockSignal = createRecallLockAcquisitionSignal(signal, lockWaitMilliseconds);
    const releaseLock = await acquireRecallConversationLock(config.lockPath, lockSignal);
    let store: ZvecConversationStore | undefined;
    try {
      const preparedIndex = await prepareIndexForWrite(signal, undefined, true);
      store = openStore('write');
      const summary = await updateConversationIndex(
        store,
        preparedIndex.tokenizer,
        preparedIndex.manifest,
        preparedIndex.embeddingModelPreflighted,
        signal,
        undefined,
        sessionPath,
      );
      if (summary.failedSessions.length > 0) {
        throw new Error(
          `Recall active session reconciliation failed for ${sessionPath}: ${summary.failedSessions[0]?.error ?? 'unknown session parsing failure'}`,
        );
      }
      return { indexSummary: summary, totalChunks: store.count() };
    } finally {
      store?.close();
      await releaseLock();
    }
  }

  return {
    search(query, limit, options = {}) {
      return runSerialized(async () => {
        const {
          mode = 'hybrid',
          scope = RecallSearchScope.PROJECT,
          invocationDirectory,
          activeSessionPath,
          signal,
        } = options;
        const searchQuery = query.trim();
        if (!searchQuery) {
          throw new Error('Recall query must not be blank');
        }
        const actualManifest = await readRequiredManifest();
        await assertRecallIndexUnlockedForSearch(config.lockPath);
        const expectedManifest = createExpectedManifest(await readCurrentEmbeddingCanary(signal));
        assertRecallIndexManifestCompatible(actualManifest, expectedManifest, config.manifestPath);
        if (activeSessionPath) {
          await reconcileActiveConversationSession(activeSessionPath, signal);
        }
        await assertRecallIndexUnlockedForSearch(config.lockPath);
        if (scope === RecallSearchScope.PROJECT && !invocationDirectory) {
          throw new Error('Project-scoped recall requires Pi trusted invocation directory context');
        }
        const invocationProject = invocationDirectory
          ? await resolveSearchProjectIdentity(invocationDirectory)
          : null;
        if (scope === RecallSearchScope.PROJECT && !invocationProject) {
          throw new Error(
            `Project-scoped recall could not resolve a project identity from Pi invocation directory ${invocationDirectory}`,
          );
        }
        const projectIdentityPredicate =
          scope === RecallSearchScope.PROJECT ? invocationProject?.projectIdentity : undefined;
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
                projectIdentityPredicate,
              ),
              lexicalCandidates: store.searchLexicalCandidates(
                searchQuery,
                config.searchCandidateLimits.lexical,
                projectIdentityPredicate,
              ),
              identifierCandidates: store.searchIdentifierCandidates(
                searchQuery,
                config.searchCandidateLimits.identifier,
                projectIdentityPredicate,
              ),
            },
            config.searchCandidateLimits.dense +
              config.searchCandidateLimits.lexical +
              config.searchCandidateLimits.identifier,
          );
          const rankedResults =
            mode === 'deep-rerank'
              ? await rerankRecallSearchResults({
                  query: searchQuery,
                  candidates: fusedCandidates,
                  resultLimit: limit,
                  reranker,
                  fetchConversationChunks: store.fetchConversationChunks,
                  ...(signal ? { signal } : {}),
                })
              : rankFusedRecallSearchResults(fusedCandidates, limit, store.fetchConversationChunks);
          const results: RecallConversationSearchResult[] = rankedResults.map((result) => ({
            ...result,
            evidenceRelation:
              !invocationProject ||
              invocationProject.projectIdentity !== result.projectAttribution?.projectIdentity
                ? RecallEvidenceRelation.UNRESTRICTED_GLOBAL
                : result.projectAttribution.identitySource ===
                      RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE ||
                    invocationProject.identitySource ===
                      RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE
                  ? RecallEvidenceRelation.CONFIGURED_PROJECT_LINEAGE
                  : invocationProject.identitySource ===
                      RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN
                    ? RecallEvidenceRelation.SAME_SESSION_ORIGIN
                    : RecallEvidenceRelation.SAME_REPOSITORY,
          }));
          return {
            results,
            totalChunks: store.count(),
            searchPolicy: {
              scope,
              invocationProjectIdentity: invocationProject?.projectIdentity ?? null,
              rankingMode: mode,
              rankFusionVersion: RECALL_RANK_FUSION_VERSION,
              reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
              rerankPolicyVersion: mode === 'deep-rerank' ? RECALL_RERANK_POLICY_VERSION : null,
              rerankerModel: mode === 'deep-rerank' ? config.rerankerModel : null,
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
        const lockSignal = createRecallLockAcquisitionSignal(
          options.signal,
          options.lockWaitMilliseconds,
        );
        const releaseLock = await acquireRecallConversationLock(config.lockPath, lockSignal);
        let store: ZvecConversationStore | undefined;
        try {
          let rebuildCanary: number[] | undefined;
          if (options.rebuild) {
            await getConversationTokenizer();
            rebuildCanary = await readCanonicalRebuildCanary(options.signal);
            await removeRecallIndexGeneration();
          }
          const preparedIndex = await prepareIndexForWrite(
            options.signal,
            rebuildCanary,
            options.requireExistingGeneration,
          );
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
    reconcileSession(sessionPath, options = {}) {
      return runSerialized(() =>
        reconcileActiveConversationSession(
          sessionPath,
          options.signal,
          options.lockWaitMilliseconds,
        ),
      );
    },
  };
}

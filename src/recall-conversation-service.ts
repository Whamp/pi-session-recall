import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  fuseRecallSearchCandidates,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import {
  indexChangedConversationSessions,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import { listIgnoredPhysicalSessionPaths } from './physical-session-ignore.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { createOctenHttpEmbeddingProvider } from './octen-http-embedding-provider.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  assertRecallIndexManifestCompatible,
  createRecallIndexManifest,
  readRecallIndexManifest,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
  type RecallIndexManifest,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import {
  readRecallIndexMaintenanceStatus,
  writeRecallIndexMaintenanceStatus,
  type RecallIndexMaintenanceStatus,
} from './recall-index-maintenance-status.js';
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
  RECALL_ACTIVE_BRANCH_PRIOR,
  type RankedRecallSearchResult,
} from './rank-recall-search-results.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';

/** Paths, one Octen profile, and bounded hybrid retrieval settings. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  databasePath: string;
  statePath: string;
  manifestPath: string;
  indexMaintenanceStatusPath: string;
  physicalSessionIgnoreStatePath: string;
  tokenizerCacheDirectory: string;
  lockPath: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingServedModelId: string;
  embeddingNativeDimensions: number;
  embeddingStoredDimensions: number;
  embeddingBatchSize: number;
  projectLineages: RecallProjectLineages;
  searchCandidateLimits: RecallSearchCandidateLimits;
  chunkPolicy: RecallChunkPolicy;
}

/** Per-channel candidate caps applied before deterministic rank fusion. */
export interface RecallSearchCandidateLimits {
  dense: number;
  lexical: number;
  identifier: number;
}

/** Trusted invocation context and cancellation for one read-only recall search. */
export interface RecallConversationSearchOptions {
  scope?: RecallSearchScope;
  invocationDirectory?: string;
  signal?: AbortSignal;
}

/** Reports the fixed hybrid rank-fusion settings used by one search. */
export interface RecallSearchPolicy {
  scope: RecallSearchScope;
  invocationProjectIdentity: ProjectIdentity | null;
  rankingMode: 'hybrid';
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  activeBranchPrior: number;
  candidateLimits: RecallSearchCandidateLimits;
}

/** One ranked result labeled by its relationship to the invoking project. */
export interface RecallConversationSearchResult extends RankedRecallSearchResult {
  evidenceRelation: RecallEvidenceRelation;
}

/** One read-only hybrid query against a compatible manually built index. */
export interface RecallConversationSearch {
  results: RecallConversationSearchResult[];
  totalChunks: number;
  searchPolicy: RecallSearchPolicy;
  indexMaintenanceStatus: RecallIndexMaintenanceStatus | null;
}

/** Options accepted only by explicit `psr` index maintenance. */
export interface RecallConversationIndexOptions {
  rebuild?: boolean;
  signal?: AbortSignal;
  onProgress?: (event: RecallIndexProgressEvent) => void;
  optimize?: boolean;
}

/** Options accepted only by standalone `psr optimize` maintenance. */
export interface RecallConversationOptimizeOptions {
  signal?: AbortSignal;
  onProgress?: (event: RecallIndexProgressEvent) => void;
}

/** Counts from one completed standalone index update. */
export interface RecallConversationIndexResult {
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
}

/** Counts from one standalone zvec optimization. */
export interface RecallConversationOptimizeResult {
  totalChunks: number;
}

/** Read-only search and standalone indexing for one zvec recall collection. */
export interface RecallConversationService {
  search(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<RecallConversationSearch>;
  index(options?: RecallConversationIndexOptions): Promise<RecallConversationIndexResult>;
}

/** Standalone collection optimization capability used only by `psr`. */
export interface RecallConversationMaintenanceService extends RecallConversationService {
  optimize(options?: RecallConversationOptimizeOptions): Promise<RecallConversationOptimizeResult>;
}

/** Injectable boundaries for public-seam tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddingProvider?: RecallEmbeddingProvider;
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write') => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  getCurrentTime?: () => Date;
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
  onWait?: () => void,
): Promise<() => Promise<void>> {
  await mkdir(dirname(lockPath), { recursive: true });
  let unreadableOwnerCount = 0;
  let reportedWait = false;
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
      } else if (!isProcessAlive(ownerProcessId)) {
        await rm(lockPath, { recursive: true, force: true });
        unreadableOwnerCount = 0;
        continue;
      } else {
        unreadableOwnerCount = 0;
      }
      if (!reportedWait) {
        onWait?.();
        reportedWait = true;
      }
      await sleep(250, undefined, signal ? { signal } : undefined);
    }
  }
}

async function assertRecallIndexUnlockedForSearch(lockPath: string): Promise<void> {
  let owner: string;
  try {
    owner = await readFile(`${lockPath}/owner.json`, 'utf8');
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return;
    }
    throw error;
  }
  const ownerProcessId = readLockOwnerProcessId(owner);
  const ownerDescription = ownerProcessId ? `owned by process ${ownerProcessId}` : 'unreadable';
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
    nativeDimensions: config.embeddingNativeDimensions,
    storedDimensions: config.embeddingStoredDimensions,
    transformation: 'vendor-prefix-then-l2-v1',
  };
}

/** Creates the service used by the read-only Pi tool and standalone `psr` writer. */
export function createRecallConversationService(
  config: RecallConversationConfig,
  dependencies: RecallConversationDependencies = {},
): RecallConversationMaintenanceService {
  const embeddingProvider =
    dependencies.embeddingProvider ??
    createOctenHttpEmbeddingProvider({
      baseUrl: config.embeddingBaseUrl,
      model: config.embeddingModel,
      nativeDimensions: config.embeddingNativeDimensions,
      storedDimensions: config.embeddingStoredDimensions,
      batchSize: config.embeddingBatchSize,
    });
  const loadTokenizer =
    dependencies.loadTokenizer ??
    (() => loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory }));
  const resolveSearchProjectIdentity = createLineageResolver(
    config.projectLineages,
    dependencies.resolveProjectIdentity ?? resolveProjectIdentity,
  );
  const getCurrentTime = dependencies.getCurrentTime ?? (() => new Date());
  const openStore =
    dependencies.openStore ??
    ((mode) =>
      openZvecConversationStore({
        databasePath: config.databasePath,
        dimensions: config.embeddingStoredDimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      }));
  let tokenizer: ConversationTextTokenizer | undefined;

  function createExpectedManifest(): RecallIndexManifest {
    return createRecallIndexManifest({
      embeddingIdentity: createEmbeddingModelIdentity(config),
      ...(dependencies.tokenizerIdentity
        ? { tokenizerIdentity: dependencies.tokenizerIdentity }
        : {}),
      chunkPolicy: config.chunkPolicy,
      projectLineages: config.projectLineages,
    });
  }

  async function getTokenizer(): Promise<ConversationTextTokenizer> {
    tokenizer ??= await loadTokenizer();
    return tokenizer;
  }

  async function readCompatibleManifest(): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(config.manifestPath);
    const expected = createExpectedManifest();
    assertRecallIndexManifestCompatible(actual, expected, config.manifestPath);
    return actual;
  }

  async function prepareIndexManifest(): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(config.manifestPath);
    const expected = createExpectedManifest();
    if (!actual && (existsSync(config.databasePath) || existsSync(config.statePath))) {
      throw new Error(
        `Recall index manifest missing at ${config.manifestPath} for existing index data; rebuild with psr index --rebuild`,
      );
    }
    if (actual) {
      assertRecallIndexManifestCompatible(actual, expected, config.manifestPath);
      return actual;
    }
    await writeRecallIndexManifest(config.manifestPath, expected);
    return expected;
  }

  return {
    async search(query, limit, options = {}) {
      const searchQuery = query.trim();
      if (!searchQuery) {
        throw new Error('Recall query must not be blank');
      }
      const scope = options.scope ?? RecallSearchScope.PROJECT;
      await readCompatibleManifest();
      const indexMaintenanceStatus = await readRecallIndexMaintenanceStatus(
        config.indexMaintenanceStatusPath,
      );
      await assertRecallIndexUnlockedForSearch(config.lockPath);
      if (scope === RecallSearchScope.PROJECT && !options.invocationDirectory) {
        throw new Error('Project-scoped recall requires Pi trusted invocation directory context');
      }
      const invocationProject = options.invocationDirectory
        ? await resolveSearchProjectIdentity(options.invocationDirectory)
        : null;
      if (scope === RecallSearchScope.PROJECT && !invocationProject) {
        throw new Error(
          `Project-scoped recall could not resolve a project identity from Pi invocation directory ${options.invocationDirectory}`,
        );
      }
      const projectIdentity =
        scope === RecallSearchScope.PROJECT ? invocationProject?.projectIdentity : undefined;
      const queryEmbedding = await embeddingProvider.embedQuery(searchQuery, options.signal);
      await assertRecallIndexUnlockedForSearch(config.lockPath);
      const store = openStore('read');
      try {
        const fusedCandidates = fuseRecallSearchCandidates(
          {
            denseCandidates: store.searchDenseCandidates(
              queryEmbedding,
              config.searchCandidateLimits.dense,
              projectIdentity,
            ),
            lexicalCandidates: store.searchLexicalCandidates(
              searchQuery,
              config.searchCandidateLimits.lexical,
              projectIdentity,
            ),
            identifierCandidates: store.searchIdentifierCandidates(
              searchQuery,
              config.searchCandidateLimits.identifier,
              projectIdentity,
            ),
          },
          config.searchCandidateLimits.dense +
            config.searchCandidateLimits.lexical +
            config.searchCandidateLimits.identifier,
        );
        const results: RecallConversationSearchResult[] = rankFusedRecallSearchResults(
          fusedCandidates,
          limit,
          store.fetchConversationChunks,
        ).map((result) => ({
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
            rankingMode: 'hybrid',
            rankFusionVersion: RECALL_RANK_FUSION_VERSION,
            reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
            activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
            candidateLimits: { ...config.searchCandidateLimits },
          },
          indexMaintenanceStatus,
        };
      } finally {
        store.close();
      }
    },

    async optimize(options = {}) {
      const releaseLock = await acquireRecallConversationLock(
        config.lockPath,
        options.signal,
        options.onProgress
          ? () => options.onProgress?.({ kind: 'waiting-for-write-lock' })
          : undefined,
      );
      let store: ZvecConversationStore | undefined;
      try {
        await readCompatibleManifest();
        if (!existsSync(config.databasePath)) {
          throw new Error(`Recall index database missing at ${config.databasePath}`);
        }
        store = openStore('write');
        options.onProgress?.({ kind: 'optimizing-collection' });
        await store.optimize();
        if (options.signal?.aborted) {
          throw new Error('Recall conversation operation cancelled', {
            cause: options.signal.reason,
          });
        }
        const totalChunks = store.count();
        options.onProgress?.({ kind: 'completed' });
        return { totalChunks };
      } finally {
        store?.close();
        await releaseLock();
      }
    },

    async index(options = {}) {
      const releaseLock = await acquireRecallConversationLock(
        config.lockPath,
        options.signal,
        options.onProgress
          ? () => options.onProgress?.({ kind: 'waiting-for-write-lock' })
          : undefined,
      );
      let store: ZvecConversationStore | undefined;
      try {
        const ignoredPhysicalSessionPaths: ReadonlySet<string> = new Set(
          await listIgnoredPhysicalSessionPaths(config.physicalSessionIgnoreStatePath),
        );
        if (options.rebuild) {
          await rm(config.indexMaintenanceStatusPath, { force: true });
          await rm(config.databasePath, { recursive: true, force: true });
          await rm(config.statePath, { force: true });
          await rm(config.manifestPath, { force: true });
        }
        const [manifest, conversationTokenizer] = await Promise.all([
          prepareIndexManifest(),
          getTokenizer(),
        ]);
        store = openStore('write');
        const indexSummary = await indexChangedConversationSessions({
          sessionsDirectory: config.sessionsDirectory,
          statePath: config.statePath,
          store,
          embeddingProvider,
          tokenizer: conversationTokenizer,
          ignoredPhysicalSessionPaths,
          chunkPolicy: {
            maxTokens: manifest.chunkPolicy.maxTokens,
            overlapTokens: manifest.chunkPolicy.overlapTokens,
          },
          resolveProjectIdentity: resolveSearchProjectIdentity,
          rebuild: options.rebuild ?? false,
          ...(options.signal ? { signal: options.signal } : {}),
          ...(options.onProgress ? { onProgress: options.onProgress } : {}),
        });
        if (
          options.optimize &&
          (indexSummary.newlyEmbeddedChunks > 0 ||
            indexSummary.deletedChunks > 0 ||
            indexSummary.indexedSessions > 0)
        ) {
          options.onProgress?.({ kind: 'optimizing-collection' });
          await store.optimize();
        }
        const totalChunks = store.count();
        if (options.signal?.aborted) {
          throw new Error('Recall conversation operation cancelled', {
            cause: options.signal.reason,
          });
        }
        await writeRecallIndexMaintenanceStatus(config.indexMaintenanceStatusPath, {
          version: 1,
          completedAt: getCurrentTime().toISOString(),
          scannedSessions: indexSummary.scannedSessions,
          failedSessions: indexSummary.failedSessions.length,
        });
        options.onProgress?.({ kind: 'completed' });
        return { indexSummary, totalChunks };
      } finally {
        store?.close();
        await releaseLock();
      }
    },
  };
}

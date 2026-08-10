import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  combineCompactRecallResults,
  COMPACT_RECALL_MIXED_RESULT_POLICY_VERSION,
  type CompactRecallConversationResult,
  type CompactRecallInvocationResult,
  type CompactRecallSearchResult,
} from './combine-compact-recall-results.js';
import {
  DENSE_RECALL_CONVERSATION_STORE_IDENTITY,
  openDenseRecallConversationStore,
  type DenseRecallConversationStore,
} from './dense-recall-conversation-store.js';
import {
  indexChangedConversationSessions,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import { listIgnoredPhysicalSessionPaths } from './physical-session-ignore.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { createOctenHttpEmbeddingProvider } from './octen-http-embedding-provider.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  activateRecallDatabaseCandidate,
  activateStagedRecallDatabase,
  createRecallDatabaseCandidate,
  resolveActiveRecallDatabasePaths,
  restorePreviousRecallDatabase,
  resumeRecallDatabaseCandidate,
  stageRecallDatabaseCandidate,
  type RecallDatabaseCandidate,
  type RecallDatabasePaths,
} from './recall-database-generation.js';
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
import { openRecallCatalog } from './recall-catalog.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  createLineageResolver,
  resolveProjectIdentity,
  type ProjectIdentity,
  type RecallProjectLineages,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import {
  rankDenseRecallSearchResults,
  RECALL_ACTIVE_BRANCH_PRIOR,
} from './rank-recall-search-results.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import { searchSessionSourceFiles, type SessionSourceSearch } from './session-source-search.js';

/** Paths, one Octen profile, and bounded compact-layout retrieval settings. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  databasePath: string;
  catalogPath: string;
  statePath: string;
  manifestPath: string;
  indexMaintenanceStatusPath: string;
  physicalSessionIgnoreStatePath: string;
  tokenizerCacheDirectory: string;
  lockPath: string;
  databaseGenerationRootPath?: string;
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

/** Per-fast-store candidate caps applied before mixed-result selection. */
export interface RecallSearchCandidateLimits {
  dense: number;
  invocation: number;
}

/** Trusted invocation context and cancellation for one read-only recall search. */
export interface RecallConversationSearchOptions {
  scope?: RecallSearchScope;
  invocationDirectory?: string;
  signal?: AbortSignal;
}

/** Reports the fixed compact mixed-result settings used by one search. */
export interface RecallSearchPolicy {
  scope: RecallSearchScope;
  invocationProjectIdentity: ProjectIdentity | null;
  rankingMode: 'compact';
  mixedResultPolicyVersion: number;
  activeBranchPrior: number;
  candidateLimits: RecallSearchCandidateLimits;
}

/** One ranked dense conversation result labeled by its relationship to the invoking project. */
export type RecallConversationSearchResult = CompactRecallConversationResult;

/** One normal recall result from the dense conversation store or compact Invocation catalog. */
export type RecallNormalSearchResult = CompactRecallSearchResult;

/** One read-only combined query against a compatible compact recall database. */
export interface RecallConversationSearch {
  results: RecallNormalSearchResult[];
  totalChunks: number;
  documentCounts: { dense: number; invocations: number };
  searchPolicy: RecallSearchPolicy;
  indexMaintenanceStatus: RecallIndexMaintenanceStatus | null;
}

/** Options accepted only by explicit `psr` index maintenance. */
export interface RecallConversationIndexOptions {
  rebuild?: boolean;
  deferActivation?: boolean;
  resumeCandidate?: boolean;
  signal?: AbortSignal;
  onProgress?: (event: RecallIndexProgressEvent) => void;
}

/** Cancellation and progress options for one recall database pointer operation. */
export interface RecallDatabasePointerOperationOptions {
  signal?: AbortSignal;
  onProgress?: (event: RecallIndexProgressEvent) => void;
}

/** Database state after one completed standalone index update. */
export type RecallDatabaseTransition =
  | { kind: 'active-updated' }
  | { kind: 'candidate-activated'; previousAvailable: boolean; staleCandidatesRemoved: number }
  | { kind: 'candidate-staged'; databaseTarget: string; staleCandidatesRemoved: number }
  | { kind: 'candidate-failed'; staleCandidatesRemoved: number };

/** Counts and database activation state from one completed standalone index update. */
export interface RecallConversationIndexResult {
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
  documentCounts: { dense: number; invocations: number };
  databaseTransition: RecallDatabaseTransition;
}

/** Result of explicitly activating one certified staged recall database. */
export interface RecallConversationActivationResult {
  kind: 'staged-activated';
  previousAvailable: boolean;
}

/** Result of restoring the database immediately preceding the active generation. */
export interface RecallConversationRollbackResult {
  kind: 'previous-restored';
  requiresLegacyRelease: boolean;
}

/** Read-only indexed search and standalone indexing for one recall collection. */
export interface RecallConversationService {
  search(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<RecallConversationSearch>;
  index(options?: RecallConversationIndexOptions): Promise<RecallConversationIndexResult>;
}

/** Explicit slow source-search capability used only when the Pi tool requests it. */
export interface RecallSourceSearchService {
  searchSource(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<SessionSourceSearch>;
}

/** Complete read-only search surface exposed through the Pi recall tool. */
export interface RecallConversationToolService
  extends RecallConversationService, RecallSourceSearchService {}

/** Standalone collection maintenance capabilities used only by `psr`. */
export interface RecallConversationMaintenanceService extends RecallConversationService {
  activate(
    databaseTarget: string,
    options?: RecallDatabasePointerOperationOptions,
  ): Promise<RecallConversationActivationResult>;
  rollback(
    options?: RecallDatabasePointerOperationOptions,
  ): Promise<RecallConversationRollbackResult>;
}

/** Injectable boundaries for public-seam tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddingProvider?: RecallEmbeddingProvider;
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write', databasePath: string) => DenseRecallConversationStore;
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
): RecallConversationMaintenanceService & RecallSourceSearchService {
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
    ((mode, databasePath) =>
      openDenseRecallConversationStore({
        databasePath,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      }));
  let tokenizer: ConversationTextTokenizer | undefined;

  async function resolveSearchInvocation(options: RecallConversationSearchOptions) {
    const scope = options.scope ?? RecallSearchScope.PROJECT;
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
    return { scope, invocationProject };
  }

  function createExpectedManifest(): RecallIndexManifest {
    return createRecallIndexManifest({
      embeddingIdentity: createEmbeddingModelIdentity(config),
      ...(dependencies.tokenizerIdentity
        ? { tokenizerIdentity: dependencies.tokenizerIdentity }
        : {}),
      chunkPolicy: config.chunkPolicy,
      projectLineages: config.projectLineages,
      denseConversationStoreIdentity: DENSE_RECALL_CONVERSATION_STORE_IDENTITY,
    });
  }

  function classifyEvidenceRelation(
    projectAttribution: ResolvedProjectIdentity | null,
    invocationProject: ResolvedProjectIdentity | null,
  ): RecallEvidenceRelation {
    if (
      !invocationProject ||
      invocationProject.projectIdentity !== projectAttribution?.projectIdentity
    ) {
      return RecallEvidenceRelation.UNRESTRICTED_GLOBAL;
    }
    if (
      projectAttribution.identitySource ===
        RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE ||
      invocationProject.identitySource === RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE
    ) {
      return RecallEvidenceRelation.CONFIGURED_PROJECT_LINEAGE;
    }
    return invocationProject.identitySource === RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN
      ? RecallEvidenceRelation.SAME_SESSION_ORIGIN
      : RecallEvidenceRelation.SAME_REPOSITORY;
  }

  async function getTokenizer(): Promise<ConversationTextTokenizer> {
    tokenizer ??= await loadTokenizer();
    return tokenizer;
  }

  async function readCompatibleManifest(paths: RecallDatabasePaths): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(paths.manifestPath);
    const expected = createExpectedManifest();
    assertRecallIndexManifestCompatible(actual, expected, paths.manifestPath);
    return actual;
  }

  async function prepareIndexManifest(paths: RecallDatabasePaths): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(paths.manifestPath);
    const expected = createExpectedManifest();
    if (
      !actual &&
      (existsSync(paths.databasePath) ||
        existsSync(paths.catalogPath) ||
        existsSync(paths.statePath))
    ) {
      throw new Error(
        `Recall index manifest missing at ${paths.manifestPath} for existing index data; rebuild with psr index --rebuild`,
      );
    }
    if (actual) {
      assertRecallIndexManifestCompatible(actual, expected, paths.manifestPath);
      return actual;
    }
    await writeRecallIndexManifest(paths.manifestPath, expected);
    return expected;
  }

  return {
    async search(query, limit, options = {}) {
      const searchQuery = query.trim();
      if (!searchQuery) {
        throw new Error('Recall query must not be blank');
      }
      const activePaths = await resolveActiveRecallDatabasePaths(config);
      await readCompatibleManifest(activePaths);
      const indexMaintenanceStatus = await readRecallIndexMaintenanceStatus(
        activePaths.indexMaintenanceStatusPath,
      );
      await assertRecallIndexUnlockedForSearch(config.lockPath);
      const { scope, invocationProject } = await resolveSearchInvocation(options);
      const projectIdentity =
        scope === RecallSearchScope.PROJECT ? invocationProject?.projectIdentity : undefined;
      const queryEmbedding = await embeddingProvider.embedQuery(searchQuery, options.signal);
      await assertRecallIndexUnlockedForSearch(config.lockPath);
      const catalog = openRecallCatalog(activePaths.catalogPath, { readOnly: true });
      try {
        const store = openStore('read', activePaths.databasePath);
        try {
          const denseCandidates = store.searchDenseCandidates(
            queryEmbedding,
            config.searchCandidateLimits.dense,
            projectIdentity,
          );
          const conversations: RecallConversationSearchResult[] = rankDenseRecallSearchResults(
            denseCandidates,
            config.searchCandidateLimits.dense,
            store.fetchDocuments,
          ).map((result) => ({
            ...result,
            resultKind: 'conversation',
            evidenceRelation: classifyEvidenceRelation(
              result.projectAttribution,
              invocationProject,
            ),
          }));
          const invocations: CompactRecallInvocationResult[] = catalog
            .searchInvocations(
              searchQuery,
              config.searchCandidateLimits.invocation,
              projectIdentity,
            )
            .map((result) => ({
              ...result,
              resultKind: 'invocation',
              content: result.searchableText,
              evidenceRelation: classifyEvidenceRelation(
                result.projectAttribution,
                invocationProject,
              ),
            }));
          const documentCounts = {
            dense: store.countDocuments(),
            invocations: catalog.countInvocations(),
          };
          return {
            results: combineCompactRecallResults(conversations, invocations, limit),
            totalChunks: documentCounts.dense,
            documentCounts,
            searchPolicy: {
              scope,
              invocationProjectIdentity: invocationProject?.projectIdentity ?? null,
              rankingMode: 'compact',
              mixedResultPolicyVersion: COMPACT_RECALL_MIXED_RESULT_POLICY_VERSION,
              activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
              candidateLimits: { ...config.searchCandidateLimits },
            },
            indexMaintenanceStatus,
          };
        } finally {
          store.close();
        }
      } finally {
        catalog.close();
      }
    },

    async searchSource(query, limit, options = {}) {
      const searchQuery = query.trim();
      if (!searchQuery) {
        throw new Error('Recall query must not be blank');
      }
      const { scope, invocationProject } = await resolveSearchInvocation(options);
      const ignoredPhysicalSessionPaths = new Set(
        await listIgnoredPhysicalSessionPaths(config.physicalSessionIgnoreStatePath),
      );
      return searchSessionSourceFiles({
        sessionsDirectory: config.sessionsDirectory,
        ignoredPhysicalSessionPaths,
        query: searchQuery,
        limit,
        scope,
        invocationProjectIdentity: invocationProject?.projectIdentity ?? null,
        resolveProjectIdentity: resolveSearchProjectIdentity,
        ...(options.signal ? { signal: options.signal } : {}),
      });
    },

    async index(options = {}) {
      if (options.deferActivation && !options.rebuild) {
        throw new Error('Recall deferred activation requires a rebuild');
      }
      if (options.resumeCandidate && (!options.rebuild || !options.deferActivation)) {
        throw new Error('Recall candidate resume requires a staged rebuild');
      }
      const maintenanceLockPath =
        options.rebuild && options.deferActivation
          ? `${config.lockPath}.candidate-construction`
          : config.lockPath;
      const releaseLock = await acquireRecallConversationLock(
        maintenanceLockPath,
        options.signal,
        options.onProgress
          ? () => options.onProgress?.({ kind: 'waiting-for-write-lock' })
          : undefined,
      );
      let store: DenseRecallConversationStore | undefined;
      let candidate: RecallDatabaseCandidate | undefined;
      try {
        const ignoredPhysicalSessionPaths: ReadonlySet<string> = new Set(
          await listIgnoredPhysicalSessionPaths(config.physicalSessionIgnoreStatePath),
        );
        candidate =
          options.rebuild && config.databaseGenerationRootPath
            ? options.resumeCandidate
              ? await resumeRecallDatabaseCandidate(config)
              : await createRecallDatabaseCandidate(config)
            : undefined;
        if (candidate) {
          options.onProgress?.(
            options.resumeCandidate
              ? { kind: 'resuming-rebuild-candidate' }
              : {
                  kind: 'preparing-rebuild-candidate',
                  staleCandidatesRemoved: candidate.staleCandidatesRemoved,
                },
          );
        }
        const activePaths = candidate
          ? candidate.paths
          : await resolveActiveRecallDatabasePaths(config);
        if (options.rebuild && !candidate) {
          await rm(activePaths.indexMaintenanceStatusPath, { force: true });
          await rm(activePaths.databasePath, { recursive: true, force: true });
          await rm(activePaths.catalogPath, { force: true });
          await rm(activePaths.statePath, { force: true });
          await rm(activePaths.manifestPath, { force: true });
        }
        const [manifest, conversationTokenizer] = await Promise.all([
          prepareIndexManifest(activePaths),
          getTokenizer(),
        ]);
        store = openStore('write', activePaths.databasePath);
        const indexSummary = await indexChangedConversationSessions({
          sessionsDirectory: config.sessionsDirectory,
          catalogPath: activePaths.catalogPath,
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
        const documentCounts = {
          dense: store.countDocuments(),
          invocations: 0,
        };
        const catalog = openRecallCatalog(activePaths.catalogPath, { readOnly: true });
        try {
          documentCounts.invocations = catalog.countInvocations();
        } finally {
          catalog.close();
        }
        const totalChunks = documentCounts.dense;
        if (options.signal?.aborted) {
          throw new Error('Recall conversation operation cancelled', {
            cause: options.signal.reason,
          });
        }
        await writeRecallIndexMaintenanceStatus(activePaths.indexMaintenanceStatusPath, {
          version: 1,
          completedAt: getCurrentTime().toISOString(),
          scannedSessions: indexSummary.scannedSessions,
          failedSessions: indexSummary.failedSessions.length,
        });
        store.close();
        store = undefined;
        let databaseTransition: RecallDatabaseTransition = { kind: 'active-updated' };
        if (candidate) {
          if (indexSummary.failedSessions.length > 0) {
            options.onProgress?.({ kind: 'rebuild-candidate-failed' });
            databaseTransition = {
              kind: 'candidate-failed',
              staleCandidatesRemoved: candidate.staleCandidatesRemoved,
            };
          } else if (options.deferActivation) {
            const staged = await stageRecallDatabaseCandidate(config, candidate);
            options.onProgress?.({
              kind: 'rebuild-candidate-staged',
              databaseTarget: staged.databaseTarget,
            });
            databaseTransition = {
              kind: 'candidate-staged',
              databaseTarget: staged.databaseTarget,
              staleCandidatesRemoved: candidate.staleCandidatesRemoved,
            };
          } else {
            const activation = await activateRecallDatabaseCandidate(config, candidate);
            options.onProgress?.({
              kind: 'rebuild-candidate-activated',
              previousAvailable: activation.previousAvailable,
            });
            databaseTransition = {
              kind: 'candidate-activated',
              previousAvailable: activation.previousAvailable,
              staleCandidatesRemoved: candidate.staleCandidatesRemoved,
            };
          }
        }
        options.onProgress?.({ kind: 'completed' });
        return { indexSummary, totalChunks, documentCounts, databaseTransition };
      } catch (error) {
        if (candidate) {
          options.onProgress?.({ kind: 'rebuild-candidate-failed' });
        }
        throw error;
      } finally {
        store?.close();
        await releaseLock();
      }
    },

    async activate(databaseTarget, options = {}) {
      const releaseLock = await acquireRecallConversationLock(
        config.lockPath,
        options.signal,
        options.onProgress
          ? () => options.onProgress?.({ kind: 'waiting-for-write-lock' })
          : undefined,
      );
      try {
        const activation = await activateStagedRecallDatabase(config, databaseTarget);
        return { kind: 'staged-activated', previousAvailable: activation.previousAvailable };
      } finally {
        await releaseLock();
      }
    },

    async rollback(options = {}) {
      const releaseLock = await acquireRecallConversationLock(
        config.lockPath,
        options.signal,
        options.onProgress
          ? () => options.onProgress?.({ kind: 'waiting-for-write-lock' })
          : undefined,
      );
      try {
        const restoration = await restorePreviousRecallDatabase(config);
        return { kind: 'previous-restored', ...restoration };
      } finally {
        await releaseLock();
      }
    },
  };
}

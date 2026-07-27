import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import type {
  RecallConversationConfig,
  RecallSearchCandidateLimits,
} from './recall-conversation-config.js';
import {
  RecallBackgroundIndexProcessState,
  RecallDiagnosticErrorCategory,
  RecallDiagnosticStatus,
  RecallEvidenceRelation,
  RecallLifecycleTrigger,
  RecallManualMaintenanceTrigger,
  RecallProjectIdentitySource,
  RecallSearchScope,
} from './enums.js';
import {
  fuseRecallSearchCandidates,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  indexChangedConversationSession,
  indexChangedConversationSessions,
  readRecallConversationIndexStateSummary,
  type ConversationIndexCheckpoint,
  type ConversationIndexProgress,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import {
  markRecallBackgroundIndexGenerationDiscarded,
  readRecallBackgroundIndexGenerationStatus,
  removeRecallBackgroundIndexWorkerRequest,
  resumeRecallBackgroundIndexGeneration,
  startRecallBackgroundIndexGeneration,
  stopRecallBackgroundIndexGeneration,
  type RecallBackgroundIndexCoordinatorConfig,
  type RecallBackgroundIndexGenerationStatus,
  type RecallBackgroundIndexServiceFactory,
} from './recall-background-index-build.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import {
  createOctenEmbeddingModelProfile,
  createQwenRerankingModelProfile,
  type RecallEmbeddingModelProfile,
  type RecallQueryPlanningModelProfile,
  type RecallRerankingModelProfile,
} from './recall-model-profiles.js';
import {
  createRecallRerankingExecutionIdentity,
  type RecallEmbeddingProvider,
  type RecallIdentifiedQueryPlanningProvider,
  type RecallQueryPlanningExecutionIdentity,
  type RecallRerankingExecutionIdentity,
  type RecallRerankingProvider,
} from './recall-inference-capabilities.js';
import {
  measureRecallQueryPlanningProviderConformance,
  type RecallQueryPlanningProviderConformanceMeasurement,
} from './recall-inference-conformance.js';
import { createLlamaCppHttpEmbeddingProvider } from './llama-cpp-http-embedding-provider.js';
import { createQwenHttpRerankingProvider } from './qwen-http-reranking-provider.js';
import {
  activateStagingRecallIndexGeneration,
  discardStagingRecallIndexGeneration,
  prepareStagingRecallIndexGeneration,
  preserveStagingRecallIndexGeneration,
  readRecallIndexGenerationStatus,
  resolveActiveRecallIndexGeneration,
  type RecallIndexGenerationCoordinatorConfig,
  type RecallIndexGenerationPaths,
  type RecallIndexGenerationStatus,
} from './recall-index-generations.js';
import {
  assertRecallIndexManifestCompatible,
  calculateRecallEmbeddingCanaryCosineSimilarity,
  createRecallIndexManifest,
  readRecallIndexManifest,
  recoverRecallEmbeddingCanaryFromManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
  type RecallIndexManifest,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import {
  createRecallIndexMetrics,
  createRecallOperationDiagnostics,
  createRecallSearchDiagnosticMetrics,
  type RecallDiagnosticsClock,
  type RecallIndexDiagnosticMetrics,
  type RecallOperationDiagnostics,
  type RecallPhysicalSessionDiagnostic,
  type RecallSearchDiagnosticMetrics,
} from './recall-operation-diagnostics.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  createLineageResolver,
  resolveProjectIdentity,
  type ProjectIdentity,
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

export type {
  RecallConversationConfig,
  RecallSearchCandidateLimits,
} from './recall-conversation-config.js';

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

/** Profile, adapter, and cache identity for the reranker used by one deep search. */
export interface RecallSearchRerankerIdentity {
  profileId: string;
  adapterId: string;
  cacheIdentity: string;
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
  rerankerIdentity: RecallSearchRerankerIdentity | null;
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

interface RecallConversationIndexBaseOptions {
  signal?: AbortSignal;
  lockWaitMilliseconds?: number;
  requireExistingGeneration?: boolean;
  onProgress?: (progress: ConversationIndexProgress) => void;
  onCheckpoint?: (checkpoint: ConversationIndexCheckpoint) => void;
  optimize?: boolean;
}

interface RecallAutomaticIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: boolean;
  manualMaintenanceTrigger?: never;
}

interface RecallManualIncrementalIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: false;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX;
}

interface RecallManualRebuildIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild: true;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD;
}

/** Valid automatic or manually attributed conversation index invocation. */
export type RecallConversationIndexOptions =
  | RecallAutomaticIndexOptions
  | RecallManualIncrementalIndexOptions
  | RecallManualRebuildIndexOptions;

/** Cancellation, lock wait, and Pi lifecycle attribution for one targeted reconciliation. */
export interface RecallConversationReconcileOptions {
  lifecycleTrigger: RecallLifecycleTrigger;
  signal?: AbortSignal;
  lockWaitMilliseconds?: number;
}

/** Counts from one completed full or targeted conversation index update. */
export interface RecallConversationIndexResult {
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
}

/** Cancellation accepted by independent query planner setup verification. */
export interface RecallQueryPlanningCapabilityVerificationOptions {
  signal?: AbortSignal;
}

/** Inspectable profile, adapter, policy, and measurement from accepted planner conformance. */
export interface RecallQueryPlanningCapabilityVerification {
  profileId: string;
  model: string;
  promptPolicy: string;
  grammarVersion: string;
  executionIdentity: Readonly<RecallQueryPlanningExecutionIdentity>;
  measurement: RecallQueryPlanningProviderConformanceMeasurement;
}

/** Search plus inference verification and explicit index-generation operations. */
export interface RecallConversationService {
  verifyQueryPlanningCapability(
    options?: RecallQueryPlanningCapabilityVerificationOptions,
  ): Promise<RecallQueryPlanningCapabilityVerification>;
  search(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<RecallConversationSearch>;
  index(options?: RecallConversationIndexOptions): Promise<RecallConversationIndexResult>;
  startBackgroundIndexGeneration(): Promise<RecallBackgroundIndexGenerationStatus>;
  resumeBackgroundIndexGeneration(): Promise<RecallBackgroundIndexGenerationStatus>;
  readBackgroundIndexGenerationStatus(): Promise<RecallBackgroundIndexGenerationStatus | null>;
  stopBackgroundIndexGeneration(): Promise<RecallBackgroundIndexGenerationStatus>;
  readIndexGenerationStatus(): Promise<RecallIndexGenerationStatus>;
  discardStagingIndexGeneration(): Promise<boolean>;
  reconcileSession(
    sessionPath: string,
    options: RecallConversationReconcileOptions,
  ): Promise<RecallConversationIndexResult>;
}

/** Injectable inference, tokenizer, and zvec boundaries used by tests and bounded evaluation. */
export interface RecallConversationDependencies {
  /** Model semantics authoritative for manifest compatibility across embedded and HTTP execution. */
  embeddingProfile?: RecallEmbeddingModelProfile;
  /** Capability-specific embedding provider; preferred over the legacy shared operation. */
  embeddingProvider?: RecallEmbeddingProvider;
  /** @deprecated Use embeddingProvider so query and document semantics stay distinct. */
  embeddings?: LocalEmbeddingClient;
  /** Model semantics recorded in deep-search policy without affecting vector compatibility. */
  rerankingProfile?: RecallRerankingModelProfile;
  reranker?: RecallRerankingProvider;
  /** Explicit identity for an injected custom reranker adapter. */
  rerankerExecutionIdentity?: RecallRerankingExecutionIdentity;
  /** Optional query planner semantics verified independently and excluded from vector identity. */
  queryPlanningProfile?: RecallQueryPlanningModelProfile;
  /** Optional identified planner adapter; query-planned retrieval remains outside this ticket. */
  queryPlanner?: RecallIdentifiedQueryPlanningProvider;
  /** Profile-owned tokenizer identity recorded with chunk geometry in the index manifest. */
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write', databasePath?: string) => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  diagnostics?: RecallOperationDiagnostics;
  diagnosticsClock?: RecallDiagnosticsClock;
  notifyWarning?: (message: string) => void;
  /** Reconstructs custom inference and indexing dependencies inside the detached worker. */
  backgroundIndexServiceFactory?: RecallBackgroundIndexServiceFactory;
}

interface LiveSessionReconciliationDiagnosticRunOptions {
  diagnostics: RecallOperationDiagnostics;
  lifecycleTrigger: RecallLifecycleTrigger;
  sessionPath: string;
  signal?: AbortSignal;
  reconcile: (
    diagnosticMetrics: RecallIndexDiagnosticMetrics,
  ) => Promise<RecallConversationIndexResult>;
}

interface RecallSearchDiagnosticRunOptions {
  diagnostics: RecallOperationDiagnostics;
  searchMode: RecallSearchMode;
  recallScope: RecallSearchScope;
  signal?: AbortSignal;
  search: (diagnosticMetrics: RecallSearchDiagnosticMetrics) => Promise<RecallConversationSearch>;
}

interface ManualIndexDiagnosticRunOptions {
  diagnostics: RecallOperationDiagnostics;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger;
  signal?: AbortSignal;
  runIndexMaintenance: (
    diagnosticMetrics: RecallIndexDiagnosticMetrics,
    onPhysicalSessionCheck: (completion: RecallPhysicalSessionDiagnostic) => void,
    runOptimizationWithDiagnostics: (optimize: () => Promise<void>) => Promise<void>,
  ) => Promise<RecallConversationIndexResult>;
}

function assertRecallManualMaintenanceTriggerMatchesIndexOptions(
  options: RecallConversationIndexOptions,
): void {
  if (!options.manualMaintenanceTrigger) {
    return;
  }
  const expectedTrigger = options.rebuild
    ? RecallManualMaintenanceTrigger.MANUAL_REBUILD
    : RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX;
  if (options.manualMaintenanceTrigger !== expectedTrigger) {
    throw new Error(
      `Recall manual maintenance trigger mismatch: expected ${expectedTrigger}, received ${options.manualMaintenanceTrigger}`,
    );
  }
}

function isRecallOperationCancelled(error: unknown, signal?: AbortSignal): boolean {
  return (
    signal?.aborted === true ||
    (error instanceof Error &&
      (error.message === 'Recall conversation operation cancelled' ||
        error.message === 'Recall conversation indexing cancelled'))
  );
}

async function runLiveSessionReconciliationWithDiagnostics(
  options: LiveSessionReconciliationDiagnosticRunOptions,
): Promise<RecallConversationIndexResult> {
  const diagnosticOperation = options.diagnostics.startLiveSessionReconciliation({
    lifecycleTrigger: options.lifecycleTrigger,
    sessionPath: options.sessionPath,
  });
  const diagnosticMetrics = createRecallIndexMetrics();
  try {
    const result = await options.reconcile(diagnosticMetrics);
    diagnosticOperation.complete({
      status: RecallDiagnosticStatus.SUCCEEDED,
      metrics: diagnosticMetrics,
      scannedSessionCount: result.indexSummary.scannedSessions,
      indexedSessionCount: result.indexSummary.indexedSessions,
      removedSessionCount: result.indexSummary.removedSessions,
      failedSessionCount: result.indexSummary.failedSessions.length,
      cacheHitCount: result.indexSummary.cacheHits,
      newEmbeddingCount: result.indexSummary.newlyEmbeddedChunks,
      embeddingRequestCount: result.indexSummary.embeddingRequestCount,
      deletedDocumentCount: result.indexSummary.deletedChunks,
      totalDocumentCount: result.totalChunks,
    });
    return result;
  } catch (error) {
    const cancelled = isRecallOperationCancelled(error, options.signal);
    diagnosticOperation.complete({
      status: cancelled ? RecallDiagnosticStatus.CANCELLED : RecallDiagnosticStatus.FAILED,
      errorCategory: cancelled
        ? RecallDiagnosticErrorCategory.OPERATION_CANCELLED
        : RecallDiagnosticErrorCategory.OPERATION_FAILED,
      metrics: diagnosticMetrics,
      scannedSessionCount: diagnosticMetrics.sourceByteSize === null ? 0 : 1,
      indexedSessionCount: 0,
      removedSessionCount: 0,
      failedSessionCount: cancelled ? 0 : 1,
      cacheHitCount: diagnosticMetrics.cacheHitCount,
      newEmbeddingCount: diagnosticMetrics.newEmbeddingCount,
      embeddingRequestCount: diagnosticMetrics.embeddingRequestCount,
      deletedDocumentCount: diagnosticMetrics.deletedDocumentCount,
      totalDocumentCount: null,
    });
    throw error;
  }
}

async function runManualIndexWithDiagnostics(
  options: ManualIndexDiagnosticRunOptions,
): Promise<RecallConversationIndexResult> {
  const diagnosticOperation = options.diagnostics.startManualIndexMaintenance({
    manualMaintenanceTrigger: options.manualMaintenanceTrigger,
  });
  const diagnosticMetrics = createRecallIndexMetrics();
  async function runOptimizationWithDiagnostics(optimize: () => Promise<void>): Promise<void> {
    const optimizationDiagnostic = diagnosticOperation.startOptimization();
    try {
      await optimize();
    } catch (error) {
      const cancelled = isRecallOperationCancelled(error, options.signal);
      optimizationDiagnostic.complete({
        status: cancelled ? RecallDiagnosticStatus.CANCELLED : RecallDiagnosticStatus.FAILED,
        errorCategory: cancelled
          ? RecallDiagnosticErrorCategory.OPERATION_CANCELLED
          : RecallDiagnosticErrorCategory.OPERATION_FAILED,
      });
      throw error;
    }
    optimizationDiagnostic.complete({ status: RecallDiagnosticStatus.SUCCEEDED });
  }

  let result: RecallConversationIndexResult;
  try {
    result = await options.runIndexMaintenance(
      diagnosticMetrics,
      (completion) => {
        diagnosticOperation.recordPhysicalSessionCheck(completion);
      },
      runOptimizationWithDiagnostics,
    );
  } catch (error) {
    const cancelled = isRecallOperationCancelled(error, options.signal);
    diagnosticOperation.complete({
      status: cancelled ? RecallDiagnosticStatus.CANCELLED : RecallDiagnosticStatus.FAILED,
      errorCategory: cancelled
        ? RecallDiagnosticErrorCategory.OPERATION_CANCELLED
        : RecallDiagnosticErrorCategory.OPERATION_FAILED,
      metrics: diagnosticMetrics,
      scannedSessionCount: diagnosticMetrics.scannedSessionCount,
      indexedSessionCount: diagnosticMetrics.indexedSessionCount,
      removedSessionCount: diagnosticMetrics.removedSessionCount,
      failedSessionCount: diagnosticMetrics.failedSessionCount,
      cacheHitCount: diagnosticMetrics.cacheHitCount,
      newEmbeddingCount: diagnosticMetrics.newEmbeddingCount,
      embeddingRequestCount: diagnosticMetrics.embeddingRequestCount,
      deletedDocumentCount: diagnosticMetrics.deletedDocumentCount,
      totalDocumentCount: null,
    });
    throw error;
  }
  const failed = result.indexSummary.failedSessions.length > 0;
  diagnosticOperation.complete({
    status: failed ? RecallDiagnosticStatus.FAILED : RecallDiagnosticStatus.SUCCEEDED,
    ...(failed ? { errorCategory: RecallDiagnosticErrorCategory.OPERATION_FAILED } : {}),
    metrics: diagnosticMetrics,
    scannedSessionCount: result.indexSummary.scannedSessions,
    indexedSessionCount: result.indexSummary.indexedSessions,
    removedSessionCount: result.indexSummary.removedSessions,
    failedSessionCount: result.indexSummary.failedSessions.length,
    cacheHitCount: result.indexSummary.cacheHits,
    newEmbeddingCount: result.indexSummary.newlyEmbeddedChunks,
    embeddingRequestCount: result.indexSummary.embeddingRequestCount,
    deletedDocumentCount: result.indexSummary.deletedChunks,
    totalDocumentCount: result.totalChunks,
  });
  return result;
}

async function runRecallSearchWithDiagnostics(
  options: RecallSearchDiagnosticRunOptions,
): Promise<RecallConversationSearch> {
  const diagnosticOperation = options.diagnostics.startRecallSearch({
    searchMode: options.searchMode,
    recallScope: options.recallScope,
  });
  const diagnosticMetrics = createRecallSearchDiagnosticMetrics();
  try {
    const result = await options.search(diagnosticMetrics);
    diagnosticOperation.complete({
      status: RecallDiagnosticStatus.SUCCEEDED,
      metrics: diagnosticMetrics,
      totalDocumentCount: result.totalChunks,
    });
    return result;
  } catch (error) {
    const cancelled = isRecallOperationCancelled(error, options.signal);
    diagnosticOperation.complete({
      status: cancelled ? RecallDiagnosticStatus.CANCELLED : RecallDiagnosticStatus.FAILED,
      errorCategory: cancelled
        ? RecallDiagnosticErrorCategory.OPERATION_CANCELLED
        : RecallDiagnosticErrorCategory.OPERATION_FAILED,
      metrics: diagnosticMetrics,
      totalDocumentCount: null,
    });
    throw error;
  }
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

function readRecallRerankingExecutionIdentity(
  provider: RecallRerankingProvider,
): RecallRerankingExecutionIdentity | undefined {
  if (!('executionIdentity' in provider)) {
    return undefined;
  }
  const identity = provider.executionIdentity;
  if (
    !isUnknownRecord(identity) ||
    typeof identity.adapterId !== 'string' ||
    (identity.backend !== 'embedded' &&
      identity.backend !== 'llama-cpp-http' &&
      identity.backend !== 'custom') ||
    typeof identity.cacheIdentity !== 'string' ||
    typeof identity.modelProfileId !== 'string'
  ) {
    return undefined;
  }
  return {
    adapterId: identity.adapterId,
    backend: identity.backend,
    cacheIdentity: identity.cacheIdentity,
    modelProfileId: identity.modelProfileId,
  };
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
  const embeddingProfile =
    dependencies.embeddingProfile ??
    createOctenEmbeddingModelProfile(createEmbeddingModelIdentity(config));
  if (dependencies.embeddingProfile && !dependencies.tokenizerIdentity) {
    throw new Error(
      'Recall embedding profile configuration incomplete: tokenizer manifest identity is required',
    );
  }
  if (
    embeddingProfile.canary &&
    (embeddingProfile.canary.expectedDimensions !== embeddingProfile.identity.dimensions ||
      embeddingProfile.canary.expectedNormalization !== embeddingProfile.identity.normalization)
  ) {
    throw new Error(
      'Recall embedding profile canary incompatible: dimensions and normalization must match the embedding identity',
    );
  }
  const legacyEmbeddings = dependencies.embeddings;
  const embeddingProvider: RecallEmbeddingProvider =
    dependencies.embeddingProvider ??
    (legacyEmbeddings
      ? {
          async embedQuery(query, signal) {
            const embedding = (await legacyEmbeddings.embedTexts([query], signal))[0];
            if (!embedding) {
              throw new Error('Recall embedding response missing query vector');
            }
            return embedding;
          },
          embedDocuments(documents, signal) {
            return legacyEmbeddings.embedTexts([...documents], signal);
          },
        }
      : createLlamaCppHttpEmbeddingProvider(embeddingProfile, {
          baseUrl: config.embeddingBaseUrl,
          batchSize: config.embeddingBatchSize,
        }));
  const rerankingProfile =
    dependencies.rerankingProfile ?? createQwenRerankingModelProfile(config.rerankerModel);
  const reranker =
    dependencies.reranker ??
    createQwenHttpRerankingProvider(rerankingProfile, {
      baseUrl: config.rerankerBaseUrl,
    });
  const providerExecutionIdentity = readRecallRerankingExecutionIdentity(reranker);
  const rerankerExecutionIdentity =
    dependencies.rerankerExecutionIdentity ??
    providerExecutionIdentity ??
    createRecallRerankingExecutionIdentity(
      rerankingProfile.profileId,
      'custom-injected-reranking-v1',
      'custom',
    );
  if (rerankerExecutionIdentity.modelProfileId !== rerankingProfile.profileId) {
    throw new Error(
      `Recall reranker profile identity mismatch: expected ${rerankingProfile.profileId}, received ${rerankerExecutionIdentity.modelProfileId}`,
    );
  }
  const queryPlanningProfile = dependencies.queryPlanningProfile;
  const queryPlanner = dependencies.queryPlanner;
  if ((queryPlanningProfile && !queryPlanner) || (!queryPlanningProfile && queryPlanner)) {
    throw new Error(
      'Recall query planner configuration incomplete: profile and identified provider are both required',
    );
  }
  if (
    queryPlanningProfile &&
    queryPlanner &&
    queryPlanner.executionIdentity.modelProfileId !== queryPlanningProfile.profileId
  ) {
    throw new Error(
      `Recall query planner profile identity mismatch: expected ${queryPlanningProfile.profileId}, received ${queryPlanner.executionIdentity.modelProfileId}`,
    );
  }
  const loadTokenizer =
    dependencies.loadTokenizer ??
    (() => loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory }));
  const resolveSearchProjectIdentity = createLineageResolver(
    config.projectLineages,
    dependencies.resolveProjectIdentity ?? resolveProjectIdentity,
  );
  const embeddingProfileId = `embedding-profile-${createHash('sha256')
    .update(
      JSON.stringify({
        identity: embeddingProfile.identity,
        queryInputPrefix: embeddingProfile.queryInputPrefix,
        documentInputPrefix: embeddingProfile.documentInputPrefix,
        canary: embeddingProfile.canary ?? null,
      }),
    )
    .digest('hex')}`;
  const generationDataDirectory = dirname(config.manifestPath);
  const generationCoordinatorConfig: RecallIndexGenerationCoordinatorConfig = {
    legacyPaths: {
      databasePath: config.databasePath,
      statePath: config.statePath,
      manifestPath: config.manifestPath,
      lockPath: config.lockPath,
    },
    generationsDirectory:
      config.generationsDirectory ?? join(generationDataDirectory, 'index-generations'),
    activeGenerationPath:
      config.activeGenerationPath ?? join(generationDataDirectory, 'active-generation.json'),
    stagingGenerationPath:
      config.stagingGenerationPath ?? join(generationDataDirectory, 'staging-generation.json'),
  };
  const backgroundIndexCoordinatorConfig: RecallBackgroundIndexCoordinatorConfig = {
    serviceConfig: config,
    generationCoordinatorConfig,
    statusPath:
      config.backgroundIndexStatusPath ??
      join(generationDataDirectory, 'background-index-status.json'),
    requestPath:
      config.backgroundIndexRequestPath ??
      join(generationDataDirectory, 'background-index-request.json'),
    embeddingProfileId,
    serviceFactory: dependencies.backgroundIndexServiceFactory ?? {
      moduleUrl: import.meta.url,
      exportName: 'createRecallConversationService',
    },
  };
  const backgroundWorkerNeedsCustomFactory =
    !dependencies.backgroundIndexServiceFactory &&
    Boolean(
      dependencies.embeddingProfile ||
      dependencies.embeddingProvider ||
      dependencies.embeddings ||
      dependencies.tokenizerIdentity ||
      dependencies.loadTokenizer ||
      dependencies.openStore ||
      dependencies.resolveProjectIdentity,
    );
  function assertBackgroundIndexWorkerCanReconstructService(): void {
    if (backgroundWorkerNeedsCustomFactory) {
      throw new Error(
        'Recall background index worker cannot reconstruct injected indexing dependencies; configure backgroundIndexServiceFactory',
      );
    }
  }
  const openStore =
    dependencies.openStore ??
    ((mode, databasePath = config.databasePath) =>
      openZvecConversationStore({
        databasePath,
        dimensions: embeddingProfile.identity.dimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      }));
  const diagnosticsClock = dependencies.diagnosticsClock ?? {
    monotonicMilliseconds: () => performance.now(),
    wallClockIsoTimestamp: () => new Date().toISOString(),
  };
  const diagnostics =
    dependencies.diagnostics ??
    createRecallOperationDiagnostics({
      mode: config.diagnosticsMode,
      activeLogPath: config.diagnosticLogPath,
      retainedLogPath: config.retainedDiagnosticLogPath,
      clock: diagnosticsClock,
      notifyWarning: dependencies.notifyWarning ?? (() => undefined),
    });
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
    if (embeddingProfile.canary) {
      const firstEmbedding = await embeddingProvider.embedQuery(
        embeddingProfile.canary.query,
        signal,
      );
      const repeatedEmbedding = await embeddingProvider.embedQuery(
        embeddingProfile.canary.query,
        signal,
      );
      const repeatCosineSimilarity = calculateRecallEmbeddingCanaryCosineSimilarity(
        firstEmbedding,
        repeatedEmbedding,
        embeddingProfile.identity.dimensions,
      );
      if (repeatCosineSimilarity < embeddingProfile.canary.minimumRepeatCosineSimilarity) {
        throw new Error(
          `Recall embedding canary repeatability mismatch: expected cosine similarity at least ${embeddingProfile.canary.minimumRepeatCosineSimilarity}, received ${repeatCosineSimilarity}`,
        );
      }
      return [...firstEmbedding];
    }
    const embedding = (
      await embeddingProvider.embedDocuments([RECALL_EMBEDDING_CANARY_TEXT], signal)
    )[0];
    if (!embedding) {
      throw new Error('Recall embedding response missing canary vector');
    }
    return [...embedding];
  }

  async function readIndexEmbeddingCanary(
    signal?: AbortSignal,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  ): Promise<number[]> {
    const embeddingRequestStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
    try {
      return await readCurrentEmbeddingCanary(signal);
    } finally {
      if (diagnosticMetrics) {
        diagnosticMetrics.embeddingServerRequestMilliseconds += Math.max(
          diagnosticsClock.monotonicMilliseconds() - embeddingRequestStartedAtMilliseconds,
          0,
        );
      }
    }
  }

  function createExpectedManifest(canaryEmbedding: readonly number[]): RecallIndexManifest {
    return createRecallIndexManifest({
      embeddingIdentity: embeddingProfile.identity,
      canaryEmbedding,
      ...(embeddingProfile.canary ? { embeddingCanary: embeddingProfile.canary } : {}),
      ...(dependencies.tokenizerIdentity
        ? { tokenizerIdentity: dependencies.tokenizerIdentity }
        : {}),
      projectLineages: config.projectLineages,
      ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
    });
  }

  async function readCanonicalRebuildCanary(
    activeManifestPath: string | undefined,
    signal?: AbortSignal,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  ): Promise<number[]> {
    const currentEmbeddingCanary = await readIndexEmbeddingCanary(signal, diagnosticMetrics);
    const currentManifest = createExpectedManifest(currentEmbeddingCanary);
    const currentCanary = currentManifest.embedding.canaryVector;
    let previousCanary:
      | {
          dimensions: number;
          canaryVector: number[];
          canaryMinimumCosineSimilarity: number;
        }
      | undefined;
    if (activeManifestPath) {
      try {
        const actualManifest = await readRecallIndexManifest(activeManifestPath);
        previousCanary = actualManifest?.embedding;
      } catch (error) {
        if (
          !(error instanceof Error) ||
          !error.message.startsWith(`Recall index manifest invalid at ${activeManifestPath}:`)
        ) {
          throw error;
        }
        previousCanary =
          (await recoverRecallEmbeddingCanaryFromManifest(
            activeManifestPath,
            embeddingProfile.identity.dimensions,
          )) ?? undefined;
      }
    }
    if (!previousCanary || previousCanary.dimensions !== embeddingProfile.identity.dimensions) {
      return currentCanary;
    }
    const cosineSimilarity = calculateRecallEmbeddingCanaryCosineSimilarity(
      previousCanary.canaryVector,
      currentCanary,
      embeddingProfile.identity.dimensions,
    );
    return cosineSimilarity >= previousCanary.canaryMinimumCosineSimilarity
      ? [...previousCanary.canaryVector]
      : currentCanary;
  }

  async function readRequiredManifest(
    generationPaths: RecallIndexGenerationPaths,
  ): Promise<RecallIndexManifest> {
    const actual = await readRecallIndexManifest(generationPaths.manifestPath);
    if (!actual) {
      throw new Error(
        `Recall index manifest missing at ${generationPaths.manifestPath}; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    return actual;
  }

  async function prepareIndexForWrite(
    generationPaths: RecallIndexGenerationPaths,
    signal?: AbortSignal,
    preflightedCanary?: readonly number[],
    requireExistingGeneration = false,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  ): Promise<{
    tokenizer: ConversationTextTokenizer;
    manifest: RecallIndexManifest;
    embeddingModelPreflighted: boolean;
  }> {
    const actual = await readRecallIndexManifest(generationPaths.manifestPath);
    if (!actual && requireExistingGeneration) {
      throw new Error(
        `Recall automatic session ingestion requires an existing index generation at ${generationPaths.manifestPath}; initialize it with /pi-session-recall-index --rebuild`,
      );
    }
    if (
      !actual &&
      (existsSync(generationPaths.databasePath) || existsSync(generationPaths.statePath))
    ) {
      throw new Error(
        `Recall index manifest missing at ${generationPaths.manifestPath} for existing index data; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    if (actual) {
      const expected = createExpectedManifest(actual.embedding.canaryVector);
      assertRecallIndexManifestCompatible(actual, expected, generationPaths.manifestPath);
      const tokenizer = await getConversationTokenizer();
      return { tokenizer, manifest: actual, embeddingModelPreflighted: false };
    }
    const expected = createExpectedManifest(
      preflightedCanary ?? (await readIndexEmbeddingCanary(signal, diagnosticMetrics)),
    );
    const tokenizer = await getConversationTokenizer();
    await writeRecallIndexManifest(generationPaths.manifestPath, expected);
    return { tokenizer, manifest: expected, embeddingModelPreflighted: true };
  }

  async function updateConversationIndex(
    generationPaths: RecallIndexGenerationPaths,
    store: ZvecConversationStore,
    tokenizer: ConversationTextTokenizer,
    manifest: RecallIndexManifest,
    embeddingModelPreflighted: boolean,
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
    onCheckpoint?: (checkpoint: ConversationIndexCheckpoint) => void,
    sessionPath?: string,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
    onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void,
  ): Promise<ConversationIndexSummary> {
    let modelPreflighted = embeddingModelPreflighted;
    async function embedTextsAfterModelPreflight(
      texts: string[],
      embeddingSignal?: AbortSignal,
    ): Promise<number[][]> {
      if (!modelPreflighted) {
        const expected = createExpectedManifest(await readCurrentEmbeddingCanary(embeddingSignal));
        assertRecallIndexManifestCompatible(manifest, expected, generationPaths.manifestPath);
        modelPreflighted = true;
      }
      return embeddingProvider.embedDocuments(texts, embeddingSignal);
    }
    const preflightedEmbeddings: LocalEmbeddingClient = {
      embedTexts: embedTextsAfterModelPreflight,
    };
    const embeddingCache = createEmbeddingVectorCache({
      cacheDirectory: config.embeddingCacheDirectory,
      identity: createEmbeddingVectorCacheIdentity(manifest),
      embeddingRequestBatchSize: config.embeddingBatchSize,
      embeddings: preflightedEmbeddings,
      diagnosticsClock,
    });
    const indexerOptions = {
      statePath: generationPaths.statePath,
      store,
      embeddingCache,
      tokenizer,
      chunkPolicy: {
        maxTokens: manifest.chunkPolicy.maxTokens,
        overlapTokens: manifest.chunkPolicy.overlapTokens,
      },
      resolveProjectIdentity: resolveSearchProjectIdentity,
      ...(signal ? { signal } : {}),
      ...(diagnosticMetrics ? { diagnosticMetrics, diagnosticsClock } : {}),
      ...(onPhysicalSessionCheck ? { onPhysicalSessionCheck } : {}),
      ...(onCheckpoint ? { onCheckpoint } : {}),
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
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  ): Promise<RecallConversationIndexResult> {
    const activeGeneration = await resolveActiveRecallIndexGeneration(generationCoordinatorConfig);
    const generationPaths = activeGeneration?.paths ?? generationCoordinatorConfig.legacyPaths;
    const lockSignal = createRecallLockAcquisitionSignal(signal, lockWaitMilliseconds);
    const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
    let releaseLock: (() => Promise<void>) | undefined;
    try {
      releaseLock = await acquireRecallConversationLock(generationPaths.lockPath, lockSignal);
    } finally {
      if (diagnosticMetrics) {
        diagnosticMetrics.writerLockWaitMilliseconds += Math.max(
          diagnosticsClock.monotonicMilliseconds() - lockStartedAtMilliseconds,
          0,
        );
      }
    }
    let store: ZvecConversationStore | undefined;
    try {
      const preparationStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
      let preparedIndex: Awaited<ReturnType<typeof prepareIndexForWrite>>;
      try {
        preparedIndex = await prepareIndexForWrite(generationPaths, signal, undefined, true);
        store = openStore('write', generationPaths.databasePath);
      } finally {
        if (diagnosticMetrics) {
          diagnosticMetrics.manifestStorePreparationMilliseconds += Math.max(
            diagnosticsClock.monotonicMilliseconds() - preparationStartedAtMilliseconds,
            0,
          );
        }
      }
      const summary = await updateConversationIndex(
        generationPaths,
        store,
        preparedIndex.tokenizer,
        preparedIndex.manifest,
        preparedIndex.embeddingModelPreflighted,
        signal,
        undefined,
        undefined,
        sessionPath,
        diagnosticMetrics,
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
    async verifyQueryPlanningCapability(options = {}) {
      if (!queryPlanningProfile || !queryPlanner) {
        throw new Error(
          'Recall query planner is not configured: select a profile and adapter before verification',
        );
      }
      const measurement = await measureRecallQueryPlanningProviderConformance({
        provider: queryPlanner,
        profile: queryPlanningProfile,
        query: queryPlanningProfile.conformanceCanary.query,
        recallIntent: queryPlanningProfile.conformanceCanary.recallIntent,
        protectedTerms: queryPlanningProfile.conformanceCanary.protectedTerms,
        ...(options.signal ? { signal: options.signal } : {}),
      });
      return {
        profileId: queryPlanningProfile.profileId,
        model: queryPlanningProfile.model,
        promptPolicy: queryPlanningProfile.promptPolicy,
        grammarVersion: queryPlanningProfile.grammarVersion,
        executionIdentity: queryPlanner.executionIdentity,
        measurement,
      };
    },
    search(query, limit, options = {}) {
      const {
        mode = 'hybrid',
        scope = RecallSearchScope.PROJECT,
        invocationDirectory,
        activeSessionPath,
        signal,
      } = options;
      return runRecallSearchWithDiagnostics({
        diagnostics,
        searchMode: mode,
        recallScope: scope,
        ...(signal ? { signal } : {}),
        search: (diagnosticMetrics) =>
          runSerialized(async () => {
            const searchQuery = query.trim();
            if (!searchQuery) {
              throw new Error('Recall query must not be blank');
            }
            const activeGeneration = await resolveActiveRecallIndexGeneration(
              generationCoordinatorConfig,
            );
            const generationPaths =
              activeGeneration?.paths ?? generationCoordinatorConfig.legacyPaths;
            let actualManifest: RecallIndexManifest;
            const manifestReadStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
            try {
              actualManifest = await readRequiredManifest(generationPaths);
            } finally {
              diagnosticMetrics.embeddingModelVerificationMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - manifestReadStartedAtMilliseconds,
                0,
              );
            }
            await assertRecallIndexUnlockedForSearch(generationPaths.lockPath);
            const canaryVerificationStartedAtMilliseconds =
              diagnosticsClock.monotonicMilliseconds();
            try {
              const expectedManifest = createExpectedManifest(
                await readCurrentEmbeddingCanary(signal),
              );
              assertRecallIndexManifestCompatible(
                actualManifest,
                expectedManifest,
                generationPaths.manifestPath,
              );
            } finally {
              diagnosticMetrics.embeddingModelVerificationMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - canaryVerificationStartedAtMilliseconds,
                0,
              );
            }
            if (activeSessionPath) {
              diagnosticMetrics.freshnessBarrierRan = true;
              const freshnessStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              try {
                await runLiveSessionReconciliationWithDiagnostics({
                  diagnostics,
                  lifecycleTrigger: RecallLifecycleTrigger.ACTIVE_SESSION_FRESHNESS,
                  sessionPath: activeSessionPath,
                  ...(signal ? { signal } : {}),
                  reconcile: (liveSessionDiagnosticMetrics) =>
                    reconcileActiveConversationSession(
                      activeSessionPath,
                      signal,
                      undefined,
                      liveSessionDiagnosticMetrics,
                    ),
                });
              } finally {
                diagnosticMetrics.activeSessionFreshnessMilliseconds += Math.max(
                  diagnosticsClock.monotonicMilliseconds() - freshnessStartedAtMilliseconds,
                  0,
                );
              }
            }
            await assertRecallIndexUnlockedForSearch(generationPaths.lockPath);
            if (scope === RecallSearchScope.PROJECT && !invocationDirectory) {
              throw new Error(
                'Project-scoped recall requires Pi trusted invocation directory context',
              );
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
            const store = openStore('read', generationPaths.databasePath);
            try {
              const queryEmbeddingStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              let queryEmbedding: number[] | undefined;
              try {
                queryEmbedding = await embeddingProvider.embedQuery(searchQuery, signal);
              } finally {
                diagnosticMetrics.queryEmbeddingMilliseconds += Math.max(
                  diagnosticsClock.monotonicMilliseconds() - queryEmbeddingStartedAtMilliseconds,
                  0,
                );
              }
              if (!queryEmbedding) {
                throw new Error('Recall embedding response missing query vector');
              }
              const retrievalStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              const deepRerankStartedWithMilliseconds = diagnosticMetrics.deepRerankMilliseconds;
              try {
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
                const diagnosticReranker: RecallRerankingProvider = {
                  async rerankDocuments(rerankerQuery, documents, rerankerSignal) {
                    const deepRerankStartedAtMilliseconds =
                      diagnosticsClock.monotonicMilliseconds();
                    try {
                      return await reranker.rerankDocuments(
                        rerankerQuery,
                        documents,
                        rerankerSignal,
                      );
                    } finally {
                      diagnosticMetrics.deepRerankMilliseconds += Math.max(
                        diagnosticsClock.monotonicMilliseconds() - deepRerankStartedAtMilliseconds,
                        0,
                      );
                    }
                  },
                };
                const rankedResults =
                  mode === 'deep-rerank'
                    ? await rerankRecallSearchResults({
                        query: searchQuery,
                        candidates: fusedCandidates,
                        resultLimit: limit,
                        reranker: diagnosticReranker,
                        fetchConversationChunks: store.fetchConversationChunks,
                        ...(signal ? { signal } : {}),
                      })
                    : rankFusedRecallSearchResults(
                        fusedCandidates,
                        limit,
                        store.fetchConversationChunks,
                      );
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
                    rerankPolicyVersion:
                      mode === 'deep-rerank' ? RECALL_RERANK_POLICY_VERSION : null,
                    rerankerModel: mode === 'deep-rerank' ? rerankingProfile.model : null,
                    rerankerIdentity:
                      mode === 'deep-rerank'
                        ? {
                            profileId: rerankingProfile.profileId,
                            adapterId: rerankerExecutionIdentity.adapterId,
                            cacheIdentity: rerankerExecutionIdentity.cacheIdentity,
                          }
                        : null,
                    activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
                    candidateLimits: { ...config.searchCandidateLimits },
                  },
                };
              } finally {
                const retrievalElapsedMilliseconds = Math.max(
                  diagnosticsClock.monotonicMilliseconds() - retrievalStartedAtMilliseconds,
                  0,
                );
                const deepRerankElapsedMilliseconds = Math.max(
                  diagnosticMetrics.deepRerankMilliseconds - deepRerankStartedWithMilliseconds,
                  0,
                );
                diagnosticMetrics.retrievalRankingMilliseconds += Math.max(
                  retrievalElapsedMilliseconds - deepRerankElapsedMilliseconds,
                  0,
                );
              }
            } finally {
              store.close();
            }
          }),
      });
    },
    async index(options = {}) {
      assertRecallManualMaintenanceTriggerMatchesIndexOptions(options);
      const runConversationIndexMaintenance = async (
        diagnosticMetrics?: RecallIndexDiagnosticMetrics,
        onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void,
        runOptimizationWithDiagnostics?: (optimize: () => Promise<void>) => Promise<void>,
      ): Promise<RecallConversationIndexResult> => {
        async function optimizeConversationStore(store: ZvecConversationStore): Promise<void> {
          const optimizationStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
          if (diagnosticMetrics) {
            diagnosticMetrics.optimizationRan = true;
          }
          try {
            const optimizeStore = () => store.optimize();
            if (runOptimizationWithDiagnostics) {
              await runOptimizationWithDiagnostics(optimizeStore);
            } else {
              await optimizeStore();
            }
          } finally {
            if (diagnosticMetrics) {
              diagnosticMetrics.optimizationMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - optimizationStartedAtMilliseconds,
                0,
              );
            }
          }
        }

        if (options.rebuild) {
          const stagingBuildLockPath = join(
            generationCoordinatorConfig.generationsDirectory,
            'staging-operation.lock',
          );
          const stagingBuildSignal = createRecallLockAcquisitionSignal(
            options.signal,
            options.lockWaitMilliseconds,
          );
          const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
          let releaseStagingBuildLock: (() => Promise<void>) | undefined;
          try {
            releaseStagingBuildLock = await acquireRecallConversationLock(
              stagingBuildLockPath,
              stagingBuildSignal,
            );
          } finally {
            if (diagnosticMetrics) {
              diagnosticMetrics.writerLockWaitMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - lockStartedAtMilliseconds,
                0,
              );
            }
          }

          let stagingGeneration:
            | Awaited<ReturnType<typeof prepareStagingRecallIndexGeneration>>
            | undefined;
          let releaseGenerationLock: (() => Promise<void>) | undefined;
          let store: ZvecConversationStore | undefined;
          try {
            const preparationStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
            const embeddingServerMillisecondsBeforePreparation =
              diagnosticMetrics?.embeddingServerRequestMilliseconds ?? 0;
            let preparedIndex: Awaited<ReturnType<typeof prepareIndexForWrite>> | undefined;
            try {
              const activeGeneration = await resolveActiveRecallIndexGeneration(
                generationCoordinatorConfig,
              );
              await getConversationTokenizer();
              const rebuildCanary = await readCanonicalRebuildCanary(
                activeGeneration?.paths.manifestPath,
                options.signal,
                diagnosticMetrics,
              );
              stagingGeneration = await prepareStagingRecallIndexGeneration(
                generationCoordinatorConfig,
                embeddingProfileId,
              );
              releaseGenerationLock = await acquireRecallConversationLock(
                stagingGeneration.paths.lockPath,
                stagingBuildSignal,
              );
              preparedIndex = await prepareIndexForWrite(
                stagingGeneration.paths,
                options.signal,
                rebuildCanary,
                false,
                diagnosticMetrics,
              );
              store = openStore('write', stagingGeneration.paths.databasePath);
            } finally {
              if (diagnosticMetrics) {
                const preparationElapsedMilliseconds = Math.max(
                  diagnosticsClock.monotonicMilliseconds() - preparationStartedAtMilliseconds,
                  0,
                );
                const embeddingServerMillisecondsDuringPreparation = Math.max(
                  diagnosticMetrics.embeddingServerRequestMilliseconds -
                    embeddingServerMillisecondsBeforePreparation,
                  0,
                );
                diagnosticMetrics.manifestStorePreparationMilliseconds += Math.max(
                  preparationElapsedMilliseconds - embeddingServerMillisecondsDuringPreparation,
                  0,
                );
              }
            }
            if (!stagingGeneration || !preparedIndex || !store) {
              throw new Error('Recall staging generation preparation did not complete');
            }
            const indexSummary = await updateConversationIndex(
              stagingGeneration.paths,
              store,
              preparedIndex.tokenizer,
              preparedIndex.manifest,
              preparedIndex.embeddingModelPreflighted,
              options.signal,
              options.onProgress,
              options.onCheckpoint,
              undefined,
              diagnosticMetrics,
              onPhysicalSessionCheck,
            );
            if (indexSummary.failedSessions.length > 0) {
              throw new Error(
                `Recall staging generation build failed: ${indexSummary.failedSessions[0]?.sessionPath}: ${indexSummary.failedSessions[0]?.error}`,
              );
            }

            const catchUpSummary = await updateConversationIndex(
              stagingGeneration.paths,
              store,
              preparedIndex.tokenizer,
              preparedIndex.manifest,
              true,
              options.signal,
              undefined,
              options.onCheckpoint,
              undefined,
              diagnosticMetrics,
              onPhysicalSessionCheck,
            );
            if (catchUpSummary.failedSessions.length > 0) {
              throw new Error(
                `Recall staging generation catch-up failed: ${catchUpSummary.failedSessions[0]?.sessionPath}: ${catchUpSummary.failedSessions[0]?.error}`,
              );
            }
            indexSummary.indexedSessions += catchUpSummary.indexedSessions;
            indexSummary.removedSessions += catchUpSummary.removedSessions;
            indexSummary.cacheHits += catchUpSummary.cacheHits;
            indexSummary.newlyEmbeddedChunks += catchUpSummary.newlyEmbeddedChunks;
            indexSummary.embeddingRequestCount += catchUpSummary.embeddingRequestCount;
            indexSummary.deletedChunks += catchUpSummary.deletedChunks;

            await optimizeConversationStore(store);
            const actualManifest = await readRequiredManifest(stagingGeneration.paths);
            const expectedManifest = createExpectedManifest(
              await readIndexEmbeddingCanary(options.signal, diagnosticMetrics),
            );
            assertRecallIndexManifestCompatible(
              actualManifest,
              expectedManifest,
              stagingGeneration.paths.manifestPath,
            );
            if (!existsSync(stagingGeneration.paths.statePath)) {
              throw new Error(
                `Recall staging generation session state missing at ${stagingGeneration.paths.statePath}`,
              );
            }
            const stateSummary = await readRecallConversationIndexStateSummary(
              stagingGeneration.paths.statePath,
            );
            const documentCount = store.count();
            if (documentCount !== stateSummary.documentIds.length) {
              throw new Error(
                `Recall staging generation document count mismatch: session state has ${stateSummary.documentIds.length}, zvec has ${documentCount}`,
              );
            }
            for (let start = 0; start < stateSummary.documentIds.length; start += 256) {
              const documentIds = stateSummary.documentIds.slice(start, start + 256);
              const chunksById = store.fetchConversationChunks(documentIds);
              if (chunksById.size !== documentIds.length) {
                throw new Error(
                  `Recall staging generation store health check failed: fetched ${chunksById.size} of ${documentIds.length} session-state documents`,
                );
              }
              const denseDocumentIds = documentIds.filter(
                (documentId) => chunksById.get(documentId)?.isDenseSearchable === true,
              );
              const vectorsById = store.fetchVectors(denseDocumentIds);
              if (vectorsById.size !== denseDocumentIds.length) {
                throw new Error(
                  `Recall staging generation vector count mismatch: fetched ${vectorsById.size} of ${denseDocumentIds.length} dense vectors`,
                );
              }
              for (const [documentId, vector] of vectorsById) {
                if (
                  vector.length !== embeddingProfile.identity.dimensions ||
                  vector.some((value) => !Number.isFinite(value))
                ) {
                  throw new Error(
                    `Recall staging generation vector invalid for ${documentId}: expected ${embeddingProfile.identity.dimensions} finite dimensions, received ${vector.length}`,
                  );
                }
              }
            }
            store.close();
            store = undefined;
            await releaseGenerationLock();
            releaseGenerationLock = undefined;
            await activateStagingRecallIndexGeneration(
              generationCoordinatorConfig,
              stagingGeneration.generationId,
              embeddingProfileId,
            );
            return { indexSummary, totalChunks: documentCount };
          } catch (error) {
            if (stagingGeneration) {
              await preserveStagingRecallIndexGeneration(
                generationCoordinatorConfig,
                stagingGeneration.generationId,
              );
            }
            throw error;
          } finally {
            store?.close();
            await releaseGenerationLock?.();
            await releaseStagingBuildLock?.();
          }
        }

        return runSerialized(async () => {
          const activeGeneration = await resolveActiveRecallIndexGeneration(
            generationCoordinatorConfig,
          );
          const generationPaths =
            activeGeneration?.paths ?? generationCoordinatorConfig.legacyPaths;
          const lockSignal = createRecallLockAcquisitionSignal(
            options.signal,
            options.lockWaitMilliseconds,
          );
          const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
          let releaseLock: (() => Promise<void>) | undefined;
          try {
            releaseLock = await acquireRecallConversationLock(generationPaths.lockPath, lockSignal);
          } finally {
            if (diagnosticMetrics) {
              diagnosticMetrics.writerLockWaitMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - lockStartedAtMilliseconds,
                0,
              );
            }
          }
          let store: ZvecConversationStore | undefined;
          try {
            const preparedIndex = await prepareIndexForWrite(
              generationPaths,
              options.signal,
              undefined,
              options.requireExistingGeneration,
              diagnosticMetrics,
            );
            store = openStore('write', generationPaths.databasePath);
            const indexSummary = await updateConversationIndex(
              generationPaths,
              store,
              preparedIndex.tokenizer,
              preparedIndex.manifest,
              preparedIndex.embeddingModelPreflighted,
              options.signal,
              options.onProgress,
              options.onCheckpoint,
              undefined,
              diagnosticMetrics,
              onPhysicalSessionCheck,
            );
            if (
              options.optimize &&
              (indexSummary.cacheHits > 0 ||
                indexSummary.newlyEmbeddedChunks > 0 ||
                indexSummary.deletedChunks > 0)
            ) {
              await optimizeConversationStore(store);
            }
            return { indexSummary, totalChunks: store.count() };
          } finally {
            store?.close();
            await releaseLock();
          }
        });
      };
      return options.manualMaintenanceTrigger
        ? runManualIndexWithDiagnostics({
            diagnostics,
            manualMaintenanceTrigger: options.manualMaintenanceTrigger,
            ...(options.signal ? { signal: options.signal } : {}),
            runIndexMaintenance: runConversationIndexMaintenance,
          })
        : runConversationIndexMaintenance();
    },
    startBackgroundIndexGeneration() {
      assertBackgroundIndexWorkerCanReconstructService();
      return startRecallBackgroundIndexGeneration(backgroundIndexCoordinatorConfig);
    },
    resumeBackgroundIndexGeneration() {
      assertBackgroundIndexWorkerCanReconstructService();
      return resumeRecallBackgroundIndexGeneration(backgroundIndexCoordinatorConfig);
    },
    readBackgroundIndexGenerationStatus() {
      return readRecallBackgroundIndexGenerationStatus(backgroundIndexCoordinatorConfig);
    },
    stopBackgroundIndexGeneration() {
      return stopRecallBackgroundIndexGeneration(backgroundIndexCoordinatorConfig);
    },
    readIndexGenerationStatus() {
      return readRecallIndexGenerationStatus(generationCoordinatorConfig);
    },
    async discardStagingIndexGeneration() {
      const backgroundStatus = await readRecallBackgroundIndexGenerationStatus(
        backgroundIndexCoordinatorConfig,
      );
      if (
        backgroundStatus &&
        [
          RecallBackgroundIndexProcessState.STARTING,
          RecallBackgroundIndexProcessState.RUNNING,
          RecallBackgroundIndexProcessState.STOPPING,
        ].includes(backgroundStatus.processState)
      ) {
        throw new Error(
          `Recall staging generation is owned by ${backgroundStatus.processState} worker ${backgroundStatus.processId}; stop it before discard`,
        );
      }
      const stagingBuildLockPath = join(
        generationCoordinatorConfig.generationsDirectory,
        'staging-operation.lock',
      );
      const releaseLock = await acquireRecallConversationLock(stagingBuildLockPath);
      try {
        const discarded = await discardStagingRecallIndexGeneration(generationCoordinatorConfig);
        if (discarded) {
          await markRecallBackgroundIndexGenerationDiscarded(backgroundIndexCoordinatorConfig);
          await removeRecallBackgroundIndexWorkerRequest(
            backgroundIndexCoordinatorConfig.requestPath,
          );
        }
        return discarded;
      } finally {
        await releaseLock();
      }
    },
    reconcileSession(sessionPath, options) {
      return runLiveSessionReconciliationWithDiagnostics({
        diagnostics,
        lifecycleTrigger: options.lifecycleTrigger,
        sessionPath,
        ...(options.signal ? { signal: options.signal } : {}),
        reconcile: (diagnosticMetrics) =>
          runSerialized(() =>
            reconcileActiveConversationSession(
              sessionPath,
              options.signal,
              options.lockWaitMilliseconds,
              diagnosticMetrics,
            ),
          ),
      });
    },
  };
}

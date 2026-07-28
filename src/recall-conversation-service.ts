import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { adoptLegacyRecallGeneration } from './adopt-legacy-recall-generation.js';
import { collectRetiredRecallGenerations } from './collect-retired-recall-generations.js';
import {
  coordinateRecallReadWindow,
  coordinateRecallWriteWindow,
  createRecallWriteWindowAcquisitionSignal,
  type RecallWriteWindow,
} from './coordinate-recall-write-window.js';
import {
  createRecallDetachedWorkerSignal,
  type RecallDetachedWorkerSignal,
} from './create-recall-detached-worker-signal.js';
import { createRecallSessionProjectionBaseline } from './create-recall-session-projection-baseline.js';
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
  RecallGenerationCutoverState,
  RecallInferenceBackend,
  RecallManualMaintenanceTrigger,
  RecallProjectIdentitySource,
  RecallSearchScope,
  RecallSessionProjectionKind,
} from './enums.js';
import {
  fuseRecallSearchCandidates,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import {
  markRecallBackgroundIndexGenerationDiscarded,
  readRecallBackgroundIndexGenerationStatus,
  readRecallBackgroundIndexStatusRecord,
  resumeRecallBackgroundIndexGeneration,
  startRecallBackgroundIndexGeneration,
  stopRecallBackgroundIndexGeneration,
  type RecallBackgroundIndexCoordinatorConfig,
  type RecallBackgroundIndexGenerationStatus,
  type RecallBackgroundIndexServiceFactory,
} from './recall-background-index-build.js';
import {
  inspectRecallConversationCorpus,
  MAX_RECALL_FIRST_INDEX_SAMPLE_SESSION_COUNT,
  selectRecallConversationCorpusSample,
  type RecallConversationCorpusInspection,
} from './recall-conversation-corpus.js';
import {
  indexChangedConversationSessions,
  type ConversationIndexCheckpoint,
  type ConversationIndexProgress,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type { LocalEmbeddingClient } from './local-embedding-client.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
import {
  assertRecallIndexManifestCompatible,
  calculateRecallEmbeddingCanaryCosineSimilarity,
  createRecallIndexManifest,
  readRecallIndexManifest,
  readRecallSearchManifest,
  recoverRecallEmbeddingCanaryFromManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
  type RecallIndexManifest,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import {
  decodeRecallBacklogSummary,
  readRecallActiveGenerationPointer,
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  readRecallMaterialBacklogWarning,
  resolveRecallGenerationDirectory,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
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
  type RecallPlannedRetrievalQuery,
  type RecallQueryPlanningExecutionIdentity,
  type RecallRerankingExecutionIdentity,
  type RecallRerankingProvider,
} from './recall-inference-capabilities.js';
import { clearPendingRecallEmbeddingReplacement } from './recall-inference-configuration.js';
import {
  measureRecallQueryPlanningProviderConformance,
  measureRecallRerankingProviderConformance,
  type RecallQueryPlanningProviderConformanceMeasurement,
  type RecallRerankingProviderConformanceMeasurement,
} from './recall-inference-conformance.js';
import { rebuildRecallGeneration } from './rebuild-recall-generation.js';
import {
  recallRebuildOwnershipLockPath,
  tryAcquireRecallRebuildOwnershipLock,
} from './recall-rebuild-ownership-lock.js';
import {
  createLogicalSessionProjectionId,
  type RecallSessionProjection,
} from './recall-session-projection.js';
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
import { rollbackRecallGeneration } from './rollback-recall-generation.js';
import {
  rankFusedRecallSearchResults,
  rerankRecallSearchResults,
  RECALL_ACTIVE_BRANCH_PRIOR,
  RECALL_RERANK_POLICY_VERSION,
  type RankedRecallSearchResult,
} from './rank-recall-search-results.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

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
}

interface RecallAutomaticIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: false;
  manualMaintenanceTrigger?: never;
  optimize?: never;
}

interface RecallAutomaticRebuildIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild: true;
  manualMaintenanceTrigger?: never;
  optimize?: boolean;
  generationId?: string;
  resumeGenerationId?: string;
}

interface RecallManualIncrementalIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: false;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX;
  optimize?: boolean;
}

interface RecallManualRebuildIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild: true;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD;
  optimize?: boolean;
  generationId?: string;
  resumeGenerationId?: string;
}

/** Valid automatic or manually attributed conversation index invocation. */
export type RecallConversationIndexOptions =
  | RecallAutomaticIndexOptions
  | RecallAutomaticRebuildIndexOptions
  | RecallManualIncrementalIndexOptions
  | RecallManualRebuildIndexOptions;

/** Counts from one completed full or targeted conversation index update. */
export interface RecallConversationIndexResult {
  indexSummary: ConversationIndexSummary;
  totalChunks: number;
}

/** Generation paths used by one ordinary update or isolated replacement build. */
interface RecallIndexTargetPaths {
  databasePath: string;
  statePath: string;
  manifestPath: string;
}

/** Verified embedding profile and tokenizer identity accepted before guided setup persists it. */
export interface RecallEmbeddingCapabilityVerification {
  embeddingProfileId: string;
  model: string;
  dimensions: number;
  normalization: 'l2' | null;
  tokenizerModel: string;
}

/** Bound and cancellation for one optional first-index measurement sample. */
export interface RecallFirstIndexSampleOptions {
  maximumSessionCount?: number;
  signal?: AbortSignal;
}

/** Cold start, throughput, cache reuse, and duration range from one bounded corpus sample. */
export interface RecallFirstIndexSampleMeasurement {
  corpus: RecallConversationCorpusInspection;
  sampledSessionCount: number;
  sampledSourceByteSize: number;
  sampledDenseDocumentCount: number;
  coldStartMilliseconds: number;
  measuredSampleMilliseconds: number;
  sourceBytesPerSecond: number;
  denseDocumentsPerSecond: number;
  cacheHitCount: number;
  newlyEmbeddedDocumentCount: number;
  embeddingRequestCount: number;
  estimatedDurationMilliseconds: { minimum: number; maximum: number };
}

/** Independent fixed-score fixture required to accept one reranking adapter. */
export interface RecallRerankingCapabilityVerificationOptions {
  query: string;
  documents: readonly string[];
  expectedScores: readonly number[];
  maximumAbsoluteDifference?: number;
  signal?: AbortSignal;
}

/** Inspectable profile, adapter, score policy, and measurement from reranker conformance. */
export interface RecallRerankingCapabilityVerification {
  profileId: string;
  model: string;
  scorePolicy: string;
  executionIdentity: Readonly<RecallRerankingExecutionIdentity>;
  measurement: RecallRerankingProviderConformanceMeasurement;
}

/** Cancellation accepted by independent query planner setup verification. */
export interface RecallQueryPlanningCapabilityVerificationOptions {
  expectedPlan?: readonly Readonly<RecallPlannedRetrievalQuery>[];
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

/** Public projection of #59's active and building generation registry states. */
export interface RecallIndexGenerationStatus {
  active: {
    kind?: 'legacy' | 'managed';
    generationId: string;
    embeddingProfileId: string;
    status?: 'active';
    manifestPath?: string;
  } | null;
  staging: {
    kind?: 'managed';
    generationId: string;
    embeddingProfileId: string;
    status: 'building' | 'resumable';
    manifestPath?: string;
  } | null;
}

/** Search, inference verification, detached rebuild control, and generation recovery. */
export interface RecallConversationService {
  verifyEmbeddingCapability(options?: {
    signal?: AbortSignal;
  }): Promise<RecallEmbeddingCapabilityVerification>;
  inspectConversationCorpus(): Promise<RecallConversationCorpusInspection>;
  measureFirstIndexSample(
    options?: RecallFirstIndexSampleOptions,
  ): Promise<RecallFirstIndexSampleMeasurement>;
  verifyRerankingCapability(
    options: RecallRerankingCapabilityVerificationOptions,
  ): Promise<RecallRerankingCapabilityVerification>;
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
  /** Restores the bounded rollback generation and republishes retained markers. */
  rollback(): Promise<void>;
  /** Explicitly adopts the exact pre-generation version-5 layout as read-only. */
  adoptLegacy(): Promise<void>;
  /** Collects only expired validated generations after replay completes. */
  collectRetired(): Promise<void>;
}

/** Injectable inference, tokenizer, zvec, generation worker, and diagnostic boundaries. */
export interface RecallConversationDependencies {
  embeddingProfile?: RecallEmbeddingModelProfile;
  embeddingProvider?: RecallEmbeddingProvider;
  /** @deprecated Use embeddingProvider so query and document semantics stay distinct. */
  embeddings?: LocalEmbeddingClient;
  rerankingProfile?: RecallRerankingModelProfile | null;
  reranker?: RecallRerankingProvider | null;
  rerankerExecutionIdentity?: RecallRerankingExecutionIdentity | null;
  queryPlanningProfile?: RecallQueryPlanningModelProfile;
  queryPlanner?: RecallIdentifiedQueryPlanningProvider;
  tokenizerIdentity?: RecallTokenizerManifestIdentity;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write', databasePath?: string) => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  diagnostics?: RecallOperationDiagnostics;
  diagnosticsClock?: RecallDiagnosticsClock;
  notifyWarning?: (message: string) => void;
  workerSignal?: RecallDetachedWorkerSignal;
  /** Reconstructs the same configured inference runtime inside a detached rebuild worker. */
  backgroundIndexServiceFactory?: RecallBackgroundIndexServiceFactory;
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

function throwIfRecallSearchCancelled(signal?: AbortSignal): void {
  if (!signal?.aborted) {
    return;
  }
  if (signal.reason instanceof Error) {
    throw signal.reason;
  }
  throw new Error('Recall conversation operation cancelled', { cause: signal.reason });
}

interface RunRecallStoreOptimizationOptions {
  optimize(): Promise<void>;
  diagnosticsClock: RecallDiagnosticsClock;
  diagnosticMetrics?: RecallIndexDiagnosticMetrics;
  runOptimizationWithDiagnostics?: (optimize: () => Promise<void>) => Promise<void>;
}

async function runRecallStoreOptimization(
  options: RunRecallStoreOptimizationOptions,
): Promise<void> {
  const optimizationStartedAtMilliseconds = options.diagnosticsClock.monotonicMilliseconds();
  if (options.diagnosticMetrics) {
    options.diagnosticMetrics.optimizationRan = true;
  }
  try {
    if (options.runOptimizationWithDiagnostics) {
      await options.runOptimizationWithDiagnostics(() => options.optimize());
    } else {
      await options.optimize();
    }
  } finally {
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.optimizationMilliseconds += Math.max(
        options.diagnosticsClock.monotonicMilliseconds() - optimizationStartedAtMilliseconds,
        0,
      );
    }
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

function closeRecallWriteStore(
  writeWindow: RecallWriteWindow,
  failureMessage: string,
  store?: ZvecConversationStore,
): void {
  try {
    store?.close();
  } catch (error) {
    writeWindow.retainRecoveryRequired();
    throw error instanceof Error ? error : new Error(failureMessage, { cause: error });
  }
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
    typeof identity.cacheIdentity !== 'string' ||
    typeof identity.modelProfileId !== 'string'
  ) {
    return undefined;
  }
  let backend: RecallInferenceBackend;
  if (identity.backend === RecallInferenceBackend.EMBEDDED) {
    backend = RecallInferenceBackend.EMBEDDED;
  } else if (identity.backend === RecallInferenceBackend.LLAMA_CPP_HTTP) {
    backend = RecallInferenceBackend.LLAMA_CPP_HTTP;
  } else if (identity.backend === RecallInferenceBackend.CUSTOM) {
    backend = RecallInferenceBackend.CUSTOM;
  } else {
    return undefined;
  }
  return {
    adapterId: identity.adapterId,
    backend,
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

interface ApprovedRecallRebuildSnapshot {
  projections: RecallSessionProjection[];
  eligibleContributorEntryIdsBySessionPath: Map<string, Map<string, ReadonlySet<string>>>;
}

async function readApprovedRecallRebuildSnapshot(
  projectionDatabasePath: string,
  generationId: string,
): Promise<ApprovedRecallRebuildSnapshot> {
  const store = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const physicalProjections = store.listPhysicalProjections();
    const projections: RecallSessionProjection[] = [...physicalProjections];
    const eligibleContributorEntryIdsBySessionPath = new Map<
      string,
      Map<string, ReadonlySet<string>>
    >();
    for (const physicalProjection of physicalProjections) {
      const logicalProjectionIds = physicalProjection.logicalSessionIds.map((logicalSessionId) =>
        createLogicalSessionProjectionId(physicalProjection.physicalSessionId, logicalSessionId),
      );
      const fetched = store.fetchProjections(logicalProjectionIds);
      const eligibleByLogicalSessionId = new Map<string, ReadonlySet<string>>();
      for (const projectionId of logicalProjectionIds) {
        const projection = fetched.get(projectionId);
        if (projection?.projectionKind !== RecallSessionProjectionKind.LOGICAL_SESSION) {
          throw new Error(`Recall rebuild approved logical projection missing: ${projectionId}`);
        }
        projections.push(projection);
        eligibleByLogicalSessionId.set(
          projection.logicalSessionId,
          new Set(projection.eligibleContributorEntryIds),
        );
      }
      eligibleContributorEntryIdsBySessionPath.set(
        physicalProjection.sourcePath,
        eligibleByLogicalSessionId,
      );
    }
    return { projections, eligibleContributorEntryIdsBySessionPath };
  } finally {
    store.close();
  }
}

function retargetRecallRebuildProjection(
  projection: RecallSessionProjection,
  generationId: string,
): RecallSessionProjection {
  return {
    ...projection,
    generationId,
    markerCheckpoint: { ...projection.markerCheckpoint, generationId },
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
  const rerankingDisabled =
    dependencies.rerankingProfile === null && dependencies.reranker === null;
  if (
    (dependencies.rerankingProfile === null) !== (dependencies.reranker === null) ||
    (rerankingDisabled && dependencies.rerankerExecutionIdentity)
  ) {
    throw new Error(
      'Recall reranker configuration incomplete: profile and provider must both be configured or both be null',
    );
  }
  let rerankingProfile: RecallRerankingModelProfile | null;
  let reranker: RecallRerankingProvider | null;
  if (rerankingDisabled) {
    rerankingProfile = null;
    reranker = null;
  } else {
    rerankingProfile =
      dependencies.rerankingProfile ?? createQwenRerankingModelProfile(config.rerankerModel);
    reranker =
      dependencies.reranker ??
      createQwenHttpRerankingProvider(rerankingProfile, { baseUrl: config.rerankerBaseUrl });
  }
  const providerExecutionIdentity = reranker
    ? readRecallRerankingExecutionIdentity(reranker)
    : null;
  const rerankerExecutionIdentity = rerankingProfile
    ? (dependencies.rerankerExecutionIdentity ??
      providerExecutionIdentity ??
      createRecallRerankingExecutionIdentity(
        rerankingProfile.profileId,
        'custom-injected-reranking-v1',
        RecallInferenceBackend.CUSTOM,
      ))
    : null;
  if (
    rerankingProfile &&
    rerankerExecutionIdentity &&
    rerankerExecutionIdentity.modelProfileId !== rerankingProfile.profileId
  ) {
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
  const workerSignal =
    dependencies.workerSignal ?? createRecallDetachedWorkerSignal(config.workerOwnershipLockPath);
  const backgroundIndexStatusPath =
    config.backgroundIndexStatusPath ?? join(config.dataDirectory, 'background-index-status.json');
  const backgroundIndexRequestPath =
    config.backgroundIndexRequestPath ??
    join(config.dataDirectory, 'background-index-request.json');
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
  const backgroundIndexCoordinatorConfig: RecallBackgroundIndexCoordinatorConfig = {
    serviceConfig: config,
    generationService: { readIndexGenerationStatus: readCanonicalIndexGenerationStatus },
    statusPath: backgroundIndexStatusPath,
    requestPath: backgroundIndexRequestPath,
    embeddingProfileId,
    serviceFactory: dependencies.backgroundIndexServiceFactory ?? {
      moduleUrl: import.meta.url,
      exportName: 'createRecallConversationService',
    },
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

  async function readCanonicalIndexGenerationStatus(): Promise<RecallIndexGenerationStatus> {
    const [pointer, registry, backgroundStatus] = await Promise.all([
      readRecallActiveGenerationPointer(config.activeGenerationPointerPath),
      readRecallGenerationRegistry(config.generationRegistryPath),
      readRecallBackgroundIndexStatusRecord(backgroundIndexStatusPath),
    ]);
    const activeEntry = pointer
      ? registry?.generations.find(
          ({ generationId }) => generationId === pointer.activeGenerationId,
        )
      : undefined;
    const buildingEntry = registry?.buildingGenerationId
      ? registry.generations.find(
          ({ generationId }) => generationId === registry.buildingGenerationId,
        )
      : undefined;
    const resumableEntry = registry?.generations
      .filter(({ state }) => state === RecallGenerationCutoverState.FAILED)
      .toSorted(
        (left, right) =>
          right.stateChangedAtEpochMilliseconds - left.stateChangedAtEpochMilliseconds,
      )[0];
    const stagingEntry = buildingEntry ?? resumableEntry;
    const backgroundOwnsBuildingEntry =
      buildingEntry !== undefined &&
      backgroundStatus?.generationId === buildingEntry.generationId &&
      (backgroundStatus.processState === RecallBackgroundIndexProcessState.STARTING ||
        backgroundStatus.processState === RecallBackgroundIndexProcessState.RUNNING ||
        backgroundStatus.processState === RecallBackgroundIndexProcessState.STOPPING);
    const stagingStatus = buildingEntry && backgroundOwnsBuildingEntry ? 'building' : 'resumable';
    const activeGenerationDirectory = pointer
      ? await resolveRecallGenerationDirectory(
          config.generationRootDirectory,
          pointer.activeGenerationId,
        )
      : null;
    const stagingGenerationDirectory = stagingEntry
      ? await resolveRecallGenerationDirectory(
          config.generationRootDirectory,
          stagingEntry.generationId,
        )
      : null;
    return {
      active: pointer
        ? {
            kind:
              activeEntry?.state === RecallGenerationCutoverState.LEGACY_READ_ONLY
                ? 'legacy'
                : 'managed',
            generationId: pointer.activeGenerationId,
            embeddingProfileId: activeEntry?.embeddingProfileId ?? embeddingProfileId,
            status: 'active',
            manifestPath: join(activeGenerationDirectory!, 'index-manifest.json'),
          }
        : null,
      staging: stagingEntry
        ? {
            kind: 'managed',
            generationId: stagingEntry.generationId,
            embeddingProfileId: stagingEntry.embeddingProfileId ?? embeddingProfileId,
            status: stagingStatus,
            manifestPath: join(stagingGenerationDirectory!, 'index-manifest.json'),
          }
        : null,
    };
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
    activeManifestPath: string | null,
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
    if (activeManifestPath !== null) {
      try {
        const actualManifest = await readRecallSearchManifest(activeManifestPath);
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

  async function readRequiredManifest(manifestPath: string): Promise<RecallIndexManifest> {
    const actual = await readRecallSearchManifest(manifestPath);
    if (!actual) {
      throw new Error(
        `Recall index manifest missing at ${manifestPath}; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    return actual;
  }

  async function prepareIndexForWrite(
    targetPaths: RecallIndexTargetPaths,
    signal?: AbortSignal,
    preflightedCanary?: readonly number[],
    requireExistingGeneration = false,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  ): Promise<{
    tokenizer: ConversationTextTokenizer;
    manifest: RecallIndexManifest;
    embeddingModelPreflighted: boolean;
  }> {
    const actual = await readRecallIndexManifest(targetPaths.manifestPath);
    if (!actual && requireExistingGeneration) {
      throw new Error(
        `Recall automatic session ingestion requires an existing index generation at ${targetPaths.manifestPath}; initialize it with /pi-session-recall-index --rebuild`,
      );
    }
    if (!actual && (existsSync(targetPaths.databasePath) || existsSync(targetPaths.statePath))) {
      throw new Error(
        `Recall index manifest missing at ${targetPaths.manifestPath} for existing index data; reindex with /pi-session-recall-index --rebuild`,
      );
    }
    const tokenizer = await getConversationTokenizer();
    if (actual) {
      const expected = createExpectedManifest(actual.embedding.canaryVector);
      assertRecallIndexManifestCompatible(actual, expected, targetPaths.manifestPath);
      return { tokenizer, manifest: actual, embeddingModelPreflighted: false };
    }
    const expected = createExpectedManifest(
      preflightedCanary ?? (await readIndexEmbeddingCanary(signal, diagnosticMetrics)),
    );
    await writeRecallIndexManifest(targetPaths.manifestPath, expected);
    return { tokenizer, manifest: expected, embeddingModelPreflighted: true };
  }

  async function updateConversationIndex(
    store: ZvecConversationStore,
    tokenizer: ConversationTextTokenizer,
    manifest: RecallIndexManifest,
    embeddingModelPreflighted: boolean,
    targetPaths: RecallIndexTargetPaths,
    signal?: AbortSignal,
    onProgress?: (progress: ConversationIndexProgress) => void,
    onCheckpoint?: (checkpoint: ConversationIndexCheckpoint) => void,
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
    onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void,
    eligibleContributorEntryIdsBySessionPath?: ReadonlyMap<
      string,
      ReadonlyMap<string, ReadonlySet<string>>
    >,
  ): Promise<ConversationIndexSummary> {
    let modelPreflighted = embeddingModelPreflighted;
    async function embedTextsAfterModelPreflight(
      texts: string[],
      embeddingSignal?: AbortSignal,
    ): Promise<number[][]> {
      if (!modelPreflighted) {
        const expected = createExpectedManifest(await readCurrentEmbeddingCanary(embeddingSignal));
        assertRecallIndexManifestCompatible(manifest, expected, targetPaths.manifestPath);
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
      statePath: targetPaths.statePath,
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
      ...(eligibleContributorEntryIdsBySessionPath
        ? { eligibleContributorEntryIdsBySessionPath }
        : {}),
    };
    return indexChangedConversationSessions({
      ...indexerOptions,
      sessionsDirectory: config.sessionsDirectory,
      ...(onProgress ? { onProgress } : {}),
    });
  }

  return {
    async verifyEmbeddingCapability(options = {}) {
      const canary = await readCurrentEmbeddingCanary(options.signal);
      await getConversationTokenizer();
      const manifest = createExpectedManifest(canary);
      return {
        embeddingProfileId,
        model: embeddingProfile.identity.requestModel,
        dimensions: embeddingProfile.identity.dimensions,
        normalization: embeddingProfile.identity.normalization ?? null,
        tokenizerModel: manifest.tokenizer.model,
      };
    },
    async inspectConversationCorpus() {
      return (await inspectRecallConversationCorpus(config.sessionsDirectory)).inspection;
    },
    measureFirstIndexSample(options = {}) {
      return runSerialized(async () => {
        const activePointer = await readRecallActiveGenerationPointer(
          config.activeGenerationPointerPath,
        );
        if (activePointer) {
          throw new Error(
            `Recall first-index sample requires no active generation; active generation ${activePointer.activeGenerationId} already exists`,
          );
        }
        const maximumSessionCount =
          options.maximumSessionCount ?? MAX_RECALL_FIRST_INDEX_SAMPLE_SESSION_COUNT;
        const corpus = await inspectRecallConversationCorpus(config.sessionsDirectory);
        const sampleFiles = selectRecallConversationCorpusSample(corpus.files, maximumSessionCount);
        const coldStartStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
        const canary = await readCurrentEmbeddingCanary(options.signal);
        const tokenizer = await getConversationTokenizer();
        const manifest = createExpectedManifest(canary);
        const coldStartMilliseconds = Math.max(
          diagnosticsClock.monotonicMilliseconds() - coldStartStartedAtMilliseconds,
          0,
        );
        const embeddingCache = createEmbeddingVectorCache({
          cacheDirectory: config.embeddingCacheDirectory,
          identity: createEmbeddingVectorCacheIdentity(manifest),
          embeddingRequestBatchSize: config.embeddingBatchSize,
          embeddings: { embedTexts: embeddingProvider.embedDocuments.bind(embeddingProvider) },
          diagnosticsClock,
        });
        let sampledDenseDocumentCount = 0;
        let cacheHitCount = 0;
        let newlyEmbeddedDocumentCount = 0;
        let embeddingRequestCount = 0;
        const measuredSampleStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
        for (const file of sampleFiles) {
          if (options.signal?.aborted) {
            throw new Error('Recall first-index sample cancelled', {
              cause: options.signal.reason,
            });
          }
          const chunks = await readSessionConversationChunks(file.sessionPath, {
            tokenizer,
            maxTokens: manifest.chunkPolicy.maxTokens,
            overlapTokens: manifest.chunkPolicy.overlapTokens,
          });
          const denseDocuments = chunks
            .filter((chunk) => chunk.isDenseSearchable)
            .map((chunk) => chunk.content);
          sampledDenseDocumentCount += denseDocuments.length;
          for (let start = 0; start < denseDocuments.length; start += 128) {
            const cacheResult = await embeddingCache.resolveEmbeddingVectors(
              denseDocuments.slice(start, start + 128),
              options.signal,
            );
            cacheHitCount += cacheResult.cacheHits;
            newlyEmbeddedDocumentCount += cacheResult.newlyEmbeddedChunks;
            embeddingRequestCount += cacheResult.embeddingRequestCount;
          }
        }
        const measuredSampleMilliseconds = Math.max(
          diagnosticsClock.monotonicMilliseconds() - measuredSampleStartedAtMilliseconds,
          0,
        );
        const sampledSourceByteSize = sampleFiles.reduce(
          (total, file) => total + file.sourceByteSize,
          0,
        );
        const measuredSampleSeconds = measuredSampleMilliseconds / 1_000;
        const sourceBytesPerSecond =
          measuredSampleSeconds > 0 ? sampledSourceByteSize / measuredSampleSeconds : 0;
        const denseDocumentsPerSecond =
          measuredSampleSeconds > 0 ? sampledDenseDocumentCount / measuredSampleSeconds : 0;
        const projectedVariableMilliseconds =
          sampledSourceByteSize > 0
            ? measuredSampleMilliseconds *
              (corpus.inspection.sourceByteSize / sampledSourceByteSize)
            : 0;
        const projectedDurationMilliseconds = coldStartMilliseconds + projectedVariableMilliseconds;
        return {
          corpus: corpus.inspection,
          sampledSessionCount: sampleFiles.length,
          sampledSourceByteSize,
          sampledDenseDocumentCount,
          coldStartMilliseconds,
          measuredSampleMilliseconds,
          sourceBytesPerSecond,
          denseDocumentsPerSecond,
          cacheHitCount,
          newlyEmbeddedDocumentCount,
          embeddingRequestCount,
          estimatedDurationMilliseconds: {
            minimum: Math.floor(projectedDurationMilliseconds * 0.8),
            maximum: Math.ceil(projectedDurationMilliseconds * 1.25),
          },
        };
      });
    },
    async verifyRerankingCapability(options) {
      if (!rerankingProfile || !reranker || !rerankerExecutionIdentity) {
        throw new Error(
          'Recall reranking is not configured: select a profile and adapter before verification',
        );
      }
      const measurement = await measureRecallRerankingProviderConformance({
        provider: {
          executionIdentity: rerankerExecutionIdentity,
          rerankDocuments: reranker.rerankDocuments.bind(reranker),
        },
        profile: rerankingProfile,
        query: options.query,
        documents: options.documents,
        expectedScores: options.expectedScores,
        ...(options.maximumAbsoluteDifference === undefined
          ? {}
          : { maximumAbsoluteDifference: options.maximumAbsoluteDifference }),
        ...(options.signal ? { signal: options.signal } : {}),
      });
      return {
        profileId: rerankingProfile.profileId,
        model: rerankingProfile.model,
        scorePolicy: rerankingProfile.scorePolicy,
        executionIdentity: rerankerExecutionIdentity,
        measurement,
      };
    },
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
        ...(options.expectedPlan ? { expectedPlan: options.expectedPlan } : {}),
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
        signal,
      } = options;
      if (
        mode === 'deep-rerank' &&
        (!rerankingProfile || !reranker || !rerankerExecutionIdentity)
      ) {
        throw new Error(
          'Recall reranking is not configured: select and verify a reranking capability before deep-rerank search',
        );
      }
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

            const canaryVerificationStartedAtMilliseconds =
              diagnosticsClock.monotonicMilliseconds();
            let expectedManifest: RecallIndexManifest;
            try {
              expectedManifest = createExpectedManifest(await readCurrentEmbeddingCanary(signal));
            } finally {
              diagnosticMetrics.embeddingModelVerificationMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - canaryVerificationStartedAtMilliseconds,
                0,
              );
            }
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

            return coordinateRecallReadWindow(
              {
                lockPath: config.lockPath,
                waitMilliseconds: config.searchWriteWindowWaitMilliseconds,
                ...(signal ? { signal } : {}),
              },
              async () => {
                const manifestReadStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
                const activeGeneration = await readRecallActiveGenerationSelection(
                  config.activeGenerationPointerPath,
                  config.generationRootDirectory,
                );
                try {
                  const actualManifest = await readRequiredManifest(activeGeneration.manifestPath);
                  assertRecallIndexManifestCompatible(
                    actualManifest,
                    expectedManifest,
                    activeGeneration.manifestPath,
                  );
                } finally {
                  diagnosticMetrics.embeddingModelVerificationMilliseconds += Math.max(
                    diagnosticsClock.monotonicMilliseconds() - manifestReadStartedAtMilliseconds,
                    0,
                  );
                }
                try {
                  const warning = await readRecallMaterialBacklogWarning(
                    config.backlogSummaryPath,
                    activeGeneration.activeGenerationId,
                  );
                  if (warning) {
                    try {
                      dependencies.notifyWarning?.(warning);
                    } catch (warningError) {
                      void warningError;
                    }
                  }
                } catch (backlogError) {
                  void backlogError;
                }

                const store = openStore('read', activeGeneration.databasePath);
                try {
                  const retrievalStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
                  const deepRerankStartedWithMilliseconds =
                    diagnosticMetrics.deepRerankMilliseconds;
                  try {
                    throwIfRecallSearchCancelled(signal);
                    const denseCandidates = await store.searchDenseCandidates(
                      queryEmbedding,
                      config.searchCandidateLimits.dense,
                      projectIdentityPredicate,
                    );
                    throwIfRecallSearchCancelled(signal);
                    const lexicalCandidates = await store.searchLexicalCandidates(
                      searchQuery,
                      config.searchCandidateLimits.lexical,
                      projectIdentityPredicate,
                    );
                    throwIfRecallSearchCancelled(signal);
                    const identifierCandidates = await store.searchIdentifierCandidates(
                      searchQuery,
                      config.searchCandidateLimits.identifier,
                      projectIdentityPredicate,
                    );
                    throwIfRecallSearchCancelled(signal);
                    const fusedCandidates = fuseRecallSearchCandidates(
                      { denseCandidates, lexicalCandidates, identifierCandidates },
                      config.searchCandidateLimits.dense +
                        config.searchCandidateLimits.lexical +
                        config.searchCandidateLimits.identifier,
                    );
                    const diagnosticReranker: RecallRerankingProvider = {
                      async rerankDocuments(rerankerQuery, documents, rerankerSignal) {
                        const deepRerankStartedAtMilliseconds =
                          diagnosticsClock.monotonicMilliseconds();
                        try {
                          if (!reranker) {
                            throw new Error(
                              'Recall reranking became unavailable during deep-rerank search',
                            );
                          }
                          return await reranker.rerankDocuments(
                            rerankerQuery,
                            documents,
                            rerankerSignal,
                          );
                        } finally {
                          diagnosticMetrics.deepRerankMilliseconds += Math.max(
                            diagnosticsClock.monotonicMilliseconds() -
                              deepRerankStartedAtMilliseconds,
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
                    const results: RecallConversationSearchResult[] = rankedResults.map(
                      (result) => ({
                        ...result,
                        evidenceRelation:
                          !invocationProject ||
                          invocationProject.projectIdentity !==
                            result.projectAttribution?.projectIdentity
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
                      }),
                    );
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
                        rerankerModel:
                          mode === 'deep-rerank' && rerankingProfile
                            ? rerankingProfile.model
                            : null,
                        rerankerIdentity:
                          mode === 'deep-rerank' && rerankingProfile && rerankerExecutionIdentity
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
              },
            );
          }),
      });
    },
    async index(options = {}) {
      assertRecallManualMaintenanceTriggerMatchesIndexOptions(options);
      if (options.rebuild && options.generationId && options.resumeGenerationId) {
        throw new Error('Recall rebuild cannot create and resume a generation simultaneously');
      }
      const runConversationIndexMaintenance = async (
        diagnosticMetrics?: RecallIndexDiagnosticMetrics,
        onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void,
        runOptimizationWithDiagnostics?: (optimize: () => Promise<void>) => Promise<void>,
      ): Promise<RecallConversationIndexResult> => {
        if (options.rebuild) {
          const startingPointer = await readRecallActiveGenerationPointer(
            config.activeGenerationPointerPath,
          );
          const startingGeneration = startingPointer
            ? await readRecallActiveGenerationSelection(
                config.activeGenerationPointerPath,
                config.generationRootDirectory,
              )
            : null;
          const activeManifestPath = startingGeneration?.manifestPath ?? null;
          let rebuiltProjectionIds: string[] = [];
          let rebuiltPhysicalProjectionCount = 0;
          const rebuilt = await rebuildRecallGeneration({
            generationRootDirectory: config.generationRootDirectory,
            activeGenerationPointerPath: config.activeGenerationPointerPath,
            generationRegistryPath: config.generationRegistryPath,
            backlogSummaryPath: config.backlogSummaryPath,
            markerSpoolDirectory: config.markerSpoolDirectory,
            lockPath: config.lockPath,
            workerSignal,
            embeddingProfileId,
            ...(options.generationId ? { generationId: options.generationId } : {}),
            ...(options.resumeGenerationId
              ? {
                  generationId: options.resumeGenerationId,
                  resumeExistingGeneration: true,
                }
              : {}),
            ...(options.signal ? { signal: options.signal } : {}),
            async captureBuildSnapshot() {
              return startingGeneration && existsSync(startingGeneration.projectionDatabasePath)
                ? readApprovedRecallRebuildSnapshot(
                    startingGeneration.projectionDatabasePath,
                    startingGeneration.activeGenerationId,
                  )
                : null;
            },
            async buildGeneration(paths, approvedRebuildSnapshot) {
              const preparationStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              const embeddingServerMillisecondsBeforePreparation =
                diagnosticMetrics?.embeddingServerRequestMilliseconds ?? 0;
              const targetPaths: RecallIndexTargetPaths = paths;
              let store: ZvecConversationStore | undefined;
              let preparedIndex: Awaited<ReturnType<typeof prepareIndexForWrite>>;
              try {
                const rebuildCanary = await readCanonicalRebuildCanary(
                  activeManifestPath,
                  options.signal,
                  diagnosticMetrics,
                );
                preparedIndex = await prepareIndexForWrite(
                  targetPaths,
                  options.signal,
                  rebuildCanary,
                  false,
                  diagnosticMetrics,
                );
                store = openStore('write', paths.databasePath);
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
              try {
                const reproducedApprovedSourcePaths = new Set<string>();
                const indexedSourceByteSizes = new Map<string, number>();
                const rebuildSourceOutcomes = new Map<string, string>();
                const recordPhysicalSessionCheck = (
                  completion: RecallPhysicalSessionDiagnostic,
                ): void => {
                  rebuildSourceOutcomes.set(
                    completion.sessionPath,
                    `${completion.status}:${completion.failedSessionCount}`,
                  );
                  if (
                    completion.status === RecallDiagnosticStatus.SUCCEEDED &&
                    completion.failedSessionCount === 0
                  ) {
                    const sourceByteSize = completion.metrics.sourceByteSize;
                    if (sourceByteSize === null) {
                      throw new Error(
                        `Recall rebuild indexed source size missing: ${completion.sessionPath}`,
                      );
                    }
                    reproducedApprovedSourcePaths.add(completion.sessionPath);
                    indexedSourceByteSizes.set(completion.sessionPath, sourceByteSize);
                  }
                  onPhysicalSessionCheck?.(completion);
                };
                const indexSummary = await updateConversationIndex(
                  store,
                  preparedIndex.tokenizer,
                  preparedIndex.manifest,
                  preparedIndex.embeddingModelPreflighted,
                  targetPaths,
                  options.signal,
                  options.onProgress,
                  options.onCheckpoint,
                  diagnosticMetrics,
                  recordPhysicalSessionCheck,
                  approvedRebuildSnapshot?.eligibleContributorEntryIdsBySessionPath,
                );
                for (const approvedSourcePath of approvedRebuildSnapshot?.eligibleContributorEntryIdsBySessionPath.keys() ??
                  []) {
                  if (!reproducedApprovedSourcePaths.has(approvedSourcePath)) {
                    throw new Error(
                      `Recall rebuild approved physical source was not reproduced: ${approvedSourcePath}; observed ${rebuildSourceOutcomes.get(approvedSourcePath) ?? 'no physical-session check'}; failure ${indexSummary.failedSessions[0]?.error ?? 'none'}`,
                    );
                  }
                }
                const result = { indexSummary, totalChunks: store.count() };
                const replacementProjections: RecallSessionProjection[] =
                  approvedRebuildSnapshot == null
                    ? (
                        await Promise.all(
                          [...reproducedApprovedSourcePaths].toSorted().map((sessionPath) => {
                            const expectedSourceByteSize = indexedSourceByteSizes.get(sessionPath);
                            if (expectedSourceByteSize === undefined) {
                              throw new Error(
                                `Recall rebuild indexed source size unavailable: ${sessionPath}`,
                              );
                            }
                            return createRecallSessionProjectionBaseline({
                              physicalSessionPath: sessionPath,
                              generationId: paths.generationId,
                              tokenizer: preparedIndex.tokenizer,
                              expectedSourceByteSize,
                              ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
                            });
                          }),
                        )
                      ).flat()
                    : approvedRebuildSnapshot.projections.map((projection) =>
                        retargetRecallRebuildProjection(projection, paths.generationId),
                      );
                rebuiltProjectionIds = replacementProjections.map(
                  ({ projectionId }) => projectionId,
                );
                rebuiltPhysicalProjectionCount = replacementProjections.filter(
                  ({ projectionKind }) =>
                    projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
                ).length;
                const storeToClose = store;
                store = undefined;
                const shouldOptimize =
                  options.optimize === true &&
                  (indexSummary.cacheHits > 0 ||
                    indexSummary.newlyEmbeddedChunks > 0 ||
                    indexSummary.deletedChunks > 0);
                return {
                  result,
                  ...(shouldOptimize
                    ? {
                        async optimize() {
                          await runRecallStoreOptimization({
                            optimize: () => storeToClose.optimize(),
                            diagnosticsClock,
                            ...(diagnosticMetrics ? { diagnosticMetrics } : {}),
                            ...(runOptimizationWithDiagnostics
                              ? { runOptimizationWithDiagnostics }
                              : {}),
                          });
                        },
                      }
                    : {}),
                  async close() {
                    storeToClose.close();
                    const projectionStore = openZvecSessionProjectionStore({
                      databasePath: paths.projectionDatabasePath,
                      generationId: paths.generationId,
                      createIfMissing: true,
                      readOnly: false,
                    });
                    try {
                      await projectionStore.upsertProjections(replacementProjections);
                    } finally {
                      projectionStore.close();
                    }
                  },
                };
              } catch (error) {
                store?.close();
                throw error;
              }
            },
            async validateGeneration(paths, result, approvedRebuildSnapshot) {
              const manifest = await readRecallIndexManifest(paths.manifestPath);
              if (!manifest) {
                throw new Error('Recall replacement generation manifest missing during validation');
              }
              const validationStore = openStore('read', paths.databasePath);
              try {
                if (validationStore.count() !== result.totalChunks) {
                  throw new Error('Recall replacement generation count changed during validation');
                }
              } finally {
                validationStore.close();
              }
              const projectionValidationStore = openZvecSessionProjectionStore({
                databasePath: paths.projectionDatabasePath,
                generationId: paths.generationId,
                createIfMissing: false,
                readOnly: true,
              });
              try {
                const physicalProjections = projectionValidationStore.listPhysicalProjections();
                const expectedPhysicalProjectionCount =
                  approvedRebuildSnapshot?.eligibleContributorEntryIdsBySessionPath.size ??
                  rebuiltPhysicalProjectionCount;
                if (physicalProjections.length !== expectedPhysicalProjectionCount) {
                  throw new Error(
                    'Recall replacement generation projection snapshot changed during validation',
                  );
                }
                const expectedProjectionIds =
                  approvedRebuildSnapshot?.projections.map(({ projectionId }) => projectionId) ??
                  rebuiltProjectionIds;
                if (
                  projectionValidationStore.fetchProjections(expectedProjectionIds).size !==
                  expectedProjectionIds.length
                ) {
                  throw new Error(
                    'Recall replacement generation logical projection snapshot incomplete',
                  );
                }
              } finally {
                projectionValidationStore.close();
              }
              return {
                indexManifestFingerprint: createHash('sha256')
                  .update(await readFile(paths.manifestPath))
                  .digest('hex'),
              };
            },
          });
          return rebuilt.result;
        }

        return runSerialized(async () => {
          const lockSignal = createRecallWriteWindowAcquisitionSignal(
            options.signal,
            options.lockWaitMilliseconds,
          );
          const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
          return coordinateRecallWriteWindow(
            {
              lockPath: config.lockPath,
              allowRecovery: false,
              ...(lockSignal ? { signal: lockSignal } : {}),
            },
            async (writeWindow) => {
              const activeGeneration = await readRecallActiveGenerationSelection(
                config.activeGenerationPointerPath,
                config.generationRootDirectory,
              );
              const registry = await readRecallGenerationRegistry(config.generationRegistryPath);
              const activeRegistryEntry = registry?.generations.find(
                ({ generationId }) => generationId === activeGeneration.activeGenerationId,
              );
              if (registry?.buildingGenerationId != null) {
                throw new Error(
                  'Recall incremental commits are frozen while a replacement generation builds',
                );
              }
              if (activeRegistryEntry?.state === RecallGenerationCutoverState.LEGACY_READ_ONLY) {
                throw new Error(
                  'Recall adopted legacy generation is read-only; run an explicit rebuild',
                );
              }
              const targetPaths: RecallIndexTargetPaths = activeGeneration;
              if (diagnosticMetrics) {
                diagnosticMetrics.writerLockWaitMilliseconds += Math.max(
                  diagnosticsClock.monotonicMilliseconds() - lockStartedAtMilliseconds,
                  0,
                );
              }
              let store: ZvecConversationStore | undefined;
              try {
                const preparedIndex = await prepareIndexForWrite(
                  targetPaths,
                  options.signal,
                  undefined,
                  options.requireExistingGeneration,
                  diagnosticMetrics,
                );
                store = openStore('write', targetPaths.databasePath);
                const indexSummary = await updateConversationIndex(
                  store,
                  preparedIndex.tokenizer,
                  preparedIndex.manifest,
                  preparedIndex.embeddingModelPreflighted,
                  targetPaths,
                  options.signal,
                  options.onProgress,
                  options.onCheckpoint,
                  diagnosticMetrics,
                  onPhysicalSessionCheck,
                );
                if (options.optimize === true) {
                  await runRecallStoreOptimization({
                    optimize: () => store?.optimize() ?? Promise.resolve(),
                    diagnosticsClock,
                    ...(diagnosticMetrics ? { diagnosticMetrics } : {}),
                    ...(runOptimizationWithDiagnostics ? { runOptimizationWithDiagnostics } : {}),
                  });
                }
                return { indexSummary, totalChunks: store.count() };
              } finally {
                closeRecallWriteStore(writeWindow, 'Recall maintenance store close failed', store);
              }
            },
          );
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
    readIndexGenerationStatus: readCanonicalIndexGenerationStatus,
    async discardStagingIndexGeneration() {
      const status = await readCanonicalIndexGenerationStatus();
      if (!status.staging) {
        return false;
      }
      const ownership = await tryAcquireRecallRebuildOwnershipLock(
        recallRebuildOwnershipLockPath(config.lockPath),
      );
      if (!ownership) {
        throw new Error(
          `Recall staging generation ${status.staging.generationId} is owned by a live rebuild; stop it before discard`,
        );
      }
      try {
        await coordinateRecallWriteWindow(
          { lockPath: config.lockPath, allowRecovery: false },
          async () => {
            const registry = await readRecallGenerationRegistry(config.generationRegistryPath);
            if (!registry) {
              throw new Error('Recall generation registry missing during staging discard');
            }
            if (registry.activeGenerationId === status.staging?.generationId) {
              throw new Error('Recall active generation cannot be discarded as staging');
            }
            const remainingGenerations = registry.generations.filter(
              ({ generationId }) => generationId !== status.staging?.generationId,
            );
            await writeRecallGenerationRegistry(config.generationRegistryPath, {
              ...registry,
              buildingGenerationId:
                registry.buildingGenerationId === status.staging?.generationId
                  ? null
                  : registry.buildingGenerationId,
              generations: remainingGenerations,
            });
            if (registry.activeGenerationId) {
              let backlogSummary;
              try {
                backlogSummary = decodeRecallBacklogSummary(
                  await readFile(config.backlogSummaryPath, 'utf8'),
                );
              } catch (error) {
                if (readNodeErrorCode(error) !== 'ENOENT') {
                  throw error;
                }
              }
              if (backlogSummary) {
                const activeEntry = remainingGenerations.find(
                  ({ generationId }) => generationId === registry.activeGenerationId,
                );
                await writeRecallBacklogSummary(config.backlogSummaryPath, {
                  ...backlogSummary,
                  buildingGenerationId: null,
                  generationState: activeEntry?.state ?? RecallGenerationCutoverState.ACTIVE,
                  rebuildAgeMilliseconds: null,
                  lastFailureCategory: null,
                  observedAtEpochMilliseconds: Date.now(),
                });
              }
            }
            await clearPendingRecallEmbeddingReplacement(
              join(dirname(config.manifestPath), 'inference-configuration.json'),
              {
                generationRegistryPath: config.generationRegistryPath,
              },
            );
          },
        );
        const stagingGenerationDirectory = await resolveRecallGenerationDirectory(
          config.generationRootDirectory,
          status.staging.generationId,
        );
        await rm(stagingGenerationDirectory, { recursive: true, force: true });
        await markRecallBackgroundIndexGenerationDiscarded(backgroundIndexCoordinatorConfig);
        workerSignal.signalDetachedWorker();
        return true;
      } finally {
        await ownership.release();
      }
    },
    async rollback() {
      await rollbackRecallGeneration({
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRegistryPath: config.generationRegistryPath,
        generationRootDirectory: config.generationRootDirectory,
        backlogSummaryPath: config.backlogSummaryPath,
        markerSpoolDirectory: config.markerSpoolDirectory,
        retainedMarkerDirectory: join(config.markerControlDirectory, 'rollback-retained'),
        lockPath: config.lockPath,
        workerSignal,
      });
    },
    async adoptLegacy() {
      await adoptLegacyRecallGeneration({
        dataDirectory: config.dataDirectory,
        legacyDatabasePath: config.databasePath,
        legacyStatePath: config.statePath,
        legacyManifestPath: config.manifestPath,
        generationRootDirectory: config.generationRootDirectory,
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRegistryPath: config.generationRegistryPath,
        backlogSummaryPath: config.backlogSummaryPath,
        backupEvidencePath: join(config.dataDirectory, 'legacy-adoption-backup.json'),
        lockPath: config.lockPath,
        async validateLegacyDatabase(databasePath) {
          const legacyStore = openStore('read', databasePath);
          try {
            legacyStore.count();
          } finally {
            legacyStore.close();
          }
        },
      });
    },
    async collectRetired() {
      await collectRetiredRecallGenerations({
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRegistryPath: config.generationRegistryPath,
        generationRootDirectory: config.generationRootDirectory,
        lockPath: config.lockPath,
        retainedMarkerDirectory: join(config.markerControlDirectory, 'rollback-retained'),
      });
    },
  };
}

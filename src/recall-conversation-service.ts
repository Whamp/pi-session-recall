import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { adoptLegacyRecallGeneration } from './adopt-legacy-recall-generation.js';
import { collectRetiredRecallGenerations } from './collect-retired-recall-generations.js';
import {
  coordinateRecallReadWindow,
  coordinateRecallWriteWindow,
  createRecallWriteWindowAcquisitionSignal,
  type RecallWriteWindow,
} from './coordinate-recall-write-window.js';
import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import {
  RecallDiagnosticErrorCategory,
  RecallDiagnosticStatus,
  type RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallGenerationCutoverState,
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
  readRecallSearchManifest,
  recoverRecallEmbeddingCanaryFromManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
  type RecallIndexManifest,
} from './recall-index-manifest.js';
import {
  readRecallActiveGenerationPointer,
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  readRecallMaterialBacklogWarning,
} from './recall-generation-state.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  createRecallDetachedWorkerSignal,
  type RecallDetachedWorkerSignal,
} from './publish-recall-work-marker.js';
import { rebuildRecallGeneration } from './rebuild-recall-generation.js';
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
import {
  createLineageResolver,
  resolveProjectIdentity,
  type ProjectIdentity,
  type RecallProjectLineages,
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
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

/** Runtime paths, bounded retrieval channels, and local embedding plus reranker identity. */
export interface RecallConversationConfig {
  sessionsDirectory: string;
  dataDirectory: string;
  databasePath: string;
  projectionDatabasePath: string;
  statePath: string;
  manifestPath: string;
  tokenizerCacheDirectory: string;
  embeddingCacheDirectory: string;
  lockPath: string;
  diagnosticsMode: RecallDiagnosticsMode;
  diagnosticLogPath: string;
  retainedDiagnosticLogPath: string;
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  markerControlDirectory: string;
  workerOwnershipLockPath: string;
  generationRootDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  incrementalDiagnosticLogPath: string;
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
  searchWriteWindowWaitMilliseconds: number;
  confirmedDeletionMaxMissingSourceCount: number;
  confirmedDeletionMaxMissingSourceRatio: number;
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

interface RecallConversationIndexBaseOptions {
  signal?: AbortSignal;
  lockWaitMilliseconds?: number;
  requireExistingGeneration?: boolean;
  onProgress?: (progress: ConversationIndexProgress) => void;
}

interface RecallAutomaticIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: false;
  manualMaintenanceTrigger?: never;
}

interface RecallAutomaticRebuildIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild: true;
  manualMaintenanceTrigger?: never;
  optimize?: boolean;
}

interface RecallManualIncrementalIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild?: false;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX;
}

interface RecallManualRebuildIndexOptions extends RecallConversationIndexBaseOptions {
  rebuild: true;
  manualMaintenanceTrigger: RecallManualMaintenanceTrigger.MANUAL_REBUILD;
  optimize?: boolean;
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

/** Read-only search plus explicit full index maintenance exposed by the extension. */
export interface RecallConversationService {
  search(
    query: string,
    limit: number,
    options?: RecallConversationSearchOptions,
  ): Promise<RecallConversationSearch>;
  index(options?: RecallConversationIndexOptions): Promise<RecallConversationIndexResult>;
  /** Restores the bounded rollback generation and republishes retained markers. */
  rollback?(): Promise<void>;
  /** Explicitly adopts the exact pre-generation version-5 layout as read-only. */
  adoptLegacy?(): Promise<void>;
  /** Collects only expired validated generations after replay completes. */
  collectRetired?(): Promise<void>;
}

/** Injectable local model, tokenizer, and zvec boundaries used by tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddings?: LocalEmbeddingClient;
  reranker?: LocalRerankerClient;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write', databasePath?: string) => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  diagnostics?: RecallOperationDiagnostics;
  diagnosticsClock?: RecallDiagnosticsClock;
  notifyWarning?: (message: string) => void;
  workerSignal?: RecallDetachedWorkerSignal;
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
    ((mode, databasePath = config.databasePath) =>
      openZvecConversationStore({
        databasePath,
        dimensions: config.embeddingDimensions,
        createIfMissing: mode === 'write',
        readOnly: mode === 'read',
      }));
  const diagnosticsClock = dependencies.diagnosticsClock ?? {
    monotonicMilliseconds: () => performance.now(),
    wallClockIsoTimestamp: () => new Date().toISOString(),
  };
  const workerSignal =
    dependencies.workerSignal ?? createRecallDetachedWorkerSignal(config.workerOwnershipLockPath);
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
    const embedding = (await embeddings.embedTexts([RECALL_EMBEDDING_CANARY_TEXT], signal))[0];
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
      embeddingIdentity: createEmbeddingModelIdentity(config),
      canaryEmbedding,
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
            config.embeddingDimensions,
          )) ?? undefined;
      }
    }
    if (!previousCanary || previousCanary.dimensions !== config.embeddingDimensions) {
      return currentCanary;
    }
    const cosineSimilarity = calculateRecallEmbeddingCanaryCosineSimilarity(
      previousCanary.canaryVector,
      currentCanary,
      config.embeddingDimensions,
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
    search(query, limit, options = {}) {
      const {
        mode = 'hybrid',
        scope = RecallSearchScope.PROJECT,
        invocationDirectory,
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
              queryEmbedding = (await embeddings.embedTexts([searchQuery], signal))[0];
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
                    const diagnosticReranker: LocalRerankerClient = {
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
                        rerankerModel: mode === 'deep-rerank' ? config.rerankerModel : null,
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
          const rebuilt = await rebuildRecallGeneration({
            generationRootDirectory: config.generationRootDirectory,
            activeGenerationPointerPath: config.activeGenerationPointerPath,
            generationRegistryPath: config.generationRegistryPath,
            backlogSummaryPath: config.backlogSummaryPath,
            markerSpoolDirectory: config.markerSpoolDirectory,
            lockPath: config.lockPath,
            workerSignal,
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
                await getConversationTokenizer();
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
                const indexSummary = await updateConversationIndex(
                  store,
                  preparedIndex.tokenizer,
                  preparedIndex.manifest,
                  preparedIndex.embeddingModelPreflighted,
                  targetPaths,
                  options.signal,
                  options.onProgress,
                  diagnosticMetrics,
                  onPhysicalSessionCheck,
                  approvedRebuildSnapshot?.eligibleContributorEntryIdsBySessionPath,
                );
                const result = { indexSummary, totalChunks: store.count() };
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
                          const optimizationStartedAtMilliseconds =
                            diagnosticsClock.monotonicMilliseconds();
                          if (diagnosticMetrics) {
                            diagnosticMetrics.optimizationRan = true;
                          }
                          try {
                            const optimizeStore = () => storeToClose.optimize();
                            if (runOptimizationWithDiagnostics) {
                              await runOptimizationWithDiagnostics(optimizeStore);
                            } else {
                              await optimizeStore();
                            }
                          } finally {
                            if (diagnosticMetrics) {
                              diagnosticMetrics.optimizationMilliseconds += Math.max(
                                diagnosticsClock.monotonicMilliseconds() -
                                  optimizationStartedAtMilliseconds,
                                0,
                              );
                            }
                          }
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
                      await projectionStore.upsertProjections(
                        approvedRebuildSnapshot?.projections.map((projection) =>
                          retargetRecallRebuildProjection(projection, paths.generationId),
                        ) ?? [],
                      );
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
                  approvedRebuildSnapshot?.eligibleContributorEntryIdsBySessionPath.size ?? 0;
                if (physicalProjections.length !== expectedPhysicalProjectionCount) {
                  throw new Error(
                    'Recall replacement generation projection snapshot changed during validation',
                  );
                }
                const expectedProjectionIds =
                  approvedRebuildSnapshot?.projections.map(({ projectionId }) => projectionId) ??
                  [];
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
                  diagnosticMetrics,
                  onPhysicalSessionCheck,
                );
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
    async rollback() {
      await rollbackRecallGeneration({
        activeGenerationPointerPath: config.activeGenerationPointerPath,
        generationRegistryPath: config.generationRegistryPath,
        generationRootDirectory: config.generationRootDirectory,
        backlogSummaryPath: config.backlogSummaryPath,
        markerSpoolDirectory: config.markerSpoolDirectory,
        retainedMarkerDirectory: join(config.markerControlDirectory, 'rollback-retained'),
        lockPath: config.lockPath,
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

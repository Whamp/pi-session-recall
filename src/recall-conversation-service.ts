import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  createEmbeddingVectorCache,
  createEmbeddingVectorCacheIdentity,
} from './embedding-vector-cache.js';
import {
  RecallDiagnosticErrorCategory,
  RecallDiagnosticStatus,
  type RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallLifecycleTrigger,
  RecallManualMaintenanceTrigger,
  RecallProjectIdentitySource,
  RecallRankedListSource,
  RecallSearchScope,
} from './enums.js';
import {
  fuseRecallRankedLists,
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
  type RecallRankedList,
} from './fuse-recall-ranked-lists.js';
import {
  indexChangedConversationSession,
  indexChangedConversationSessions,
  type ConversationIndexProgress,
  type ConversationIndexSummary,
} from './incremental-session-indexer.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createLocalEmbeddingClient, type LocalEmbeddingClient } from './local-embedding-client.js';
import { createLocalRerankerClient, type LocalRerankerClient } from './local-reranker-client.js';
import { loadOctenConversationTokenizer } from './octen-conversation-tokenizer.js';
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
} from './recall-index-manifest.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
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
  type RecallProjectLineages,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import {
  QUERY_PLANNED_RECALL_FUSED_RANK_BLEND,
  QUERY_PLANNED_RECALL_RERANK_POLICY_VERSION,
  rankFusedRecallSearchResults,
  rerankRecallSearchResults,
  RECALL_ACTIVE_BRANCH_PRIOR,
  RECALL_RERANK_POLICY_VERSION,
  type RankedRecallSearchResult,
  type RecallFusedRankBlendBand,
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
  diagnosticsMode: RecallDiagnosticsMode;
  diagnosticLogPath: string;
  retainedDiagnosticLogPath: string;
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
  fusedPoolLimit?: number;
  rerankPoolLimit?: number;
  chunkPolicy?: RecallChunkPolicy;
}

/** Per-channel candidate caps applied before recall rank fusion. */
export interface RecallSearchCandidateLimits {
  dense: number;
  lexical: number;
  identifier: number;
}

/** Maximum agent-supplied retrieval queries admitted by one query-planned recall search. */
export const MAX_AGENT_RECALL_PLANNED_QUERY_COUNT = 10;

/** Fixed project-eligible candidate admission limit for every query-planned ranked list. */
export const QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT = 20;

/** Duplicate evidence group limit applied before query-planned reranking. */
export const QUERY_PLANNED_RECALL_GROUP_LIMIT = 40;

/** Final result cap applied before query-planned neighbor context expansion. */
export const QUERY_PLANNED_RECALL_FINAL_RESULT_LIMIT = 5;

/** One agent-supplied retrieval query routed only to its declared channel. */
export interface RecallPlannedRetrievalQuery {
  type: 'lex' | 'vec' | 'hyde';
  query: string;
}

/** User-selected ranking depth for hybrid-only, local Qwen, or agent-planned recall search. */
export type RecallSearchMode = 'hybrid' | 'deep-rerank' | 'query-planned';

/** One project-eligible ranked-list admission trace from query-planned retrieval. */
export interface RecallRankedListTrace {
  source: RecallRankedListSource;
  query: string;
  weight: number;
  candidateLimit: number;
  admittedCandidateCount: number;
}

/** QMD fusion constants used for one agent-supplied query plan. */
export interface RecallQueryPlannedFusionPolicy {
  reciprocalRankConstant: number;
  submittedQueryListWeight: number;
  plannedQueryListWeight: number;
  rankOneBonus: number;
  rankTwoOrThreeBonus: number;
}

/** Reranker model and position-aware blend profile used by query-planned recall. */
export interface RecallQueryPlannedRerankerProfile {
  model: string;
  policyVersion: number;
  fusedRankBlend: readonly RecallFusedRankBlendBand[];
}

/** Agent plan, intent, ranked-list traces, fusion constants, and reranker profile. */
export interface RecallQueryPlanEvidence {
  source: 'agent';
  intent: string | null;
  plannedQueries: readonly RecallPlannedRetrievalQuery[];
  rankedLists: readonly RecallRankedListTrace[];
  fusionPolicy: RecallQueryPlannedFusionPolicy;
  rerankerProfile: RecallQueryPlannedRerankerProfile;
}

/** Trusted invocation context, cancellation, and ranking depth for one recall search. */
export interface RecallConversationSearchOptions {
  mode?: RecallSearchMode;
  scope?: RecallSearchScope;
  invocationDirectory?: string;
  activeSessionPath?: string;
  plan?: readonly RecallPlannedRetrievalQuery[];
  intent?: string;
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
  fusedPoolLimit: number;
  rerankPoolLimit: number;
  finalResultLimit: number;
  queryPlan?: RecallQueryPlanEvidence;
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
    options: RecallConversationReconcileOptions,
  ): Promise<RecallConversationIndexResult>;
}

/** Injectable local model, tokenizer, and zvec boundaries used by tests and bounded evaluation. */
export interface RecallConversationDependencies {
  embeddings?: LocalEmbeddingClient;
  reranker?: LocalRerankerClient;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
  openStore?: (mode: 'read' | 'write') => ZvecConversationStore;
  resolveProjectIdentity?: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  diagnostics?: RecallOperationDiagnostics;
  diagnosticsClock?: RecallDiagnosticsClock;
  notifyWarning?: (message: string) => void;
}

interface ValidatedRecallQueryPlanningOptions {
  plannedQueries: RecallPlannedRetrievalQuery[];
  intent: string | null;
}

const QUERY_PLANNED_RECALL_SUBMITTED_QUERY_LIST_WEIGHT = 2;
const QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT = 1;
const QUERY_PLANNED_RECALL_RANK_ONE_BONUS = 0.05;
const QUERY_PLANNED_RECALL_RANK_TWO_OR_THREE_BONUS = 0.02;

function validateRecallQueryPlanningOptions(
  mode: RecallSearchMode,
  plan: readonly RecallPlannedRetrievalQuery[] | undefined,
  intent: string | undefined,
): ValidatedRecallQueryPlanningOptions {
  if (mode === 'hybrid' && intent !== undefined) {
    throw new Error(
      'Recall hybrid mode does not accept intent; remove intent or choose deep-rerank or query-planned mode',
    );
  }
  if (mode !== 'query-planned' && plan !== undefined) {
    throw new Error(
      'Recall query plan is supported only in query-planned mode; remove plan or choose query-planned mode',
    );
  }
  const normalizedIntent = intent?.trim() ?? null;
  if (intent !== undefined && !normalizedIntent) {
    throw new Error('Recall intent must not be blank');
  }
  if (mode !== 'query-planned') {
    return { plannedQueries: [], intent: normalizedIntent };
  }
  if (!Array.isArray(plan)) {
    throw new Error(
      'Recall query plan invalid: provide plan with 1 to 10 { type: lex|vec|hyde, query: string } entries',
    );
  }
  if (plan.length < 1 || plan.length > MAX_AGENT_RECALL_PLANNED_QUERY_COUNT) {
    throw new Error(
      `Recall query plan invalid: expected 1 to ${MAX_AGENT_RECALL_PLANNED_QUERY_COUNT} entries, received ${plan.length}`,
    );
  }
  const plannedQueries: RecallPlannedRetrievalQuery[] = [];
  for (const [index, plannedQuery] of plan.entries()) {
    if (!isUnknownRecord(plannedQuery)) {
      throw new Error(
        `Recall query plan invalid at entry ${index + 1}: expected { type: lex|vec|hyde, query: string }`,
      );
    }
    const unexpectedKeys = Object.keys(plannedQuery).filter(
      (key) => key !== 'type' && key !== 'query',
    );
    if (unexpectedKeys.length > 0) {
      throw new Error(
        `Recall query plan invalid at entry ${index + 1}: unsupported field ${unexpectedKeys[0]}; use only type and query`,
      );
    }
    const queryType = plannedQuery.type;
    if (queryType !== 'lex' && queryType !== 'vec' && queryType !== 'hyde') {
      throw new Error(
        `Recall query plan invalid at entry ${index + 1}: unsupported type ${String(queryType)}; use lex, vec, or hyde`,
      );
    }
    if (typeof plannedQuery.query !== 'string' || !plannedQuery.query.trim()) {
      throw new Error(
        `Recall query plan invalid at entry ${index + 1}: query must be a nonblank string`,
      );
    }
    if (/\r|\n/u.test(plannedQuery.query)) {
      throw new Error(`Recall query plan invalid at entry ${index + 1}: query must be single-line`);
    }
    plannedQueries.push({ type: queryType, query: plannedQuery.query.trim() });
  }
  return { plannedQueries, intent: normalizedIntent };
}

function createRecallRerankerQuery(query: string, intent: string | null): string {
  return intent ? `${intent}\n\n${query}` : query;
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
    try {
      const actualManifest = await readRecallIndexManifest(config.manifestPath);
      previousCanary = actualManifest?.embedding;
    } catch (error) {
      if (
        !(error instanceof Error) ||
        !error.message.startsWith(`Recall index manifest invalid at ${config.manifestPath}:`)
      ) {
        throw error;
      }
      previousCanary =
        (await recoverRecallEmbeddingCanaryFromManifest(
          config.manifestPath,
          config.embeddingDimensions,
        )) ?? undefined;
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
    diagnosticMetrics?: RecallIndexDiagnosticMetrics,
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
      preflightedCanary ?? (await readIndexEmbeddingCanary(signal, diagnosticMetrics)),
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
      diagnosticsClock,
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
      ...(diagnosticMetrics ? { diagnosticMetrics, diagnosticsClock } : {}),
      ...(onPhysicalSessionCheck ? { onPhysicalSessionCheck } : {}),
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
    const lockSignal = createRecallLockAcquisitionSignal(signal, lockWaitMilliseconds);
    const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
    let releaseLock: (() => Promise<void>) | undefined;
    try {
      releaseLock = await acquireRecallConversationLock(config.lockPath, lockSignal);
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
        preparedIndex = await prepareIndexForWrite(signal, undefined, true);
        store = openStore('write');
      } finally {
        if (diagnosticMetrics) {
          diagnosticMetrics.manifestStorePreparationMilliseconds += Math.max(
            diagnosticsClock.monotonicMilliseconds() - preparationStartedAtMilliseconds,
            0,
          );
        }
      }
      const summary = await updateConversationIndex(
        store,
        preparedIndex.tokenizer,
        preparedIndex.manifest,
        preparedIndex.embeddingModelPreflighted,
        signal,
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
    async search(query, limit, options = {}) {
      const {
        mode = 'hybrid',
        scope = RecallSearchScope.PROJECT,
        invocationDirectory,
        activeSessionPath,
        plan,
        intent,
        signal,
      } = options;
      const queryPlanning = validateRecallQueryPlanningOptions(mode, plan, intent);
      return await runRecallSearchWithDiagnostics({
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
            let actualManifest: RecallIndexManifest;
            const manifestReadStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
            try {
              actualManifest = await readRequiredManifest();
            } finally {
              diagnosticMetrics.embeddingModelVerificationMilliseconds += Math.max(
                diagnosticsClock.monotonicMilliseconds() - manifestReadStartedAtMilliseconds,
                0,
              );
            }
            await assertRecallIndexUnlockedForSearch(config.lockPath);
            const canaryVerificationStartedAtMilliseconds =
              diagnosticsClock.monotonicMilliseconds();
            try {
              const expectedManifest = createExpectedManifest(
                await readCurrentEmbeddingCanary(signal),
              );
              assertRecallIndexManifestCompatible(
                actualManifest,
                expectedManifest,
                config.manifestPath,
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
            await assertRecallIndexUnlockedForSearch(config.lockPath);
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
            const store = openStore('read');
            try {
              const semanticQueryTexts = [
                searchQuery,
                ...queryPlanning.plannedQueries
                  .filter(
                    (plannedQuery) => plannedQuery.type === 'vec' || plannedQuery.type === 'hyde',
                  )
                  .map((plannedQuery) => plannedQuery.query),
              ];
              const queryEmbeddingStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              let queryEmbeddings: number[][];
              try {
                queryEmbeddings = await embeddings.embedTexts(semanticQueryTexts, signal);
              } finally {
                diagnosticMetrics.queryEmbeddingMilliseconds += Math.max(
                  diagnosticsClock.monotonicMilliseconds() - queryEmbeddingStartedAtMilliseconds,
                  0,
                );
              }
              if (queryEmbeddings.length !== semanticQueryTexts.length) {
                throw new Error(
                  `Recall embedding response vector count mismatch: expected ${semanticQueryTexts.length}, received ${queryEmbeddings.length}`,
                );
              }
              const queryEmbedding = queryEmbeddings[0];
              if (!queryEmbedding) {
                throw new Error('Recall embedding response missing submitted query vector');
              }
              const retrievalStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              const deepRerankStartedWithMilliseconds = diagnosticMetrics.deepRerankMilliseconds;
              try {
                const isQueryPlanned = mode === 'query-planned';
                const submittedQueryCandidateLimits = isQueryPlanned
                  ? {
                      dense: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                      lexical: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                      identifier: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                    }
                  : config.searchCandidateLimits;
                const submittedQueryListWeight = isQueryPlanned
                  ? QUERY_PLANNED_RECALL_SUBMITTED_QUERY_LIST_WEIGHT
                  : 1;
                const maximumCurrentCandidateCount =
                  config.searchCandidateLimits.dense +
                  config.searchCandidateLimits.lexical +
                  config.searchCandidateLimits.identifier;
                const fusedPoolLimit = isQueryPlanned
                  ? QUERY_PLANNED_RECALL_GROUP_LIMIT
                  : (config.fusedPoolLimit ?? maximumCurrentCandidateCount);
                const rerankPoolLimit = isQueryPlanned
                  ? QUERY_PLANNED_RECALL_GROUP_LIMIT
                  : (config.rerankPoolLimit ?? fusedPoolLimit);
                if (
                  !Number.isInteger(rerankPoolLimit) ||
                  rerankPoolLimit < 1 ||
                  rerankPoolLimit > fusedPoolLimit
                ) {
                  throw new Error(
                    `Recall rerank pool limit invalid: expected an integer from 1 to fused pool limit ${fusedPoolLimit}`,
                  );
                }
                const denseCandidates = store.searchDenseCandidates(
                  queryEmbedding,
                  submittedQueryCandidateLimits.dense,
                  projectIdentityPredicate,
                );
                const lexicalCandidates = store.searchLexicalCandidates(
                  searchQuery,
                  submittedQueryCandidateLimits.lexical,
                  projectIdentityPredicate,
                );
                const identifierCandidates = store.searchIdentifierCandidates(
                  searchQuery,
                  submittedQueryCandidateLimits.identifier,
                  projectIdentityPredicate,
                );
                const rankedLists: RecallRankedList[] = [
                  {
                    source: RecallRankedListSource.DENSE,
                    query: searchQuery,
                    weight: submittedQueryListWeight,
                    candidateLimit: submittedQueryCandidateLimits.dense,
                    higherNativeScoresRankFirst: false,
                    candidates: denseCandidates.map(({ cosineDistance, ...document }) => ({
                      document,
                      nativeScore: cosineDistance,
                    })),
                  },
                  {
                    source: RecallRankedListSource.LEXICAL,
                    query: searchQuery,
                    weight: submittedQueryListWeight,
                    candidateLimit: submittedQueryCandidateLimits.lexical,
                    higherNativeScoresRankFirst: true,
                    candidates: lexicalCandidates.map(({ fullTextScore, ...document }) => ({
                      document,
                      nativeScore: fullTextScore,
                    })),
                  },
                  {
                    source: RecallRankedListSource.IDENTIFIER,
                    query: searchQuery,
                    weight: submittedQueryListWeight,
                    candidateLimit: submittedQueryCandidateLimits.identifier,
                    higherNativeScoresRankFirst: true,
                    candidates: identifierCandidates.map(({ fullTextScore, ...document }) => ({
                      document,
                      nativeScore: fullTextScore,
                    })),
                  },
                ];
                const rankedListTraces: RecallRankedListTrace[] = rankedLists.map((list) => ({
                  source: list.source,
                  query: list.query,
                  weight: list.weight,
                  candidateLimit: list.candidateLimit,
                  admittedCandidateCount: list.candidates.length,
                }));
                let semanticEmbeddingIndex = 1;
                for (const plannedQuery of queryPlanning.plannedQueries) {
                  if (plannedQuery.type === 'lex') {
                    const plannedCandidates = store.searchLexicalCandidates(
                      plannedQuery.query,
                      QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                      projectIdentityPredicate,
                    );
                    rankedLists.push({
                      source: RecallRankedListSource.PLANNED_LEX,
                      query: plannedQuery.query,
                      weight: QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT,
                      candidateLimit: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                      higherNativeScoresRankFirst: true,
                      candidates: plannedCandidates.map(({ fullTextScore, ...document }) => ({
                        document,
                        nativeScore: fullTextScore,
                      })),
                    });
                    rankedListTraces.push({
                      source: RecallRankedListSource.PLANNED_LEX,
                      query: plannedQuery.query,
                      weight: QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT,
                      candidateLimit: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                      admittedCandidateCount: plannedCandidates.length,
                    });
                    continue;
                  }
                  const plannedEmbedding = queryEmbeddings[semanticEmbeddingIndex];
                  semanticEmbeddingIndex += 1;
                  if (!plannedEmbedding) {
                    throw new Error(
                      `Recall embedding response missing ${plannedQuery.type} plan vector at entry ${semanticEmbeddingIndex - 1}`,
                    );
                  }
                  const plannedCandidates = store.searchDenseCandidates(
                    plannedEmbedding,
                    QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                    projectIdentityPredicate,
                  );
                  const plannedSource =
                    plannedQuery.type === 'vec'
                      ? RecallRankedListSource.PLANNED_VEC
                      : RecallRankedListSource.PLANNED_HYDE;
                  rankedLists.push({
                    source: plannedSource,
                    query: plannedQuery.query,
                    weight: QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT,
                    candidateLimit: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                    higherNativeScoresRankFirst: false,
                    candidates: plannedCandidates.map(({ cosineDistance, ...document }) => ({
                      document,
                      nativeScore: cosineDistance,
                    })),
                  });
                  rankedListTraces.push({
                    source: plannedSource,
                    query: plannedQuery.query,
                    weight: QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT,
                    candidateLimit: QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT,
                    admittedCandidateCount: plannedCandidates.length,
                  });
                }
                const preGroupingDocumentLimit = isQueryPlanned
                  ? rankedLists.length * QUERY_PLANNED_RECALL_LIST_CANDIDATE_LIMIT
                  : fusedPoolLimit;
                const fusedCandidates = fuseRecallRankedLists(
                  rankedLists,
                  preGroupingDocumentLimit,
                  isQueryPlanned
                    ? {
                        rankOne: QUERY_PLANNED_RECALL_RANK_ONE_BONUS,
                        rankTwoOrThree: QUERY_PLANNED_RECALL_RANK_TWO_OR_THREE_BONUS,
                      }
                    : undefined,
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
                        diagnosticsClock.monotonicMilliseconds() - deepRerankStartedAtMilliseconds,
                        0,
                      );
                    }
                  },
                };
                const finalResultLimit = isQueryPlanned
                  ? Math.min(limit, QUERY_PLANNED_RECALL_FINAL_RESULT_LIMIT)
                  : limit;
                let rankedResults: RankedRecallSearchResult[];
                if (mode === 'hybrid') {
                  rankedResults = rankFusedRecallSearchResults(
                    fusedCandidates,
                    finalResultLimit,
                    store.fetchConversationChunks,
                  );
                } else {
                  try {
                    rankedResults = await rerankRecallSearchResults({
                      query: createRecallRerankerQuery(searchQuery, queryPlanning.intent),
                      candidates: fusedCandidates,
                      rerankPoolLimit,
                      resultLimit: finalResultLimit,
                      reranker: diagnosticReranker,
                      fetchConversationChunks: store.fetchConversationChunks,
                      ...(isQueryPlanned ? { useQueryPlannedPositionBlend: true } : {}),
                      ...(signal ? { signal } : {}),
                    });
                  } catch (error: unknown) {
                    if (!isQueryPlanned || isRecallOperationCancelled(error, signal)) {
                      throw error;
                    }
                    const reason = error instanceof Error ? error.message : String(error);
                    throw new Error(
                      `Recall query-planned reranking failed: verify the configured reranker and retry; ${reason}`,
                      { cause: error },
                    );
                  }
                }
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
                      mode === 'query-planned'
                        ? QUERY_PLANNED_RECALL_RERANK_POLICY_VERSION
                        : mode === 'deep-rerank'
                          ? RECALL_RERANK_POLICY_VERSION
                          : null,
                    rerankerModel: mode === 'hybrid' ? null : config.rerankerModel,
                    activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
                    candidateLimits: { ...submittedQueryCandidateLimits },
                    fusedPoolLimit,
                    rerankPoolLimit,
                    finalResultLimit,
                    ...(isQueryPlanned
                      ? {
                          queryPlan: {
                            source: 'agent',
                            intent: queryPlanning.intent,
                            plannedQueries: queryPlanning.plannedQueries,
                            rankedLists: rankedListTraces,
                            fusionPolicy: {
                              reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
                              submittedQueryListWeight:
                                QUERY_PLANNED_RECALL_SUBMITTED_QUERY_LIST_WEIGHT,
                              plannedQueryListWeight:
                                QUERY_PLANNED_RECALL_PLANNED_QUERY_LIST_WEIGHT,
                              rankOneBonus: QUERY_PLANNED_RECALL_RANK_ONE_BONUS,
                              rankTwoOrThreeBonus: QUERY_PLANNED_RECALL_RANK_TWO_OR_THREE_BONUS,
                            },
                            rerankerProfile: {
                              model: config.rerankerModel,
                              policyVersion: QUERY_PLANNED_RECALL_RERANK_POLICY_VERSION,
                              fusedRankBlend: QUERY_PLANNED_RECALL_FUSED_RANK_BLEND,
                            },
                          },
                        }
                      : {}),
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
      const runConversationIndexMaintenance = (
        diagnosticMetrics?: RecallIndexDiagnosticMetrics,
        onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void,
        runOptimizationWithDiagnostics?: (optimize: () => Promise<void>) => Promise<void>,
      ) =>
        runSerialized(async () => {
          const lockSignal = createRecallLockAcquisitionSignal(
            options.signal,
            options.lockWaitMilliseconds,
          );
          const lockStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
          let releaseLock: (() => Promise<void>) | undefined;
          try {
            releaseLock = await acquireRecallConversationLock(config.lockPath, lockSignal);
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
            const embeddingServerMillisecondsBeforePreparation =
              diagnosticMetrics?.embeddingServerRequestMilliseconds ?? 0;
            let preparedIndex: Awaited<ReturnType<typeof prepareIndexForWrite>>;
            try {
              let rebuildCanary: number[] | undefined;
              if (options.rebuild) {
                await getConversationTokenizer();
                rebuildCanary = await readCanonicalRebuildCanary(options.signal, diagnosticMetrics);
                await removeRecallIndexGeneration();
              }
              preparedIndex = await prepareIndexForWrite(
                options.signal,
                rebuildCanary,
                options.requireExistingGeneration,
                diagnosticMetrics,
              );
              store = openStore('write');
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
            const indexSummary = await updateConversationIndex(
              store,
              preparedIndex.tokenizer,
              preparedIndex.manifest,
              preparedIndex.embeddingModelPreflighted,
              options.signal,
              options.onProgress,
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
              const optimizationStartedAtMilliseconds = diagnosticsClock.monotonicMilliseconds();
              const storeToOptimize = store;
              if (diagnosticMetrics) {
                diagnosticMetrics.optimizationRan = true;
              }
              try {
                const optimizeStore = () => storeToOptimize.optimize();
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
            return { indexSummary, totalChunks: store.count() };
          } finally {
            store?.close();
            await releaseLock();
          }
        });
      return options.manualMaintenanceTrigger
        ? runManualIndexWithDiagnostics({
            diagnostics,
            manualMaintenanceTrigger: options.manualMaintenanceTrigger,
            ...(options.signal ? { signal: options.signal } : {}),
            runIndexMaintenance: runConversationIndexMaintenance,
          })
        : runConversationIndexMaintenance();
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

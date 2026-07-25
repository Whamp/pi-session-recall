import { mkdir, rm } from 'node:fs/promises';
import { basename, isAbsolute, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

import {
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-search-candidates.js';
import type { ConversationIndexSummary } from './incremental-session-indexer.js';
import { createLocalEmbeddingClient, type LocalEmbeddingClient } from './local-embedding-client.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import {
  measureRecallQuality,
  type RecallQualitySearchObservation,
} from './measure-recall-quality.js';
import type {
  LoadedRecallQualityCorpus,
  RecallQualityChunkPolicy,
} from './recall-quality-corpus.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  type RecallConversationDependencies,
} from './recall-conversation-service.js';
import {
  selectRecallQualityPolicy,
  type RecallQualityConfigurationMeasurement,
  type RecallQualityPolicySelection,
} from './select-recall-quality-policy.js';
import { RECALL_ACTIVE_BRANCH_PRIOR } from './rank-recall-search-results.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const RECALL_QUALITY_FULL_POOL_LIMIT = 200;
const RECALL_QUALITY_WORK_DIRECTORY_NAME = 'recall-quality-evaluation';

/** Local model and tokenizer boundaries that make the bounded runner integration-testable. */
export interface RecallQualityEvaluationDependencies {
  embeddings?: LocalEmbeddingClient;
  loadTokenizer?: () => Promise<ConversationTextTokenizer>;
}

/** Inputs for one bounded evaluation run over a checksum-fixed corpus. */
export interface RunRecallQualityEvaluationOptions {
  corpus: LoadedRecallQualityCorpus;
  baseConfig: RecallConversationConfig;
  workDirectory: string;
  dependencies?: RecallQualityEvaluationDependencies;
}

/** Index work performed once for one required chunk policy. */
export interface RecallQualityIndexRun {
  chunkPolicy: RecallQualityChunkPolicy;
  totalChunks: number;
  indexLatencyMilliseconds: number;
  indexSummary: ConversationIndexSummary;
}

/** Exact work counters proving the run stayed inside the fixed evaluation bounds. */
export interface RecallQualityExecutedWork {
  sessionFiles: number;
  evaluationCases: number;
  indexRuns: number;
  executedSearchRequests: number;
  rerankerRequests: number;
  chunkEmbeddingRequests: number;
  maximumCandidatesPerSearch: number;
}

/** Ranking constants that bind quality evidence to the hybrid policy it measured. */
export interface RecallQualityRankingIdentity {
  rankingMode: 'hybrid';
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  activeBranchPrior: number;
}

/** Raw measured configurations, gate selection, and bounded-work evidence from one run. */
export interface RecallQualityEvaluationResult {
  version: 3;
  rankingIdentity: RecallQualityRankingIdentity;
  startedAt: string;
  completedAt: string;
  durationMilliseconds: number;
  specificationPath: string;
  specificationSha256: string;
  corpusId: string;
  indexRuns: RecallQualityIndexRun[];
  configurations: RecallQualityConfigurationMeasurement[];
  selection: RecallQualityPolicySelection;
  boundedWork: RecallQualityExecutedWork;
}

function isPathInside(parentPath: string, childPath: string): boolean {
  const pathFromParent = relative(parentPath, childPath);
  return pathFromParent === '' || (!pathFromParent.startsWith('..') && !isAbsolute(pathFromParent));
}

function assertSafeRecallQualityPaths(
  workDirectory: string,
  corpus: LoadedRecallQualityCorpus,
  baseConfig: RecallConversationConfig,
): string {
  const resolvedWorkDirectory = resolve(workDirectory);
  if (basename(resolvedWorkDirectory) !== RECALL_QUALITY_WORK_DIRECTORY_NAME) {
    throw new Error(
      `Recall quality work directory invalid: basename must be ${RECALL_QUALITY_WORK_DIRECTORY_NAME}`,
    );
  }
  const protectedPaths = [
    corpus.sessionDirectory,
    baseConfig.sessionsDirectory,
    baseConfig.databasePath,
    baseConfig.statePath,
    baseConfig.manifestPath,
    baseConfig.embeddingCacheDirectory,
    baseConfig.lockPath,
  ].map((path) => resolve(path));
  for (const protectedPath of protectedPaths) {
    if (
      isPathInside(resolvedWorkDirectory, protectedPath) ||
      isPathInside(protectedPath, resolvedWorkDirectory)
    ) {
      throw new Error(
        `Recall quality work directory overlaps protected path: ${resolvedWorkDirectory} and ${protectedPath}`,
      );
    }
  }
  return resolvedWorkDirectory;
}

function createEvaluationEmbeddingClient(
  config: RecallConversationConfig,
  dependency?: LocalEmbeddingClient,
): LocalEmbeddingClient {
  return (
    dependency ??
    createLocalEmbeddingClient({
      baseUrl: config.embeddingBaseUrl,
      model: config.embeddingModel,
      dimensions: config.embeddingDimensions,
      batchSize: config.embeddingBatchSize,
    })
  );
}

function createChunkPolicyConfig(
  baseConfig: RecallConversationConfig,
  corpus: LoadedRecallQualityCorpus,
  workDirectory: string,
  chunkPolicy: RecallQualityChunkPolicy,
  candidateCount: number,
): RecallConversationConfig {
  const policyDirectory = join(workDirectory, chunkPolicy.id);
  return {
    ...baseConfig,
    sessionsDirectory: corpus.sessionDirectory,
    databasePath: join(policyDirectory, 'zvec'),
    statePath: join(policyDirectory, 'index-state.json'),
    manifestPath: join(policyDirectory, 'index-manifest.json'),
    embeddingCacheDirectory: join(policyDirectory, 'embedding-cache'),
    lockPath: join(policyDirectory, 'operation.lock'),
    chunkPolicy: {
      maxTokens: chunkPolicy.maxTokens,
      overlapTokens: chunkPolicy.overlapTokens,
    },
    searchCandidateLimits: {
      dense: candidateCount,
      lexical: candidateCount,
      identifier: candidateCount,
    },
  };
}

function createServiceDependencies(
  embeddings: LocalEmbeddingClient,
  reranker: LocalRerankerClient,
  loadTokenizer?: () => Promise<ConversationTextTokenizer>,
): RecallConversationDependencies {
  return {
    embeddings,
    reranker,
    ...(loadTokenizer ? { loadTokenizer } : {}),
  };
}

function assertBoundedCandidateCount(candidateCount: number): void {
  const maximumCandidates = candidateCount * 3;
  if (maximumCandidates > RECALL_QUALITY_FULL_POOL_LIMIT) {
    throw new Error(
      `Recall quality candidate bound exceeded: ${candidateCount} per channel can produce ${maximumCandidates}, above full-pool limit ${RECALL_QUALITY_FULL_POOL_LIMIT}`,
    );
  }
}

/** Runs exactly the declared policy/count grid without reading or writing the production corpus. */
export async function runRecallQualityEvaluation(
  options: RunRecallQualityEvaluationOptions,
): Promise<RecallQualityEvaluationResult> {
  const startedAt = new Date();
  const started = performance.now();
  const workDirectory = assertSafeRecallQualityPaths(
    options.workDirectory,
    options.corpus,
    options.baseConfig,
  );
  const { specification } = options.corpus;
  for (const candidateCount of specification.candidateCounts) {
    assertBoundedCandidateCount(candidateCount);
  }
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true });

  const embeddings = createEvaluationEmbeddingClient(
    options.baseConfig,
    options.dependencies?.embeddings,
  );
  const indexRuns: RecallQualityIndexRun[] = [];
  const configurations: RecallQualityConfigurationMeasurement[] = [];
  let executedSearchRequests = 0;
  let rerankerRequests = 0;
  const rejectingReranker: LocalRerankerClient = {
    async rerankDocuments() {
      rerankerRequests += 1;
      throw new Error('Recall quality evaluation attempted an unexpected reranker request');
    },
  };

  for (const chunkPolicy of specification.chunkPolicies) {
    const firstCandidateCount = specification.candidateCounts[0];
    if (firstCandidateCount === undefined) {
      throw new Error('Recall quality evaluation requires at least one candidate count');
    }
    const indexConfig = createChunkPolicyConfig(
      options.baseConfig,
      options.corpus,
      workDirectory,
      chunkPolicy,
      firstCandidateCount,
    );
    const indexService = createRecallConversationService(
      indexConfig,
      createServiceDependencies(embeddings, rejectingReranker, options.dependencies?.loadTokenizer),
    );
    const indexStarted = performance.now();
    const indexed = await indexService.index({ optimize: true });
    const indexLatencyMilliseconds = performance.now() - indexStarted;
    if (indexed.indexSummary.failedSessions.length > 0) {
      const failures = indexed.indexSummary.failedSessions
        .map(({ sessionPath, error }) => `${sessionPath}: ${error}`)
        .join('; ');
      throw new Error(`Recall quality bounded index failed: ${failures}`);
    }
    if (indexed.indexSummary.scannedSessions !== options.corpus.sessionFiles.length) {
      throw new Error(
        `Recall quality bounded index scan mismatch: expected ${options.corpus.sessionFiles.length}, scanned ${indexed.indexSummary.scannedSessions}`,
      );
    }
    indexRuns.push({
      chunkPolicy: { ...chunkPolicy },
      totalChunks: indexed.totalChunks,
      indexLatencyMilliseconds,
      indexSummary: indexed.indexSummary,
    });

    for (const candidateCount of specification.candidateCounts) {
      const searchConfig = createChunkPolicyConfig(
        options.baseConfig,
        options.corpus,
        workDirectory,
        chunkPolicy,
        candidateCount,
      );
      const searchService = createRecallConversationService(
        searchConfig,
        createServiceDependencies(
          embeddings,
          rejectingReranker,
          options.dependencies?.loadTokenizer,
        ),
      );
      const warmupCase = specification.cases[0];
      if (!warmupCase) {
        throw new Error('Recall quality evaluation requires at least one case');
      }
      for (
        let warmupIndex = 0;
        warmupIndex < specification.warmupQueriesPerCombination;
        warmupIndex += 1
      ) {
        await searchService.search(warmupCase.query, RECALL_QUALITY_FULL_POOL_LIMIT, {
          mode: 'hybrid',
        });
        executedSearchRequests += 1;
      }

      const observations: RecallQualitySearchObservation[] = [];
      for (const evaluationCase of specification.cases) {
        const queryStarted = performance.now();
        const search = await searchService.search(
          evaluationCase.query,
          RECALL_QUALITY_FULL_POOL_LIMIT,
          { mode: 'hybrid' },
        );
        const queryLatencyMilliseconds = performance.now() - queryStarted;
        executedSearchRequests += 1;
        observations.push({
          evaluationCase,
          results: search.results,
          queryLatencyMilliseconds,
        });
      }
      configurations.push({
        chunkPolicy: { ...chunkPolicy },
        candidateCount,
        totalChunks: indexed.totalChunks,
        indexLatencyMilliseconds,
        measurement: measureRecallQuality(observations, specification.finalCounts),
      });
    }
  }

  if (rerankerRequests !== 0) {
    throw new Error(
      `Recall quality reranker request bound exceeded: executed ${rerankerRequests}, maximum 0`,
    );
  }
  if (executedSearchRequests > specification.bounds.maximumSearchRequests) {
    throw new Error(
      `Recall quality search bound exceeded after run: executed ${executedSearchRequests}, maximum ${specification.bounds.maximumSearchRequests}`,
    );
  }
  const completedAt = new Date();
  return {
    version: 3,
    rankingIdentity: {
      rankingMode: 'hybrid',
      rankFusionVersion: RECALL_RANK_FUSION_VERSION,
      reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
      activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
    },
    startedAt: startedAt.toISOString(),
    completedAt: completedAt.toISOString(),
    durationMilliseconds: performance.now() - started,
    specificationPath: options.corpus.specificationPath,
    specificationSha256: options.corpus.specificationSha256,
    corpusId: specification.corpus.id,
    indexRuns,
    configurations,
    selection: selectRecallQualityPolicy(configurations, specification.qualityGate),
    boundedWork: {
      sessionFiles: options.corpus.sessionFiles.length,
      evaluationCases: specification.cases.length,
      indexRuns: indexRuns.length,
      executedSearchRequests,
      rerankerRequests,
      chunkEmbeddingRequests: indexRuns.reduce(
        (total, indexRun) => total + indexRun.indexSummary.embeddingRequestCount,
        0,
      ),
      maximumCandidatesPerSearch: Math.max(...specification.candidateCounts) * 3,
    },
  };
}

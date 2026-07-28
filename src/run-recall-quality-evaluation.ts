import { execFile } from 'node:child_process';
import { mkdir, rm } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';

import { RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import {
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-ranked-lists.js';
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
  type RecallConversationSearchOptions,
} from './recall-conversation-service.js';
import {
  selectRecallQualityPolicy,
  type RecallQualityConfigurationMeasurement,
  type RecallQualityPolicySelection,
} from './select-recall-quality-policy.js';
import { RECALL_ACTIVE_BRANCH_PRIOR } from './rank-recall-search-results.js';
import {
  createLineageDigest,
  normalizeRecallProjectLineages,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  PROJECT_IDENTITY_POLICY_VERSION,
  PROJECT_LINEAGE_POLICY_VERSION,
  PROJECT_SCOPE_POLICY_VERSION,
  parseProjectIdentity,
  resolveProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const RECALL_QUALITY_FULL_POOL_LIMIT = 200;
const RECALL_QUALITY_WORK_DIRECTORY_NAME = 'recall-quality-evaluation';

/** Profile-aware inference and tokenizer boundaries for bounded quality evaluation. */
export interface RecallQualityEvaluationDependencies extends Pick<
  RecallConversationDependencies,
  'embeddingProfile' | 'embeddingProvider' | 'tokenizerIdentity' | 'loadTokenizer'
> {
  /** @deprecated Use embeddingProvider for profile-aware query and document semantics. */
  embeddings?: LocalEmbeddingClient;
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
  repositoryIdentityResolutions: number;
}

/** Project admission, lineage, ranking, and result counts bound to one evaluation result. */
export interface RecallQualityEvaluationIdentity {
  defaultScope: RecallSearchScope.PROJECT;
  projectScopePolicyVersion: number;
  projectIdentityPolicyVersion: number;
  projectIdentityMetadataSchemaVersion: number;
  lineagePolicyVersion: number;
  lineageDigest: string;
  rankingMode: 'hybrid';
  rankFusionVersion: number;
  reciprocalRankConstant: number;
  activeBranchPrior: number;
  candidateLimits: { dense: 8; lexical: 8; identifier: 8 };
  fusedPoolLimit: 24;
  rerankPoolLimit: 24;
  finalResultCount: 5;
}

/** Raw measured configurations, gate selection, and bounded-work evidence from one run. */
export interface RecallQualityEvaluationResult {
  version: 4;
  evaluationIdentity: RecallQualityEvaluationIdentity;
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
  const evaluationDirectory = dirname(corpus.specificationPath);
  if (!isPathInside(evaluationDirectory, resolvedWorkDirectory)) {
    throw new Error(
      `Recall quality work directory must stay inside evaluation data area: ${resolvedWorkDirectory} is outside ${evaluationDirectory}`,
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
    projectLineages: normalizeRecallProjectLineages(corpus.specification.projectLineages),
    chunkPolicy: {
      maxTokens: chunkPolicy.maxTokens,
      overlapTokens: chunkPolicy.overlapTokens,
    },
    searchCandidateLimits: {
      dense: candidateCount,
      lexical: candidateCount,
      identifier: candidateCount,
    },
    fusedPoolLimit: candidateCount * 3,
    rerankPoolLimit: candidateCount * 3,
  };
}

interface EvaluationProjectResolver {
  resolveProjectIdentity: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>;
  repositoryIdentityResolutions: number;
}

async function createEvaluationProjectResolver(
  corpus: LoadedRecallQualityCorpus,
  workDirectory: string,
): Promise<EvaluationProjectResolver> {
  const actualDirectories = new Map<string, string>();
  const resolvedFixtures = new Map<string, ResolvedProjectIdentity>();
  let repositoryIdentityResolutions = 0;
  for (const [index, fixture] of corpus.specification.projectIdentityFixtures.entries()) {
    if (fixture.identitySource === RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN) {
      resolvedFixtures.set(fixture.workingDirectory, {
        projectIdentity: parseProjectIdentity(fixture.projectIdentity),
        identitySource: fixture.identitySource,
      });
      continue;
    }
    const actualDirectory = join(workDirectory, 'project-fixtures', String(index));
    const worktreeSource = fixture.worktreeOf
      ? actualDirectories.get(fixture.worktreeOf)
      : undefined;
    if (fixture.worktreeOf && !worktreeSource) {
      throw new Error(
        `Recall quality repository fixture invalid: worktree source ${fixture.worktreeOf} must precede ${fixture.workingDirectory}`,
      );
    }
    if (worktreeSource) {
      await mkdir(dirname(actualDirectory), { recursive: true });
      await EXEC_FILE_ASYNC(
        'git',
        ['worktree', 'add', actualDirectory, '-b', `quality-fixture-${index}`],
        { cwd: worktreeSource },
      );
    } else {
      await mkdir(actualDirectory, { recursive: true });
      await EXEC_FILE_ASYNC('git', ['init'], { cwd: actualDirectory });
      if (fixture.origin) {
        await EXEC_FILE_ASYNC('git', ['remote', 'add', 'origin', fixture.origin], {
          cwd: actualDirectory,
        });
      }
      await EXEC_FILE_ASYNC(
        'git',
        [
          '-c',
          'user.name=Recall Quality',
          '-c',
          'user.email=recall-quality@example.test',
          'commit',
          '--allow-empty',
          '-m',
          'fixture',
        ],
        { cwd: actualDirectory },
      );
    }
    actualDirectories.set(fixture.workingDirectory, actualDirectory);
    const resolvedIdentity = await resolveProjectIdentity(actualDirectory);
    repositoryIdentityResolutions += 1;
    if (
      !resolvedIdentity ||
      resolvedIdentity.projectIdentity !== fixture.projectIdentity ||
      resolvedIdentity.identitySource !== fixture.identitySource
    ) {
      throw new Error(
        `Recall quality repository identity mismatch for ${fixture.workingDirectory}: expected ${fixture.projectIdentity} from ${fixture.identitySource}, received ${resolvedIdentity?.projectIdentity ?? 'unresolved'} from ${resolvedIdentity?.identitySource ?? 'none'}`,
      );
    }
    resolvedFixtures.set(fixture.workingDirectory, resolvedIdentity);
  }
  return {
    async resolveProjectIdentity(workingDirectory) {
      return resolvedFixtures.get(workingDirectory) ?? null;
    },
    repositoryIdentityResolutions,
  };
}

function createServiceDependencies(
  reranker: LocalRerankerClient,
  resolveProjectIdentity: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>,
  evaluationDependencies?: RecallQualityEvaluationDependencies,
  embeddings?: LocalEmbeddingClient,
): RecallConversationDependencies {
  return {
    ...(evaluationDependencies?.embeddingProfile
      ? { embeddingProfile: evaluationDependencies.embeddingProfile }
      : {}),
    ...(evaluationDependencies?.embeddingProvider
      ? { embeddingProvider: evaluationDependencies.embeddingProvider }
      : {}),
    ...(evaluationDependencies?.tokenizerIdentity
      ? { tokenizerIdentity: evaluationDependencies.tokenizerIdentity }
      : {}),
    ...(embeddings ? { embeddings } : {}),
    reranker,
    resolveProjectIdentity,
    ...(evaluationDependencies?.loadTokenizer
      ? { loadTokenizer: evaluationDependencies.loadTokenizer }
      : {}),
  };
}

function createEvaluationSearchOptions(
  evaluationCase: LoadedRecallQualityCorpus['specification']['cases'][number],
): RecallConversationSearchOptions {
  return {
    mode: 'hybrid',
    scope: evaluationCase.scope,
    ...(evaluationCase.invocationDirectory
      ? { invocationDirectory: evaluationCase.invocationDirectory }
      : {}),
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

  const embeddings = options.dependencies?.embeddingProvider
    ? undefined
    : createEvaluationEmbeddingClient(options.baseConfig, options.dependencies?.embeddings);
  const projectResolver = await createEvaluationProjectResolver(options.corpus, workDirectory);
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
      createServiceDependencies(
        rejectingReranker,
        projectResolver.resolveProjectIdentity,
        options.dependencies,
        embeddings,
      ),
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
          rejectingReranker,
          projectResolver.resolveProjectIdentity,
          options.dependencies,
          embeddings,
        ),
      );
      const warmupCases = specification.cases.filter(
        (evaluationCase, index, cases) =>
          cases.findIndex(({ scope }) => scope === evaluationCase.scope) === index,
      );
      if (warmupCases.length === 0) {
        throw new Error('Recall quality evaluation requires at least one case');
      }
      for (const warmupCase of warmupCases) {
        for (
          let warmupIndex = 0;
          warmupIndex < specification.warmupQueriesPerCombination;
          warmupIndex += 1
        ) {
          await searchService.search(
            warmupCase.query,
            RECALL_QUALITY_FULL_POOL_LIMIT,
            createEvaluationSearchOptions(warmupCase),
          );
          executedSearchRequests += 1;
        }
      }

      const observations: RecallQualitySearchObservation[] = [];
      for (const evaluationCase of specification.cases) {
        const queryStarted = performance.now();
        const search = await searchService.search(
          evaluationCase.query,
          RECALL_QUALITY_FULL_POOL_LIMIT,
          createEvaluationSearchOptions(evaluationCase),
        );
        const queryLatencyMilliseconds = performance.now() - queryStarted;
        executedSearchRequests += 1;
        let globalControlResults;
        if (evaluationCase.preLimitChannelProof) {
          const globalControl = await searchService.search(
            evaluationCase.query,
            RECALL_QUALITY_FULL_POOL_LIMIT,
            {
              mode: 'hybrid',
              scope: RecallSearchScope.GLOBAL,
              ...(evaluationCase.invocationDirectory
                ? { invocationDirectory: evaluationCase.invocationDirectory }
                : {}),
            },
          );
          globalControlResults = globalControl.results;
          executedSearchRequests += 1;
        }
        observations.push({
          evaluationCase,
          results: search.results,
          searchPolicy: {
            scope: search.searchPolicy.scope,
            invocationProjectIdentity: search.searchPolicy.invocationProjectIdentity,
          },
          ...(globalControlResults ? { globalControlResults } : {}),
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
  const chunkEmbeddingRequests = indexRuns.reduce(
    (total, indexRun) => total + indexRun.indexSummary.embeddingRequestCount,
    0,
  );
  if (chunkEmbeddingRequests > specification.bounds.maximumChunkEmbeddingRequests) {
    throw new Error(
      `Recall quality chunk-embedding request bound exceeded: executed ${chunkEmbeddingRequests}, maximum ${specification.bounds.maximumChunkEmbeddingRequests}`,
    );
  }
  const completedAt = new Date();
  return {
    version: 4,
    evaluationIdentity: {
      defaultScope: RecallSearchScope.PROJECT,
      projectScopePolicyVersion: PROJECT_SCOPE_POLICY_VERSION,
      projectIdentityPolicyVersion: PROJECT_IDENTITY_POLICY_VERSION,
      projectIdentityMetadataSchemaVersion: PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
      lineagePolicyVersion: PROJECT_LINEAGE_POLICY_VERSION,
      lineageDigest: createLineageDigest(
        normalizeRecallProjectLineages(specification.projectLineages),
      ),
      rankingMode: 'hybrid',
      rankFusionVersion: RECALL_RANK_FUSION_VERSION,
      reciprocalRankConstant: RECALL_RRF_RANK_CONSTANT,
      activeBranchPrior: RECALL_ACTIVE_BRANCH_PRIOR,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      fusedPoolLimit: 24,
      rerankPoolLimit: 24,
      finalResultCount: 5,
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
      chunkEmbeddingRequests,
      maximumCandidatesPerSearch: Math.max(...specification.candidateCounts) * 3,
      repositoryIdentityResolutions: projectResolver.repositoryIdentityResolutions,
    },
  };
}

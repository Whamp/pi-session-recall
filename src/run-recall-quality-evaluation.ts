import { execFile } from 'node:child_process';
import { copyFile, mkdir, readdir, rm, stat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { promisify } from 'node:util';

import { assertRecallTestDataRoot } from './assert-recall-test-data-root.js';
import {
  PROJECT_SCOPE_POLICY_VERSION,
  RecallProjectIdentitySource,
  RecallSearchScope,
} from './enums.js';
import {
  RECALL_RANK_FUSION_VERSION,
  RECALL_RRF_RANK_CONSTANT,
} from './fuse-recall-ranked-lists.js';
import type { ConversationIndexSummary } from './incremental-session-indexer.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
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
import type {
  RecallEmbeddingProvider,
  RecallIdentifiedQueryPlanningProvider,
  RecallIdentifiedRerankingProvider,
  RecallQueryPlanningExecutionIdentity,
  RecallRerankingExecutionIdentity,
} from './recall-inference-capabilities.js';
import {
  selectRecallQualityPolicy,
  type RecallQualityConfigurationMeasurement,
  type RecallQualityPolicySelection,
} from './select-recall-quality-policy.js';
import { RECALL_ACTIVE_BRANCH_PRIOR } from './rank-recall-search-results.js';
import {
  createOctenEmbeddingModelProfile,
  type RecallQueryPlanningModelProfile,
  type RecallRerankingModelProfile,
} from './recall-model-profiles.js';
import { RECALL_GENERATION_FORMAT_VERSION } from './recall-generation-manifest.js';
import {
  createRecallGenerationComponentPaths,
  readRecallGenerationStoreRecordMembership,
  RECALL_GENERATION_STORE_FORMAT_VERSION,
  type RecallGenerationStoreCounts,
} from './recall-generation-stores.js';
import { RECALL_GENERATION_VALIDATION_RECEIPT_VERSION } from './recall-generation-validation-receipt.js';
import { INCREMENTAL_RECALL_ELIGIBILITY_POLICY_VERSION } from './reduce-recall-eligibility.js';
import {
  createLineageDigest,
  normalizeRecallProjectLineages,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
  PROJECT_IDENTITY_POLICY_VERSION,
  PROJECT_LINEAGE_POLICY_VERSION,
  parseProjectIdentity,
  resolveProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const RECALL_QUALITY_FULL_POOL_LIMIT = 200;
const RECALL_QUALITY_WORK_DIRECTORY_NAME = 'recall-quality-evaluation';
const RECALL_QUALITY_GENERATION_ID = 'generation_quality_active';

/** Profile-aware inference and tokenizer boundaries for bounded quality evaluation. */
export type RecallQualityEvaluationDependencies = Pick<
  RecallConversationDependencies,
  'embeddingProfile' | 'embeddingProvider' | 'tokenizerIdentity' | 'loadTokenizer'
>;

/** Live planner and reranker boundaries for an optional committed-corpus query-planned lane. */
export interface RecallQualityQueryPlannedDependencies {
  queryPlanningProfile: RecallQueryPlanningModelProfile;
  queryPlanner: RecallIdentifiedQueryPlanningProvider;
  rerankingProfile: RecallRerankingModelProfile;
  reranker: RecallIdentifiedRerankingProvider;
}

/** Inputs for one bounded evaluation run over a checksum-fixed corpus. */
export interface RunRecallQualityEvaluationOptions {
  corpus: LoadedRecallQualityCorpus;
  baseConfig: RecallConversationConfig;
  workDirectory: string;
  dependencies?: RecallQualityEvaluationDependencies;
  queryPlannedDependencies?: RecallQualityQueryPlannedDependencies;
}

/** Validated target generation built once for one required chunk policy. */
export interface RecallQualityIndexRun {
  chunkPolicy: RecallQualityChunkPolicy;
  generationId: string;
  manifestFingerprint: string;
  startingSnapshotFingerprint: string;
  storeCounts: RecallGenerationStoreCounts;
  totalChunks: number;
  generationSizeBytes: number;
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

/** Coherent generation and incremental eligibility contracts measured by quality evidence. */
export interface RecallQualityStorageIdentity {
  generationFormatVersion: number;
  generationStoreFormatVersion: number;
  validationReceiptVersion: number;
  incrementalEligibilityPolicyVersion: number;
}

/** Live query-planned quality measured independently over the committed corpus. */
export interface RecallQualityQueryPlannedResult {
  executionIdentity: {
    queryPlanning: Readonly<RecallQueryPlanningExecutionIdentity>;
    reranking: Readonly<RecallRerankingExecutionIdentity>;
  };
  configurations: RecallQualityConfigurationMeasurement[];
  selection: RecallQualityPolicySelection;
  boundedWork: {
    executedSearchRequests: number;
    plannerRequests: number;
    rerankerRequests: number;
  };
}

/** Raw measured configurations, gate selection, and bounded-work evidence from one run. */
export interface RecallQualityEvaluationResult {
  version: 6;
  storageIdentity: RecallQualityStorageIdentity;
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
  queryPlanned?: RecallQualityQueryPlannedResult;
}

function isPathInside(parentPath: string, childPath: string): boolean {
  const pathFromParent = relative(parentPath, childPath);
  return pathFromParent === '' || (!pathFromParent.startsWith('..') && !isAbsolute(pathFromParent));
}

async function measureRecallGenerationOnDiskBytes(generationDirectory: string): Promise<number> {
  let totalBytes = 0;
  for (const entry of await readdir(generationDirectory, { withFileTypes: true })) {
    const entryPath = join(generationDirectory, entry.name);
    if (entry.isDirectory()) {
      totalBytes += await measureRecallGenerationOnDiskBytes(entryPath);
    } else if (entry.isFile()) {
      totalBytes += (await stat(entryPath)).size;
    }
  }
  return totalBytes;
}

async function assertSafeRecallQualityPaths(
  workDirectory: string,
  corpus: LoadedRecallQualityCorpus,
  baseConfig: RecallConversationConfig,
): Promise<string> {
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
  return assertRecallTestDataRoot({
    testDataRoot: resolvedWorkDirectory,
    repositoryRoot: dirname(evaluationDirectory),
    configuredProtectedPaths: [
      corpus.sessionDirectory,
      baseConfig.sessionsDirectory,
      baseConfig.dataDirectory,
      baseConfig.databasePath,
      baseConfig.projectionDatabasePath,
      baseConfig.statePath,
      baseConfig.manifestPath,
      baseConfig.tokenizerCacheDirectory,
      baseConfig.lockPath,
      baseConfig.diagnosticLogPath,
      baseConfig.retainedDiagnosticLogPath,
      baseConfig.markerSpoolDirectory,
      baseConfig.markerQuarantineDirectory,
      baseConfig.markerControlDirectory,
      baseConfig.workerOwnershipLockPath,
      baseConfig.generationRootDirectory,
      baseConfig.activeGenerationPointerPath,
      baseConfig.generationRegistryPath,
      baseConfig.backlogSummaryPath,
      baseConfig.incrementalDiagnosticLogPath,
    ],
  });
}

function createEvaluationEmbeddingProvider(
  config: RecallConversationConfig,
): RecallEmbeddingProvider {
  return createLlamaCppHttpEmbeddingProvider(
    createOctenEmbeddingModelProfile({
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: config.embeddingPooling,
    }),
    {
      baseUrl: config.embeddingBaseUrl,
      batchSize: config.embeddingBatchSize,
    },
  );
}

function createChunkPolicyConfig(
  baseConfig: RecallConversationConfig,
  corpus: LoadedRecallQualityCorpus,
  sessionsDirectory: string,
  workDirectory: string,
  chunkPolicy: RecallQualityChunkPolicy,
  candidateCount: number,
): RecallConversationConfig {
  const policyDirectory = join(workDirectory, chunkPolicy.id);
  const generationRootDirectory = join(policyDirectory, 'generations');
  const generationDirectory = join(generationRootDirectory, RECALL_QUALITY_GENERATION_ID);
  return {
    ...baseConfig,
    sessionsDirectory,
    databasePath: join(generationDirectory, 'zvec'),
    projectionDatabasePath: join(generationDirectory, 'session-projections'),
    statePath: join(generationDirectory, 'index-state.json'),
    manifestPath: join(generationDirectory, 'index-manifest.json'),
    lockPath: join(policyDirectory, 'operation.lock'),
    generationRootDirectory,
    activeGenerationPointerPath: join(policyDirectory, 'active-generation.json'),
    generationRegistryPath: join(policyDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(policyDirectory, 'backlog-summary.json'),
    markerSpoolDirectory: join(policyDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(policyDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(policyDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(policyDirectory, 'incremental-worker.lock'),
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

async function createDisposableRecallQualitySourceSnapshot(
  corpus: LoadedRecallQualityCorpus,
  workDirectory: string,
): Promise<string> {
  const snapshotDirectory = join(workDirectory, 'source-snapshot');
  await mkdir(snapshotDirectory, { recursive: true });
  await Promise.all(
    corpus.sessionFiles.map(({ path }) => copyFile(path, join(snapshotDirectory, basename(path)))),
  );
  return snapshotDirectory;
}

interface MeasuredRecallQualityDependencies {
  dependencies?: RecallQualityEvaluationDependencies;
  readDocumentEmbeddingRequestCount(): number;
}

function createMeasuredRecallQualityDependencies(
  embeddingProvider: RecallEmbeddingProvider,
  dependencies?: RecallQualityEvaluationDependencies,
): MeasuredRecallQualityDependencies {
  let documentEmbeddingRequestCount = 0;
  return {
    dependencies: {
      ...dependencies,
      embeddingProvider: {
        embedQuery: (query, signal) => embeddingProvider.embedQuery(query, signal),
        async embedDocuments(documents, signal) {
          documentEmbeddingRequestCount += 1;
          return embeddingProvider.embedDocuments(documents, signal);
        },
      },
    },
    readDocumentEmbeddingRequestCount: () => documentEmbeddingRequestCount,
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
  reranker: RecallConversationDependencies['reranker'],
  resolveProjectIdentity: (workingDirectory: string) => Promise<ResolvedProjectIdentity | null>,
  evaluationDependencies?: RecallQualityEvaluationDependencies,
  queryPlannedDependencies?: RecallQualityQueryPlannedDependencies,
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
    rerankingProfile: queryPlannedDependencies?.rerankingProfile ?? null,
    reranker: reranker ?? null,
    rerankerExecutionIdentity: queryPlannedDependencies?.reranker.executionIdentity ?? null,
    ...(queryPlannedDependencies
      ? {
          queryPlanningProfile: queryPlannedDependencies.queryPlanningProfile,
          queryPlanner: queryPlannedDependencies.queryPlanner,
        }
      : {}),
    resolveProjectIdentity,
    ...(evaluationDependencies?.loadTokenizer
      ? { loadTokenizer: evaluationDependencies.loadTokenizer }
      : {}),
  };
}

function createEvaluationSearchOptions(
  evaluationCase: LoadedRecallQualityCorpus['specification']['cases'][number],
  mode: 'hybrid' | 'query-planned' = 'hybrid',
): RecallConversationSearchOptions {
  return {
    mode,
    scope: evaluationCase.scope,
    ...(evaluationCase.invocationDirectory
      ? { invocationDirectory: evaluationCase.invocationDirectory }
      : {}),
  };
}

function createHybridPreLimitControlSearchOptions(
  evaluationCase: LoadedRecallQualityCorpus['specification']['cases'][number],
): RecallConversationSearchOptions {
  return {
    ...createEvaluationSearchOptions(evaluationCase),
    scope: RecallSearchScope.GLOBAL,
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
  const workDirectory = await assertSafeRecallQualityPaths(
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
  const sourceSnapshotDirectory = await createDisposableRecallQualitySourceSnapshot(
    options.corpus,
    workDirectory,
  );

  const embeddingProvider =
    options.dependencies?.embeddingProvider ??
    createEvaluationEmbeddingProvider(options.baseConfig);
  const projectResolver = await createEvaluationProjectResolver(options.corpus, workDirectory);
  const indexRuns: RecallQualityIndexRun[] = [];
  const configurations: RecallQualityConfigurationMeasurement[] = [];
  const queryPlannedConfigurations: RecallQualityConfigurationMeasurement[] = [];
  let executedSearchRequests = 0;
  let queryPlannedExecutedSearchRequests = 0;
  let queryPlannedPlannerRequests = 0;
  let queryPlannedRerankerRequests = 0;
  const configuredQueryPlannedDependencies = options.queryPlannedDependencies;
  const measuredQueryPlannedDependencies: RecallQualityQueryPlannedDependencies | undefined =
    configuredQueryPlannedDependencies
      ? {
          ...configuredQueryPlannedDependencies,
          queryPlanner: {
            get executionIdentity() {
              return configuredQueryPlannedDependencies.queryPlanner.executionIdentity;
            },
            async planRecallQuery(request, signal) {
              queryPlannedPlannerRequests += 1;
              return configuredQueryPlannedDependencies.queryPlanner.planRecallQuery(
                request,
                signal,
              );
            },
          },
          reranker: {
            get executionIdentity() {
              return configuredQueryPlannedDependencies.reranker.executionIdentity;
            },
            async rerankDocuments(query, documents, signal) {
              queryPlannedRerankerRequests += 1;
              return configuredQueryPlannedDependencies.reranker.rerankDocuments(
                query,
                documents,
                signal,
              );
            },
          },
        }
      : undefined;

  for (const chunkPolicy of specification.chunkPolicies) {
    const firstCandidateCount = specification.candidateCounts[0];
    if (firstCandidateCount === undefined) {
      throw new Error('Recall quality evaluation requires at least one candidate count');
    }
    const indexConfig = createChunkPolicyConfig(
      options.baseConfig,
      options.corpus,
      sourceSnapshotDirectory,
      workDirectory,
      chunkPolicy,
      firstCandidateCount,
    );
    const measuredDependencies = createMeasuredRecallQualityDependencies(
      embeddingProvider,
      options.dependencies,
    );
    const indexService = createRecallConversationService(
      indexConfig,
      createServiceDependencies(
        null,
        projectResolver.resolveProjectIdentity,
        measuredDependencies.dependencies,
      ),
    );
    const physicalSessionPaths = options.corpus.sessionFiles.map(({ fileName }) =>
      join(sourceSnapshotDirectory, fileName),
    );
    const indexStarted = performance.now();
    const opened = await indexService.createRecallGenerationFromPhysicalSources({
      generationId: RECALL_QUALITY_GENERATION_ID,
      physicalSessionPaths,
    });
    await indexService.activateValidatedRecallGeneration(RECALL_QUALITY_GENERATION_ID);
    const indexLatencyMilliseconds = performance.now() - indexStarted;
    const recordMembership = await readRecallGenerationStoreRecordMembership(
      createRecallGenerationComponentPaths(opened.generationDirectory),
    );
    const totalChunks = recordMembership.lexicalSource.filter((recordId) =>
      recordId.startsWith('occurrence_'),
    ).length;
    const generationSizeBytes = await measureRecallGenerationOnDiskBytes(
      opened.generationDirectory,
    );
    const indexSummary: ConversationIndexSummary = {
      scannedSessions: physicalSessionPaths.length,
      indexedSessions: physicalSessionPaths.length,
      removedSessions: 0,
      newlyEmbeddedChunks: opened.storeCounts.dense,
      embeddingRequestCount: measuredDependencies.readDocumentEmbeddingRequestCount(),
      deletedChunks: 0,
      failedSessions: [],
    };
    indexRuns.push({
      chunkPolicy: { ...chunkPolicy },
      generationId: opened.generationId,
      manifestFingerprint: opened.manifestFingerprint,
      startingSnapshotFingerprint: opened.startingSnapshotFingerprint,
      storeCounts: opened.storeCounts,
      totalChunks,
      generationSizeBytes,
      indexLatencyMilliseconds,
      indexSummary,
    });
    await rm(sourceSnapshotDirectory, { recursive: true, force: true });

    for (const candidateCount of specification.candidateCounts) {
      const searchConfig = createChunkPolicyConfig(
        options.baseConfig,
        options.corpus,
        sourceSnapshotDirectory,
        workDirectory,
        chunkPolicy,
        candidateCount,
      );
      const searchService = createRecallConversationService(
        searchConfig,
        createServiceDependencies(
          measuredQueryPlannedDependencies?.reranker ?? null,
          projectResolver.resolveProjectIdentity,
          { ...options.dependencies, embeddingProvider },
          measuredQueryPlannedDependencies,
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
            createHybridPreLimitControlSearchOptions(evaluationCase),
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
        totalChunks,
        indexLatencyMilliseconds,
        measurement: measureRecallQuality(observations, specification.finalCounts),
      });

      if (measuredQueryPlannedDependencies) {
        const queryPlannedObservations: RecallQualitySearchObservation[] = [];
        for (const evaluationCase of specification.cases) {
          const queryStarted = performance.now();
          const search = await searchService.search(
            evaluationCase.query,
            RECALL_QUALITY_FULL_POOL_LIMIT,
            createEvaluationSearchOptions(evaluationCase, 'query-planned'),
          );
          const queryLatencyMilliseconds = performance.now() - queryStarted;
          queryPlannedExecutedSearchRequests += 1;
          let globalControlResults;
          if (evaluationCase.preLimitChannelProof) {
            const globalControl = await searchService.search(
              evaluationCase.query,
              RECALL_QUALITY_FULL_POOL_LIMIT,
              createHybridPreLimitControlSearchOptions(evaluationCase),
            );
            globalControlResults = globalControl.results;
            queryPlannedExecutedSearchRequests += 1;
          }
          queryPlannedObservations.push({
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
        queryPlannedConfigurations.push({
          chunkPolicy: { ...chunkPolicy },
          candidateCount,
          totalChunks,
          indexLatencyMilliseconds,
          measurement: measureRecallQuality(queryPlannedObservations, specification.finalCounts),
        });
      }
    }
  }

  if (options.queryPlannedDependencies) {
    if (queryPlannedPlannerRequests < 1 || queryPlannedRerankerRequests < 1) {
      throw new Error(
        'Recall quality query-planned coverage failed: live planner and reranker must both execute',
      );
    }
    if (queryPlannedExecutedSearchRequests > specification.bounds.maximumSearchRequests) {
      throw new Error(
        `Recall quality query-planned search bound exceeded after run: executed ${queryPlannedExecutedSearchRequests}, maximum ${specification.bounds.maximumSearchRequests}`,
      );
    }
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
    version: 6,
    storageIdentity: {
      generationFormatVersion: RECALL_GENERATION_FORMAT_VERSION,
      generationStoreFormatVersion: RECALL_GENERATION_STORE_FORMAT_VERSION,
      validationReceiptVersion: RECALL_GENERATION_VALIDATION_RECEIPT_VERSION,
      incrementalEligibilityPolicyVersion: INCREMENTAL_RECALL_ELIGIBILITY_POLICY_VERSION,
    },
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
      rerankerRequests: 0,
      chunkEmbeddingRequests,
      maximumCandidatesPerSearch: Math.max(...specification.candidateCounts) * 3,
      repositoryIdentityResolutions: projectResolver.repositoryIdentityResolutions,
    },
    ...(options.queryPlannedDependencies
      ? {
          queryPlanned: {
            executionIdentity: {
              queryPlanning: options.queryPlannedDependencies.queryPlanner.executionIdentity,
              reranking: options.queryPlannedDependencies.reranker.executionIdentity,
            },
            configurations: queryPlannedConfigurations,
            selection: selectRecallQualityPolicy(
              queryPlannedConfigurations,
              specification.qualityGate,
            ),
            boundedWork: {
              executedSearchRequests: queryPlannedExecutedSearchRequests,
              plannerRequests: queryPlannedPlannerRequests,
              rerankerRequests: queryPlannedRerankerRequests,
            },
          },
        }
      : {}),
  };
}

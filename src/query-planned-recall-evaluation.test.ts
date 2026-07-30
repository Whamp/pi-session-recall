import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { chmod, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, relative } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallControlKind,
  RecallEvidenceRelation,
  RecallInferenceBackend,
  RecallSearchScope,
} from './enums.js';
import {
  createLiveQueryPlannedEvaluationConfigurationIdentity,
  createPublishableLiveQueryPlannedProfileAcceptance,
  createPublishableQueryPlannedRecallEvaluationEvidence,
  createPublishableQueryPlannedRecallPlanIdentity,
  formatPublishableLiveQueryPlannedProfileAcceptanceReport,
  formatPublishableQueryPlannedRecallEvaluationReport,
  loadPrivateQueryPlannedRecallPlans,
  runLiveQueryPlannedProfileEvaluation,
  runPrivateQueryPlannedRecallEvaluation,
  selectQueryPlannedSourceProvenance,
  type LiveQueryPlannedProfileEvaluationResult,
  type RunLiveQueryPlannedProfileEvaluationOptions,
} from './query-planned-recall-evaluation.js';
import {
  createPublishableQueryPlannedRecallControls,
  loadPrivateQueryPlannedRecallCorpus,
  runPrivateQueryPlannedRecallBaseline,
} from './query-planned-recall-baseline.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { DEFAULT_RECALL_CHUNK_POLICY } from './recall-index-manifest.js';
import {
  createRecallQueryPlanningExecutionIdentity,
  createRecallRerankingExecutionIdentity,
} from './recall-inference-capabilities.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createSha256(content: string): string {
  return createHash('sha256').update(content).digest('hex');
}

void test('per-source provenance uses each ranked match or its own candidate-admission match', () => {
  assert.deepEqual(
    selectQueryPlannedSourceProvenance(
      [{ provenancePassed: true }, null],
      [{ provenancePassed: false }, { provenancePassed: true }],
    ),
    [
      { selectedFrom: 'ranked_result', passed: true },
      { selectedFrom: 'candidate_admission', passed: true },
    ],
  );
  assert.equal(
    selectQueryPlannedSourceProvenance(
      [{ provenancePassed: true }, null],
      [{ provenancePassed: true }, { provenancePassed: false }],
    ).every(({ passed }) => passed),
    false,
  );
  assert.equal(
    selectQueryPlannedSourceProvenance(
      [{ provenancePassed: false }, null],
      [{ provenancePassed: true }, { provenancePassed: true }],
    ).every(({ passed }) => passed),
    false,
  );
});

async function readEvaluationContainmentFileTree(
  rootDirectory: string,
): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  async function visit(directory: string): Promise<void> {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const entryPath = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(entryPath);
      } else if (entry.isFile()) {
        files.set(relative(rootDirectory, entryPath), (await readFile(entryPath)).toString('hex'));
      }
    }
  }
  await visit(rootDirectory);
  return files;
}

function createEvaluationContainmentTreeSha256(files: ReadonlyMap<string, string>): string {
  const hash = createHash('sha256');
  for (const [path, content] of [...files.entries()].toSorted(([left], [right]) =>
    left.localeCompare(right),
  )) {
    hash.update(path);
    hash.update('\0');
    hash.update(content);
    hash.update('\0');
  }
  return hash.digest('hex');
}

void test('committed query-planned recall evidence is deterministic and records a passing source-admission gate', async () => {
  const projectDirectory = dirname(dirname(fileURLToPath(import.meta.url)));
  const evidenceContent = await readFile(
    join(projectDirectory, 'docs', 'evaluation', 'query-planned-recall-quality.json'),
    'utf8',
  );
  const reportContent = await readFile(
    join(projectDirectory, 'docs', 'evaluation', 'query-planned-recall-quality.md'),
    'utf8',
  );
  const evidence: unknown = JSON.parse(evidenceContent);
  if (!isUnknownRecord(evidence)) {
    throw new Error('Committed query-planned recall evidence invalid: expected an object');
  }
  const evaluation = Reflect.get(evidence, 'evaluation');
  if (!isUnknownRecord(evaluation)) {
    throw new Error('Committed query-planned recall evidence invalid: expected evaluation data');
  }

  assert.equal(
    createSha256(evidenceContent),
    'a377d2197d142d048281f31d92f99115e1b4a680a68eed493969e4822658ad2d',
  );
  assert.equal(
    createSha256(reportContent),
    '49440877ab6f1a93694ce5ef0f600ecc000694ffd3cccf41c31f9dc0eb6c79d5',
  );
  assert.equal(Reflect.get(evaluation, 'executedSearchRequests'), 32);
  assert.deepEqual(Reflect.get(evaluation, 'contributionCounts'), {
    newCandidateAdmission: 4,
    rankingOnlyPromotion: 1,
    preservedExistingSuccess: 1,
    noImprovement: 2,
  });
  assert.match(reportContent, /New candidate admission beyond both controls: 4/u);
  assert.match(reportContent, /Ranking-only promotion of an already admitted source: 1/u);
});

void test('fixed private plans prove new source admission through deterministic public searches without publishing private text', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'query-planned-recall-evaluation-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const privateDirectory = join(directory, '.recall-data', 'query-planned-recall');
  const snapshotDirectory = join(privateDirectory, 'snapshots');
  await mkdir(snapshotDirectory, { recursive: true, mode: 0o700 });
  await chmod(join(directory, '.recall-data'), 0o700);
  await chmod(privateDirectory, 0o700);
  await chmod(snapshotDirectory, 0o700);

  const distractorRecords = Array.from({ length: 40 }, (_, index) => ({
    type: 'message',
    id: `distractor-${String(index).padStart(2, '0')}`,
    parentId: index === 0 ? null : `distractor-${String(index - 1).padStart(2, '0')}`,
    timestamp: `2026-08-01T10:${String(index + 1).padStart(2, '0')}:00.000Z`,
    message: {
      role: 'assistant',
      content: `Private symptom wording distractor ${index}.`,
    },
  }));
  const snapshotContent = `${[
    {
      type: 'session',
      version: 3,
      id: 'private-session',
      timestamp: '2026-08-01T10:00:00.000Z',
      cwd: '/private/project',
    },
    ...distractorRecords,
    {
      type: 'message',
      id: 'expected-entry',
      parentId: 'distractor-39',
      timestamp: '2026-08-01T11:00:00.000Z',
      message: { role: 'assistant', content: 'Private mechanism phrase.' },
    },
  ]
    .map((record) => JSON.stringify(record))
    .join('\n')}\n`;
  const snapshotPath = join(snapshotDirectory, 'snapshot-001.jsonl');
  await writeFile(snapshotPath, snapshotContent, { mode: 0o600 });
  const query = 'private symptom wording';
  const privateCase = {
    id: 'case-001',
    category: QueryPlannedRecallCaseCategory.SYMPTOM_TO_MECHANISM,
    controlKind: QueryPlannedRecallControlKind.DIFFICULT_CASE,
    query,
    querySha256: createSha256(query),
    scope: RecallSearchScope.GLOBAL,
    expectedSources: [
      {
        snapshotId: 'snapshot-001',
        entryId: 'expected-entry',
        requiredText: ['Private mechanism phrase'],
        expectedSessionOrigin: '/private/project',
        expectedEvidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
        expectedEvidenceKind: 'conversation',
        expectedBranch: 'active',
      },
    ],
    relevantDistractors: [],
    plannedRetrievalLists: { lexical: 1, semantic: 1, hypotheticalAnswer: 0 },
    retrievalWorkMatchedCandidateLimits: { dense: 34, lexical: 33, identifier: 33 },
  };
  const manifestContent = `${JSON.stringify(
    {
      version: 1,
      corpus: {
        id: 'private-query-planned-recall-v1',
        snapshotDirectory: 'snapshots',
        snapshots: [
          {
            id: 'snapshot-001',
            fileName: 'snapshot-001.jsonl',
            sha256: createSha256(snapshotContent),
          },
        ],
      },
      policy: {
        normalCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
        plannedCandidateLimit: 20,
        finalResultLimit: 5,
      },
      cases: [privateCase],
    },
    null,
    2,
  )}\n`;
  const manifestPath = join(privateDirectory, 'manifest.json');
  await writeFile(manifestPath, manifestContent, { mode: 0o600 });
  const corpus = await loadPrivateQueryPlannedRecallCorpus(manifestPath);

  const lexicalPlanQuery = 'Private mechanism';
  const semanticPlanQuery = 'mechanism behind the symptom';
  const plansContent = `${JSON.stringify(
    {
      version: 1,
      corpusId: corpus.manifest.corpus.id,
      privateManifestSha256: corpus.manifestSha256,
      cases: [
        {
          caseId: 'case-001',
          queries: [
            { type: 'lex', query: lexicalPlanQuery },
            { type: 'vec', query: semanticPlanQuery },
          ],
        },
      ],
    },
    null,
    2,
  )}\n`;
  const plansPath = join(privateDirectory, 'plans.json');
  await writeFile(plansPath, plansContent, { mode: 0o600 });

  const plans = await loadPrivateQueryPlannedRecallPlans(plansPath, corpus);
  const identity = createPublishableQueryPlannedRecallPlanIdentity(plans);
  const published = JSON.stringify(identity);

  assert.equal(plans.sha256, createSha256(plansContent));
  assert.equal(identity.source, 'agent');
  assert.equal(identity.planSha256, createSha256(plansContent));
  assert.deepEqual(identity.cases[0], {
    caseId: 'case-001',
    plannedQueries: [
      { type: 'lex', querySha256: createSha256(lexicalPlanQuery) },
      { type: 'vec', querySha256: createSha256(semanticPlanQuery) },
    ],
  });
  assert.equal(published.includes(lexicalPlanQuery), false);
  assert.equal(published.includes(semanticPlanQuery), false);
  const immutablePrivateInputBytes = new Map<string, string>();
  for (const path of [snapshotPath, manifestPath, plansPath]) {
    immutablePrivateInputBytes.set(path, (await readFile(path)).toString('hex'));
  }

  const productionSessionsDirectory = join(directory, 'production-sessions');
  const productionDataDirectory = join(directory, 'production-recall');
  await mkdir(productionSessionsDirectory);
  await writeFile(
    join(productionSessionsDirectory, 'production.jsonl'),
    `${[
      {
        type: 'session',
        version: 3,
        id: 'production-session',
        timestamp: '2026-08-01T09:00:00.000Z',
        cwd: '/production/project',
      },
      {
        type: 'message',
        id: 'production-entry',
        parentId: null,
        timestamp: '2026-08-01T09:01:00.000Z',
        message: { role: 'assistant', content: 'Production-only copper albatross evidence.' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  const baseConfig = await loadRecallConversationConfig({
    homeDirectory: directory,
    environment: {
      PI_RECALL_DATA_DIRECTORY: productionDataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: productionSessionsDirectory,
      PI_RECALL_EMBEDDING_BASE_URL: 'deterministic://query-planned-evaluation',
      PI_RECALL_EMBEDDING_MODEL: 'deterministic-token-hash-v1',
      PI_RECALL_EMBEDDING_SERVED_MODEL_ID: 'deterministic-token-hash-v1',
      PI_RECALL_EMBEDDING_ARTIFACT: 'none',
      PI_RECALL_EMBEDDING_QUANTIZATION: 'none',
      PI_RECALL_EMBEDDING_POOLING: 'token-hash',
      PI_RECALL_EMBEDDING_DIMENSIONS: '64',
      PI_RECALL_EMBEDDING_BATCH_SIZE: '64',
      PI_RECALL_RERANKER_BASE_URL: 'deterministic://query-planned-evaluation',
      PI_RECALL_RERANKER_MODEL: 'deterministic-source-admission-v1',
    },
  });
  const loadTokenizer = async () => ({
    encodeConversationText(text: string) {
      return { ids: Array.from(text.split(/\\s+/u).filter(Boolean).keys()) };
    },
  });
  const embeddingProvider = {
    async embedQuery() {
      return [1, ...Array.from({ length: 63 }, () => 0)];
    },
    async embedDocuments(documents: readonly string[]) {
      return documents.map(() => [1, ...Array.from({ length: 63 }, () => 0)]);
    },
  };
  const conformanceFixture = {
    query: 'source provenance',
    documents: ['relevant', 'irrelevant'],
    expectedScores: [0.9, 0.1],
    maximumAbsoluteDifference: 0.001,
  };
  const evaluationConfigurationIdentity = createLiveQueryPlannedEvaluationConfigurationIdentity(
    baseConfig,
    conformanceFixture,
  );
  assert.deepEqual(
    evaluationConfigurationIdentity,
    createLiveQueryPlannedEvaluationConfigurationIdentity(
      { ...baseConfig, projectLineages: normalizeRecallProjectLineages({}) },
      { ...conformanceFixture },
    ),
  );
  assert.deepEqual(
    evaluationConfigurationIdentity,
    createLiveQueryPlannedEvaluationConfigurationIdentity(
      { ...baseConfig, chunkPolicy: DEFAULT_RECALL_CHUNK_POLICY },
      conformanceFixture,
    ),
  );
  assert.notDeepEqual(
    evaluationConfigurationIdentity,
    createLiveQueryPlannedEvaluationConfigurationIdentity(
      { ...baseConfig, embeddingBatchSize: 32 },
      conformanceFixture,
    ),
  );
  assert.notDeepEqual(
    evaluationConfigurationIdentity,
    createLiveQueryPlannedEvaluationConfigurationIdentity(baseConfig, {
      ...conformanceFixture,
      maximumAbsoluteDifference: 0.01,
    }),
  );

  const productionService = createRecallConversationService(baseConfig, {
    embeddingProvider,
    loadTokenizer,
    rerankingProfile: null,
    reranker: null,
  });
  await productionService.index({ rebuild: true });
  assert.equal((await productionService.readIndexGenerationStatus()).active?.kind, 'managed');
  let productionFilesBeforeEvaluation =
    await readEvaluationContainmentFileTree(productionDataDirectory);
  const assertProductionRemainsIsolated = async (): Promise<void> => {
    const productionFilesAfterEvaluation =
      await readEvaluationContainmentFileTree(productionDataDirectory);
    assert.equal(
      createEvaluationContainmentTreeSha256(productionFilesAfterEvaluation),
      createEvaluationContainmentTreeSha256(productionFilesBeforeEvaluation),
    );
    for (const [path, expectedBytes] of immutablePrivateInputBytes) {
      assert.equal((await readFile(path)).toString('hex'), expectedBytes);
    }
    const productionSearch = await productionService.search('Private mechanism phrase', 20, {
      scope: RecallSearchScope.GLOBAL,
    });
    assert.equal(
      productionSearch.results.some(({ content }) => content.includes('Private mechanism phrase')),
      false,
    );
    assert.equal(
      productionSearch.results
        .flatMap((result) => [result, ...result.duplicateOccurrences])
        .some(({ sessionPath }) => sessionPath === snapshotPath),
      false,
    );
    productionFilesBeforeEvaluation =
      await readEvaluationContainmentFileTree(productionDataDirectory);
  };

  const baselineWorkDirectory = join(privateDirectory, 'baseline-work');
  await runPrivateQueryPlannedRecallBaseline({
    corpus,
    baseConfig,
    workDirectory: baselineWorkDirectory,
    dependencies: { embeddingProvider, loadTokenizer },
  });
  await assertProductionRemainsIsolated();

  const deterministicWorkDirectory = join(privateDirectory, 'evaluation-work');
  const evaluation = await runPrivateQueryPlannedRecallEvaluation({
    corpus,
    plans,
    baseConfig,
    workDirectory: deterministicWorkDirectory,
    dependencies: { loadTokenizer },
  });
  await assertProductionRemainsIsolated();
  const measuredCase = evaluation.cases[0];
  assert.equal(evaluation.executedSearchRequests, 3);
  assert.equal(evaluation.indexedSnapshotCount, 1);
  assert.deepEqual(evaluation.indexedSnapshotSha256, [createSha256(snapshotContent)]);
  assert.equal(
    measuredCase?.normal.outcome,
    QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS,
  );
  assert.equal(
    measuredCase?.retrievalWorkMatched.outcome,
    QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS,
  );
  assert.equal(measuredCase?.queryPlanned.outcome, QueryPlannedRecallBaselineOutcome.SUCCESS);
  assert.equal(measuredCase?.queryPlanned.candidateAdmissionVerified, true);
  assert.deepEqual(measuredCase?.queryPlanned.sourceProvenance, [
    { selectedFrom: 'ranked_result', passed: true },
  ]);
  assert.equal(measuredCase?.queryPlanned.fusedPoolLimit, 100);
  assert.equal(measuredCase?.queryPlanned.rerankPoolLimit, 40);
  assert.deepEqual(measuredCase?.queryPlanned.rerankerPolicy, {
    version: 2,
    activeBranchPrior: 0.01,
    fusedRankBlend: [
      { firstRank: 1, lastRank: 3, retrievalWeight: 0.75, rerankerWeight: 0.25 },
      { firstRank: 4, lastRank: 10, retrievalWeight: 0.6, rerankerWeight: 0.4 },
      { firstRank: 11, lastRank: null, retrievalWeight: 0.4, rerankerWeight: 0.6 },
    ],
  });
  assert.equal(measuredCase?.contribution.newCandidateAdmission, true);
  assert.equal(measuredCase?.contribution.rankingOnlyPromotion, false);
  assert.deepEqual(evaluation.contributionCounts, {
    newCandidateAdmission: 1,
    rankingOnlyPromotion: 0,
    preservedExistingSuccess: 0,
    noImprovement: 0,
  });
  assert.equal(JSON.stringify(evaluation).includes(lexicalPlanQuery), false);
  assert.equal(JSON.stringify(evaluation).includes('Private mechanism phrase'), false);

  const controls = createPublishableQueryPlannedRecallControls(corpus);
  const evidence = createPublishableQueryPlannedRecallEvaluationEvidence(controls, evaluation, {
    recordedAgainstCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf',
  });
  const report = formatPublishableQueryPlannedRecallEvaluationReport(evidence);
  assert.equal(evidence.evaluation.contributionCounts.newCandidateAdmission, 1);
  assert.equal(evidence.evaluation.providerIdentity.embeddingPolicy, 'deterministic-token-hash-v1');
  assert.match(report, /new candidate admission/u);
  assert.match(report, /normal hybrid/u);
  assert.match(report, /retrieval-work-matched original query/u);
  assert.match(report, /neutral-fused-order-v1/u);
  assert.match(report, /Fused-document limits before duplicate grouping: 100/u);
  assert.match(report, /Duplicate-group rerank limit \/ final results: 40 \/ 5/u);
  assert.equal(JSON.stringify(evidence).includes(lexicalPlanQuery), false);
  assert.equal(report.includes('Private mechanism phrase'), false);
  const evidenceEnvironment = {
    recordedAgainstCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf',
  };
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallEvaluationEvidence(
        controls,
        {
          ...evaluation,
          contributionCounts: {
            ...evaluation.contributionCounts,
            newCandidateAdmission: 0,
            noImprovement: 1,
          },
        },
        evidenceEnvironment,
      ),
    /requires at least one new candidate admission/u,
  );
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallEvaluationEvidence(
        controls,
        { ...evaluation, indexedSnapshotSha256: ['0'.repeat(64)] },
        evidenceEnvironment,
      ),
    /indexed snapshots must exactly match manifest hashes/u,
  );
  const controlCase = controls.cases[0];
  const planCase = evaluation.planIdentity.cases[0];
  if (!controlCase || !planCase || !measuredCase) {
    throw new Error('Query-planned recall evaluation fixture requires one measured case');
  }
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallEvaluationEvidence(
        { ...controls, cases: [controlCase, controlCase] },
        evaluation,
        evidenceEnvironment,
      ),
    /Evaluation case coverage invalid/u,
  );
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallEvaluationEvidence(
        controls,
        {
          ...evaluation,
          planIdentity: {
            ...evaluation.planIdentity,
            cases: [planCase, planCase],
          },
        },
        evidenceEnvironment,
      ),
    /Evaluation case coverage invalid/u,
  );
  for (const cases of [
    [measuredCase, measuredCase],
    [{ ...measuredCase, caseId: 'case-999' }],
    [],
  ]) {
    assert.throws(
      () =>
        createPublishableQueryPlannedRecallEvaluationEvidence(
          controls,
          { ...evaluation, cases },
          evidenceEnvironment,
        ),
      /Evaluation case coverage invalid/u,
    );
  }

  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  let conformancePlannerRequestCount = 0;
  const liveProgress: string[] = [];
  const liveWorkDirectory = join(privateDirectory, 'live-evaluation-work');
  const fixtureQueryPlanningExecutionIdentity = {
    ...createRecallQueryPlanningExecutionIdentity(
      queryPlanningProfile,
      'fixture-live-planner-v1',
      'fixture-live-planner-configuration-v1',
      RecallInferenceBackend.EMBEDDED,
      1_000,
      '1',
    ),
    computeBackend: 'cpu',
    deviceNames: ['Fixture CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    contextSize: 2_048,
    threads: null,
    nodeLlamaCppVersion: '3.18.1',
    idleTimeoutMilliseconds: 300_000,
    physicalDeviceIdentity: ['fixture cpu'],
    probedComputeBackends: [],
  };
  const fixtureRerankingExecutionIdentity = {
    ...createRecallRerankingExecutionIdentity(
      rerankingProfile,
      'fixture-live-reranker-v1',
      'fixture-live-reranker-configuration-v1',
      RecallInferenceBackend.EMBEDDED,
      1_000,
      '1',
    ),
    computeBackend: 'cpu',
    deviceNames: ['Fixture CPU'],
    devicePolicy: 'cpu',
    fallbackFromComputeBackend: null,
    contextSize: 4_096,
    threads: null,
    nodeLlamaCppVersion: '3.18.1',
    parallelism: 1,
    idleTimeoutMilliseconds: 300_000,
    physicalDeviceIdentity: ['fixture cpu'],
    probedComputeBackends: [],
  };
  const liveEvaluationOptions = {
    corpus,
    baseConfig,
    workDirectory: liveWorkDirectory,
    profileRun: {
      id: 'fixture-embedded-cpu',
      backend: RecallInferenceBackend.EMBEDDED,
      deviceClass: 'cpu',
      device: 'cpu',
      backendVersion: 'fixture-backend-v1',
    },
    evaluationConfiguration: {
      version: 1,
      effectiveConfigurationIdentity: 'fixture-effective-evaluation-config-v1',
      rerankerConformanceFixtureIdentity: 'fixture-reranker-conformance-v1',
    },
    software: {
      repositoryCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf',
      backendVersion: 'fixture-backend-v1',
      nodeVersion: process.version,
      platform: process.platform,
      architecture: process.arch,
    },
    queryPlanningProfile,
    queryPlanner: {
      executionIdentity: fixtureQueryPlanningExecutionIdentity,
      async planRecallQuery(request) {
        if (request.query === queryPlanningProfile.conformanceCanary.query) {
          conformancePlannerRequestCount += 1;
          if (conformancePlannerRequestCount === 2) {
            throw new Error('fixture warm planner output invalid');
          }
          return [
            { type: 'lex', query: 'Copper Finch records' },
            { type: 'vec', query: 'Copper Finch evidence' },
          ];
        }
        return [
          { type: 'lex', query: lexicalPlanQuery },
          { type: 'vec', query: semanticPlanQuery },
        ];
      },
    },
    rerankingProfile,
    reranker: {
      executionIdentity: fixtureRerankingExecutionIdentity,
      async rerankDocuments(rerankerQuery, documents) {
        if (rerankerQuery === 'source provenance') {
          return [0.9, 0.1];
        }
        return documents.map((document) =>
          document.includes('Private mechanism phrase.') ? 1 : 0,
        );
      },
    },
    rerankerConformance: {
      query: 'source provenance',
      documents: ['Source provenance is retained.', 'The navigation bar is blue.'],
      expectedScores: [0.9, 0.1],
      maximumAbsoluteDifference: 0,
    },
    reportProgress(message) {
      liveProgress.push(message);
    },
    dependencies: {
      async loadTokenizer() {
        return {
          encodeConversationText(text: string) {
            return { ids: Array.from(text.split(/\\s+/u).filter(Boolean).keys()) };
          },
        };
      },
    },
  } satisfies RunLiveQueryPlannedProfileEvaluationOptions;
  const liveEvaluation = await runLiveQueryPlannedProfileEvaluation(liveEvaluationOptions);
  await assertProductionRemainsIsolated();
  const privateEvaluationFiles = await readEvaluationContainmentFileTree(privateDirectory);
  const immutablePrivateFiles = new Set([
    relative(privateDirectory, snapshotPath),
    relative(privateDirectory, manifestPath),
    relative(privateDirectory, plansPath),
  ]);
  const workDirectoryPrefixes = [
    relative(privateDirectory, baselineWorkDirectory),
    relative(privateDirectory, deterministicWorkDirectory),
    relative(privateDirectory, liveWorkDirectory),
  ].map((path) => `${path}/`);
  for (const path of privateEvaluationFiles.keys()) {
    assert.ok(
      immutablePrivateFiles.has(path) ||
        workDirectoryPrefixes.some((prefix) => path.startsWith(prefix)),
      `Private evaluation write escaped its work area: ${path}`,
    );
  }
  for (const prefix of workDirectoryPrefixes) {
    assert.ok(
      [...privateEvaluationFiles.keys()].some((path) => path.startsWith(prefix)),
      `Private evaluation arm did not write below ${prefix}`,
    );
  }

  const publishedLiveEvaluation = JSON.stringify(liveEvaluation);
  assert.equal(liveEvaluation.capabilityConformance.queryPlanning.measurement.plannedQueryCount, 2);
  assert.equal(liveEvaluation.capabilityConformance.reranking.measurement.documentCount, 2);
  assert.equal(liveEvaluation.latency.warmPlanningSucceeded, false);
  assert.equal(liveEvaluation.quality.newCandidateAdmissionCount, 1);
  assert.equal(liveEvaluation.quality.noImprovementCount, 0);
  assert.equal(liveEvaluation.quality.plannerFallbackCount, 0);
  assert.equal(liveEvaluation.cases[0]?.planSource, 'planner');
  assert.deepEqual(liveEvaluation.cases[0]?.plannedQueries, [
    { type: 'lex', querySha256: createSha256(lexicalPlanQuery) },
    { type: 'vec', querySha256: createSha256(semanticPlanQuery) },
  ]);
  assert.equal(liveEvaluation.cases[0]?.queryPlanned.candidateAdmissionVerified, true);
  assert.equal(liveEvaluation.cases[0]?.queryPlanned.fusedPoolLimit, 100);
  assert.equal(liveEvaluation.cases[0]?.queryPlanned.rerankPoolLimit, 40);
  assert.deepEqual(liveProgress, [
    'Verifying live profile fixture-embedded-cpu capabilities',
    'Indexing live profile fixture-embedded-cpu private corpus',
    'Evaluating live profile fixture-embedded-cpu case 1/1',
  ]);
  assert.equal(publishedLiveEvaluation.includes(lexicalPlanQuery), false);
  assert.equal(publishedLiveEvaluation.includes(semanticPlanQuery), false);
  assert.equal(publishedLiveEvaluation.includes('Private mechanism phrase'), false);

  const fallbackCorpus = {
    ...corpus,
    manifest: {
      ...corpus.manifest,
      cases: corpus.manifest.cases.map((evaluationCase) => ({
        ...evaluationCase,
        query: lexicalPlanQuery,
      })),
    },
  };
  const fallbackEvaluation = await runLiveQueryPlannedProfileEvaluation({
    ...liveEvaluationOptions,
    corpus: fallbackCorpus,
    workDirectory: join(privateDirectory, 'live-fallback-evaluation-work'),
    profileRun: { ...liveEvaluationOptions.profileRun, id: 'fixture-embedded-cpu-fallback' },
    queryPlanner: {
      executionIdentity: fixtureQueryPlanningExecutionIdentity,
      async planRecallQuery(request) {
        if (request.query === queryPlanningProfile.conformanceCanary.query) {
          return [
            { type: 'lex', query: 'Copper Finch records' },
            { type: 'vec', query: 'Copper Finch evidence' },
          ];
        }
        throw new Error('fixture planner unavailable');
      },
    },
    reportProgress() {},
  });
  const fallbackCase = fallbackEvaluation.cases[0];
  assert.equal(fallbackCase?.planSource, 'fallback');
  assert.equal(fallbackCase?.queryPlanned.candidateAdmissionVerified, true);
  assert.ok(
    fallbackCase?.queryPlanned.candidateAdmissionSourceRanks.every((rank) => rank !== null),
  );
  assert.equal(fallbackCase?.queryPlanned.outcome, QueryPlannedRecallBaselineOutcome.SUCCESS);
  assert.deepEqual(fallbackCase?.contribution, {
    newCandidateAdmission: false,
    rankingOnlyPromotion: false,
    preservedExistingSuccess: false,
    noImprovement: true,
  });
  assert.equal(fallbackEvaluation.quality.newCandidateAdmissionCount, 0);
  assert.equal(fallbackEvaluation.quality.noImprovementCount, 1);
  assert.equal(fallbackEvaluation.quality.plannerFallbackCount, 1);

  function createMeasuredProfileRun(
    id: string,
    backend: RecallInferenceBackend,
    deviceClass: 'cpu' | 'accelerated',
    adapterIds: { queryPlanning: string; reranking: string },
  ) {
    const embeddedDevice = deviceClass === 'cpu' ? 'cpu' : 'vulkan';
    const device =
      backend === RecallInferenceBackend.EMBEDDED ? embeddedDevice : 'fixture HTTP CPU';
    const backendVersion =
      backend === RecallInferenceBackend.EMBEDDED ? 'fixture-embedded-v1' : 'fixture-http-v1';
    const embeddedPhysicalDeviceIdentity =
      backend === RecallInferenceBackend.EMBEDDED
        ? {
            computeBackend: embeddedDevice,
            deviceNames: [deviceClass === 'cpu' ? 'Fixture CPU' : 'Fixture GPU'],
            devicePolicy: embeddedDevice,
            fallbackFromComputeBackend: null,
            contextSize: 4_096,
            threads: null,
            nodeLlamaCppVersion: '3.18.1',
            parallelism: 1,
            idleTimeoutMilliseconds: 300_000,
            physicalDeviceIdentity: [deviceClass === 'cpu' ? 'fixture cpu' : 'fixture gpu'],
            probedComputeBackends: [],
          }
        : {};
    const queryPlanningExecutionIdentity = {
      ...createRecallQueryPlanningExecutionIdentity(
        queryPlanningProfile,
        adapterIds.queryPlanning,
        `${adapterIds.queryPlanning}-configuration`,
        backend,
        1_000,
        '1',
      ),
      ...embeddedPhysicalDeviceIdentity,
    };
    const rerankingExecutionIdentity = {
      ...createRecallRerankingExecutionIdentity(
        rerankingProfile,
        adapterIds.reranking,
        `${adapterIds.reranking}-configuration`,
        backend,
        1_000,
        '1',
      ),
      ...embeddedPhysicalDeviceIdentity,
    };
    return {
      ...liveEvaluation,
      profileRun: { id, backend, deviceClass, device, backendVersion },
      profileIdentity: {
        ...liveEvaluation.profileIdentity,
        software: {
          ...liveEvaluation.profileIdentity.software,
          backendVersion,
        },
        queryPlanning: {
          ...liveEvaluation.profileIdentity.queryPlanning,
          executionIdentity: queryPlanningExecutionIdentity,
        },
        reranking: {
          ...liveEvaluation.profileIdentity.reranking,
          executionIdentity: rerankingExecutionIdentity,
        },
      },
      capabilityConformance: {
        queryPlanning: {
          ...liveEvaluation.capabilityConformance.queryPlanning,
          executionIdentity: queryPlanningExecutionIdentity,
        },
        reranking: {
          ...liveEvaluation.capabilityConformance.reranking,
          executionIdentity: rerankingExecutionIdentity,
        },
      },
      committedCorpus: {
        corpusId: 'recall-quality-project-scoped-bounded-v3',
        specificationSha256: '3'.repeat(64),
        caseCount: 17,
        qualityPassed: true,
        candidatePoolRecall: 1,
        finalRecall: 1,
        contextUsefulness: 1,
        sourceOccurrencePreservation: 1,
        sessionOriginVerification: 1,
        evidenceRelationVerification: 1,
        contributingEntryVerification: 1,
        branchVerification: 1,
        policyFailureCaseIds: [],
        queryLatencyMilliseconds: { median: 10, p95: 20 },
        executedSearchRequests: 18,
        plannerRequests: 17,
        rerankerRequests: 18,
      },
    };
  }
  const embeddedAdapters = {
    queryPlanning: 'node-llama-cpp-qmd-query-planning-v1',
    reranking: 'node-llama-cpp-qwen-reranking-logit-recovery-v1',
  };
  const measuredProfileRuns = [
    createMeasuredProfileRun(
      'embedded-cpu',
      RecallInferenceBackend.EMBEDDED,
      'cpu',
      embeddedAdapters,
    ),
    createMeasuredProfileRun(
      'embedded-accelerated',
      RecallInferenceBackend.EMBEDDED,
      'accelerated',
      embeddedAdapters,
    ),
    createMeasuredProfileRun('http-cpu', RecallInferenceBackend.LLAMA_CPP_HTTP, 'cpu', {
      queryPlanning: 'llama-cpp-http-query-planning-v1',
      reranking: 'llama-cpp-http-reranking-v1',
    }),
  ];
  const expectedProfileRuns = measuredProfileRuns.map(({ profileRun }) => profileRun);
  const expectedCorpus = {
    id: liveEvaluation.corpus.id,
    privateManifestSha256: liveEvaluation.corpus.privateManifestSha256,
    snapshotSha256: liveEvaluation.corpus.snapshotSha256,
    cases: liveEvaluation.cases.map((measuredCase) => ({
      caseId: measuredCase.caseId,
      category: measuredCase.category,
      controlKind: measuredCase.controlKind,
      expectedSourceCount: measuredCase.queryPlanned.expectedSourceRanks.length,
    })),
  };
  const acceptance = createPublishableLiveQueryPlannedProfileAcceptance({
    recordedAgainstCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf',
    defaultSearchMode: 'hybrid',
    expectedCorpus,
    committedCorpus: [
      {
        evidenceKind: 'accepted-hybrid-baseline',
        deviceClass: 'baseline',
        profileId: 'deterministic-fixture-v1',
        evidenceSha256: '0'.repeat(64),
        qualityPassed: true,
        candidatePoolRecall: 1,
        finalRecall: 1,
      },
      {
        evidenceKind: 'live-profile-candidate',
        deviceClass: 'cpu',
        profileId: 'embeddinggemma-300m-q8-0-v1',
        evidenceSha256: '1'.repeat(64),
        qualityPassed: false,
        candidatePoolRecall: 0.941,
        finalRecall: 0.941,
      },
      {
        evidenceKind: 'live-profile-candidate',
        deviceClass: 'accelerated',
        profileId: 'embeddinggemma-300m-q8-0-v1',
        evidenceSha256: '2'.repeat(64),
        qualityPassed: false,
        candidatePoolRecall: 0.941,
        finalRecall: 0.941,
      },
    ],
    expectedProfileRuns,
    profileRuns: measuredProfileRuns,
    requiredSuccessfulBaselineControlCount: 0,
    privacyAudit: { checkedValueCount: 7, leakCount: 0 },
    failureSemantics: {
      plannerFallbackPublicServicePassed: true,
      rerankerFailurePublicServicePassed: true,
      piToolContractPassed: true,
    },
  });
  assert.equal(acceptance.releaseDecision, 'approved-explicit-fallback');
  assert.equal(acceptance.defaultSearchMode, 'hybrid');
  assert.equal(acceptance.aggregateQuality.newCandidateAdmissionCount, 3);
  assert.equal(acceptance.aggregateQuality.noImprovementCount, 0);
  assert.equal(
    acceptance.aggregateQuality.newCandidateAdmissionCount +
      acceptance.aggregateQuality.rankingOnlyPromotionCount +
      acceptance.aggregateQuality.preservedExistingSuccessCount +
      acceptance.aggregateQuality.noImprovementCount,
    measuredProfileRuns.reduce((total, run) => total + run.cases.length, 0),
  );
  assert.equal(acceptance.profileRuns.length, 3);
  const acceptanceReport = formatPublishableLiveQueryPlannedProfileAcceptanceReport(acceptance);
  assert.match(acceptanceReport, /Approved as an explicit fallback after hybrid misses/u);
  assert.match(acceptanceReport, /Hybrid remains the default/u);
  assert.match(
    acceptanceReport,
    /fused-document limits 100; duplicate-group rerank\/final limits 40\/5/u,
  );
  assert.match(acceptanceReport, /embedded-cpu/u);
  assert.match(acceptanceReport, /Cold planning/u);
  assert.match(acceptanceReport, /Live planner\/reranker quality on the committed corpus/u);
  assert.match(acceptanceReport, /hybrid 2-second latency gate does not apply/u);
  assert.match(acceptanceReport, /17 \| 100\.0% \| 100\.0%/u);
  assert.match(acceptanceReport, /Candidate work/u);
  assert.match(
    acceptanceReport,
    /Every matrix row uses `deterministic-token-hash-v1` test embeddings/u,
  );
  assert.match(acceptanceReport, /Planner\/reranker backend/u);
  assert.match(acceptanceReport, /does not measure end-to-end production inference/u);
  assert.match(acceptanceReport, /No improvement: 0/u);
  assert.match(acceptanceReport, /Planner fallbacks/u);
  assert.equal(acceptanceReport.includes(lexicalPlanQuery), false);
  assert.equal(acceptanceReport.includes('Private mechanism phrase'), false);

  const createAcceptanceWithProfileRuns = (
    profileRuns: readonly LiveQueryPlannedProfileEvaluationResult[],
  ) =>
    createPublishableLiveQueryPlannedProfileAcceptance({
      recordedAgainstCommit: acceptance.recordedAgainstCommit,
      defaultSearchMode: acceptance.defaultSearchMode,
      committedCorpus: acceptance.committedCorpus,
      expectedCorpus,
      expectedProfileRuns,
      profileRuns,
      requiredSuccessfulBaselineControlCount: 0,
      privacyAudit: acceptance.privacyAudit,
      failureSemantics: acceptance.failureSemantics,
    });
  const embeddedCpuRun = measuredProfileRuns[0];
  const embeddedAcceleratedRun = measuredProfileRuns[1];
  const httpRun = measuredProfileRuns[2];
  if (!embeddedCpuRun || !embeddedAcceleratedRun || !httpRun) {
    throw new Error('Expected complete measured profile fixture matrix');
  }
  const { committedCorpus: omittedCommittedCorpus, ...runWithoutCommittedCorpus } = embeddedCpuRun;
  void omittedCommittedCorpus;
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([runWithoutCommittedCorpus, embeddedAcceleratedRun, httpRun]),
    /committed-corpus planner\/reranker quality invalid/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        embeddedCpuRun,
        {
          ...embeddedAcceleratedRun,
          profileRun: {
            ...embeddedAcceleratedRun.profileRun,
            id: 'embedded-cpu-duplicate',
            deviceClass: 'cpu',
            device: 'cpu',
          },
        },
        { ...httpRun, profileRun: { ...httpRun.profileRun, deviceClass: 'accelerated' } },
      ]),
    /missing or substituted profile tuple/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        {
          ...embeddedCpuRun,
          profileRun: { ...embeddedCpuRun.profileRun, deviceClass: 'accelerated' },
        },
        {
          ...embeddedAcceleratedRun,
          profileRun: { ...embeddedAcceleratedRun.profileRun, deviceClass: 'cpu' },
        },
        httpRun,
      ]),
    /missing or substituted profile tuple/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        embeddedCpuRun,
        embeddedAcceleratedRun,
        { ...embeddedCpuRun, profileRun: { ...embeddedCpuRun.profileRun, id: 'duplicate-run' } },
      ]),
    /missing or substituted profile tuple/u,
  );
  assert.throws(
    () => createAcceptanceWithProfileRuns([...measuredProfileRuns, httpRun]),
    /duplicate or extra profile tuples/u,
  );

  function replaceEmbeddedPhysicalDeviceIdentity(
    run: LiveQueryPlannedProfileEvaluationResult,
    changes: Record<string, unknown>,
  ): LiveQueryPlannedProfileEvaluationResult {
    const queryPlanningExecutionIdentity = {
      ...run.profileIdentity.queryPlanning.executionIdentity,
      ...changes,
    };
    const rerankingExecutionIdentity = {
      ...run.profileIdentity.reranking.executionIdentity,
      ...changes,
    };
    return {
      ...run,
      profileIdentity: {
        ...run.profileIdentity,
        queryPlanning: {
          ...run.profileIdentity.queryPlanning,
          executionIdentity: queryPlanningExecutionIdentity,
        },
        reranking: {
          ...run.profileIdentity.reranking,
          executionIdentity: rerankingExecutionIdentity,
        },
      },
      capabilityConformance: {
        queryPlanning: {
          ...run.capabilityConformance.queryPlanning,
          executionIdentity: queryPlanningExecutionIdentity,
        },
        reranking: {
          ...run.capabilityConformance.reranking,
          executionIdentity: rerankingExecutionIdentity,
        },
      },
    };
  }
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        embeddedCpuRun,
        replaceEmbeddedPhysicalDeviceIdentity(embeddedAcceleratedRun, { deviceNames: [] }),
        httpRun,
      ]),
    /resolved physical device identity mismatch/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        embeddedCpuRun,
        replaceEmbeddedPhysicalDeviceIdentity(embeddedAcceleratedRun, {
          computeBackend: 'pending',
        }),
        httpRun,
      ]),
    /resolved physical device identity mismatch/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        embeddedCpuRun,
        replaceEmbeddedPhysicalDeviceIdentity(embeddedAcceleratedRun, {
          computeBackend: 'metal',
        }),
        httpRun,
      ]),
    /resolved physical device identity mismatch/u,
  );

  assert.throws(
    () =>
      createAcceptanceWithProfileRuns(
        acceptance.profileRuns.map((run) => ({
          ...run,
          quality: {
            ...run.quality,
            newCandidateAdmissionCount: 0,
            preservedExistingSuccessCount: 0,
          },
        })),
      ),
    /quality does not match recomputed case evidence/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        {
          ...embeddedCpuRun,
          corpus: {
            ...embeddedCpuRun.corpus,
            snapshotSha256: ['f'.repeat(64)],
          },
        },
        embeddedAcceleratedRun,
        httpRun,
      ]),
    /corpus identity mismatch/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        {
          ...embeddedCpuRun,
          cases: [...embeddedCpuRun.cases, ...embeddedCpuRun.cases],
        },
        embeddedAcceleratedRun,
        httpRun,
      ]),
    /case IDs must be unique/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        {
          ...embeddedCpuRun,
          cases: embeddedCpuRun.cases.map((measuredCase) => ({
            ...measuredCase,
            queryPlanned: {
              ...measuredCase.queryPlanned,
              expectedSourceRanks: [],
              candidateAdmissionSourceRanks: [],
              sourceProvenance: [],
            },
          })),
        },
        embeddedAcceleratedRun,
        httpRun,
      ]),
    /source evidence length mismatch/u,
  );
  assert.throws(
    () =>
      createAcceptanceWithProfileRuns([
        {
          ...embeddedCpuRun,
          cases: embeddedCpuRun.cases.map((measuredCase) => ({
            ...measuredCase,
            contribution: {
              newCandidateAdmission: false,
              rankingOnlyPromotion: false,
              preservedExistingSuccess: false,
              noImprovement: true,
            },
          })),
        },
        embeddedAcceleratedRun,
        httpRun,
      ]),
    /contribution does not match recomputed case evidence/u,
  );

  await chmod(plansPath, 0o644);
  await assert.rejects(
    () => loadPrivateQueryPlannedRecallPlans(plansPath, corpus),
    /Private query-planned recall plan permissions invalid/u,
  );
});

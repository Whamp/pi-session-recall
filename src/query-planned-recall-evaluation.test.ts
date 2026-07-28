import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { chmod, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  QueryPlannedRecallCaseCategory,
  QueryPlannedRecallBaselineOutcome,
  QueryPlannedRecallControlKind,
  RecallDiagnosticsMode,
  RecallEvidenceRelation,
  RecallInferenceBackend,
  RecallSearchScope,
} from './enums.js';
import {
  createPublishableLiveQueryPlannedProfileAcceptance,
  createPublishableQueryPlannedRecallEvaluationEvidence,
  createPublishableQueryPlannedRecallPlanIdentity,
  formatPublishableLiveQueryPlannedProfileAcceptanceReport,
  formatPublishableQueryPlannedRecallEvaluationReport,
  loadPrivateQueryPlannedRecallPlans,
  runLiveQueryPlannedProfileEvaluation,
  runPrivateQueryPlannedRecallEvaluation,
} from './query-planned-recall-evaluation.js';
import {
  createPublishableQueryPlannedRecallControls,
  loadPrivateQueryPlannedRecallCorpus,
} from './query-planned-recall-baseline.js';
import { isUnknownRecord } from './is-unknown-record.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createSha256(content: string): string {
  return createHash('sha256').update(content).digest('hex');
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
    '367bad22570ccc5badd26093cc4d2c912e0be08950a6bc3a67f9b557961c863d',
  );
  assert.equal(
    createSha256(reportContent),
    'dcbc12e33f5dee8d4c7b1b984030397062935d72a79189348103b4ceecdbb9ce',
  );
  assert.equal(Reflect.get(evaluation, 'executedSearchRequests'), 32);
  assert.deepEqual(Reflect.get(evaluation, 'contributionCounts'), {
    newCandidateAdmission: 4,
    rankingOnlyPromotion: 1,
    preservedExistingSuccess: 2,
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

  const baseConfig: RecallConversationConfig = {
    sessionsDirectory: join(directory, 'must-not-scan-production-sessions'),
    databasePath: join(directory, 'unused-zvec'),
    statePath: join(directory, 'unused-state.json'),
    manifestPath: join(directory, 'unused-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'unused-tokenizers'),
    embeddingCacheDirectory: join(directory, 'unused-embedding-cache'),
    lockPath: join(directory, 'unused.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'unused-diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'unused-diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'deterministic://query-planned-evaluation',
    embeddingModel: 'deterministic-token-hash-v1',
    embeddingServedModelId: 'deterministic-token-hash-v1',
    embeddingArtifact: 'none',
    embeddingQuantization: 'none',
    embeddingPooling: 'token-hash',
    embeddingDimensions: 64,
    embeddingBatchSize: 64,
    rerankerBaseUrl: 'deterministic://query-planned-evaluation',
    rerankerModel: 'deterministic-source-admission-v1',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
  const evaluation = await runPrivateQueryPlannedRecallEvaluation({
    corpus,
    plans,
    baseConfig,
    workDirectory: join(privateDirectory, 'evaluation-work'),
    dependencies: {
      async loadTokenizer() {
        return {
          encodeConversationText(text: string) {
            return { ids: Array.from(text.split(/\\s+/u).filter(Boolean).keys()) };
          },
        };
      },
    },
  });
  const measuredCase = evaluation.cases[0];
  assert.equal(evaluation.executedSearchRequests, 4);
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
  assert.equal(JSON.stringify(evidence).includes(lexicalPlanQuery), false);
  assert.equal(report.includes('Private mechanism phrase'), false);
  assert.throws(
    () =>
      createPublishableQueryPlannedRecallEvaluationEvidence(
        controls,
        {
          ...evaluation,
          contributionCounts: {
            ...evaluation.contributionCounts,
            newCandidateAdmission: 0,
          },
        },
        { recordedAgainstCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf' },
      ),
    /requires at least one new candidate admission/u,
  );

  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  let conformancePlannerRequestCount = 0;
  const liveEvaluation = await runLiveQueryPlannedProfileEvaluation({
    corpus,
    baseConfig,
    workDirectory: join(privateDirectory, 'live-evaluation-work'),
    profileRun: {
      id: 'fixture-embedded-cpu',
      backend: RecallInferenceBackend.EMBEDDED,
      deviceClass: 'cpu',
      device: 'fixture CPU',
    },
    queryPlanningProfile,
    queryPlanner: {
      executionIdentity: {
        adapterId: 'fixture-live-planner-v1',
        backend: RecallInferenceBackend.EMBEDDED,
        cacheIdentity: `${queryPlanningProfile.profileId}:fixture-live-planner-v1:${queryPlanningProfile.promptPolicy}:${queryPlanningProfile.grammarVersion}`,
        modelProfileId: queryPlanningProfile.profileId,
        promptPolicy: queryPlanningProfile.promptPolicy,
        grammarVersion: queryPlanningProfile.grammarVersion,
        requestTimeoutMilliseconds: 1_000,
      },
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
      executionIdentity: {
        adapterId: 'fixture-live-reranker-v1',
        backend: RecallInferenceBackend.EMBEDDED,
        cacheIdentity: `${rerankingProfile.profileId}:fixture-live-reranker-v1`,
        modelProfileId: rerankingProfile.profileId,
      },
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
    dependencies: {
      async loadTokenizer() {
        return {
          encodeConversationText(text: string) {
            return { ids: Array.from(text.split(/\\s+/u).filter(Boolean).keys()) };
          },
        };
      },
    },
  });
  const publishedLiveEvaluation = JSON.stringify(liveEvaluation);
  assert.equal(liveEvaluation.capabilityConformance.queryPlanning.measurement.plannedQueryCount, 2);
  assert.equal(liveEvaluation.capabilityConformance.reranking.measurement.documentCount, 2);
  assert.equal(liveEvaluation.latency.warmPlanningSucceeded, false);
  assert.equal(liveEvaluation.quality.newCandidateAdmissionCount, 1);
  assert.equal(liveEvaluation.quality.plannerFallbackCount, 0);
  assert.equal(liveEvaluation.cases[0]?.planSource, 'planner');
  assert.deepEqual(liveEvaluation.cases[0]?.plannedQueries, [
    { type: 'lex', querySha256: createSha256(lexicalPlanQuery) },
    { type: 'vec', querySha256: createSha256(semanticPlanQuery) },
  ]);
  assert.equal(liveEvaluation.cases[0]?.queryPlanned.candidateAdmissionVerified, true);
  assert.equal(publishedLiveEvaluation.includes(lexicalPlanQuery), false);
  assert.equal(publishedLiveEvaluation.includes(semanticPlanQuery), false);
  assert.equal(publishedLiveEvaluation.includes('Private mechanism phrase'), false);

  function createMeasuredProfileRun(
    id: string,
    backend: RecallInferenceBackend,
    deviceClass: 'cpu' | 'accelerated',
    adapterIds: { queryPlanning: string; reranking: string },
  ) {
    const queryPlanningExecutionIdentity = {
      ...liveEvaluation.profileIdentity.queryPlanning.executionIdentity,
      adapterId: adapterIds.queryPlanning,
      backend,
      cacheIdentity: `${queryPlanningProfile.profileId}:${adapterIds.queryPlanning}:${queryPlanningProfile.promptPolicy}:${queryPlanningProfile.grammarVersion}`,
    };
    const rerankingExecutionIdentity = {
      ...liveEvaluation.profileIdentity.reranking.executionIdentity,
      adapterId: adapterIds.reranking,
      backend,
      cacheIdentity: `${rerankingProfile.profileId}:${adapterIds.reranking}`,
    };
    return {
      ...liveEvaluation,
      profileRun: {
        id,
        backend,
        deviceClass,
        device: deviceClass === 'cpu' ? 'fixture CPU' : 'fixture accelerator',
      },
      profileIdentity: {
        ...liveEvaluation.profileIdentity,
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
    };
  }
  const embeddedAdapters = {
    queryPlanning: 'node-llama-cpp-qmd-query-planning-v1',
    reranking: 'node-llama-cpp-qwen-reranking-logit-recovery-v1',
  };
  const acceptance = createPublishableLiveQueryPlannedProfileAcceptance({
    recordedAgainstCommit: '030396576c03c705a1f3c84dce1ff639256ed2cf',
    defaultSearchMode: 'hybrid',
    committedCorpus: [
      {
        evidenceKind: 'accepted-hybrid-baseline',
        deviceClass: 'baseline',
        profileId: 'octen-embed',
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
    profileRuns: [
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
    ],
    requiredSuccessfulBaselineControlCount: 0,
    privacyAudit: { checkedValueCount: 7, leakCount: 0 },
    failureSemantics: {
      plannerFallbackPublicServicePassed: true,
      rerankerFailurePublicServicePassed: true,
      piToolContractPassed: true,
    },
  });
  assert.equal(acceptance.releaseDecision, 'approved-explicit-mode');
  assert.equal(acceptance.defaultSearchMode, 'hybrid');
  assert.equal(acceptance.aggregateQuality.newCandidateAdmissionCount, 3);
  assert.equal(acceptance.profileRuns.length, 3);
  const acceptanceReport = formatPublishableLiveQueryPlannedProfileAcceptanceReport(acceptance);
  assert.match(acceptanceReport, /Approved for explicit mode only/u);
  assert.match(acceptanceReport, /Hybrid remains the default/u);
  assert.match(acceptanceReport, /embedded-cpu/u);
  assert.match(acceptanceReport, /Cold planning/u);
  assert.match(acceptanceReport, /Candidate work/u);
  assert.match(acceptanceReport, /Planner fallbacks/u);
  assert.equal(acceptanceReport.includes(lexicalPlanQuery), false);
  assert.equal(acceptanceReport.includes('Private mechanism phrase'), false);

  await chmod(plansPath, 0o644);
  await assert.rejects(
    () => loadPrivateQueryPlannedRecallPlans(plansPath, corpus),
    /Private query-planned recall plan permissions invalid/u,
  );
});

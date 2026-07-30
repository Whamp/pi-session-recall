import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallSearchScope } from './enums.js';
import { formatRecallQualityReport } from './format-recall-quality-report.js';
import {
  parseQualityCaseId,
  parseQualityEntryId,
  type LoadedRecallQualityCorpus,
} from './recall-quality-corpus.js';
import type { RecallQualityEvaluationResult } from './run-recall-quality-evaluation.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import type { RecallQualityGateCombination } from './select-recall-quality-policy.js';

void test('recall quality report records verdict, measured counts, sources, and reproduction', () => {
  const combination: RecallQualityGateCombination = {
    chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
    candidateCount: 8,
    finalCount: 5,
    totalChunks: 72,
    indexLatencyMilliseconds: 1_200,
    candidatePoolRecall: 1,
    candidatePoolDuplicateRate: 0.1,
    finalRecall: 1,
    contextUsefulness: 1,
    sourceOccurrencePreservation: 1,
    sessionOriginVerification: 1,
    evidenceRelationVerification: 1,
    contributingEntryVerification: 1,
    branchVerification: 1,
    finalDuplicateRate: 0,
    queryLatencyMilliseconds: { median: 500, p95: 800 },
    queryLatencyByScope: {
      project: { median: 500, p95: 800 },
      global: null,
    },
    policyFailureCaseIds: [],
    gatePassed: true,
    failures: [],
  };
  const corpus: LoadedRecallQualityCorpus = {
    specificationPath: '/repo/evaluation/recall-quality-cases.json',
    specificationSha256: 'a'.repeat(64),
    sessionDirectory: '/repo/evaluation/corpus',
    sessionFiles: [
      {
        fileName: 'semantic-context.jsonl',
        path: '/repo/evaluation/corpus/semantic-context.jsonl',
        sha256: 'b'.repeat(64),
      },
    ],
    specification: {
      version: 3,
      corpus: {
        id: 'report-fixture-v1',
        sessionDirectory: 'corpus',
        sessionFiles: [{ fileName: 'semantic-context.jsonl', sha256: 'b'.repeat(64) }],
      },
      projectIdentityFixtures: [],
      projectLineages: {},
      bounds: {
        maximumSessionFiles: 1,
        maximumEvaluationCases: 1,
        maximumChunkPolicies: 1,
        maximumCandidateCounts: 1,
        maximumFinalCounts: 1,
        maximumSearchRequests: 3,
        maximumChunkEmbeddingRequests: 6,
      },
      chunkPolicies: [{ id: '512-64', maxTokens: 512, overlapTokens: 64 }],
      candidateCounts: [8],
      finalCounts: [5],
      warmupQueriesPerCombination: 0,
      qualityGate: {
        minimumCandidatePoolRecall: 1,
        minimumFinalRecall: 0.9,
        minimumContextUsefulness: 0.9,
        minimumSourceOccurrencePreservation: 1,
        maximumFinalDuplicateRate: 0,
        maximumQueryP95Milliseconds: 2_000,
      },
      cases: [
        {
          id: parseQualityCaseId('semantic-outbox'),
          category: 'semantic_paraphrase',
          query: 'How do queued jobs survive a crash?',
          scope: RecallSearchScope.PROJECT,
          invocationDirectory: '/evaluation/fulfillment',
          expectedInvocationProjectIdentity: parseRepositoryIdentity(
            'git-origin:github.com/whamp/fixture',
          ),
          expectedSources: [
            {
              sessionFile: 'semantic-context.jsonl',
              entryId: parseQualityEntryId('queue-answer'),
              requiredText: ['append-only SQLite outbox'],
              expectedSessionOrigin: '/evaluation/fulfillment',
              expectedEvidenceRelation: RecallEvidenceRelation.SAME_REPOSITORY,
              requiredContributingEntryIds: [parseQualityEntryId('queue-answer')],
              expectedBranch: 'active',
            },
          ],
          excludedSessionFiles: [],
          requiredContext: ['append-only SQLite outbox'],
          minimumPreservedSourceOccurrences: 1,
        },
      ],
    },
  };
  const result: RecallQualityEvaluationResult = {
    version: 6,
    storageIdentity: {
      generationFormatVersion: 1,
      generationStoreFormatVersion: 1,
      validationReceiptVersion: 1,
      incrementalEligibilityPolicyVersion: 1,
    },
    evaluationIdentity: {
      defaultScope: RecallSearchScope.PROJECT,
      projectScopePolicyVersion: 1,
      projectIdentityPolicyVersion: 3,
      projectIdentityMetadataSchemaVersion: 3,
      lineagePolicyVersion: 1,
      lineageDigest: 'a'.repeat(64),
      rankingMode: 'hybrid',
      rankFusionVersion: 1,
      reciprocalRankConstant: 60,
      activeBranchPrior: 0.01,
      candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      fusedPoolLimit: 24,
      rerankPoolLimit: 24,
      finalResultCount: 5,
    },
    startedAt: '2026-07-25T12:00:00.000Z',
    completedAt: '2026-07-25T12:01:00.000Z',
    durationMilliseconds: 60_000,
    specificationPath: corpus.specificationPath,
    specificationSha256: corpus.specificationSha256,
    corpusId: corpus.specification.corpus.id,
    indexRuns: [
      {
        chunkPolicy: { ...combination.chunkPolicy },
        generationId: 'generation_quality_active',
        manifestFingerprint: 'b'.repeat(64),
        startingSnapshotFingerprint: 'c'.repeat(64),
        storeCounts: { lexicalSource: 80, dense: 72, sessionProjection: 2 },
        totalChunks: 72,
        indexLatencyMilliseconds: 1_200,
        indexSummary: {
          scannedSessions: 1,
          indexedSessions: 1,
          removedSessions: 0,
          cacheHits: 0,
          newlyEmbeddedChunks: 72,
          embeddingRequestCount: 5,
          deletedChunks: 0,
          failedSessions: [],
        },
      },
    ],
    configurations: [
      {
        chunkPolicy: { ...combination.chunkPolicy },
        candidateCount: 8,
        totalChunks: 72,
        indexLatencyMilliseconds: 1_200,
        measurement: {
          caseCount: 1,
          candidatePoolRecall: 1,
          candidatePoolDuplicateRate: 0.1,
          queryLatencyMilliseconds: { median: 500, p95: 800 },
          queryLatencyByScope: {
            project: { median: 500, p95: 800 },
            global: null,
          },
          policyFailureCaseIds: [],
          missedCandidatePoolCaseIds: [],
          caseMeasurements: [],
          finalCounts: [
            {
              finalCount: 5,
              finalRecall: 1,
              contextUsefulness: 1,
              sourceOccurrencePreservation: 1,
              sessionOriginVerification: 1,
              evidenceRelationVerification: 1,
              contributingEntryVerification: 1,
              branchVerification: 1,
              finalDuplicateRate: 0,
              missedCaseIds: [],
              contextFailureCaseIds: [],
              sourceOccurrenceFailureCaseIds: [],
              sessionOriginFailureCaseIds: [],
              evidenceRelationFailureCaseIds: [],
              contributingEntryFailureCaseIds: [],
              branchFailureCaseIds: [],
              finalDuplicateSlots: 0,
              finalResultSlots: 5,
            },
          ],
        },
      },
    ],
    selection: {
      passed: true,
      selected: combination,
      blockers: [],
      combinations: [combination, { ...combination, candidateCount: 16 }],
    },
    boundedWork: {
      sessionFiles: 1,
      evaluationCases: 1,
      indexRuns: 1,
      executedSearchRequests: 1,
      rerankerRequests: 0,
      chunkEmbeddingRequests: 5,
      maximumCandidatesPerSearch: 24,
      repositoryIdentityResolutions: 0,
    },
  };

  const report = formatRecallQualityReport(result, corpus, {
    command: 'npm run evaluate:recall',
    gitCommit: 'abc1234',
    gitDirty: false,
    nodeVersion: 'v24.16.0',
    platform: 'linux',
    architecture: 'x64',
    cpuModel: 'Test CPU',
    embeddingBaseUrl: 'http://embedding.test/v1',
    embeddingModel: 'octen-embed',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingArtifact: 'Octen-Embedding-4B.Q8_0.gguf',
    embeddingDimensions: 2_560,
    rerankerBaseUrl: 'http://reranker.test/v1',
    rerankerModel: 'qwen3-rerank',
  });

  assert.match(report, /Automated gate: PASS/);
  assert.match(report, /512\/64/);
  assert.match(report, /8 candidates\/channel/);
  assert.match(
    report,
    /Ranked-list limits: dense 8, lexical 8, identifier 8; fused pool 24; rerank pool 24; final results 5/,
  );
  assert.match(report, /semantic-context\.jsonl#queue-answer/);
  assert.match(report, /coherent generation v1, store format v1, validation receipt v1/);
  assert.match(report, /c{64}/u);
  assert.match(report, /never opens the production recall generation or original Pi session files/);
  assert.match(report, /npm run evaluate:recall/);
  assert.match(report, /Full corpus backfill remains blocked pending human approval\./);
  assert.match(report, /no discriminating quality variance/);
});

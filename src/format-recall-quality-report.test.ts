import assert from 'node:assert/strict';
import test from 'node:test';

import { formatRecallQualityReport } from './format-recall-quality-report.js';
import type { LoadedRecallQualityCorpus } from './recall-quality-corpus.js';
import type { RecallQualityEvaluationResult } from './run-recall-quality-evaluation.js';
import type { RecallQualityGateCombination } from './select-recall-quality-policy.js';

void test('recall quality report records verdict, measured counts, sources, and reproduction', () => {
  const combination: RecallQualityGateCombination = {
    chunkPolicy: { id: '768-96', maxTokens: 768, overlapTokens: 96 },
    candidateCount: 8,
    finalCount: 5,
    totalChunks: 72,
    indexLatencyMilliseconds: 1_200,
    preRerankRecall: 1,
    preRerankDuplicateRate: 0.1,
    postRerankRecall: 1,
    contextUsefulness: 1,
    sourceOccurrencePreservation: 1,
    postRerankDuplicateRate: 0,
    queryLatencyMilliseconds: { median: 500, p95: 800 },
    rerankerLatencyMilliseconds: { median: 300, p95: 450 },
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
      version: 1,
      corpus: {
        id: 'report-fixture-v1',
        sessionDirectory: 'corpus',
        sessionFiles: [{ fileName: 'semantic-context.jsonl', sha256: 'b'.repeat(64) }],
      },
      bounds: {
        maximumSessionFiles: 1,
        maximumEvaluationCases: 1,
        maximumChunkPolicies: 3,
        maximumCandidateCounts: 1,
        maximumFinalCounts: 1,
        maximumSearchRequests: 3,
      },
      chunkPolicies: [
        { id: '512-64', maxTokens: 512, overlapTokens: 64 },
        { id: '768-96', maxTokens: 768, overlapTokens: 96 },
        { id: '1024-128', maxTokens: 1_024, overlapTokens: 128 },
      ],
      candidateCounts: [8],
      finalCounts: [5],
      warmupQueriesPerCombination: 0,
      qualityGate: {
        minimumPreRerankRecall: 1,
        minimumPostRerankRecall: 0.9,
        minimumContextUsefulness: 0.9,
        minimumSourceOccurrencePreservation: 1,
        maximumPostRerankDuplicateRate: 0,
        maximumQueryP95Milliseconds: 2_000,
        maximumRerankerP95Milliseconds: 1_500,
      },
      cases: [
        {
          id: 'semantic-outbox',
          category: 'semantic_paraphrase',
          query: 'How do queued jobs survive a crash?',
          expectedSources: [
            {
              sessionFile: 'semantic-context.jsonl',
              entryId: 'queue-answer',
              requiredText: ['append-only SQLite outbox'],
              expectedBranch: 'active',
            },
          ],
          requiredContext: ['append-only SQLite outbox'],
          minimumPreservedSourceOccurrences: 1,
        },
      ],
    },
  };
  const result: RecallQualityEvaluationResult = {
    version: 1,
    startedAt: '2026-07-25T12:00:00.000Z',
    completedAt: '2026-07-25T12:01:00.000Z',
    durationMilliseconds: 60_000,
    specificationPath: corpus.specificationPath,
    specificationSha256: corpus.specificationSha256,
    corpusId: corpus.specification.corpus.id,
    indexRuns: [
      {
        chunkPolicy: { ...combination.chunkPolicy },
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
          preRerankRecall: 1,
          preRerankDuplicateRate: 0.1,
          queryLatencyMilliseconds: { median: 500, p95: 800 },
          rerankerLatencyMilliseconds: { median: 300, p95: 450 },
          missedPreRerankCaseIds: [],
          caseMeasurements: [],
          finalCounts: [
            {
              finalCount: 5,
              postRerankRecall: 1,
              contextUsefulness: 1,
              sourceOccurrencePreservation: 1,
              postRerankDuplicateRate: 0,
              missedCaseIds: [],
              contextFailureCaseIds: [],
              sourceOccurrenceFailureCaseIds: [],
              postRerankDuplicateSlots: 0,
              postRerankResultSlots: 5,
            },
          ],
        },
      },
    ],
    selection: { passed: true, selected: combination, blockers: [], combinations: [combination] },
    boundedWork: {
      sessionFiles: 1,
      evaluationCases: 1,
      indexRuns: 3,
      executedSearchRequests: 3,
      rerankerRequests: 3,
      chunkEmbeddingRequests: 15,
      maximumCandidatesPerSearch: 24,
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
  assert.match(report, /768\/96/);
  assert.match(report, /8 candidates\/channel/);
  assert.match(report, /semantic-context\.jsonl#queue-answer/);
  assert.match(report, /npm run evaluate:recall/);
  assert.match(report, /Full corpus backfill remains blocked pending human approval\./);
});

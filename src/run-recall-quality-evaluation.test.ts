import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { LocalEmbeddingClient } from './local-embedding-client.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import { loadRecallQualityCorpus } from './recall-quality-corpus.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';
import { RECALL_EMBEDDING_CANARY_TEXT } from './recall-index-manifest.js';
import { runRecallQualityEvaluation } from './run-recall-quality-evaluation.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

void test('recall quality runner indexes and searches only the bounded declared corpus', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'run-recall-quality-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evaluationDirectory = join(directory, 'evaluation');
  const corpusDirectory = join(evaluationDirectory, 'corpus');
  await mkdir(corpusDirectory, { recursive: true });
  const sessionContent =
    [
      {
        type: 'session',
        version: 3,
        id: 'bounded-session',
        timestamp: '2026-07-20T10:00:00Z',
        cwd: '/bounded',
      },
      {
        type: 'message',
        id: 'bounded-answer',
        parentId: null,
        timestamp: '2026-07-20T10:01:00Z',
        message: {
          role: 'assistant',
          content: 'The bounded answer is quartz-heron and stays inside this fixture.',
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n';
  const sessionFileName = 'bounded.jsonl';
  await writeFile(join(corpusDirectory, sessionFileName), sessionContent);
  const sha256 = createHash('sha256').update(sessionContent).digest('hex');
  const specification = {
    version: 1,
    corpus: {
      id: 'bounded-runner-v1',
      sessionDirectory: 'corpus',
      sessionFiles: [{ fileName: sessionFileName, sha256 }],
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
      { id: '1024-128', maxTokens: 1024, overlapTokens: 128 },
    ],
    candidateCounts: [2],
    finalCounts: [1],
    warmupQueriesPerCombination: 0,
    qualityGate: {
      minimumPreRerankRecall: 1,
      minimumPostRerankRecall: 1,
      minimumContextUsefulness: 1,
      minimumSourceOccurrencePreservation: 1,
      maximumPostRerankDuplicateRate: 0,
      maximumQueryP95Milliseconds: 10_000,
      maximumRerankerP95Milliseconds: 10_000,
    },
    cases: [
      {
        id: 'bounded-answer',
        category: 'exact_identifier',
        query: 'quartz-heron',
        expectedSources: [
          {
            sessionFile: sessionFileName,
            entryId: 'bounded-answer',
            requiredText: ['quartz-heron'],
            expectedBranch: 'active',
          },
        ],
        requiredContext: ['quartz-heron'],
        minimumPreservedSourceOccurrences: 1,
      },
    ],
  };
  const specificationPath = join(evaluationDirectory, 'recall-quality-cases.json');
  await writeFile(specificationPath, `${JSON.stringify(specification, null, 2)}\n`);
  const corpus = await loadRecallQualityCorpus(specificationPath);
  const baseConfig: RecallConversationConfig = {
    sessionsDirectory: join(directory, 'must-not-scan-production-sessions'),
    databasePath: join(directory, 'unused-zvec'),
    statePath: join(directory, 'unused-state.json'),
    manifestPath: join(directory, 'unused-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'unused-tokenizers'),
    embeddingCacheDirectory: join(directory, 'unused-embedding-cache'),
    lockPath: join(directory, 'unused.lock'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-embedding',
    embeddingServedModelId: 'test-embedding-served',
    embeddingArtifact: 'test-embedding.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker',
    searchCandidateLimits: { dense: 2, lexical: 2, identifier: 2 },
  };
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
    },
  };
  const reranker: LocalRerankerClient = {
    async rerankDocuments(query, documents) {
      void query;
      return documents.map((document) => (document.includes('quartz-heron') ? 1 : 0));
    },
  };
  const tokenizer: ConversationTextTokenizer = {
    encodeConversationText(text) {
      return {
        ids: text
          .split(/\s+/u)
          .filter(Boolean)
          .map((_, index) => index),
      };
    },
  };

  const result = await runRecallQualityEvaluation({
    corpus,
    baseConfig,
    workDirectory: join(directory, 'recall-quality-evaluation'),
    dependencies: {
      embeddings,
      reranker,
      async loadTokenizer() {
        return tokenizer;
      },
    },
  });

  assert.equal(result.boundedWork.indexRuns, 3);
  assert.equal(result.boundedWork.executedSearchRequests, 3);
  assert.equal(result.indexRuns.length, 3);
  assert.ok(result.indexRuns.every(({ indexSummary }) => indexSummary.scannedSessions === 1));
  assert.equal(result.configurations.length, 3);
  assert.equal(result.selection.passed, true);
  assert.equal(result.selection.selected?.candidateCount, 2);
  assert.equal(result.selection.selected?.finalCount, 1);
});

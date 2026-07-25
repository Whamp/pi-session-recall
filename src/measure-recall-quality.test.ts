import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import { measureRecallQuality } from './measure-recall-quality.js';
import type { RecallQualityEvaluationCase } from './recall-quality-corpus.js';
import type { RerankedRecallSearchResult } from './rerank-recall-search-results.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';

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

function createSearchResult(
  chunk: SessionConversationChunk,
  fusedScore: number,
): RecallSearchResult {
  return {
    ...chunk,
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    fusedScore,
  };
}

void test('recall quality measures pre/post rerank duplicates and preserved source occurrences', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'measure-recall-quality-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const content = 'Release Meridian used sha256:4c91d7e2 and rollback tag meridian-safe-3.';
  const sessionFiles = [
    { fileName: 'copy-a.jsonl', sessionId: 'copy-a', entryId: 'entry-a' },
    { fileName: 'copy-b.jsonl', sessionId: 'copy-b', entryId: 'entry-b' },
  ];
  const chunks: SessionConversationChunk[] = [];
  for (const sessionFile of sessionFiles) {
    const sessionPath = join(directory, sessionFile.fileName);
    await writeFile(
      sessionPath,
      [
        {
          type: 'session',
          version: 3,
          id: sessionFile.sessionId,
          timestamp: '2026-07-20T10:00:00Z',
          cwd: '/evaluation',
        },
        {
          type: 'message',
          id: sessionFile.entryId,
          parentId: null,
          timestamp: '2026-07-20T10:01:00Z',
          message: { role: 'assistant', content },
        },
      ]
        .map((record) => JSON.stringify(record))
        .join('\n') + '\n',
    );
    chunks.push(...(await readSessionConversationChunks(sessionPath, { tokenizer })));
  }
  const first = chunks[0];
  const second = chunks[1];
  assert.ok(first);
  assert.ok(second);
  const representative = createSearchResult(first, 0.03);
  const duplicate = createSearchResult(second, 0.02);
  const rerankedResult: RerankedRecallSearchResult = {
    ...representative,
    rerankerScore: 0.99,
    activeBranchPrior: 0.01,
    rankingScore: 1,
    duplicateOccurrences: [duplicate],
    neighborContext: null,
  };
  const evaluationCase: RecallQualityEvaluationCase = {
    id: 'duplicate-meridian',
    category: 'duplicate_content',
    query: 'What release checksum did Meridian use?',
    expectedSources: [
      {
        sessionFile: 'copy-a.jsonl',
        entryId: 'entry-a',
        requiredText: ['sha256:4c91d7e2'],
        expectedBranch: 'active',
      },
      {
        sessionFile: 'copy-b.jsonl',
        entryId: 'entry-b',
        requiredText: ['sha256:4c91d7e2'],
        expectedBranch: 'active',
      },
    ],
    requiredContext: ['sha256:4c91d7e2', 'meridian-safe-3'],
    minimumPreservedSourceOccurrences: 2,
  };

  const measurement = measureRecallQuality(
    [
      {
        evaluationCase,
        results: [rerankedResult],
        queryLatencyMilliseconds: 120,
        rerankerLatencyMilliseconds: 80,
      },
    ],
    [1, 2],
  );

  assert.equal(measurement.preRerankRecall, 1);
  assert.equal(measurement.preRerankDuplicateRate, 0.5);
  assert.deepEqual(measurement.queryLatencyMilliseconds, { median: 120, p95: 120 });
  assert.deepEqual(measurement.rerankerLatencyMilliseconds, { median: 80, p95: 80 });
  assert.deepEqual(
    measurement.finalCounts.map((finalCount) => ({
      finalCount: finalCount.finalCount,
      postRerankRecall: finalCount.postRerankRecall,
      contextUsefulness: finalCount.contextUsefulness,
      sourceOccurrencePreservation: finalCount.sourceOccurrencePreservation,
      postRerankDuplicateRate: finalCount.postRerankDuplicateRate,
    })),
    [
      {
        finalCount: 1,
        postRerankRecall: 1,
        contextUsefulness: 1,
        sourceOccurrencePreservation: 1,
        postRerankDuplicateRate: 0,
      },
      {
        finalCount: 2,
        postRerankRecall: 1,
        contextUsefulness: 1,
        sourceOccurrencePreservation: 1,
        postRerankDuplicateRate: 0,
      },
    ],
  );
});

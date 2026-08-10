import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallEvidenceRelation, RecallSearchScope } from './enums.js';
import { measureRecallQuality } from './measure-recall-quality.js';
import {
  parseQualityCaseId,
  parseQualityEntryId,
  type RecallQualityEvaluationCase,
} from './recall-quality-corpus.js';
import type {
  RankedRecallSearchResult,
  RecallDenseSearchResult,
} from './rank-recall-search-results.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
    };
  },
};

function createSearchResult(
  chunk: SessionConversationChunk,
  cosineDistance: number,
): RecallDenseSearchResult {
  return {
    ...chunk,
    cosineDistance,
    denseRank: 1,
    denseReciprocalRankScore: 1 - cosineDistance,
  };
}

void test('recall quality measures candidate-pool and final source preservation', async (t) => {
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
  const rerankedResult: RankedRecallSearchResult & {
    resultKind: 'conversation';
    evidenceRelation: RecallEvidenceRelation;
  } = {
    ...representative,
    resultKind: 'conversation',
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    activeBranchPrior: 0.01,
    rankingScore: 1,
    duplicateOccurrences: [duplicate],
    neighborContext: null,
  };
  const evaluationCase: RecallQualityEvaluationCase = {
    id: parseQualityCaseId('duplicate-meridian'),
    category: 'duplicate_content',
    query: 'What release checksum did Meridian use?',
    scope: RecallSearchScope.GLOBAL,
    expectedSources: [
      {
        sessionFile: 'copy-a.jsonl',
        entryId: parseQualityEntryId('entry-a'),
        requiredText: ['sha256:4c91d7e2'],
        expectedSessionOrigin: '/evaluation',
        expectedEvidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
        requiredContributingEntryIds: [parseQualityEntryId('entry-a')],
        expectedBranch: 'active',
      },
      {
        sessionFile: 'copy-b.jsonl',
        entryId: parseQualityEntryId('entry-b'),
        requiredText: ['sha256:4c91d7e2'],
        expectedSessionOrigin: '/evaluation',
        expectedEvidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
        requiredContributingEntryIds: [parseQualityEntryId('entry-b')],
        expectedBranch: 'active',
      },
    ],
    excludedSessionFiles: [],
    requiredContext: ['sha256:4c91d7e2', 'meridian-safe-3'],
    minimumPreservedSourceOccurrences: 2,
  };

  const measurement = measureRecallQuality(
    [
      {
        evaluationCase,
        results: [rerankedResult],
        searchPolicy: {
          scope: RecallSearchScope.GLOBAL,
          invocationProjectIdentity: null,
        },
        queryLatencyMilliseconds: 120,
      },
    ],
    [1, 2],
  );

  assert.equal(measurement.candidatePoolRecall, 1);
  assert.equal(measurement.candidatePoolDuplicateRate, 0.5);
  assert.deepEqual(measurement.queryLatencyMilliseconds, { median: 120, p95: 120 });
  assert.deepEqual(measurement.caseMeasurements[0]?.finalCounts[0], {
    finalCount: 1,
    finalRecalled: true,
    contextUseful: true,
    sourceOccurrencesPreserved: true,
    preservedSourceOccurrences: 2,
    sessionOriginsVerified: true,
    evidenceRelationsVerified: true,
    contributingEntriesVerified: true,
    branchesVerified: true,
    finalDuplicateSlots: 0,
    finalResultSlots: 1,
  });
  assert.deepEqual(
    measurement.finalCounts.map((finalCount) => ({
      finalCount: finalCount.finalCount,
      finalRecall: finalCount.finalRecall,
      contextUsefulness: finalCount.contextUsefulness,
      sourceOccurrencePreservation: finalCount.sourceOccurrencePreservation,
      sessionOriginVerification: finalCount.sessionOriginVerification,
      evidenceRelationVerification: finalCount.evidenceRelationVerification,
      contributingEntryVerification: finalCount.contributingEntryVerification,
      branchVerification: finalCount.branchVerification,
      finalDuplicateRate: finalCount.finalDuplicateRate,
    })),
    [1, 2].map((finalCount) => ({
      finalCount,
      finalRecall: 1,
      contextUsefulness: 1,
      sourceOccurrencePreservation: 1,
      sessionOriginVerification: 1,
      evidenceRelationVerification: 1,
      contributingEntryVerification: 1,
      branchVerification: 1,
      finalDuplicateRate: 0,
    })),
  );

  const unrelatedSameEntry: RankedRecallSearchResult & {
    resultKind: 'conversation';
    evidenceRelation: RecallEvidenceRelation;
  } = {
    ...rerankedResult,
    id: 'unrelated-same-entry',
    checksum: 'unrelated-checksum',
    content: 'Release planning introduction without the required checksum or rollback tag.',
    duplicateOccurrences: [],
  };
  const unrelatedMeasurement = measureRecallQuality(
    [
      {
        evaluationCase,
        results: [unrelatedSameEntry],
        searchPolicy: {
          scope: RecallSearchScope.GLOBAL,
          invocationProjectIdentity: null,
        },
        queryLatencyMilliseconds: 10,
      },
    ],
    [1],
  );

  assert.equal(unrelatedMeasurement.candidatePoolRecall, 0);
  assert.equal(unrelatedMeasurement.finalCounts[0]?.finalRecall, 0);
  assert.equal(unrelatedMeasurement.finalCounts[0]?.sourceOccurrencePreservation, 0);
});

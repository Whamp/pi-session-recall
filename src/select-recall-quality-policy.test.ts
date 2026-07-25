import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallQualityMeasurement } from './measure-recall-quality.js';
import type { RecallQualityGate } from './recall-quality-corpus.js';
import { selectRecallQualityPolicy } from './select-recall-quality-policy.js';

const gate: RecallQualityGate = {
  minimumPreRerankRecall: 1,
  minimumPostRerankRecall: 0.9,
  minimumContextUsefulness: 0.9,
  minimumSourceOccurrencePreservation: 1,
  maximumPostRerankDuplicateRate: 0,
  maximumQueryP95Milliseconds: 2_000,
  maximumRerankerP95Milliseconds: 1_500,
};

function createMeasurement(
  preRerankRecall: number,
  finalCounts: Array<{
    finalCount: number;
    postRerankRecall: number;
    contextUsefulness: number;
  }>,
): RecallQualityMeasurement {
  return {
    caseCount: 10,
    preRerankRecall,
    preRerankDuplicateRate: 0.2,
    queryLatencyMilliseconds: { median: 700, p95: 900 },
    rerankerLatencyMilliseconds: { median: 500, p95: 650 },
    missedPreRerankCaseIds: preRerankRecall === 1 ? [] : ['semantic-miss'],
    caseMeasurements: [],
    finalCounts: finalCounts.map((measurement) => ({
      ...measurement,
      sourceOccurrencePreservation: 1,
      postRerankDuplicateRate: 0,
      missedCaseIds: measurement.postRerankRecall >= 0.9 ? [] : ['semantic-miss'],
      contextFailureCaseIds: measurement.contextUsefulness >= 0.9 ? [] : ['context-miss'],
      sourceOccurrenceFailureCaseIds: [],
      postRerankDuplicateSlots: 0,
      postRerankResultSlots: measurement.finalCount * 10,
    })),
  };
}

void test('recall quality policy selects the smallest measured counts that pass every gate', () => {
  const selection = selectRecallQualityPolicy(
    [
      {
        chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
        candidateCount: 4,
        totalChunks: 90,
        indexLatencyMilliseconds: 1_200,
        measurement: createMeasurement(0.9, [
          { finalCount: 3, postRerankRecall: 0.8, contextUsefulness: 0.8 },
          { finalCount: 5, postRerankRecall: 0.9, contextUsefulness: 0.9 },
        ]),
      },
      {
        chunkPolicy: { id: '768-96', maxTokens: 768, overlapTokens: 96 },
        candidateCount: 8,
        totalChunks: 76,
        indexLatencyMilliseconds: 1_000,
        measurement: createMeasurement(1, [
          { finalCount: 3, postRerankRecall: 0.9, contextUsefulness: 0.8 },
          { finalCount: 5, postRerankRecall: 1, contextUsefulness: 0.9 },
        ]),
      },
      {
        chunkPolicy: { id: '1024-128', maxTokens: 1_024, overlapTokens: 128 },
        candidateCount: 16,
        totalChunks: 68,
        indexLatencyMilliseconds: 900,
        measurement: createMeasurement(1, [
          { finalCount: 3, postRerankRecall: 1, contextUsefulness: 1 },
        ]),
      },
    ],
    gate,
  );

  assert.equal(selection.passed, true);
  assert.deepEqual(selection.selected?.chunkPolicy, {
    id: '768-96',
    maxTokens: 768,
    overlapTokens: 96,
  });
  assert.equal(selection.selected?.candidateCount, 8);
  assert.equal(selection.selected?.finalCount, 5);
  assert.deepEqual(selection.blockers, []);
});

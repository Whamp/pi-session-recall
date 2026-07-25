import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallQualityMeasurement } from './measure-recall-quality.js';
import { parseQualityCaseId, type RecallQualityGate } from './recall-quality-corpus.js';
import { selectRecallQualityPolicy } from './select-recall-quality-policy.js';

const gate: RecallQualityGate = {
  minimumCandidatePoolRecall: 1,
  minimumFinalRecall: 0.9,
  minimumContextUsefulness: 0.9,
  minimumSourceOccurrencePreservation: 1,
  maximumFinalDuplicateRate: 0,
  maximumQueryP95Milliseconds: 2_000,
};

function createMeasurement(
  candidatePoolRecall: number,
  finalCounts: Array<{
    finalCount: number;
    finalRecall: number;
    contextUsefulness: number;
  }>,
): RecallQualityMeasurement {
  return {
    caseCount: 10,
    candidatePoolRecall,
    candidatePoolDuplicateRate: 0.2,
    queryLatencyMilliseconds: { median: 700, p95: 900 },
    queryLatencyByScope: {
      project: { median: 700, p95: 900 },
      global: { median: 600, p95: 800 },
    },
    policyFailureCaseIds: [],
    missedCandidatePoolCaseIds:
      candidatePoolRecall === 1 ? [] : [parseQualityCaseId('semantic-miss')],
    caseMeasurements: [],
    finalCounts: finalCounts.map((measurement) => ({
      ...measurement,
      sourceOccurrencePreservation: 1,
      sessionOriginVerification: 1,
      evidenceRelationVerification: 1,
      contributingEntryVerification: 1,
      branchVerification: 1,
      finalDuplicateRate: 0,
      missedCaseIds: measurement.finalRecall >= 0.9 ? [] : [parseQualityCaseId('semantic-miss')],
      contextFailureCaseIds:
        measurement.contextUsefulness >= 0.9 ? [] : [parseQualityCaseId('context-miss')],
      sourceOccurrenceFailureCaseIds: [],
      sessionOriginFailureCaseIds: [],
      evidenceRelationFailureCaseIds: [],
      contributingEntryFailureCaseIds: [],
      branchFailureCaseIds: [],
      finalDuplicateSlots: 0,
      finalResultSlots: measurement.finalCount * 10,
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
          { finalCount: 3, finalRecall: 0.8, contextUsefulness: 0.8 },
          { finalCount: 5, finalRecall: 0.9, contextUsefulness: 0.9 },
        ]),
      },
      {
        chunkPolicy: { id: '768-96', maxTokens: 768, overlapTokens: 96 },
        candidateCount: 8,
        totalChunks: 76,
        indexLatencyMilliseconds: 1_000,
        measurement: createMeasurement(1, [
          { finalCount: 3, finalRecall: 0.9, contextUsefulness: 0.8 },
          { finalCount: 5, finalRecall: 1, contextUsefulness: 0.9 },
        ]),
      },
      {
        chunkPolicy: { id: '1024-128', maxTokens: 1_024, overlapTokens: 128 },
        candidateCount: 16,
        totalChunks: 68,
        indexLatencyMilliseconds: 900,
        measurement: createMeasurement(1, [
          { finalCount: 3, finalRecall: 1, contextUsefulness: 1 },
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

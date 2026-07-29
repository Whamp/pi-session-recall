import assert from 'node:assert/strict';
import test from 'node:test';

import { QueryPlannedRecallBaselineOutcome, QueryPlannedRecallControlKind } from './enums.js';
import { classifyQueryPlannedRecallContribution } from './query-planned-recall-contribution.js';

void test('successful baseline control is preserved only when the measured normal arm also succeeds', () => {
  const contribution = classifyQueryPlannedRecallContribution({
    controlKind: QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL,
    normalOutcome: QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
    retrievalWorkMatchedOutcome: QueryPlannedRecallBaselineOutcome.SUCCESS,
    queryPlannedOutcome: QueryPlannedRecallBaselineOutcome.SUCCESS,
    candidateAdmissionVerified: false,
  });

  assert.deepEqual(contribution, {
    newCandidateAdmission: false,
    rankingOnlyPromotion: true,
    preservedExistingSuccess: false,
    noImprovement: false,
  });
});

void test('contribution classification is exclusive and exhaustive for every control and outcome combination', () => {
  const outcomes = Object.values(QueryPlannedRecallBaselineOutcome);
  const controlKinds = Object.values(QueryPlannedRecallControlKind);
  for (const controlKind of controlKinds) {
    for (const normalOutcome of outcomes) {
      for (const retrievalWorkMatchedOutcome of outcomes) {
        for (const queryPlannedOutcome of outcomes) {
          for (const candidateAdmissionVerified of [false, true]) {
            const contribution = classifyQueryPlannedRecallContribution({
              controlKind,
              normalOutcome,
              retrievalWorkMatchedOutcome,
              queryPlannedOutcome,
              candidateAdmissionVerified,
            });
            assert.equal(
              Object.values(contribution).filter(Boolean).length,
              1,
              JSON.stringify({
                controlKind,
                normalOutcome,
                retrievalWorkMatchedOutcome,
                queryPlannedOutcome,
                candidateAdmissionVerified,
              }),
            );
          }
        }
      }
    }
  }

  assert.deepEqual(
    classifyQueryPlannedRecallContribution({
      controlKind: QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL,
      normalOutcome: QueryPlannedRecallBaselineOutcome.SUCCESS,
      retrievalWorkMatchedOutcome: QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
      queryPlannedOutcome: QueryPlannedRecallBaselineOutcome.SUCCESS,
      candidateAdmissionVerified: true,
    }),
    {
      newCandidateAdmission: false,
      rankingOnlyPromotion: false,
      preservedExistingSuccess: true,
      noImprovement: false,
    },
  );
});

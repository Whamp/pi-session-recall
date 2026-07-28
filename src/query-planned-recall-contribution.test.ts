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
  });
});

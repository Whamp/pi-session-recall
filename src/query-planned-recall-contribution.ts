import { QueryPlannedRecallBaselineOutcome, QueryPlannedRecallControlKind } from './enums.js';

/** Inputs that distinguish query-planning admission, promotion, and same-run preservation. */
export interface QueryPlannedRecallContributionInput {
  controlKind: QueryPlannedRecallControlKind;
  normalOutcome: QueryPlannedRecallBaselineOutcome;
  retrievalWorkMatchedOutcome: QueryPlannedRecallBaselineOutcome;
  queryPlannedOutcome: QueryPlannedRecallBaselineOutcome;
  candidateAdmissionVerified: boolean;
}

/** Classifies query-planned recall gains without treating a historical label as measured preservation. */
export function classifyQueryPlannedRecallContribution(
  input: QueryPlannedRecallContributionInput,
): {
  newCandidateAdmission: boolean;
  rankingOnlyPromotion: boolean;
  preservedExistingSuccess: boolean;
} {
  const newCandidateAdmission =
    input.candidateAdmissionVerified &&
    input.normalOutcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS &&
    input.retrievalWorkMatchedOutcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS;
  return {
    newCandidateAdmission,
    rankingOnlyPromotion:
      input.queryPlannedOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS &&
      !newCandidateAdmission &&
      [input.normalOutcome, input.retrievalWorkMatchedOutcome].includes(
        QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
      ),
    preservedExistingSuccess:
      input.controlKind === QueryPlannedRecallControlKind.SUCCESSFUL_BASELINE_CONTROL &&
      input.normalOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS &&
      input.queryPlannedOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS,
  };
}

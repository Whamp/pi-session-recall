import { QueryPlannedRecallBaselineOutcome } from './enums.js';
import type { QueryPlannedRecallControlKind } from './enums.js';

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
  noImprovement: boolean;
} {
  const newCandidateAdmission =
    input.candidateAdmissionVerified &&
    input.queryPlannedOutcome !== QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS &&
    input.normalOutcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS &&
    input.retrievalWorkMatchedOutcome === QueryPlannedRecallBaselineOutcome.CANDIDATE_UNION_MISS;
  const preservedExistingSuccess =
    !newCandidateAdmission &&
    input.normalOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS &&
    input.queryPlannedOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS;
  const rankingOnlyPromotion =
    !newCandidateAdmission &&
    !preservedExistingSuccess &&
    input.queryPlannedOutcome === QueryPlannedRecallBaselineOutcome.SUCCESS &&
    [input.normalOutcome, input.retrievalWorkMatchedOutcome].includes(
      QueryPlannedRecallBaselineOutcome.FINAL_RANK_MISS,
    );
  return {
    newCandidateAdmission,
    rankingOnlyPromotion,
    preservedExistingSuccess,
    noImprovement: !newCandidateAdmission && !rankingOnlyPromotion && !preservedExistingSuccess,
  };
}

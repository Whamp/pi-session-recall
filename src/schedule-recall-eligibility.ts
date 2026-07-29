import { RecallEligibilityThreshold } from './enums.js';

const EXPLICIT_EXIT_QUIET_MILLISECONDS = 60_000;
const LARGE_PREPARED_TRANSFER_QUIET_MILLISECONDS = 5 * 60_000;
const CRASH_ONLY_QUIESCENCE_MILLISECONDS = 30 * 60_000;
const LARGE_PREPARED_DOCUMENT_COUNT = 32;

/** Clock and observed work used to select one re-measurable quiet-period threshold. */
export interface ScheduleRecallEligibilityInput {
  lastGrowthAtEpochMilliseconds: number;
  explicitExitObserved: boolean;
  preparedDocumentCount: number;
  nowEpochMilliseconds: () => number;
}

/** Readiness diagnostic recording both elapsed quiet and the measured candidate that fired. */
export interface RecallEligibilitySchedule {
  ready: boolean;
  threshold: RecallEligibilityThreshold;
  thresholdMilliseconds: number;
  quietMilliseconds: number;
}

function selectRecallEligibilityThreshold(input: ScheduleRecallEligibilityInput): {
  threshold: RecallEligibilityThreshold;
  thresholdMilliseconds: number;
} {
  if (!input.explicitExitObserved) {
    return {
      threshold: RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
      thresholdMilliseconds: CRASH_ONLY_QUIESCENCE_MILLISECONDS,
    };
  }
  if (input.preparedDocumentCount > LARGE_PREPARED_DOCUMENT_COUNT) {
    return {
      threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
      thresholdMilliseconds: LARGE_PREPARED_TRANSFER_QUIET_MILLISECONDS,
    };
  }
  return {
    threshold: RecallEligibilityThreshold.EXPLICIT_EXIT_QUIET,
    thresholdMilliseconds: EXPLICIT_EXIT_QUIET_MILLISECONDS,
  };
}

/** Selects 60-second exit, 5-minute large-transfer, or 30-minute crash-only quiet readiness. */
export function scheduleRecallEligibility(
  input: ScheduleRecallEligibilityInput,
): RecallEligibilitySchedule {
  if (
    !Number.isSafeInteger(input.lastGrowthAtEpochMilliseconds) ||
    input.lastGrowthAtEpochMilliseconds < 0 ||
    !Number.isSafeInteger(input.preparedDocumentCount) ||
    input.preparedDocumentCount < 0
  ) {
    throw new Error('Recall eligibility schedule input invalid');
  }
  const nowEpochMilliseconds = input.nowEpochMilliseconds();
  if (!Number.isSafeInteger(nowEpochMilliseconds) || nowEpochMilliseconds < 0) {
    throw new Error('Recall eligibility schedule clock invalid');
  }
  const quietMilliseconds = Math.max(0, nowEpochMilliseconds - input.lastGrowthAtEpochMilliseconds);
  const selected = selectRecallEligibilityThreshold(input);
  return {
    ready: quietMilliseconds >= selected.thresholdMilliseconds,
    ...selected,
    quietMilliseconds,
  };
}

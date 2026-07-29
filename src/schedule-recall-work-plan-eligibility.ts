import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { RecallWorkMarkerTrigger } from './enums.js';
import {
  scheduleRecallEligibility,
  type RecallEligibilitySchedule,
} from './schedule-recall-eligibility.js';

/** Durable quiet-period inputs reconstructed for one physical-session work plan. */
export interface ScheduleRecallWorkPlanEligibilityInput {
  workPlan: RecallMarkerReplayWorkPlan;
  sourceModifiedAtEpochMilliseconds: number;
  preparedDocumentCount: number;
  nowEpochMilliseconds: () => number;
}

/** Reconstructed eligibility decision with the exact epoch deadline for a later worker. */
export interface RecallWorkPlanEligibilitySchedule extends RecallEligibilitySchedule {
  readyAtEpochMilliseconds: number;
}

function markerEstablishesExplicitRecallExit(
  marker: RecallMarkerReplayWorkPlan['workItems'][number]['marker'],
): boolean {
  return (
    marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION ||
    marker.trigger.kind === RecallWorkMarkerTrigger.BRANCH_EXIT ||
    marker.trigger.kind === RecallWorkMarkerTrigger.DEPARTURE
  );
}

function readRecallWorkPlanQuietAnchor(input: ScheduleRecallWorkPlanEligibilityInput): number {
  if (
    !Number.isSafeInteger(input.sourceModifiedAtEpochMilliseconds) ||
    input.sourceModifiedAtEpochMilliseconds < 0
  ) {
    throw new Error('Recall work plan source modified time invalid');
  }
  let quietAnchorEpochMilliseconds = input.sourceModifiedAtEpochMilliseconds;
  for (const { marker } of input.workPlan.workItems) {
    if (marker.trigger.kind !== RecallWorkMarkerTrigger.ARRIVAL) {
      quietAnchorEpochMilliseconds = Math.max(
        quietAnchorEpochMilliseconds,
        marker.createdAtEpochMilliseconds,
      );
    }
  }
  return quietAnchorEpochMilliseconds;
}

/** Schedules one marker-backed transfer from source metadata without reading session payload text. */
export function scheduleRecallWorkPlanEligibility(
  input: ScheduleRecallWorkPlanEligibilityInput,
): RecallWorkPlanEligibilitySchedule {
  const lastGrowthAtEpochMilliseconds = readRecallWorkPlanQuietAnchor(input);
  const schedule = scheduleRecallEligibility({
    lastGrowthAtEpochMilliseconds,
    explicitExitObserved: input.workPlan.workItems.some(({ marker }) =>
      markerEstablishesExplicitRecallExit(marker),
    ),
    preparedDocumentCount: input.preparedDocumentCount,
    nowEpochMilliseconds: input.nowEpochMilliseconds,
  });
  const readyAtEpochMilliseconds = lastGrowthAtEpochMilliseconds + schedule.thresholdMilliseconds;
  if (!Number.isSafeInteger(readyAtEpochMilliseconds)) {
    throw new Error('Recall work plan eligibility deadline invalid');
  }
  return { ...schedule, readyAtEpochMilliseconds };
}

import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { RecallEligibilityThreshold, RecallWorkMarkerTrigger } from './enums.js';
import {
  createRecallWorkMarkerId,
  type RecallWorkMarker,
  type RecallWorkMarkerIdentity,
  type RecallWorkMarkerTriggerPayload,
} from './recall-work-marker.js';
import { scheduleRecallWorkPlanEligibility } from './schedule-recall-work-plan-eligibility.js';

function createMarker(
  runtimeSequence: number,
  createdAtEpochMilliseconds: number,
  trigger: RecallWorkMarkerTriggerPayload,
): RecallWorkMarker {
  const identity: RecallWorkMarkerIdentity = {
    version: 1,
    physicalSessionId: 'physical-session-1',
    physicalSessionPath: '/sessions/physical-session-1.jsonl',
    runtimeInstanceId: 'runtime-1',
    runtimeSequence,
    createdAtEpochMilliseconds,
    trigger,
  };
  return { ...identity, markerId: createRecallWorkMarkerId(identity) };
}

function createWorkPlan(markers: readonly RecallWorkMarker[]): RecallMarkerReplayWorkPlan {
  return {
    targetGenerationId: 'generation-1',
    markerSpoolDirectory: '/recall/markers',
    discoveredMarkerCount: markers.length,
    sourceMarkerIds: markers.map(({ markerId }) => markerId),
    workItems: markers.map((marker) => ({ marker, coveredMarkerIds: [marker.markerId] })),
    quarantineDiagnostics: [],
  };
}

void test('arrival does not move an explicit-exit quiet anchor', () => {
  const departure = createMarker(1, 1_000, { kind: RecallWorkMarkerTrigger.DEPARTURE });
  const arrival = createMarker(2, 50_000, { kind: RecallWorkMarkerTrigger.ARRIVAL });
  const schedule = scheduleRecallWorkPlanEligibility({
    workPlan: createWorkPlan([departure, arrival]),
    sourceModifiedAtEpochMilliseconds: 1_000,
    preparedDocumentCount: 0,
    nowEpochMilliseconds: () => 61_000,
  });

  assert.equal(schedule.ready, true);
  assert.equal(schedule.threshold, RecallEligibilityThreshold.EXPLICIT_EXIT_QUIET);
  assert.equal(schedule.readyAtEpochMilliseconds, 61_000);
});

void test('source growth moves the crash-only deadline forward', () => {
  const activity = createMarker(1, 1_000, { kind: RecallWorkMarkerTrigger.ACTIVITY });
  const schedule = scheduleRecallWorkPlanEligibility({
    workPlan: createWorkPlan([activity]),
    sourceModifiedAtEpochMilliseconds: 20_000,
    preparedDocumentCount: 0,
    nowEpochMilliseconds: () => 30 * 60_000,
  });

  assert.equal(schedule.ready, false);
  assert.equal(schedule.threshold, RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE);
  assert.equal(schedule.readyAtEpochMilliseconds, 20_000 + 30 * 60_000);
});

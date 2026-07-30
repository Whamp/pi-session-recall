import assert from 'node:assert/strict';
import test from 'node:test';

import { decideConfirmedSessionDeletion } from './confirmed-session-deletion-policy.js';
import {
  RecallConfirmedDeletionDecisionKind,
  RecallMetadataSweepStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';

void test('confirmed deletion requires absence in a distinct later healthy metadata sweep', () => {
  const projection: PhysicalSessionProjection = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: 'physical-projection',
    generationId: 'generation',
    physicalSessionId: 'physical-session',
    sourcePath: '/disposable/sessions/source.jsonl',
    sourceDevice: '10',
    sourceInode: '20',
    appendCursorBytes: 10,
    appendCursorLines: 1,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: null,
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: { generationId: 'generation', coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
  const first = decideConfirmedSessionDeletion({
    projection,
    sweepId: 'sweep-1',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 100,
    sourceObservation: null,
  });
  assert.equal(first.kind, RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING);
  assert.ok('nextProjection' in first);
  const repeated = decideConfirmedSessionDeletion({
    projection: first.nextProjection,
    sweepId: 'sweep-1',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 200,
    sourceObservation: null,
  });
  assert.equal(repeated.kind, RecallConfirmedDeletionDecisionKind.NO_CHANGE);
  const confirmed = decideConfirmedSessionDeletion({
    projection: first.nextProjection,
    sweepId: 'sweep-2',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 300,
    sourceObservation: null,
  });
  assert.equal(confirmed.kind, RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION);
});

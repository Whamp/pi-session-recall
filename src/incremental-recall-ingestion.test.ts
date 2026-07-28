import assert from 'node:assert/strict';
import test from 'node:test';

import {
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallProjectedEntryDescriptor,
} from './recall-session-projection.js';
import { reduceRecallEligibility } from './reduce-recall-eligibility.js';
import type { RecallWorkMarker, RecallWorkMarkerTriggerPayload } from './recall-work-marker.js';

const GENERATION_ID = 'generation_acceptance';
const PHYSICAL_SESSION_ID = 'physical_acceptance';
const SESSION_PATH = '/isolated/acceptance/session.jsonl';

function createAcceptanceEntry(
  entryId: string,
  parentEntryId: string | null,
  entryType = 'message',
  firstKeptEntryId: string | null = null,
  hasRetainedTail = false,
): RecallProjectedEntryDescriptor {
  const sourceLine = Number(entryId.replaceAll(/\D/gu, '')) || entryId.length;
  return {
    entryId,
    parentEntryId,
    entryType,
    timestamp: '2026-01-01T00:00:00Z',
    messageRole: entryType === 'message' ? 'user' : null,
    branchSummaryFromEntryId: null,
    sourceLine,
    startByte: sourceLine * 100,
    endByte: sourceLine * 100 + 50,
    sourceFingerprint: 'b'.repeat(64),
    firstKeptEntryId,
    hasRetainedTail,
    toolCalls: [],
    toolResult: null,
  };
}

function createAcceptancePhysicalProjection(): PhysicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(PHYSICAL_SESSION_ID),
    generationId: GENERATION_ID,
    physicalSessionId: PHYSICAL_SESSION_ID,
    sourcePath: SESSION_PATH,
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 1,
    appendCursorLines: 1,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: null,
    logicalSessionIds: ['logical_acceptance'],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: { generationId: GENERATION_ID, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createAcceptanceLogicalProjection(
  entries: readonly RecallProjectedEntryDescriptor[],
  effectiveLeafEntryId = entries.at(-1)?.entryId ?? null,
): LogicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(PHYSICAL_SESSION_ID, 'logical_acceptance'),
    generationId: GENERATION_ID,
    physicalSessionId: PHYSICAL_SESSION_ID,
    physicalProjectionId: createPhysicalSessionProjectionId(PHYSICAL_SESSION_ID),
    logicalSessionId: 'logical_acceptance',
    effectiveLeafEntryId,
    activeContextBoundary:
      entries.length === 0
        ? null
        : {
            firstEntryId: entries[0]?.entryId ?? '',
            lastEntryId: effectiveLeafEntryId ?? '',
          },
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 1,
      sourceFingerprint: 'a'.repeat(64),
      cwd: '/isolated/acceptance',
      parentSessionPath: null,
    },
    entryDescriptors: [...entries],
    eligibleContributorEntryIds: [],
    eligibleSpans: [],
    labels: [],
    markerCheckpoint: { generationId: GENERATION_ID, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createAcceptanceMarker(
  markerId: string,
  trigger: RecallWorkMarkerTriggerPayload,
  runtimeInstanceId = 'runtime-1',
  runtimeSequence = 1,
): RecallWorkMarker {
  return {
    version: 1,
    markerId,
    physicalSessionId: PHYSICAL_SESSION_ID,
    physicalSessionPath: SESSION_PATH,
    runtimeInstanceId,
    runtimeSequence,
    createdAtEpochMilliseconds: runtimeSequence,
    trigger,
  };
}

interface LifecycleAcceptanceScenario {
  name: string;
  entries: readonly RecallProjectedEntryDescriptor[];
  effectiveLeafEntryId?: string;
  markers: readonly RecallWorkMarker[];
  quiescenceObserved: boolean;
  expectedNewlyEligibleEntryIds: readonly string[];
}

const linearEntries = [
  createAcceptanceEntry('e1', null),
  createAcceptanceEntry('e2', 'e1'),
  createAcceptanceEntry('e3', 'e2'),
];
const branchedEntries = [
  createAcceptanceEntry('e1', null),
  createAcceptanceEntry('e2', 'e1'),
  createAcceptanceEntry('e3', 'e2'),
  createAcceptanceEntry('e4', 'e2'),
  createAcceptanceEntry('s5', 'e4', 'branch_summary'),
];

const lifecycleAcceptanceScenarios: readonly LifecycleAcceptanceScenario[] = [
  ...['activity', 'arrival', 'reload', 'fork', 'clone'].map((name, index) => ({
    name,
    entries: linearEntries,
    markers: [
      createAcceptanceMarker(
        `${name}-marker`,
        {
          kind:
            name === 'activity'
              ? RecallWorkMarkerTrigger.ACTIVITY
              : RecallWorkMarkerTrigger.ARRIVAL,
        },
        `runtime-${name}`,
        index + 1,
      ),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: [],
  })),
  {
    name: 'repeated compaction',
    entries: [
      ...linearEntries,
      createAcceptanceEntry('c4', 'e3', 'compaction', 'e3'),
      createAcceptanceEntry('c5', 'c4', 'compaction', 'c4'),
    ],
    markers: [
      createAcceptanceMarker('compaction-1', {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        compactionEntryId: 'c4',
      }),
      createAcceptanceMarker(
        'compaction-2',
        { kind: RecallWorkMarkerTrigger.COMPACTION, compactionEntryId: 'c5' },
        'runtime-1',
        2,
      ),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['c4', 'c5', 'e1', 'e2', 'e3'],
  },
  {
    name: 'split compaction with retained tail',
    entries: [...linearEntries, createAcceptanceEntry('c4', 'e3', 'compaction', null, true)],
    markers: [
      createAcceptanceMarker('split-compaction', {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        compactionEntryId: 'c4',
      }),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['c4', 'e1', 'e2', 'e3'],
  },
  {
    name: 'branch exit without summary',
    entries: branchedEntries,
    effectiveLeafEntryId: 'e4',
    markers: [
      createAcceptanceMarker('branch-without-summary', {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        oldLeafEntryId: 'e3',
        newLeafEntryId: 'e4',
      }),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['e3'],
  },
  {
    name: 'branch exit with summary',
    entries: branchedEntries,
    effectiveLeafEntryId: 's5',
    markers: [
      createAcceptanceMarker('branch-with-summary', {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        oldLeafEntryId: 'e3',
        newLeafEntryId: 's5',
        summaryEntryId: 's5',
      }),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['e3', 's5'],
  },
  ...['departure', 'switch', 'clean exit'].map((name, index) => ({
    name,
    entries: linearEntries,
    markers: [
      createAcceptanceMarker(
        `${name}-marker`,
        { kind: RecallWorkMarkerTrigger.DEPARTURE },
        `runtime-${name}`,
        index + 1,
      ),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['e1', 'e2', 'e3'],
  })),
  ...['forced exit', 'quiescence'].map((name) => ({
    name,
    entries: linearEntries,
    markers: [],
    quiescenceObserved: true,
    expectedNewlyEligibleEntryIds: ['e1', 'e2', 'e3'],
  })),
  {
    name: 'concurrent runtimes preserve both branch exits',
    entries: branchedEntries,
    effectiveLeafEntryId: 's5',
    markers: [
      createAcceptanceMarker(
        'runtime-a-exit',
        {
          kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
          oldLeafEntryId: 'e3',
          newLeafEntryId: 'e4',
        },
        'runtime-a',
      ),
      createAcceptanceMarker(
        'runtime-b-exit',
        {
          kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
          oldLeafEntryId: 'e4',
          newLeafEntryId: 'e3',
        },
        'runtime-b',
      ),
    ],
    quiescenceObserved: false,
    expectedNewlyEligibleEntryIds: ['e3', 'e4'],
  },
];

void test('prototype lifecycle transition table is executable at the production eligibility seam', () => {
  assert.deepEqual(
    lifecycleAcceptanceScenarios.map(({ name }) => name),
    [
      'activity',
      'arrival',
      'reload',
      'fork',
      'clone',
      'repeated compaction',
      'split compaction with retained tail',
      'branch exit without summary',
      'branch exit with summary',
      'departure',
      'switch',
      'clean exit',
      'forced exit',
      'quiescence',
      'concurrent runtimes preserve both branch exits',
    ],
  );
  for (const scenario of lifecycleAcceptanceScenarios) {
    const result = reduceRecallEligibility({
      physicalProjection: createAcceptancePhysicalProjection(),
      logicalProjection: createAcceptanceLogicalProjection(
        scenario.entries,
        scenario.effectiveLeafEntryId,
      ),
      markers: scenario.markers,
      quiescenceObserved: scenario.quiescenceObserved,
    });
    assert.deepEqual(
      result.newlyEligibleContributorEntryIds.toSorted(),
      scenario.expectedNewlyEligibleEntryIds.toSorted(),
      scenario.name,
    );
  }
});

import assert from 'node:assert/strict';
import test from 'node:test';

import fc from 'fast-check';

import {
  RecallConfirmedDeletionPhase,
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

const generationId = 'generation';
const physicalSessionId = 'physical';

function descriptor(
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

function physical(
  sourceAvailability = RecallSourceAvailability.PRESENT,
): PhysicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId,
    physicalSessionId,
    sourcePath: '/isolated/session.jsonl',
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 1,
    appendCursorLines: 1,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: null,
    logicalSessionIds: ['logical'],
    sourceAvailability,
    sourceMissingObservedAtEpochMilliseconds:
      sourceAvailability === RecallSourceAvailability.PRESENT ? null : 1,
    sourceMissingObservationCount:
      sourceAvailability === RecallSourceAvailability.PRESENT
        ? 0
        : sourceAvailability === RecallSourceAvailability.SOURCE_MISSING
          ? 1
          : 2,
    sourceMissingSweepId:
      sourceAvailability === RecallSourceAvailability.PRESENT ? null : 'sweep-1',
    deletionCheckpoint:
      sourceAvailability === RecallSourceAvailability.DELETION_CONFIRMED
        ? {
            confirmedSweepId: 'sweep-1',
            phase: RecallConfirmedDeletionPhase.EVIDENCE,
            deletedEvidenceCount: 0,
            deletedLogicalProjectionCount: 0,
            pendingEvidenceIds: [],
            pendingLogicalProjectionIds: [],
          }
        : null,
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function logical(
  entries: RecallProjectedEntryDescriptor[],
  effectiveLeafEntryId = entries.at(-1)?.entryId ?? null,
  eligibleContributorEntryIds: string[] = [],
): LogicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(physicalSessionId, 'logical'),
    generationId,
    physicalSessionId,
    physicalProjectionId: createPhysicalSessionProjectionId(physicalSessionId),
    logicalSessionId: 'logical',
    effectiveLeafEntryId,
    activeContextBoundary: entries.length
      ? { firstEntryId: entries[0]?.entryId ?? '', lastEntryId: effectiveLeafEntryId ?? '' }
      : null,
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 1,
      sourceFingerprint: 'a'.repeat(64),
      cwd: '/isolated',
      parentSessionPath: null,
    },
    entryDescriptors: entries,
    eligibleContributorEntryIds,
    eligibleSpans: [],
    labels: [],
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function marker(
  markerId: string,
  trigger: RecallWorkMarkerTriggerPayload,
  runtimeInstanceId = 'runtime-1',
  runtimeSequence = 1,
): RecallWorkMarker {
  return {
    version: 1,
    markerId,
    physicalSessionId,
    physicalSessionPath: '/isolated/session.jsonl',
    runtimeInstanceId,
    runtimeSequence,
    createdAtEpochMilliseconds: runtimeSequence,
    trigger,
  };
}

interface ReducerScenario {
  name: string;
  projection: LogicalSessionProjection;
  markers?: RecallWorkMarker[];
  quiescenceObserved?: boolean;
  expected: string[];
  expectedRepairState?: RecallProjectionRepairState;
}

const linear = [descriptor('e1', null), descriptor('e2', 'e1'), descriptor('e3', 'e2')];
const scenarios: ReducerScenario[] = [
  {
    name: 'initial batch and activity retain the active tail',
    projection: logical(linear),
    expected: [],
  },
  {
    name: 'firstKept compaction admits summarized ancestors and its summary',
    projection: logical([...linear, descriptor('c4', 'e3', 'compaction', 'e3')], 'c4'),
    markers: [
      marker('compact', {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        logicalSessionId: 'logical',
        compactionEntryId: 'c4',
      }),
    ],
    expected: ['c4', 'e1', 'e2'],
  },
  {
    name: 'retainedTail compaction admits all prior source entries and its summary',
    projection: logical([...linear, descriptor('c4', 'e3', 'compaction', null, true)], 'c4'),
    markers: [
      marker('compact', {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        logicalSessionId: 'logical',
        compactionEntryId: 'c4',
      }),
    ],
    expected: ['c4', 'e1', 'e2', 'e3'],
  },
  {
    name: 'branch exit without summary admits only the abandoned path',
    projection: logical(
      [
        descriptor('e1', null),
        descriptor('e2', 'e1'),
        descriptor('e3', 'e2'),
        descriptor('e4', 'e1'),
      ],
      'e4',
    ),
    markers: [
      marker('branch', {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        logicalSessionId: 'logical',
        oldLeafEntryId: 'e3',
        newLeafEntryId: 'e4',
      }),
    ],
    expected: ['e2', 'e3'],
  },
  {
    name: 'branch exit with summary admits abandoned evidence and summary immediately',
    projection: logical(
      [
        descriptor('e1', null),
        descriptor('e2', 'e1'),
        descriptor('e3', 'e2'),
        descriptor('e4', 'e1'),
        descriptor('s5', 'e4', 'branch_summary'),
      ],
      's5',
    ),
    markers: [
      marker('branch', {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        logicalSessionId: 'logical',
        oldLeafEntryId: 'e3',
        newLeafEntryId: 'e4',
        summaryEntryId: 's5',
      }),
    ],
    expected: ['e2', 'e3', 's5'],
  },
  {
    name: 'departure switch and clean quit admit the remaining active tail',
    projection: logical(linear),
    markers: [
      marker('departure', {
        kind: RecallWorkMarkerTrigger.DEPARTURE,
        logicalSessionId: 'logical',
        leafEntryId: 'e3',
      }),
    ],
    expected: ['e1', 'e2', 'e3'],
  },
  {
    name: 'departure admits only the event-time tail when the source grows later',
    projection: logical([...linear, descriptor('e4', 'e3')], 'e4'),
    markers: [
      marker('departure-before-later-append', {
        kind: RecallWorkMarkerTrigger.DEPARTURE,
        logicalSessionId: 'logical',
        leafEntryId: 'e3',
      }),
    ],
    expected: ['e1', 'e2', 'e3'],
  },
  {
    name: 'unbounded departure without a prior runtime leaf requires reconciliation',
    projection: logical([...linear, descriptor('e4', 'e3')], 'e4'),
    markers: [
      marker('unbounded-departure', { kind: RecallWorkMarkerTrigger.DEPARTURE }, 'runtime-new'),
    ],
    expectedRepairState: RecallProjectionRepairState.REQUIRES_RECONCILIATION,
    expected: [],
  },
  {
    name: 'arrival reload fork clone and import never admit a copied active tail',
    projection: logical(linear),
    markers: [marker('arrival', { kind: RecallWorkMarkerTrigger.ARRIVAL })],
    expected: [],
  },
  {
    name: 'crash and force-exit sustained quiescence admit the remaining active tail',
    projection: logical(linear),
    quiescenceObserved: true,
    expected: ['e1', 'e2', 'e3'],
  },
];

for (const scenario of scenarios) {
  void test(scenario.name, () => {
    const result = reduceRecallEligibility({
      physicalProjection: physical(),
      logicalProjection: scenario.projection,
      markers: scenario.markers ?? [],
      quiescenceObserved: scenario.quiescenceObserved ?? false,
    });
    assert.equal(result.deletionConfirmed, false);
    assert.deepEqual(
      result.logicalProjection?.eligibleContributorEntryIds.toSorted(),
      scenario.expected,
    );
    if (scenario.expectedRepairState !== undefined) {
      assert.equal(result.logicalProjection?.repairState, scenario.expectedRepairState);
    }
  });
}

void test('repeated overlapping and split-turn compaction is monotonic and replay-idempotent', () => {
  const projection = logical(
    [
      descriptor('e1', null),
      descriptor('e2', 'e1'),
      descriptor('e3', 'e2'),
      descriptor('c4', 'e3', 'compaction', 'e3'),
      descriptor('c5', 'c4', 'compaction', 'e3'),
    ],
    'c5',
  );
  const markers = [
    marker(
      'c4-marker',
      {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        logicalSessionId: 'logical',
        compactionEntryId: 'c4',
      },
      'a',
      1,
    ),
    marker(
      'c5-marker',
      {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        logicalSessionId: 'logical',
        compactionEntryId: 'c5',
      },
      'b',
      1,
    ),
  ];
  const once = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers,
    quiescenceObserved: false,
  });
  assert.ok(once.logicalProjection);
  const replay = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: once.logicalProjection,
    markers,
    quiescenceObserved: false,
  });
  assert.deepEqual(replay.logicalProjection, once.logicalProjection);
  assert.deepEqual(once.newlyEligibleContributorEntryIds.toSorted(), ['c4', 'c5', 'e1', 'e2']);
  assert.equal(once.logicalProjection.eligibleContributorEntryIds.includes('e3'), false);

  const coalescedLatest = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers: [markers[1]].filter(
      (candidate): candidate is RecallWorkMarker => candidate !== undefined,
    ),
    quiescenceObserved: false,
  });
  assert.deepEqual(coalescedLatest.newlyEligibleContributorEntryIds.toSorted(), [
    'c4',
    'c5',
    'e1',
    'e2',
  ]);
});

void test('uncovered late compaction remains eligible below the runtime sequence checkpoint', () => {
  const projection = logical(
    [
      descriptor('e1', null),
      descriptor('e2', 'e1'),
      descriptor('compaction-late', 'e2', 'compaction', 'e2'),
    ],
    'compaction-late',
  );
  projection.markerCheckpoint = {
    generationId,
    coveredMarkerIds: ['sequence-2'],
    runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 2 }],
  };
  const lateCompaction = marker(
    'late-compaction',
    {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'logical',
      compactionEntryId: 'compaction-late',
    },
    'runtime-1',
    1,
  );

  const result = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers: [lateCompaction],
    quiescenceObserved: false,
  });

  assert.deepEqual(result.newlyEligibleContributorEntryIds, ['e1', 'compaction-late']);
  assert.deepEqual(result.logicalProjection?.markerCheckpoint.coveredMarkerIds, [
    'late-compaction',
    'sequence-2',
  ]);
  assert.deepEqual(result.logicalProjection?.markerCheckpoint.runtimeSequences, [
    { runtimeInstanceId: 'runtime-1', sequence: 2 },
  ]);
});

void test('uncovered late branch exit remains eligible below the runtime sequence checkpoint', () => {
  const projection = logical(
    [
      descriptor('e1', null),
      descriptor('e2', 'e1'),
      descriptor('e3', 'e2'),
      descriptor('e4', 'e1'),
    ],
    'e4',
  );
  projection.markerCheckpoint = {
    generationId,
    coveredMarkerIds: ['sequence-2'],
    runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 2 }],
  };
  const lateBranchExit = marker(
    'late-branch-exit',
    {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical',
      oldLeafEntryId: 'e3',
      newLeafEntryId: 'e4',
    },
    'runtime-1',
    1,
  );

  const result = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers: [lateBranchExit],
    quiescenceObserved: false,
  });

  assert.deepEqual(result.newlyEligibleContributorEntryIds, ['e2', 'e3']);
  assert.deepEqual(result.logicalProjection?.markerCheckpoint.coveredMarkerIds, [
    'late-branch-exit',
    'sequence-2',
  ]);
  assert.deepEqual(result.logicalProjection?.markerCheckpoint.runtimeSequences, [
    { runtimeInstanceId: 'runtime-1', sequence: 2 },
  ]);
});

void test('concurrent runtimes union eligibility without choosing one authoritative leaf', () => {
  const projection = logical([
    descriptor('e1', null),
    descriptor('e2', 'e1'),
    descriptor('e3', 'e2'),
    descriptor('e4', 'e1'),
    descriptor('e5', 'e4'),
  ]);
  const branchA = marker(
    'a',
    {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical',
      oldLeafEntryId: 'e3',
      newLeafEntryId: 'e5',
    },
    'runtime-a',
    2,
  );
  const branchB = marker(
    'b',
    {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical',
      oldLeafEntryId: 'e5',
      newLeafEntryId: 'e3',
    },
    'runtime-b',
    4,
  );
  const first = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers: [branchA, branchB],
    quiescenceObserved: false,
  });
  const reversed = reduceRecallEligibility({
    physicalProjection: physical(),
    logicalProjection: projection,
    markers: [branchB, branchA],
    quiescenceObserved: false,
  });
  assert.deepEqual(first.logicalProjection?.eligibleContributorEntryIds.toSorted(), [
    'e2',
    'e3',
    'e4',
    'e5',
  ]);
  assert.deepEqual(first.logicalProjection?.runtimeLeafObservations, [
    { runtimeInstanceId: 'runtime-a', leafEntryId: 'e5' },
    { runtimeInstanceId: 'runtime-b', leafEntryId: 'e3' },
  ]);
  assert.deepEqual(reversed.logicalProjection?.eligibleContributorEntryIds.toSorted(), [
    'e2',
    'e3',
    'e4',
    'e5',
  ]);
});

void test('eligibility unions only grow and reducer replay is idempotent for arbitrary valid marker sequences', () => {
  const entries = [
    descriptor('e1', null),
    descriptor('e2', 'e1'),
    descriptor('e3', 'e2'),
    descriptor('e4', 'e1'),
    descriptor('e5', 'e4'),
    descriptor('c6', 'e5', 'compaction', 'e5'),
  ];
  fc.assert(
    fc.property(
      fc.array(fc.boolean(), { minLength: entries.length, maxLength: entries.length }),
      fc.array(fc.constantFrom('activity', 'arrival', 'branch', 'compaction', 'departure'), {
        maxLength: 20,
      }),
      (eligibleFlags, triggerKinds) => {
        const initiallyEligible = entries
          .filter((entry, index) => eligibleFlags[index])
          .map(({ entryId }) => entryId);
        const markers = triggerKinds.map((kind, index) => {
          let trigger: RecallWorkMarkerTriggerPayload;
          switch (kind) {
            case 'branch':
              trigger = {
                kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
                logicalSessionId: 'logical',
                oldLeafEntryId: 'e3',
                newLeafEntryId: 'e5',
              };
              break;
            case 'compaction':
              trigger = {
                kind: RecallWorkMarkerTrigger.COMPACTION,
                logicalSessionId: 'logical',
                compactionEntryId: 'c6',
              };
              break;
            case 'departure':
              trigger = { kind: RecallWorkMarkerTrigger.DEPARTURE };
              break;
            case 'arrival':
              trigger = { kind: RecallWorkMarkerTrigger.ARRIVAL };
              break;
            case 'activity':
              trigger = { kind: RecallWorkMarkerTrigger.ACTIVITY };
              break;
            default:
              throw new Error('Reducer property trigger unsupported');
          }
          return marker(`property-${index}`, trigger, 'property-runtime', index + 1);
        });
        const once = reduceRecallEligibility({
          physicalProjection: physical(),
          logicalProjection: logical(entries, 'c6', initiallyEligible),
          markers,
          quiescenceObserved: false,
        });
        assert.ok(once.logicalProjection);
        for (const entryId of initiallyEligible) {
          assert.equal(once.logicalProjection.eligibleContributorEntryIds.includes(entryId), true);
        }
        const replay = reduceRecallEligibility({
          physicalProjection: physical(),
          logicalProjection: once.logicalProjection,
          markers,
          quiescenceObserved: false,
        });
        assert.deepEqual(replay.logicalProjection, once.logicalProjection);
      },
    ),
    { numRuns: 100 },
  );
});

void test('marker interleavings preserving each runtime sequence produce the same monotonic union', () => {
  const projection = logical([
    descriptor('e1', null),
    descriptor('e2', 'e1'),
    descriptor('e3', 'e2'),
    descriptor('e4', 'e1'),
    descriptor('e5', 'e4'),
  ]);
  const runtimeA = [
    marker(
      'a1',
      {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        logicalSessionId: 'logical',
        oldLeafEntryId: 'e3',
        newLeafEntryId: 'e5',
      },
      'runtime-a',
      1,
    ),
    marker('a2', { kind: RecallWorkMarkerTrigger.DEPARTURE }, 'runtime-a', 2),
  ];
  const runtimeB = [
    marker(
      'b1',
      {
        kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
        logicalSessionId: 'logical',
        oldLeafEntryId: 'e5',
        newLeafEntryId: 'e3',
      },
      'runtime-b',
      1,
    ),
  ];
  const interleavings = [
    [runtimeA[0], runtimeA[1], runtimeB[0]],
    [runtimeA[0], runtimeB[0], runtimeA[1]],
    [runtimeB[0], runtimeA[0], runtimeA[1]],
  ];
  fc.assert(
    fc.property(fc.integer({ min: 0, max: interleavings.length - 1 }), (index) => {
      const markers =
        interleavings[index]?.filter(
          (candidate): candidate is RecallWorkMarker => candidate !== undefined,
        ) ?? [];
      const result = reduceRecallEligibility({
        physicalProjection: physical(),
        logicalProjection: projection,
        markers,
        quiescenceObserved: false,
      });
      assert.deepEqual(result.logicalProjection?.eligibleContributorEntryIds.toSorted(), [
        'e1',
        'e2',
        'e3',
        'e4',
        'e5',
      ]);
    }),
  );
});

void test('confirmed deletion is the sole transition that retracts eligibility and projection state', () => {
  const result = reduceRecallEligibility({
    physicalProjection: physical(RecallSourceAvailability.DELETION_CONFIRMED),
    logicalProjection: logical(linear, 'e3', ['e1', 'e2']),
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(result.deletionConfirmed, true);
  assert.equal(result.logicalProjection, null);
  assert.deepEqual(result.newlyEligibleContributorEntryIds, []);
});

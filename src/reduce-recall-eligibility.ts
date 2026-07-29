import {
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  readProjectedActiveContextPath,
  readProjectedRecallSessionEntryPath,
} from './parse-recall-session-record.js';
import {
  mergeRecallMarkerCheckpoint,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
  type RecallEligibleSourceSpan,
  type RecallProjectedEntryDescriptor,
} from './recall-session-projection.js';
import {
  isBoundedRecallDepartureMarkerTrigger,
  type RecallWorkMarker,
} from './recall-work-marker.js';

/** Version of monotonic compaction, branch-exit, departure, and quiescence eligibility rules. */
export const INCREMENTAL_RECALL_ELIGIBILITY_POLICY_VERSION = 1;

/** Ordered lifecycle facts applied to one logical projection without I/O or global runtime authority. */
export interface ReduceRecallEligibilityInput {
  physicalProjection: PhysicalSessionProjection;
  logicalProjection: LogicalSessionProjection;
  markers: readonly RecallWorkMarker[];
  quiescenceObserved: boolean;
}

/** Monotonic contributor union and newly crossed source spans for one reducer application. */
export interface ReduceRecallEligibilityResult {
  logicalProjection: LogicalSessionProjection | null;
  newlyEligibleContributorEntryIds: string[];
  newlyEligibleSpans: RecallEligibleSourceSpan[];
  deletionConfirmed: boolean;
}

function addCompactionEligibility(
  compactionEntryId: string,
  entriesById: ReadonlyMap<string, RecallProjectedEntryDescriptor>,
  eligibleEntryIds: Set<string>,
): RecallProjectedEntryDescriptor | null {
  const compaction = entriesById.get(compactionEntryId);
  if (compaction?.entryType !== 'compaction') {
    return null;
  }
  const ancestorPath =
    compaction.parentEntryId === null
      ? []
      : readProjectedRecallSessionEntryPath(compaction.parentEntryId, entriesById);
  if (ancestorPath === null) {
    return null;
  }
  let summarized = ancestorPath;
  if (!compaction.hasRetainedTail) {
    const firstKeptIndex = ancestorPath.findIndex(
      ({ entryId }) => entryId === compaction.firstKeptEntryId,
    );
    if (firstKeptIndex < 0) {
      return null;
    }
    summarized = ancestorPath.slice(0, firstKeptIndex);
  }
  for (const entry of summarized) {
    eligibleEntryIds.add(entry.entryId);
  }
  for (const priorCompaction of ancestorPath.filter(
    ({ entryType }) => entryType === 'compaction',
  )) {
    eligibleEntryIds.add(priorCompaction.entryId);
  }
  eligibleEntryIds.add(compaction.entryId);
  return compaction;
}

function addBranchExitEligibility(
  oldLeafEntryId: string | null,
  newLeafEntryId: string | null,
  entriesById: ReadonlyMap<string, RecallProjectedEntryDescriptor>,
  eligibleEntryIds: Set<string>,
  summaryEntryId?: string,
): boolean {
  const oldPath =
    oldLeafEntryId === null ? [] : readProjectedRecallSessionEntryPath(oldLeafEntryId, entriesById);
  const newPath =
    newLeafEntryId === null ? [] : readProjectedRecallSessionEntryPath(newLeafEntryId, entriesById);
  if (oldPath === null || newPath === null) {
    return false;
  }
  const newPathIds = new Set(newPath.map(({ entryId }) => entryId));
  for (const entry of oldPath) {
    if (!newPathIds.has(entry.entryId)) {
      eligibleEntryIds.add(entry.entryId);
    }
  }
  if (summaryEntryId !== undefined) {
    if (!entriesById.has(summaryEntryId)) {
      return false;
    }
    eligibleEntryIds.add(summaryEntryId);
  }
  return true;
}

function createEligibleSourceSpan(
  descriptor: RecallProjectedEntryDescriptor,
): RecallEligibleSourceSpan {
  return {
    startByte: descriptor.startByte,
    endByte: descriptor.endByte,
    startEntryId: descriptor.entryId,
    endEntryId: descriptor.entryId,
    contributorEntryIds: [descriptor.entryId],
  };
}

function markLogicalProjectionMalformed(
  logicalProjection: LogicalSessionProjection,
): LogicalSessionProjection {
  return {
    ...logicalProjection,
    repairState: RecallProjectionRepairState.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.MALFORMED_GRAPH,
  };
}

/** Unions compaction, branch-exit, departure, and quiescence eligibility without selecting a global runtime leaf. */
export function reduceRecallEligibility(
  input: ReduceRecallEligibilityInput,
): ReduceRecallEligibilityResult {
  if (input.physicalProjection.sourceAvailability === RecallSourceAvailability.DELETION_CONFIRMED) {
    return {
      logicalProjection: null,
      newlyEligibleContributorEntryIds: [],
      newlyEligibleSpans: [],
      deletionConfirmed: true,
    };
  }
  const projection = input.logicalProjection;
  const entriesById = new Map(
    projection.entryDescriptors.map((descriptor) => [descriptor.entryId, descriptor]),
  );
  const previousEligibleEntryIds = new Set(projection.eligibleContributorEntryIds);
  const eligibleEntryIds = new Set(previousEligibleEntryIds);
  const coveredMarkerIds = new Set(projection.markerCheckpoint.coveredMarkerIds);
  const unprocessedMarkers = input.markers.filter(
    (marker) => !coveredMarkerIds.has(marker.markerId),
  );
  let malformed = false;
  let compactionBoundary = projection.compactionBoundary;
  const runtimeLeafEntryIds = new Map(
    projection.runtimeLeafObservations.map(({ runtimeInstanceId, leafEntryId }) => [
      runtimeInstanceId,
      leafEntryId,
    ]),
  );
  const preservedBranchExits = [...projection.preservedBranchExits];
  for (const marker of unprocessedMarkers) {
    switch (marker.trigger.kind) {
      case RecallWorkMarkerTrigger.ACTIVITY:
      case RecallWorkMarkerTrigger.ARRIVAL:
        break;
      case RecallWorkMarkerTrigger.COMPACTION: {
        const compaction = addCompactionEligibility(
          marker.trigger.compactionEntryId,
          entriesById,
          eligibleEntryIds,
        );
        if (compaction === null) {
          malformed = true;
        } else {
          compactionBoundary = {
            compactionEntryId: compaction.entryId,
            firstRetainedEntryId: compaction.firstKeptEntryId ?? compaction.entryId,
          };
          runtimeLeafEntryIds.set(marker.runtimeInstanceId, compaction.entryId);
        }
        break;
      }
      case RecallWorkMarkerTrigger.BRANCH_EXIT:
        if (
          !addBranchExitEligibility(
            marker.trigger.oldLeafEntryId,
            marker.trigger.newLeafEntryId,
            entriesById,
            eligibleEntryIds,
            marker.trigger.summaryEntryId,
          )
        ) {
          malformed = true;
        } else {
          preservedBranchExits.push({
            oldLeafEntryId: marker.trigger.oldLeafEntryId,
            newLeafEntryId: marker.trigger.newLeafEntryId,
            summaryEntryId: marker.trigger.summaryEntryId ?? null,
          });
          runtimeLeafEntryIds.set(marker.runtimeInstanceId, marker.trigger.newLeafEntryId);
        }
        break;
      case RecallWorkMarkerTrigger.DEPARTURE: {
        const departureLeafEntryId = isBoundedRecallDepartureMarkerTrigger(marker.trigger)
          ? marker.trigger.leafEntryId
          : (runtimeLeafEntryIds.get(marker.runtimeInstanceId) ?? null);
        if (departureLeafEntryId === null) {
          // Unbounded departures without a prior runtime leaf must not fall back to the
          // projection's latest effective leaf: later appends from another runtime would
          // inherit the shorter explicit-exit window.
          malformed = true;
          break;
        }
        const activeContext = readProjectedActiveContextPath(departureLeafEntryId, entriesById);
        if (activeContext === null) {
          malformed = true;
        } else {
          for (const entry of activeContext) {
            eligibleEntryIds.add(entry.entryId);
          }
        }
        break;
      }
      default:
        malformed = true;
    }
  }
  if (input.quiescenceObserved) {
    const observedLeaves = new Set<string | null>([
      projection.effectiveLeafEntryId,
      ...runtimeLeafEntryIds.values(),
    ]);
    for (const observedLeaf of observedLeaves) {
      const activeContext = readProjectedActiveContextPath(observedLeaf, entriesById);
      if (activeContext === null) {
        malformed = true;
        break;
      }
      for (const entry of activeContext) {
        eligibleEntryIds.add(entry.entryId);
      }
    }
  }
  if (malformed) {
    return {
      logicalProjection: markLogicalProjectionMalformed(projection),
      newlyEligibleContributorEntryIds: [],
      newlyEligibleSpans: [],
      deletionConfirmed: false,
    };
  }

  const newlyEligibleDescriptors = projection.entryDescriptors.filter(
    ({ entryId }) => eligibleEntryIds.has(entryId) && !previousEligibleEntryIds.has(entryId),
  );
  const newlyEligibleSpans = newlyEligibleDescriptors.map(createEligibleSourceSpan);
  return {
    logicalProjection: {
      ...projection,
      compactionBoundary,
      runtimeLeafObservations: [...runtimeLeafEntryIds.entries()]
        .map(([runtimeInstanceId, leafEntryId]) => ({ runtimeInstanceId, leafEntryId }))
        .toSorted((left, right) => left.runtimeInstanceId.localeCompare(right.runtimeInstanceId)),
      preservedBranchExits,
      eligibleContributorEntryIds: projection.entryDescriptors
        .map(({ entryId }) => entryId)
        .filter((entryId) => eligibleEntryIds.has(entryId)),
      eligibleSpans: [...projection.eligibleSpans, ...newlyEligibleSpans],
      markerCheckpoint: mergeRecallMarkerCheckpoint({
        generationId: projection.markerCheckpoint.generationId,
        current: projection.markerCheckpoint,
        coveredMarkerIds: unprocessedMarkers.map(({ markerId }) => markerId),
        runtimeSequences: unprocessedMarkers.map(({ runtimeInstanceId, runtimeSequence }) => ({
          runtimeInstanceId,
          sequence: runtimeSequence,
        })),
      }),
    },
    newlyEligibleContributorEntryIds: newlyEligibleDescriptors.map(({ entryId }) => entryId),
    newlyEligibleSpans,
    deletionConfirmed: false,
  };
}

import {
  RecallConfirmedDeletionDecisionKind,
  RecallConfirmedDeletionHaltCategory,
  RecallConfirmedDeletionPhase,
  RecallMetadataSweepStatus,
  RecallProjectionRepairState,
  RecallSourceAvailability,
} from './enums.js';
import type { PhysicalSessionProjection } from './recall-session-projection.js';

/** Stable device and inode identity observed for one present physical session source. */
export interface ConfirmedSessionDeletionSourceObservation {
  sourceDevice: string;
  sourceInode: string;
}

/** Complete metadata sweep facts used by the pure confirmed source deletion policy. */
export interface DecideConfirmedSessionDeletionInput {
  projection: PhysicalSessionProjection;
  sweepId: string;
  sweepStatus: RecallMetadataSweepStatus;
  observedAtEpochMilliseconds: number;
  sourceObservation: ConfirmedSessionDeletionSourceObservation | null;
}

/** A confirmed source deletion policy decision that leaves durable state unchanged. */
export interface UnchangedConfirmedSessionDeletionDecision {
  kind:
    | RecallConfirmedDeletionDecisionKind.NO_CHANGE
    | RecallConfirmedDeletionDecisionKind.RESUME_CONFIRMED_DELETION;
}

/** A confirmed source deletion policy decision carrying the next durable physical projection. */
export interface UpdatedConfirmedSessionDeletionDecision {
  kind:
    | RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING
    | RecallConfirmedDeletionDecisionKind.CLEAR_SOURCE_MISSING
    | RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION;
  nextProjection: PhysicalSessionProjection;
}

/** A privacy-safe confirmed source deletion policy halt. */
export interface HaltedConfirmedSessionDeletionDecision {
  kind: RecallConfirmedDeletionDecisionKind.HALT;
  haltCategory: RecallConfirmedDeletionHaltCategory;
}

/** Exhaustive result of the pure confirmed source deletion policy. */
export type ConfirmedSessionDeletionDecision =
  | UnchangedConfirmedSessionDeletionDecision
  | UpdatedConfirmedSessionDeletionDecision
  | HaltedConfirmedSessionDeletionDecision;

function haltCategoryForSweepStatus(
  status: RecallMetadataSweepStatus,
): RecallConfirmedDeletionHaltCategory | null {
  switch (status) {
    case RecallMetadataSweepStatus.COMPLETE:
      return null;
    case RecallMetadataSweepStatus.ROOT_UNAVAILABLE:
      return RecallConfirmedDeletionHaltCategory.ROOT_UNAVAILABLE;
    case RecallMetadataSweepStatus.PERMISSION_DENIED:
      return RecallConfirmedDeletionHaltCategory.PERMISSION_DENIED;
    case RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS:
      return RecallConfirmedDeletionHaltCategory.SUSPICIOUS_MASS_LOSS;
    case RecallMetadataSweepStatus.CONTINUATION_REQUIRED:
      return RecallConfirmedDeletionHaltCategory.INCOMPLETE_SWEEP;
    default:
      return RecallConfirmedDeletionHaltCategory.INCOMPLETE_SWEEP;
  }
}

function sourceIdentityMatches(
  projection: PhysicalSessionProjection,
  observation: ConfirmedSessionDeletionSourceObservation,
): boolean {
  return (
    projection.sourceDevice === observation.sourceDevice &&
    projection.sourceInode === observation.sourceInode
  );
}

function clearSourceMissingProjection(
  projection: PhysicalSessionProjection,
): PhysicalSessionProjection {
  return {
    ...projection,
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
  };
}

/**
 * Decides source deletion from one complete metadata sweep without I/O.
 * Confirmation requires a distinct later healthy sweep and unchanged device/inode identity.
 */
export function decideConfirmedSessionDeletion(
  input: DecideConfirmedSessionDeletionInput,
): ConfirmedSessionDeletionDecision {
  const haltCategory = haltCategoryForSweepStatus(input.sweepStatus);
  if (haltCategory !== null) {
    return { kind: RecallConfirmedDeletionDecisionKind.HALT, haltCategory };
  }
  if (input.projection.repairState !== RecallProjectionRepairState.READY) {
    return {
      kind: RecallConfirmedDeletionDecisionKind.HALT,
      haltCategory: RecallConfirmedDeletionHaltCategory.PROJECTION_REQUIRES_RECONCILIATION,
    };
  }

  if (input.sourceObservation !== null) {
    if (!sourceIdentityMatches(input.projection, input.sourceObservation)) {
      return {
        kind: RecallConfirmedDeletionDecisionKind.HALT,
        haltCategory: RecallConfirmedDeletionHaltCategory.SOURCE_IDENTITY_CHANGED,
      };
    }
    if (input.projection.sourceAvailability === RecallSourceAvailability.DELETION_CONFIRMED) {
      return {
        kind: RecallConfirmedDeletionDecisionKind.HALT,
        haltCategory: RecallConfirmedDeletionHaltCategory.SOURCE_REAPPEARED_DURING_DELETION,
      };
    }
    if (input.projection.sourceAvailability === RecallSourceAvailability.SOURCE_MISSING) {
      return {
        kind: RecallConfirmedDeletionDecisionKind.CLEAR_SOURCE_MISSING,
        nextProjection: clearSourceMissingProjection(input.projection),
      };
    }
    return { kind: RecallConfirmedDeletionDecisionKind.NO_CHANGE };
  }

  if (input.projection.sourceAvailability === RecallSourceAvailability.PRESENT) {
    return {
      kind: RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING,
      nextProjection: {
        ...input.projection,
        sourceAvailability: RecallSourceAvailability.SOURCE_MISSING,
        sourceMissingObservedAtEpochMilliseconds: input.observedAtEpochMilliseconds,
        sourceMissingObservationCount: 1,
        sourceMissingSweepId: input.sweepId,
        deletionCheckpoint: null,
      },
    };
  }
  if (input.projection.sourceAvailability === RecallSourceAvailability.DELETION_CONFIRMED) {
    return { kind: RecallConfirmedDeletionDecisionKind.RESUME_CONFIRMED_DELETION };
  }
  if (input.projection.sourceMissingSweepId === input.sweepId) {
    return { kind: RecallConfirmedDeletionDecisionKind.NO_CHANGE };
  }
  return {
    kind: RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION,
    nextProjection: {
      ...input.projection,
      sourceAvailability: RecallSourceAvailability.DELETION_CONFIRMED,
      sourceMissingObservationCount: input.projection.sourceMissingObservationCount + 1,
      sourceMissingSweepId: input.sweepId,
      deletionCheckpoint: {
        confirmedSweepId: input.sweepId,
        phase: RecallConfirmedDeletionPhase.EVIDENCE,
        deletedEvidenceCount: 0,
        deletedLogicalProjectionCount: 0,
        pendingEvidenceIds: [],
        pendingLogicalProjectionIds: [],
      },
    },
  };
}

/** Counts and categories from one confirmed deletion reconciliation without source paths or content. */
export interface ConfirmedSessionDeletionReconciliationResult {
  halted: boolean;
  consideredPhysicalSessionCount: number;
  sourceMissingRecordedCount: number;
  sourceMissingClearedCount: number;
  confirmedSourceDeletionCount: number;
  removedEvidenceOccurrenceCount: number;
  removedLogicalProjectionCount: number;
  removedPhysicalProjectionCount: number;
  acknowledgedCheckpointCount: number;
  haltCategoryCounts: Readonly<Partial<Record<RecallConfirmedDeletionHaltCategory, number>>>;
}

import {
  RecallConfirmedDeletionDecisionKind,
  RecallConfirmedDeletionHaltCategory,
  RecallConfirmedDeletionPhase,
  RecallMetadataSweepStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';
import {
  coordinateRecallWriteWindow,
  type RecallWriteWindow,
} from './coordinate-recall-write-window.js';
import {
  readRecallActiveGenerationSelection,
  type RecallActiveGenerationSelection,
} from './recall-generation-state.js';
import {
  createLogicalSessionProjectionId,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import { acknowledgeCoveredRecallMarkers } from './recall-marker-spool.js';
import type { RecallSessionMetadataSweepResult } from './scan-recall-session-metadata.js';
import {
  CONFIRMED_DELETION_BATCH_SIZE,
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';
import {
  openZvecSessionProjectionStore,
  type ZvecSessionProjectionStore,
} from './zvec-session-projection-store.js';

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

function physicalProjectionNeedsDeletionReconciliation(
  projection: PhysicalSessionProjection,
  metadataSweep: RecallSessionMetadataSweepResult,
): boolean {
  if (
    metadataSweep.missingPhysicalSessionIds.includes(projection.physicalSessionId) ||
    projection.sourceAvailability !== RecallSourceAvailability.PRESENT
  ) {
    return true;
  }
  const observedIdentity = metadataSweep.observedKnownSourceIdentities.find(
    ({ physicalSessionId }) => physicalSessionId === projection.physicalSessionId,
  );
  return observedIdentity !== undefined && !sourceIdentityMatches(projection, observedIdentity);
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

/** Scalar progress emitted only after one durable confirmed deletion write window. */
export interface ConfirmedSessionDeletionProgress {
  phase: RecallConfirmedDeletionPhase;
  removedEvidenceOccurrenceCount: number;
  removedLogicalProjectionCount: number;
  removedPhysicalProjectionCount: number;
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

/** Pointer, sweep, stores, and scalar callbacks for one resumable confirmed deletion reconciliation. */
export interface ReconcileConfirmedSessionDeletionOptions {
  metadataSweep: RecallSessionMetadataSweepResult;
  physicalProjections: readonly PhysicalSessionProjection[];
  activeGenerationPointerPath: string;
  generationRootDirectory: string;
  lockPath: string;
  embeddingDimensions: number;
  markerWorkPlans?: readonly RecallMarkerReplayWorkPlan[];
  signal?: AbortSignal;
  acknowledgeCheckpoint?: (
    confirmedSweepId: string,
    physicalSessionProjectionId: string,
  ) => Promise<void>;
  onProgress?: (progress: ConfirmedSessionDeletionProgress) => void | Promise<void>;
  onDiagnostic?: (result: ConfirmedSessionDeletionReconciliationResult) => void;
}

interface MutableConfirmedSessionDeletionCounts {
  sourceMissingRecordedCount: number;
  sourceMissingClearedCount: number;
  confirmedSourceDeletionCount: number;
  removedEvidenceOccurrenceCount: number;
  removedLogicalProjectionCount: number;
  removedPhysicalProjectionCount: number;
  acknowledgedCheckpointCount: number;
}

type ConfirmedDeletionWindowOutcome =
  | { action: 'no_change' }
  | { action: 'source_missing_recorded' }
  | { action: 'source_missing_cleared' }
  | { action: 'deletion_confirmed' }
  | { action: 'batch_checkpointed' }
  | {
      action: 'evidence_removed';
      progress: ConfirmedSessionDeletionProgress;
    }
  | {
      action: 'logical_projections_removed';
      progress: ConfirmedSessionDeletionProgress;
    }
  | { action: 'phase_advanced' }
  | {
      action: 'physical_projection_removed';
      confirmedSweepId: string;
      progress: ConfirmedSessionDeletionProgress;
    }
  | { action: 'already_removed'; confirmedSweepId: string }
  | { action: 'halted'; haltCategory: RecallConfirmedDeletionHaltCategory };

function createConfirmedDeletionCounts(): MutableConfirmedSessionDeletionCounts {
  return {
    sourceMissingRecordedCount: 0,
    sourceMissingClearedCount: 0,
    confirmedSourceDeletionCount: 0,
    removedEvidenceOccurrenceCount: 0,
    removedLogicalProjectionCount: 0,
    removedPhysicalProjectionCount: 0,
    acknowledgedCheckpointCount: 0,
  };
}

function closeConfirmedDeletionStores(
  evidenceStore: ZvecConversationStore,
  projectionStore: ZvecSessionProjectionStore,
  writeWindow: RecallWriteWindow,
): void {
  const closeErrors: Error[] = [];
  try {
    projectionStore.close();
  } catch (error) {
    closeErrors.push(
      error instanceof Error
        ? error
        : new Error('Recall confirmed deletion projection store close failed', { cause: error }),
    );
  }
  try {
    evidenceStore.close();
  } catch (error) {
    closeErrors.push(
      error instanceof Error
        ? error
        : new Error('Recall confirmed deletion evidence store close failed', { cause: error }),
    );
  }
  if (closeErrors.length > 0) {
    writeWindow.retainRecoveryRequired();
    throw new AggregateError(closeErrors, 'Recall confirmed deletion store close failed');
  }
}

function createConfirmedDeletionResult(
  halted: boolean,
  consideredPhysicalSessionCount: number,
  counts: MutableConfirmedSessionDeletionCounts,
  haltCategory?: RecallConfirmedDeletionHaltCategory,
): ConfirmedSessionDeletionReconciliationResult {
  return Object.freeze({
    halted,
    consideredPhysicalSessionCount,
    ...counts,
    haltCategoryCounts: Object.freeze(haltCategory === undefined ? {} : { [haltCategory]: 1 }),
  });
}

function defaultConfirmedDeletionEvidenceStore(
  selection: RecallActiveGenerationSelection,
  embeddingDimensions: number,
): ZvecConversationStore {
  return openZvecConversationStore({
    databasePath: selection.databasePath,
    dimensions: embeddingDimensions,
    createIfMissing: false,
  });
}

function defaultConfirmedDeletionProjectionStore(
  selection: RecallActiveGenerationSelection,
  readOnly: boolean,
): ZvecSessionProjectionStore {
  return openZvecSessionProjectionStore({
    databasePath: selection.projectionDatabasePath,
    generationId: selection.activeGenerationId,
    createIfMissing: false,
    readOnly,
  });
}

function findConfirmedDeletionMarkerWorkPlan(
  options: ReconcileConfirmedSessionDeletionOptions,
  physicalSessionId: string,
): RecallMarkerReplayWorkPlan | undefined {
  return options.markerWorkPlans?.find(
    (workPlan) => workPlan.workItems[0]?.marker.physicalSessionId === physicalSessionId,
  );
}

function coverConfirmedDeletionMarkers(
  projection: PhysicalSessionProjection,
  workPlan?: RecallMarkerReplayWorkPlan,
): PhysicalSessionProjection {
  if (workPlan === undefined) {
    return projection;
  }
  const coveredMarkerIds = new Set(projection.markerCheckpoint.coveredMarkerIds);
  const runtimeSequences = new Map(
    projection.markerCheckpoint.runtimeSequences.map(({ runtimeInstanceId, sequence }) => [
      runtimeInstanceId,
      sequence,
    ]),
  );
  for (const workItem of workPlan.workItems) {
    for (const markerId of workItem.coveredMarkerIds) {
      coveredMarkerIds.add(markerId);
    }
    runtimeSequences.set(
      workItem.marker.runtimeInstanceId,
      Math.max(
        runtimeSequences.get(workItem.marker.runtimeInstanceId) ?? 0,
        workItem.marker.runtimeSequence,
      ),
    );
  }
  return {
    ...projection,
    markerCheckpoint: {
      generationId: projection.generationId,
      coveredMarkerIds: [...coveredMarkerIds].toSorted(),
      runtimeSequences: [...runtimeSequences.entries()]
        .map(([runtimeInstanceId, sequence]) => ({ runtimeInstanceId, sequence }))
        .toSorted((left, right) => left.runtimeInstanceId.localeCompare(right.runtimeInstanceId)),
    },
  };
}

function requireConfirmedDeletionCheckpoint(projection: PhysicalSessionProjection) {
  if (projection.deletionCheckpoint === null) {
    throw new Error('Recall confirmed deletion checkpoint missing from confirmed projection');
  }
  return projection.deletionCheckpoint;
}

async function runConfirmedDeletionWriteWindow(
  options: ReconcileConfirmedSessionDeletionOptions,
  targetGenerationId: string,
  physicalProjectionId: string,
  sourceObservation: ConfirmedSessionDeletionSourceObservation | null,
): Promise<ConfirmedDeletionWindowOutcome> {
  return coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: true,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      const selection = await readRecallActiveGenerationSelection(
        options.activeGenerationPointerPath,
        options.generationRootDirectory,
      );
      if (selection.activeGenerationId !== targetGenerationId) {
        return {
          action: 'halted',
          haltCategory: RecallConfirmedDeletionHaltCategory.ACTIVE_GENERATION_CHANGED,
        };
      }
      const evidenceStore = defaultConfirmedDeletionEvidenceStore(
        selection,
        options.embeddingDimensions,
      );
      const projectionStore = defaultConfirmedDeletionProjectionStore(selection, false);
      try {
        const projection = projectionStore
          .fetchProjections([physicalProjectionId])
          .get(physicalProjectionId);
        if (projection === undefined) {
          return { action: 'already_removed', confirmedSweepId: options.metadataSweep.sweepId };
        }
        if (projection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
          return {
            action: 'halted',
            haltCategory: RecallConfirmedDeletionHaltCategory.PROJECTION_REQUIRES_RECONCILIATION,
          };
        }
        const markerWorkPlan = findConfirmedDeletionMarkerWorkPlan(
          options,
          projection.physicalSessionId,
        );
        const decision = decideConfirmedSessionDeletion({
          projection: coverConfirmedDeletionMarkers(projection, markerWorkPlan),
          sweepId: options.metadataSweep.sweepId,
          sweepStatus: options.metadataSweep.status,
          observedAtEpochMilliseconds: Date.now(),
          sourceObservation,
        });
        switch (decision.kind) {
          case RecallConfirmedDeletionDecisionKind.HALT:
            return { action: 'halted', haltCategory: decision.haltCategory };
          case RecallConfirmedDeletionDecisionKind.NO_CHANGE:
            return { action: 'no_change' };
          case RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING:
            await projectionStore.upsertProjections([decision.nextProjection]);
            return { action: 'source_missing_recorded' };
          case RecallConfirmedDeletionDecisionKind.CLEAR_SOURCE_MISSING:
            await projectionStore.upsertProjections([decision.nextProjection]);
            return { action: 'source_missing_cleared' };
          case RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION:
            await projectionStore.upsertProjections([decision.nextProjection]);
            return { action: 'deletion_confirmed' };
          case RecallConfirmedDeletionDecisionKind.RESUME_CONFIRMED_DELETION:
            break;
          default:
            return { action: 'no_change' };
        }

        const checkpoint = requireConfirmedDeletionCheckpoint(projection);
        switch (checkpoint.phase) {
          case RecallConfirmedDeletionPhase.EVIDENCE: {
            if (checkpoint.pendingEvidenceIds.length === 0) {
              const evidenceIds = await evidenceStore.listChunkIdsByPhysicalSessionProjectionId(
                projection.projectionId,
                CONFIRMED_DELETION_BATCH_SIZE,
              );
              if (evidenceIds.length === 0) {
                await projectionStore.upsertProjections([
                  {
                    ...projection,
                    deletionCheckpoint: {
                      ...checkpoint,
                      phase: RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS,
                    },
                  },
                ]);
                return { action: 'phase_advanced' };
              }
              await projectionStore.upsertProjections([
                {
                  ...projection,
                  deletionCheckpoint: { ...checkpoint, pendingEvidenceIds: evidenceIds },
                },
              ]);
              return { action: 'batch_checkpointed' };
            }
            const remainingEvidenceIds = [
              ...evidenceStore.fetchConversationChunks(checkpoint.pendingEvidenceIds).keys(),
            ];
            await evidenceStore.deleteChunks(remainingEvidenceIds);
            const deletedEvidenceCount =
              checkpoint.deletedEvidenceCount + checkpoint.pendingEvidenceIds.length;
            await projectionStore.upsertProjections([
              {
                ...projection,
                deletionCheckpoint: {
                  ...checkpoint,
                  deletedEvidenceCount,
                  pendingEvidenceIds: [],
                },
              },
            ]);
            return {
              action: 'evidence_removed',
              progress: {
                phase: RecallConfirmedDeletionPhase.EVIDENCE,
                removedEvidenceOccurrenceCount: deletedEvidenceCount,
                removedLogicalProjectionCount: checkpoint.deletedLogicalProjectionCount,
                removedPhysicalProjectionCount: 0,
              },
            };
          }
          case RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS: {
            if (checkpoint.pendingLogicalProjectionIds.length === 0) {
              const logicalProjectionIds = projection.logicalSessionIds.map((logicalSessionId) =>
                createLogicalSessionProjectionId(projection.physicalSessionId, logicalSessionId),
              );
              const existingLogicalProjectionIds = [
                ...projectionStore.fetchProjections(logicalProjectionIds).keys(),
              ]
                .toSorted()
                .slice(0, CONFIRMED_DELETION_BATCH_SIZE);
              if (existingLogicalProjectionIds.length === 0) {
                await projectionStore.upsertProjections([
                  {
                    ...projection,
                    deletionCheckpoint: {
                      ...checkpoint,
                      phase: RecallConfirmedDeletionPhase.PHYSICAL_PROJECTION,
                    },
                  },
                ]);
                return { action: 'phase_advanced' };
              }
              await projectionStore.upsertProjections([
                {
                  ...projection,
                  deletionCheckpoint: {
                    ...checkpoint,
                    pendingLogicalProjectionIds: existingLogicalProjectionIds,
                  },
                },
              ]);
              return { action: 'batch_checkpointed' };
            }
            const remainingLogicalProjectionIds = [
              ...projectionStore.fetchProjections(checkpoint.pendingLogicalProjectionIds).keys(),
            ];
            await projectionStore.deleteProjections(remainingLogicalProjectionIds);
            const deletedLogicalProjectionCount =
              checkpoint.deletedLogicalProjectionCount +
              checkpoint.pendingLogicalProjectionIds.length;
            await projectionStore.upsertProjections([
              {
                ...projection,
                deletionCheckpoint: {
                  ...checkpoint,
                  deletedLogicalProjectionCount,
                  pendingLogicalProjectionIds: [],
                },
              },
            ]);
            return {
              action: 'logical_projections_removed',
              progress: {
                phase: RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS,
                removedEvidenceOccurrenceCount: checkpoint.deletedEvidenceCount,
                removedLogicalProjectionCount: deletedLogicalProjectionCount,
                removedPhysicalProjectionCount: 0,
              },
            };
          }
          case RecallConfirmedDeletionPhase.PHYSICAL_PROJECTION:
            await projectionStore.deleteProjections([projection.projectionId]);
            return {
              action: 'physical_projection_removed',
              confirmedSweepId: checkpoint.confirmedSweepId,
              progress: {
                phase: RecallConfirmedDeletionPhase.PHYSICAL_PROJECTION,
                removedEvidenceOccurrenceCount: checkpoint.deletedEvidenceCount,
                removedLogicalProjectionCount: checkpoint.deletedLogicalProjectionCount,
                removedPhysicalProjectionCount: 1,
              },
            };
          default:
            return { action: 'no_change' };
        }
      } finally {
        closeConfirmedDeletionStores(evidenceStore, projectionStore, writeWindow);
      }
    },
  );
}

async function acknowledgeConfirmedDeletionMarkers(
  options: ReconcileConfirmedSessionDeletionOptions,
  projection: PhysicalSessionProjection,
  confirmedSweepId: string,
): Promise<void> {
  await options.acknowledgeCheckpoint?.(confirmedSweepId, projection.projectionId);
  const workPlan = findConfirmedDeletionMarkerWorkPlan(options, projection.physicalSessionId);
  if (workPlan !== undefined) {
    await acknowledgeCoveredRecallMarkers(
      workPlan,
      coverConfirmedDeletionMarkers(projection, workPlan).markerCheckpoint,
    );
  }
}

async function observeConfirmedPhysicalProjectionRemoved(
  options: ReconcileConfirmedSessionDeletionOptions,
  targetGenerationId: string,
  physicalProjectionId: string,
): Promise<boolean> {
  const selection = await readRecallActiveGenerationSelection(
    options.activeGenerationPointerPath,
    options.generationRootDirectory,
  );
  if (selection.activeGenerationId !== targetGenerationId) {
    return false;
  }
  const projectionStore = defaultConfirmedDeletionProjectionStore(selection, true);
  try {
    return !projectionStore.fetchProjections([physicalProjectionId]).has(physicalProjectionId);
  } finally {
    projectionStore.close();
  }
}

/**
 * Reconciles one complete metadata sweep against the pointer-selected active generation.
 * Every destructive batch is checkpointed before deletion and acknowledged only after absence is observed.
 */
export async function reconcileConfirmedSessionDeletion(
  options: ReconcileConfirmedSessionDeletionOptions,
): Promise<ConfirmedSessionDeletionReconciliationResult> {
  const counts = createConfirmedDeletionCounts();
  const sweepHaltCategory = haltCategoryForSweepStatus(options.metadataSweep.status);
  if (sweepHaltCategory !== null) {
    const result = createConfirmedDeletionResult(true, 0, counts, sweepHaltCategory);
    options.onDiagnostic?.(result);
    return result;
  }
  const projectionsNeedingReconciliation = options.physicalProjections.filter((projection) =>
    physicalProjectionNeedsDeletionReconciliation(projection, options.metadataSweep),
  );
  if (projectionsNeedingReconciliation.length === 0) {
    const result = createConfirmedDeletionResult(false, 0, counts);
    options.onDiagnostic?.(result);
    return result;
  }
  const targetSelection = await readRecallActiveGenerationSelection(
    options.activeGenerationPointerPath,
    options.generationRootDirectory,
  );
  let consideredPhysicalSessionCount = 0;

  for (const suppliedProjection of projectionsNeedingReconciliation) {
    consideredPhysicalSessionCount += 1;
    if (suppliedProjection.generationId !== targetSelection.activeGenerationId) {
      const result = createConfirmedDeletionResult(
        true,
        consideredPhysicalSessionCount,
        counts,
        RecallConfirmedDeletionHaltCategory.ACTIVE_GENERATION_CHANGED,
      );
      options.onDiagnostic?.(result);
      return result;
    }
    const observedIdentity = options.metadataSweep.observedKnownSourceIdentities.find(
      ({ physicalSessionId }) => physicalSessionId === suppliedProjection.physicalSessionId,
    );
    const sourceObservation = observedIdentity
      ? {
          sourceDevice: observedIdentity.sourceDevice,
          sourceInode: observedIdentity.sourceInode,
        }
      : null;

    while (true) {
      const outcome = await runConfirmedDeletionWriteWindow(
        options,
        targetSelection.activeGenerationId,
        suppliedProjection.projectionId,
        sourceObservation,
      );
      switch (outcome.action) {
        case 'halted': {
          const result = createConfirmedDeletionResult(
            true,
            consideredPhysicalSessionCount,
            counts,
            outcome.haltCategory,
          );
          options.onDiagnostic?.(result);
          return result;
        }
        case 'no_change':
          break;
        case 'source_missing_recorded':
          counts.sourceMissingRecordedCount += 1;
          break;
        case 'source_missing_cleared':
          counts.sourceMissingClearedCount += 1;
          break;
        case 'deletion_confirmed':
          counts.confirmedSourceDeletionCount += 1;
          continue;
        case 'batch_checkpointed':
        case 'phase_advanced':
          continue;
        case 'evidence_removed':
        case 'logical_projections_removed':
          await options.onProgress?.(Object.freeze({ ...outcome.progress }));
          continue;
        case 'physical_projection_removed':
          counts.removedEvidenceOccurrenceCount += outcome.progress.removedEvidenceOccurrenceCount;
          counts.removedLogicalProjectionCount += outcome.progress.removedLogicalProjectionCount;
          counts.removedPhysicalProjectionCount += 1;
          await options.onProgress?.(Object.freeze({ ...outcome.progress }));
          if (
            !(await observeConfirmedPhysicalProjectionRemoved(
              options,
              targetSelection.activeGenerationId,
              suppliedProjection.projectionId,
            ))
          ) {
            const result = createConfirmedDeletionResult(
              true,
              consideredPhysicalSessionCount,
              counts,
              RecallConfirmedDeletionHaltCategory.ACTIVE_GENERATION_CHANGED,
            );
            options.onDiagnostic?.(result);
            return result;
          }
          await acknowledgeConfirmedDeletionMarkers(
            options,
            suppliedProjection,
            outcome.confirmedSweepId,
          );
          counts.acknowledgedCheckpointCount += 1;
          break;
        case 'already_removed':
          await acknowledgeConfirmedDeletionMarkers(
            options,
            suppliedProjection,
            outcome.confirmedSweepId,
          );
          counts.acknowledgedCheckpointCount += 1;
          break;
        default:
          break;
      }
      break;
    }
  }

  const result = createConfirmedDeletionResult(false, consideredPhysicalSessionCount, counts);
  options.onDiagnostic?.(result);
  return result;
}

/** Formats scalar confirmed deletion maintenance output without source paths or conversation content. */
export function formatConfirmedSessionDeletionResult(
  result: ConfirmedSessionDeletionReconciliationResult,
): string {
  const haltCategories = Object.entries(result.haltCategoryCounts)
    .map(([category, count]) => `${category}:${count}`)
    .join(',');
  return [
    `halted=${result.halted}`,
    `consideredPhysicalSessions=${result.consideredPhysicalSessionCount}`,
    `sourceMissingRecorded=${result.sourceMissingRecordedCount}`,
    `sourceMissingCleared=${result.sourceMissingClearedCount}`,
    `confirmedSourceDeletions=${result.confirmedSourceDeletionCount}`,
    `removedEvidenceOccurrences=${result.removedEvidenceOccurrenceCount}`,
    `removedLogicalProjections=${result.removedLogicalProjectionCount}`,
    `removedPhysicalProjections=${result.removedPhysicalProjectionCount}`,
    `acknowledgedCheckpoints=${result.acknowledgedCheckpointCount}`,
    `haltCategories=${haltCategories || 'none'}`,
  ].join(' ');
}

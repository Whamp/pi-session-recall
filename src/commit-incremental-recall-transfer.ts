import { acknowledgeCoveredRecallMarkers } from './recall-marker-spool.js';
import {
  coordinateRecallWriteWindow,
  type RecallWriteWindow,
} from './coordinate-recall-write-window.js';
import type { PreparedIncrementalRecallTransfer } from './prepare-incremental-recall-transfer.js';
import type {
  RecallMarkerCheckpoint,
  RecallSessionProjection,
} from './recall-session-projection.js';
import {
  openZvecConversationStore,
  type IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';
import {
  openZvecSessionProjectionStore,
  type ZvecSessionProjectionStore,
} from './zvec-session-projection-store.js';

/** Maximum immutable evidence documents accepted by one recall write window. */
export const INCREMENTAL_RECALL_EVIDENCE_BATCH_SIZE = 32;

/** Checked evidence methods available inside an incremental recall write window. */
export interface IncrementalRecallCommitEvidenceStore {
  upsertChunks(chunks: readonly IndexedSessionConversationChunk[]): Promise<void>;
  close(): void;
}

/** Checked projection methods used for write commit and later checkpoint observation. */
export interface IncrementalRecallCommitProjectionStore {
  upsertProjections(projections: readonly RecallSessionProjection[]): Promise<void>;
  fetchProjections(projectionIds: readonly string[]): Map<string, RecallSessionProjection>;
  close(): void;
}

/** Exclusive phase timings for one bounded recall write window. */
export interface IncrementalRecallWriteWindowDiagnostic {
  readonly documentCount: number;
  readonly recovering: boolean;
  readonly lockWaitMilliseconds: number;
  readonly evidenceOpenMilliseconds: number;
  readonly evidenceWriteMilliseconds: number;
  readonly projectionOpenMilliseconds: number;
  readonly projectionWriteMilliseconds: number;
  readonly closeMilliseconds: number;
  readonly writeWindowMilliseconds: number;
}

/** Durable commit counts and separately measured acknowledgement phases. */
export interface CommittedIncrementalRecallTransfer {
  readonly committedDocumentCount: number;
  readonly writeWindowCount: number;
  readonly acknowledgedMarkerCount: number;
  readonly checkpointObservationMilliseconds: number;
  readonly markerAcknowledgementMilliseconds: number;
  readonly writeWindowDiagnostics: readonly IncrementalRecallWriteWindowDiagnostic[];
}

/** Paths, store boundaries, lock authority, and diagnostics for one prepared transfer commit. */
export interface CommitIncrementalRecallTransferOptions {
  prepared: PreparedIncrementalRecallTransfer;
  lockPath: string;
  evidenceDatabasePath: string;
  projectionDatabasePath: string;
  embeddingDimensions: number;
  signal?: AbortSignal;
  openEvidenceStore?: () => IncrementalRecallCommitEvidenceStore;
  openProjectionStore?: (mode: 'read' | 'write') => IncrementalRecallCommitProjectionStore;
  acknowledgeMarkers?: (
    workPlan: PreparedIncrementalRecallTransfer['workPlan'],
    observedCheckpoint: RecallMarkerCheckpoint,
  ) => Promise<number>;
  monotonicMilliseconds?: () => number;
  onWriteWindowDiagnostic?: (diagnostic: IncrementalRecallWriteWindowDiagnostic) => void;
}

interface MutableWriteWindowDiagnostic {
  documentCount: number;
  recovering: boolean;
  lockWaitMilliseconds: number;
  evidenceOpenMilliseconds: number;
  evidenceWriteMilliseconds: number;
  projectionOpenMilliseconds: number;
  projectionWriteMilliseconds: number;
  closeMilliseconds: number;
  writeWindowMilliseconds: number;
}

function elapsedMilliseconds(clock: () => number, startedAtMilliseconds: number): number {
  return Math.max(clock() - startedAtMilliseconds, 0);
}

function normalizeCommitError(error: unknown, message: string): Error {
  return error instanceof Error ? error : new Error(message, { cause: error });
}

function createEvidenceBatches(
  documents: readonly IndexedSessionConversationChunk[],
): Array<readonly IndexedSessionConversationChunk[]> {
  if (documents.length === 0) {
    return [[]];
  }
  const batches: Array<readonly IndexedSessionConversationChunk[]> = [];
  for (let start = 0; start < documents.length; start += INCREMENTAL_RECALL_EVIDENCE_BATCH_SIZE) {
    batches.push(documents.slice(start, start + INCREMENTAL_RECALL_EVIDENCE_BATCH_SIZE));
  }
  return batches;
}

function closeIncrementalRecallStores(
  evidenceStore: IncrementalRecallCommitEvidenceStore | undefined,
  projectionStore: IncrementalRecallCommitProjectionStore | undefined,
  writeWindow: RecallWriteWindow,
): Error[] {
  const errors: Error[] = [];
  try {
    projectionStore?.close();
  } catch (error) {
    errors.push(normalizeCommitError(error, 'Recall projection store close failed'));
  }
  try {
    evidenceStore?.close();
  } catch (error) {
    errors.push(normalizeCommitError(error, 'Recall evidence store close failed'));
  }
  if (errors.length > 0) {
    writeWindow.retainRecoveryRequired();
  }
  return errors;
}

function throwCommitAndCloseErrors(operationError: Error | null, closeErrors: Error[]): void {
  if (operationError !== null && closeErrors.length > 0) {
    throw new AggregateError(
      [operationError, ...closeErrors],
      'Recall incremental write and close failed',
    );
  }
  if (operationError !== null) {
    throw operationError;
  }
  if (closeErrors.length > 0) {
    throw new AggregateError(closeErrors, 'Recall incremental close failed');
  }
}

function defaultEvidenceStore(
  options: CommitIncrementalRecallTransferOptions,
): IncrementalRecallCommitEvidenceStore {
  return openZvecConversationStore({
    databasePath: options.evidenceDatabasePath,
    dimensions: options.embeddingDimensions,
  });
}

function defaultProjectionStore(
  options: CommitIncrementalRecallTransferOptions,
  mode: 'read' | 'write',
): ZvecSessionProjectionStore {
  return openZvecSessionProjectionStore({
    databasePath: options.projectionDatabasePath,
    generationId: options.prepared.targetGenerationId,
    createIfMissing: mode === 'write',
    readOnly: mode === 'read',
  });
}

async function commitOneIncrementalRecallWindow(
  options: CommitIncrementalRecallTransferOptions,
  documents: readonly IndexedSessionConversationChunk[],
  commitProjection: boolean,
  windowIndexStartedAtMilliseconds: number,
  clock: () => number,
): Promise<IncrementalRecallWriteWindowDiagnostic> {
  const diagnostic: MutableWriteWindowDiagnostic = {
    documentCount: documents.length,
    recovering: false,
    lockWaitMilliseconds: 0,
    evidenceOpenMilliseconds: 0,
    evidenceWriteMilliseconds: 0,
    projectionOpenMilliseconds: 0,
    projectionWriteMilliseconds: 0,
    closeMilliseconds: 0,
    writeWindowMilliseconds: 0,
  };
  return coordinateRecallWriteWindow(
    {
      lockPath: options.lockPath,
      allowRecovery: true,
      ...(options.signal ? { signal: options.signal } : {}),
    },
    async (writeWindow) => {
      diagnostic.recovering = writeWindow.recovering;
      diagnostic.lockWaitMilliseconds = elapsedMilliseconds(
        clock,
        windowIndexStartedAtMilliseconds,
      );
      const writeWindowStartedAtMilliseconds = clock();
      let evidenceStore: IncrementalRecallCommitEvidenceStore | undefined;
      let projectionStore: IncrementalRecallCommitProjectionStore | undefined;
      let operationError: Error | null = null;
      try {
        let startedAtMilliseconds = clock();
        try {
          evidenceStore = options.openEvidenceStore?.() ?? defaultEvidenceStore(options);
        } finally {
          diagnostic.evidenceOpenMilliseconds = elapsedMilliseconds(clock, startedAtMilliseconds);
        }
        startedAtMilliseconds = clock();
        try {
          projectionStore =
            options.openProjectionStore?.('write') ?? defaultProjectionStore(options, 'write');
        } finally {
          diagnostic.projectionOpenMilliseconds = elapsedMilliseconds(clock, startedAtMilliseconds);
        }
        startedAtMilliseconds = clock();
        try {
          await evidenceStore.upsertChunks(documents);
        } finally {
          diagnostic.evidenceWriteMilliseconds = elapsedMilliseconds(clock, startedAtMilliseconds);
        }
        if (commitProjection) {
          startedAtMilliseconds = clock();
          try {
            await projectionStore.upsertProjections([
              options.prepared.checkpointIntent.physicalProjection,
              ...options.prepared.checkpointIntent.logicalProjections,
            ]);
          } finally {
            diagnostic.projectionWriteMilliseconds = elapsedMilliseconds(
              clock,
              startedAtMilliseconds,
            );
          }
        }
      } catch (error) {
        operationError = normalizeCommitError(error, 'Recall incremental write failed');
      }
      const closeStartedAtMilliseconds = clock();
      const closeErrors = closeIncrementalRecallStores(evidenceStore, projectionStore, writeWindow);
      diagnostic.closeMilliseconds = elapsedMilliseconds(clock, closeStartedAtMilliseconds);
      diagnostic.writeWindowMilliseconds = elapsedMilliseconds(
        clock,
        writeWindowStartedAtMilliseconds,
      );
      const immutableDiagnostic = Object.freeze({ ...diagnostic });
      options.onWriteWindowDiagnostic?.(immutableDiagnostic);
      throwCommitAndCloseErrors(operationError, closeErrors);
      return immutableDiagnostic;
    },
  );
}

function assertObservedTransferCheckpoint(
  options: CommitIncrementalRecallTransferOptions,
  observedProjection: RecallSessionProjection | undefined,
): void {
  if (
    observedProjection === undefined ||
    observedProjection.projectionKind !==
      options.prepared.checkpointIntent.physicalProjection.projectionKind ||
    observedProjection.generationId !== options.prepared.targetGenerationId
  ) {
    throw new Error('Recall incremental checkpoint observation missing target physical projection');
  }
  const coveredMarkerIds = new Set(observedProjection.markerCheckpoint.coveredMarkerIds);
  for (const markerId of options.prepared.workPlan.sourceMarkerIds) {
    if (!coveredMarkerIds.has(markerId)) {
      throw new Error(
        `Recall incremental checkpoint observation does not cover marker ${markerId}`,
      );
    }
  }
}

async function observeIncrementalRecallCheckpoint(
  options: CommitIncrementalRecallTransferOptions,
): Promise<RecallMarkerCheckpoint> {
  const projectionStore =
    options.openProjectionStore?.('read') ?? defaultProjectionStore(options, 'read');
  let operationError: Error | null = null;
  let observedCheckpoint: RecallMarkerCheckpoint | null = null;
  try {
    const physicalProjectionId = options.prepared.checkpointIntent.physicalProjection.projectionId;
    const observed = projectionStore.fetchProjections([physicalProjectionId]);
    const observedProjection = observed.get(physicalProjectionId);
    assertObservedTransferCheckpoint(options, observedProjection);
    observedCheckpoint = observedProjection?.markerCheckpoint ?? null;
  } catch (error) {
    operationError = normalizeCommitError(error, 'Recall checkpoint observation failed');
  }
  let closeError: Error | null = null;
  try {
    projectionStore.close();
  } catch (error) {
    closeError = normalizeCommitError(error, 'Recall checkpoint observation close failed');
  }
  throwCommitAndCloseErrors(operationError, closeError === null ? [] : [closeError]);
  if (observedCheckpoint === null) {
    throw new Error('Recall incremental checkpoint observation missing after validation');
  }
  return observedCheckpoint;
}

/**
 * Commits deterministic evidence before scalar projections in windows of at most 32 documents.
 * Marker acknowledgement occurs only after a later read observes target-generation checkpoint coverage.
 */
export async function commitIncrementalRecallTransfer(
  options: CommitIncrementalRecallTransferOptions,
): Promise<CommittedIncrementalRecallTransfer> {
  const clock = options.monotonicMilliseconds ?? (() => performance.now());
  const batches = createEvidenceBatches(options.prepared.documents);
  const diagnostics: IncrementalRecallWriteWindowDiagnostic[] = [];
  for (const [index, documents] of batches.entries()) {
    const startedAtMilliseconds = clock();
    diagnostics.push(
      await commitOneIncrementalRecallWindow(
        options,
        documents,
        index === batches.length - 1,
        startedAtMilliseconds,
        clock,
      ),
    );
  }
  const observationStartedAtMilliseconds = clock();
  const observedCheckpoint = await observeIncrementalRecallCheckpoint(options);
  const checkpointObservationMilliseconds = elapsedMilliseconds(
    clock,
    observationStartedAtMilliseconds,
  );
  const acknowledgementStartedAtMilliseconds = clock();
  const acknowledgedMarkerCount = await (options.acknowledgeMarkers?.(
    options.prepared.workPlan,
    observedCheckpoint,
  ) ?? acknowledgeCoveredRecallMarkers(options.prepared.workPlan, observedCheckpoint));
  const markerAcknowledgementMilliseconds = elapsedMilliseconds(
    clock,
    acknowledgementStartedAtMilliseconds,
  );
  return Object.freeze({
    committedDocumentCount: options.prepared.documents.length,
    writeWindowCount: batches.length,
    acknowledgedMarkerCount,
    checkpointObservationMilliseconds,
    markerAcknowledgementMilliseconds,
    writeWindowDiagnostics: Object.freeze(diagnostics),
  });
}

import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecOpen, type ZVecCollection } from '@zvec/zvec';

import {
  coordinateRecallWriteWindow,
  inspectRecallWriteWindow,
} from './coordinate-recall-write-window.js';
import {
  RecallBacklogFailureCategory,
  RecallGenerationCutoverState,
  RECALL_INDEX_MANIFEST_VERSION,
} from './enums.js';
import {
  decodeRecallBacklogSummary,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
import { readRecallGenerationManifest } from './recall-generation-manifest.js';
import {
  readRecallGenerationReplaySnapshot,
  RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
} from './recall-generation-replay-snapshot.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { recoverRecallGenerationCutoverTransition } from './recall-generation-transitions.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  recallRebuildOwnershipLockPath,
  tryAcquireRecallRebuildOwnershipLock,
} from './recall-rebuild-ownership-lock.js';

/** Paths, store recovery capabilities, and retention for repairing generation cutover state. */
export interface RecoverRecallGenerationCutoverOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  generationRootDirectory: string;
  backlogSummaryPath: string;
  lockPath: string;
  embeddingDimensions: number;
  rollbackRetentionMilliseconds?: number;
  nowEpochMilliseconds?: () => number;
  openWriteEvidenceStore?: (
    databasePath: string,
    embeddingDimensions: number,
  ) => RecallGenerationRecoveryStore;
  openWriteProjectionStore?: (
    databasePath: string,
    generationId: string,
  ) => RecallGenerationRecoveryStore;
}

/** Minimal close evidence required from a write-capable recovery store open. */
export interface RecallGenerationRecoveryStore {
  close(): void;
}

function normalizeRecoveryStoreError(error: unknown, message: string): Error {
  return error instanceof Error ? error : new Error(message, { cause: error });
}

async function recoverActiveRecallGenerationStores(
  options: RecoverRecallGenerationCutoverOptions,
  generationId: string,
): Promise<void> {
  const generationDirectory = await resolveRecallGenerationDirectory(
    options.generationRootDirectory,
    generationId,
  );
  const targetPaths = createRecallGenerationComponentPaths(generationDirectory);
  if (existsSync(targetPaths.lexicalSourceStorePath)) {
    const openedTargetStores: ZVecCollection[] = [];
    let openError: Error | null = null;
    try {
      openedTargetStores.push(ZVecOpen(targetPaths.lexicalSourceStorePath));
      openedTargetStores.push(ZVecOpen(targetPaths.denseStorePath));
      openedTargetStores.push(ZVecOpen(targetPaths.sessionProjectionStorePath));
    } catch (error) {
      openError = normalizeRecoveryStoreError(
        error,
        'Recall target generation recovery store open failed',
      );
    }
    const closeErrors: Error[] = [];
    for (const store of openedTargetStores.reverse()) {
      try {
        store.closeSync();
      } catch (error) {
        closeErrors.push(
          normalizeRecoveryStoreError(error, 'Recall target generation recovery close failed'),
        );
      }
    }
    if (openError !== null && closeErrors.length > 0) {
      throw new AggregateError(
        [openError, ...closeErrors],
        'Recall target generation recovery open and close failed',
      );
    }
    if (openError !== null) {
      throw openError;
    }
    if (closeErrors.length > 0) {
      throw new AggregateError(closeErrors, 'Recall target generation recovery close failed');
    }
    return;
  }
  let openWriteEvidenceStore = options.openWriteEvidenceStore;
  if (openWriteEvidenceStore === undefined) {
    const { openZvecConversationStore } = await import('./zvec-conversation-store.js');
    openWriteEvidenceStore = (databasePath: string, embeddingDimensions: number) =>
      openZvecConversationStore({
        databasePath,
        dimensions: embeddingDimensions,
        createIfMissing: false,
        readOnly: false,
      });
  }
  let openWriteProjectionStore = options.openWriteProjectionStore;
  if (openWriteProjectionStore === undefined) {
    const { openZvecSessionProjectionStore } = await import('./zvec-session-projection-store.js');
    openWriteProjectionStore = (databasePath: string, targetGenerationId: string) =>
      openZvecSessionProjectionStore({
        databasePath,
        generationId: targetGenerationId,
        createIfMissing: false,
        readOnly: false,
      });
  }
  let evidenceStore: RecallGenerationRecoveryStore | undefined;
  let projectionStore: RecallGenerationRecoveryStore | undefined;
  let openError: Error | null = null;
  try {
    evidenceStore = openWriteEvidenceStore(
      join(generationDirectory, 'zvec'),
      options.embeddingDimensions,
    );
    projectionStore = openWriteProjectionStore(
      join(generationDirectory, 'session-projections'),
      generationId,
    );
  } catch (error) {
    openError = normalizeRecoveryStoreError(error, 'Recall generation recovery store open failed');
  }
  const closeErrors: Error[] = [];
  try {
    projectionStore?.close();
  } catch (error) {
    closeErrors.push(
      normalizeRecoveryStoreError(error, 'Recall generation recovery projection close failed'),
    );
  }
  try {
    evidenceStore?.close();
  } catch (error) {
    closeErrors.push(
      normalizeRecoveryStoreError(error, 'Recall generation recovery evidence close failed'),
    );
  }
  if (openError !== null && closeErrors.length > 0) {
    throw new AggregateError(
      [openError, ...closeErrors],
      'Recall generation recovery open and close failed',
    );
  }
  if (openError !== null) {
    throw openError;
  }
  if (closeErrors.length > 0) {
    throw new AggregateError(closeErrors, 'Recall generation recovery close failed');
  }
}

async function validateTargetRecallCutoverArtifacts(
  options: RecoverRecallGenerationCutoverOptions,
  pointer: Awaited<ReturnType<typeof readRecallActiveGenerationPointer>>,
  registry: Awaited<ReturnType<typeof readRecallGenerationRegistry>>,
): Promise<void> {
  const readyEntry = registry?.generations.find(
    (entry) =>
      entry.indexManifestVersion === RECALL_INDEX_MANIFEST_VERSION &&
      (entry.generationId === registry.buildingGenerationId ||
        entry.generationId === pointer?.activeGenerationId) &&
      (entry.state === RecallGenerationCutoverState.READY ||
        entry.state === RecallGenerationCutoverState.REPLAY_PENDING),
  );
  if (readyEntry === undefined) {
    return;
  }
  const generationDirectory = await resolveRecallGenerationDirectory(
    options.generationRootDirectory,
    readyEntry.generationId,
  );
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  if (!existsSync(paths.lexicalSourceStorePath)) {
    return;
  }
  const [{ fingerprint }, receipt, replaySnapshot] = await Promise.all([
    readRecallGenerationManifest(paths.manifestPath),
    readRecallGenerationValidationReceipt(paths.validationReceiptPath),
    readRecallGenerationReplaySnapshot(
      join(
        generationDirectory,
        readyEntry.replaySnapshotFileName ?? RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
      ),
    ),
  ]);
  if (
    receipt.generationId !== readyEntry.generationId ||
    receipt.manifestFingerprint !== fingerprint ||
    fingerprint !== readyEntry.indexManifestFingerprint
  ) {
    throw new Error(
      `Recall target cutover recovery validation receipt mismatch for ${readyEntry.generationId}`,
    );
  }
  if (replaySnapshot.generationId !== readyEntry.generationId) {
    throw new Error(
      `Recall target cutover recovery replay snapshot mismatch for ${readyEntry.generationId}`,
    );
  }
}

async function readRecallRecoveryBacklogSummary(
  backlogSummaryPath: string,
): Promise<ReturnType<typeof decodeRecallBacklogSummary> | null> {
  try {
    return decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return null;
    }
    throw error;
  }
}

/** Recovers validated rebuild, rollback, or legacy cutover ordering and preserves unknown states. */
export async function recoverRecallGenerationCutover(
  options: RecoverRecallGenerationCutoverOptions,
): Promise<boolean> {
  const [initialPointer, initialRegistry, initialBacklog, writeWindowState] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
    readRecallRecoveryBacklogSummary(options.backlogSummaryPath),
    inspectRecallWriteWindow(options.lockPath),
  ]);
  const initiallyConsistent =
    initialPointer !== null &&
    initialRegistry?.activeGenerationId === initialPointer.activeGenerationId &&
    initialRegistry.activePointerChecksum === initialPointer.checksum;
  const initialBuildingGenerationEntry = initialRegistry?.generations.find(
    ({ generationId }) =>
      generationId === initialRegistry.buildingGenerationId &&
      initialRegistry.buildingGenerationId !== null,
  );
  const buildingGenerationPresent =
    initialBuildingGenerationEntry?.state === RecallGenerationCutoverState.BUILDING;
  const rebuildOwnershipLock = buildingGenerationPresent
    ? await tryAcquireRecallRebuildOwnershipLock(recallRebuildOwnershipLockPath(options.lockPath))
    : null;
  if (buildingGenerationPresent && rebuildOwnershipLock === null) {
    return false;
  }
  const failedBuildingGenerationRequiresRecovery =
    buildingGenerationPresent &&
    initialBacklog?.buildingGenerationId === initialBuildingGenerationEntry.generationId &&
    initialBacklog.lastFailureCategory === RecallBacklogFailureCategory.REBUILD_FAILED;
  const abandonedBuildingGenerationRequiresRecovery =
    buildingGenerationPresent && rebuildOwnershipLock !== null;
  if (
    ((initiallyConsistent &&
      initialBuildingGenerationEntry?.state !== RecallGenerationCutoverState.READY &&
      !failedBuildingGenerationRequiresRecovery &&
      !abandonedBuildingGenerationRequiresRecovery) ||
      (initialPointer === null && initialRegistry === null)) &&
    !writeWindowState.currentWindow &&
    !writeWindowState.recoveryRequired
  ) {
    await rebuildOwnershipLock?.release();
    return false;
  }

  return coordinateRecallWriteWindow(
    { lockPath: options.lockPath, allowRecovery: true },
    async (writeWindow) => {
      const [pointer, registry, backlog] = await Promise.all([
        readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
        readRecallGenerationRegistry(options.generationRegistryPath),
        readRecallRecoveryBacklogSummary(options.backlogSummaryPath),
      ]);
      await validateTargetRecallCutoverArtifacts(options, pointer, registry);
      const transition = await recoverRecallGenerationCutoverTransition({
        activeGenerationPointerPath: options.activeGenerationPointerPath,
        generationRegistryPath: options.generationRegistryPath,
        backlogSummaryPath: options.backlogSummaryPath,
        pointer,
        registry,
        backlog,
        abandonedBuildingGeneration: rebuildOwnershipLock !== null,
        recoveredAtEpochMilliseconds: options.nowEpochMilliseconds?.() ?? Date.now(),
        ...(options.rollbackRetentionMilliseconds === undefined
          ? {}
          : { rollbackRetentionMilliseconds: options.rollbackRetentionMilliseconds }),
        retainRecoveryRequired: () => writeWindow.retainRecoveryRequired(),
      });
      if (!writeWindow.recovering) {
        return transition.stateChanged;
      }
      await recoverActiveRecallGenerationStores(options, transition.activeGenerationId);
      writeWindow.attestRecoveryCompleted();
      return true;
    },
  ).finally(async () => {
    await rebuildOwnershipLock?.release();
  });
}

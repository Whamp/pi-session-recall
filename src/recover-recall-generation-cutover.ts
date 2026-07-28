import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

import {
  coordinateRecallWriteWindow,
  inspectRecallWriteWindow,
} from './coordinate-recall-write-window.js';
import { RecallBacklogFailureCategory, RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  decodeRecallBacklogSummary,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import { readNodeErrorCode } from './read-node-error-code.js';

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

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;

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

function finalizeRecoveredRecallRegistry(
  registry: RecallGenerationRegistry,
  replacement: RecallGenerationRegistryEntry,
  pointerChecksum: string,
  recoveredAt: number,
  rollbackRetentionMilliseconds: number,
): RecallGenerationRegistry {
  const previousGenerationId = registry.activeGenerationId;
  return {
    ...registry,
    activeGenerationId: replacement.generationId,
    buildingGenerationId: null,
    rollbackGenerationId: previousGenerationId,
    activePointerChecksum: pointerChecksum,
    generations: registry.generations.map((entry): RecallGenerationRegistryEntry => {
      if (entry.generationId === replacement.generationId) {
        return {
          ...replacement,
          state: RecallGenerationCutoverState.REPLAY_PENDING,
          stateChangedAtEpochMilliseconds: recoveredAt,
        };
      }
      if (entry.generationId === previousGenerationId) {
        return {
          ...entry,
          state: RecallGenerationCutoverState.ROLLBACK,
          stateChangedAtEpochMilliseconds: recoveredAt,
          retireAfterEpochMilliseconds: recoveredAt + rollbackRetentionMilliseconds,
        };
      }
      if (entry.state === RecallGenerationCutoverState.ROLLBACK) {
        return { ...entry, state: RecallGenerationCutoverState.RETIRED };
      }
      return entry;
    }),
  };
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
  const initialReadyBuildingGeneration = initialRegistry?.generations.find(
    ({ generationId }) =>
      generationId === initialRegistry.buildingGenerationId &&
      initialRegistry.buildingGenerationId !== null,
  );
  const failedBuildingGenerationRequiresRecovery =
    initialReadyBuildingGeneration?.state === RecallGenerationCutoverState.BUILDING &&
    initialBacklog?.buildingGenerationId === initialReadyBuildingGeneration.generationId &&
    initialBacklog.lastFailureCategory === RecallBacklogFailureCategory.REBUILD_FAILED;
  if (
    ((initiallyConsistent &&
      initialReadyBuildingGeneration?.state !== RecallGenerationCutoverState.READY &&
      !failedBuildingGenerationRequiresRecovery) ||
      (initialPointer === null && initialRegistry === null)) &&
    !writeWindowState.currentWindow &&
    !writeWindowState.recoveryRequired
  ) {
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
      const readyBuildingGeneration = registry?.generations.find(
        ({ generationId }) =>
          generationId === registry.buildingGenerationId && registry.buildingGenerationId !== null,
      );
      const pointerAndRegistrySelectSameGeneration =
        pointer !== null &&
        registry?.activeGenerationId === pointer.activeGenerationId &&
        registry.activePointerChecksum === pointer.checksum;
      const attestRecoveredActiveStores = async (generationId: string): Promise<boolean> => {
        if (!writeWindow.recovering) {
          return false;
        }
        await recoverActiveRecallGenerationStores(options, generationId);
        writeWindow.attestRecoveryCompleted();
        return true;
      };
      if (
        pointerAndRegistrySelectSameGeneration &&
        pointer !== null &&
        registry !== null &&
        readyBuildingGeneration?.state === RecallGenerationCutoverState.BUILDING &&
        backlog?.buildingGenerationId === readyBuildingGeneration.generationId &&
        backlog.lastFailureCategory === RecallBacklogFailureCategory.REBUILD_FAILED
      ) {
        const recoveredAt = options.nowEpochMilliseconds?.() ?? Date.now();
        const activeEntry = registry.generations.find(
          ({ generationId }) => generationId === registry.activeGenerationId,
        );
        if (activeEntry === undefined) {
          writeWindow.retainRecoveryRequired();
          throw new Error('Recall failed rebuild recovery active registry entry missing');
        }
        await writeRecallGenerationRegistry(options.generationRegistryPath, {
          ...registry,
          buildingGenerationId: null,
          generations: registry.generations.map((entry) =>
            entry.generationId === readyBuildingGeneration.generationId
              ? {
                  ...entry,
                  state: RecallGenerationCutoverState.FAILED,
                  stateChangedAtEpochMilliseconds: recoveredAt,
                }
              : entry,
          ),
        });
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: backlog.pendingEligibleSessionCount,
          oldestEligibleMarkerAgeMilliseconds: backlog.oldestEligibleMarkerAgeMilliseconds,
          activeGenerationId: pointer.activeGenerationId,
          buildingGenerationId: null,
          generationState: activeEntry.state,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: null,
          lastFailureCategory: RecallBacklogFailureCategory.REBUILD_FAILED,
          observedAtEpochMilliseconds: recoveredAt,
        });
        await attestRecoveredActiveStores(pointer.activeGenerationId);
        return true;
      }
      if (
        pointerAndRegistrySelectSameGeneration &&
        pointer !== null &&
        readyBuildingGeneration?.state !== RecallGenerationCutoverState.READY
      ) {
        return attestRecoveredActiveStores(pointer.activeGenerationId);
      }
      if (
        pointerAndRegistrySelectSameGeneration &&
        registry !== null &&
        readyBuildingGeneration?.state === RecallGenerationCutoverState.READY
      ) {
        const recoveredAt = options.nowEpochMilliseconds?.() ?? Date.now();
        const activeEntry = registry.generations.find(
          ({ generationId }) => generationId === registry.activeGenerationId,
        );
        if (activeEntry === undefined) {
          writeWindow.retainRecoveryRequired();
          throw new Error('Recall generation cutover recovery active registry entry missing');
        }
        await writeRecallGenerationRegistry(options.generationRegistryPath, {
          ...registry,
          buildingGenerationId: null,
          generations: registry.generations.map((entry) =>
            entry.generationId === readyBuildingGeneration.generationId
              ? {
                  ...entry,
                  state: RecallGenerationCutoverState.RETIRED,
                  stateChangedAtEpochMilliseconds: recoveredAt,
                }
              : entry,
          ),
        });
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: 0,
          oldestEligibleMarkerAgeMilliseconds: null,
          activeGenerationId: pointer.activeGenerationId,
          buildingGenerationId: null,
          generationState: activeEntry.state,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: null,
          lastFailureCategory: null,
          observedAtEpochMilliseconds: recoveredAt,
        });
        await attestRecoveredActiveStores(pointer.activeGenerationId);
        return true;
      }
      const registrySelectedEntry = registry?.generations.find(
        ({ generationId }) => generationId === registry.activeGenerationId,
      );
      const registryFirstCutover =
        registry !== null &&
        registrySelectedEntry !== undefined &&
        ((pointer === null &&
          registrySelectedEntry.state === RecallGenerationCutoverState.LEGACY_READ_ONLY) ||
          (pointer !== null &&
            registry.rollbackGenerationId === pointer.activeGenerationId &&
            (registrySelectedEntry.state === RecallGenerationCutoverState.REPLAY_PENDING ||
              registrySelectedEntry.state === RecallGenerationCutoverState.LEGACY_READ_ONLY)));
      if (registryFirstCutover) {
        const targetPointer = createRecallActiveGenerationPointer(
          registrySelectedEntry.generationId,
        );
        await writeRecallActiveGenerationPointer(
          options.activeGenerationPointerPath,
          targetPointer,
        );
        await writeRecallBacklogSummary(options.backlogSummaryPath, {
          version: RECALL_BACKLOG_SUMMARY_VERSION,
          pendingEligibleSessionCount: 0,
          oldestEligibleMarkerAgeMilliseconds: null,
          activeGenerationId: registrySelectedEntry.generationId,
          buildingGenerationId: null,
          generationState: registrySelectedEntry.state,
          activeGenerationAgeMilliseconds: 0,
          rebuildAgeMilliseconds: null,
          lastFailureCategory: null,
          observedAtEpochMilliseconds: options.nowEpochMilliseconds?.() ?? Date.now(),
        });
        await attestRecoveredActiveStores(registrySelectedEntry.generationId);
        return true;
      }
      const replacement = registry?.generations.find(
        ({ generationId }) => generationId === pointer?.activeGenerationId,
      );
      const replacementPointer = pointer;
      if (
        !replacementPointer ||
        !registry ||
        registry.buildingGenerationId !== replacementPointer.activeGenerationId ||
        replacement?.state !== RecallGenerationCutoverState.READY ||
        replacement.validatedAtEpochMilliseconds === undefined ||
        replacement.validatedAtEpochMilliseconds === null
      ) {
        writeWindow.retainRecoveryRequired();
        throw new Error(
          'Recall generation cutover recovery found an unsupported pointer and registry state',
        );
      }
      const recoveredAt = options.nowEpochMilliseconds?.() ?? Date.now();
      const recoveredRegistry = finalizeRecoveredRecallRegistry(
        registry,
        replacement,
        replacementPointer.checksum,
        recoveredAt,
        options.rollbackRetentionMilliseconds ?? DEFAULT_ROLLBACK_RETENTION_MILLISECONDS,
      );
      await writeRecallGenerationRegistry(options.generationRegistryPath, recoveredRegistry);
      await writeRecallBacklogSummary(options.backlogSummaryPath, {
        version: RECALL_BACKLOG_SUMMARY_VERSION,
        pendingEligibleSessionCount: replacement.rebuildMarkerWatermark?.length ?? 0,
        oldestEligibleMarkerAgeMilliseconds: null,
        activeGenerationId: replacement.generationId,
        buildingGenerationId: null,
        generationState: RecallGenerationCutoverState.REPLAY_PENDING,
        activeGenerationAgeMilliseconds: 0,
        rebuildAgeMilliseconds: Math.max(
          0,
          recoveredAt - replacement.rebuildStartedAtEpochMilliseconds,
        ),
        lastFailureCategory: null,
        observedAtEpochMilliseconds: recoveredAt,
      });
      await attestRecoveredActiveStores(replacement.generationId);
      return true;
    },
  );
}

import {
  coordinateRecallWriteWindow,
  inspectRecallWriteWindow,
} from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';

/** Paths and bounded retention used to recover a pointer-swapped READY generation. */
export interface RecoverRecallGenerationCutoverOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  lockPath: string;
  rollbackRetentionMilliseconds?: number;
  nowEpochMilliseconds?: () => number;
}

const DEFAULT_ROLLBACK_RETENTION_MILLISECONDS = 7 * 24 * 60 * 60_000;

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

/** Recovers validated rebuild, rollback, or legacy cutover ordering and preserves unknown states. */
export async function recoverRecallGenerationCutover(
  options: RecoverRecallGenerationCutoverOptions,
): Promise<boolean> {
  const [initialPointer, initialRegistry, writeWindowState] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
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
  if (
    ((initiallyConsistent &&
      initialReadyBuildingGeneration?.state !== RecallGenerationCutoverState.READY) ||
      (initialPointer === null && initialRegistry === null)) &&
    !writeWindowState.currentWindow &&
    !writeWindowState.recoveryRequired
  ) {
    return false;
  }

  return coordinateRecallWriteWindow(
    { lockPath: options.lockPath, allowRecovery: true },
    async (writeWindow) => {
      const [pointer, registry] = await Promise.all([
        readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
        readRecallGenerationRegistry(options.generationRegistryPath),
      ]);
      const readyBuildingGeneration = registry?.generations.find(
        ({ generationId }) =>
          generationId === registry.buildingGenerationId && registry.buildingGenerationId !== null,
      );
      const pointerAndRegistrySelectSameGeneration =
        pointer !== null &&
        registry?.activeGenerationId === pointer.activeGenerationId &&
        registry.activePointerChecksum === pointer.checksum;
      if (
        pointerAndRegistrySelectSameGeneration &&
        readyBuildingGeneration?.state !== RecallGenerationCutoverState.READY
      ) {
        return false;
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
      return true;
    },
  );
}

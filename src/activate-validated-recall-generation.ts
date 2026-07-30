import { join } from 'node:path';

import { openValidatedRecallGeneration } from './recall-coherent-generation.js';
import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import { RecallValidatedGenerationActivationStage } from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import {
  activateReadyRecallGenerationTransition,
  prepareValidatedRecallGenerationActivationTransition,
  publishRecallGenerationActivationBacklogTransition,
} from './recall-generation-transitions.js';
import {
  listPendingRecallMarkerIds,
  listQuarantinedRecallMarkerIds,
} from './recall-generation-replay-markers.js';
import {
  RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
  RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION,
  writeRecallGenerationReplaySnapshot,
} from './recall-generation-replay-snapshot.js';

/** Configured state, marker, and fault boundaries for target generation activation. */
export interface ActivateValidatedRecallGenerationOptions {
  generation: Readonly<RecallCoherentGenerationConfig>;
  generationId: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  markerQuarantineDirectory: string;
  retainRecoveryRequired(): void;
  activationFault?: (stage: RecallValidatedGenerationActivationStage) => void | Promise<void>;
  nowEpochMilliseconds?: () => number;
}

/** Active target identity and exact fixed replay snapshot sizes after cutover. */
export interface ActivatedValidatedRecallGeneration {
  activeGenerationId: string;
  replayPendingMarkerCount: number;
  replayQuarantinedMarkerCount: number;
}

/** Authenticates, snapshots, and activates one validated target through named transitions. */
export async function activateValidatedRecallGeneration(
  options: ActivateValidatedRecallGenerationOptions,
): Promise<ActivatedValidatedRecallGeneration> {
  const opened = await openValidatedRecallGeneration(options.generation, options.generationId);
  const [expectedActivePointer, expectedRegistry, pendingMarkerIds, quarantinedMarkerIds] =
    await Promise.all([
      readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
      readRecallGenerationRegistry(options.generationRegistryPath),
      listPendingRecallMarkerIds(options.markerSpoolDirectory),
      listQuarantinedRecallMarkerIds(options.markerQuarantineDirectory),
    ]);
  const activatedAtEpochMilliseconds = options.nowEpochMilliseconds?.() ?? Date.now();
  const readiness = prepareValidatedRecallGenerationActivationTransition({
    registry: expectedRegistry,
    activePointer: expectedActivePointer,
    generationId: options.generationId,
    embeddingProfileId: options.generation.embeddingProfileId,
    indexManifestFingerprint: opened.manifestFingerprint,
    readyAtEpochMilliseconds: opened.validatedAtEpochMilliseconds,
    replayMarkerIds: pendingMarkerIds,
    replaySnapshotFileName: RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME,
  });
  await writeRecallGenerationReplaySnapshot(
    join(opened.generationDirectory, RECALL_ACTIVATION_REPLAY_SNAPSHOT_FILE_NAME),
    {
      snapshotVersion: RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION,
      generationId: options.generationId,
      pendingMarkerIds,
      quarantinedMarkerIds,
      capturedAtEpochMilliseconds: activatedAtEpochMilliseconds,
    },
  );
  const activation = await activateReadyRecallGenerationTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    expectedActivePointer,
    expectedFrozenRegistry: expectedRegistry,
    readyRegistry: readiness.readyRegistry,
    readyEntry: readiness.readyEntry,
    activatedAtEpochMilliseconds,
    async afterReadyRegistry() {
      await options.activationFault?.(
        RecallValidatedGenerationActivationStage.AFTER_READY_REGISTRY,
      );
    },
    async afterPointerSwap() {
      await options.activationFault?.(RecallValidatedGenerationActivationStage.AFTER_POINTER_SWAP);
    },
    async afterActivatedRegistry() {
      await options.activationFault?.(
        RecallValidatedGenerationActivationStage.AFTER_ACTIVATED_REGISTRY,
      );
    },
    throwIfCancelled() {},
    retainRecoveryRequired: () => options.retainRecoveryRequired(),
  });
  try {
    await publishRecallGenerationActivationBacklogTransition({
      backlogSummaryPath: options.backlogSummaryPath,
      activation,
    });
  } catch (error) {
    options.retainRecoveryRequired();
    throw error;
  }
  return {
    activeGenerationId: activation.activeGenerationId,
    replayPendingMarkerCount: pendingMarkerIds.length,
    replayQuarantinedMarkerCount: quarantinedMarkerIds.length,
  };
}

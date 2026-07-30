import { readdir } from 'node:fs/promises';
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
  RECALL_GENERATION_REPLAY_SNAPSHOT_VERSION,
  writeRecallGenerationReplaySnapshot,
} from './recall-generation-replay-snapshot.js';
import { readNodeErrorCode } from './read-node-error-code.js';

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

const PENDING_MARKER_FILE_PATTERN = /^([A-Za-z0-9_-]+)\.json$/u;
const QUARANTINED_MARKER_FILE_PATTERN = /^([A-Za-z0-9_-]+)\.json(?:\..+)?$/u;

async function listPendingMarkerIds(markerSpoolDirectory: string): Promise<string[]> {
  try {
    return (await readdir(markerSpoolDirectory, { withFileTypes: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => PENDING_MARKER_FILE_PATTERN.exec(entry.name)?.[1])
      .filter((markerId) => markerId !== undefined)
      .toSorted();
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
}

async function listQuarantinedMarkerIds(markerQuarantineDirectory: string): Promise<string[]> {
  let categories;
  try {
    categories = await readdir(markerQuarantineDirectory, { withFileTypes: true });
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    throw error;
  }
  const markerIds = (
    await Promise.all(
      categories
        .filter((entry) => entry.isDirectory())
        .map(async (category) =>
          (await readdir(join(markerQuarantineDirectory, category.name), { withFileTypes: true }))
            .filter((entry) => entry.isFile())
            .map((entry) => QUARANTINED_MARKER_FILE_PATTERN.exec(entry.name)?.[1])
            .filter((markerId) => markerId !== undefined),
        ),
    )
  ).flat();
  return [...new Set(markerIds)].toSorted();
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
      listPendingMarkerIds(options.markerSpoolDirectory),
      listQuarantinedMarkerIds(options.markerQuarantineDirectory),
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
  });
  await writeRecallGenerationReplaySnapshot(
    join(opened.generationDirectory, 'generation-replay-snapshot.json'),
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

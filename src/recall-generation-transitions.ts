import { RecallGenerationCutoverState } from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
  RECALL_BACKLOG_SUMMARY_VERSION,
  type RecallBacklogSummary,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';

/** Durable paths and external marker proof required to complete generation replay. */
export interface CompleteRecallGenerationReplayTransitionOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  nowEpochMilliseconds?: () => number;
  proveReplayWorkComplete(): Promise<boolean>;
}

function createActiveRecallGenerationBacklogSummary(
  activeGenerationId: string,
  observedAtEpochMilliseconds: number,
): RecallBacklogSummary {
  return {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 0,
    oldestEligibleMarkerAgeMilliseconds: null,
    activeGenerationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.ACTIVE,
    activeGenerationAgeMilliseconds: 0,
    rebuildAgeMilliseconds: null,
    lastFailureCategory: null,
    observedAtEpochMilliseconds,
  };
}

/**
 * Completes replay after external marker work is empty, publishing registry before backlog state.
 * Already-active generations repair their backlog summary without rechecking marker work.
 */
export async function completeRecallGenerationReplayTransition(
  options: CompleteRecallGenerationReplayTransitionOptions,
): Promise<boolean> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error(
      'Recall generation replay completion requires initialized pointer and registry',
    );
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation replay completion found pointer and registry disagreement');
  }
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  if (!activeEntry) {
    throw new Error('Recall generation replay completion active registry entry missing');
  }
  const completedAt = options.nowEpochMilliseconds?.() ?? Date.now();
  const activeBacklog = createActiveRecallGenerationBacklogSummary(
    pointer.activeGenerationId,
    completedAt,
  );
  if (activeEntry.state === RecallGenerationCutoverState.ACTIVE) {
    await writeRecallBacklogSummary(options.backlogSummaryPath, activeBacklog);
    return true;
  }
  if (activeEntry.state !== RecallGenerationCutoverState.REPLAY_PENDING) {
    return false;
  }
  if (!(await options.proveReplayWorkComplete())) {
    return false;
  }
  const activeReplacement: RecallGenerationRegistryEntry = {
    ...activeEntry,
    state: RecallGenerationCutoverState.ACTIVE,
    stateChangedAtEpochMilliseconds: completedAt,
  };
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...registry,
    generations: registry.generations.map((entry) =>
      entry.generationId === activeReplacement.generationId ? activeReplacement : entry,
    ),
  });
  await writeRecallBacklogSummary(options.backlogSummaryPath, activeBacklog);
  return true;
}

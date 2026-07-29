import { access, rm } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
  writeRecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';
import { readNodeErrorCode } from './read-node-error-code.js';

/** Conservative generation garbage-collection inputs and deterministic clock. */
export interface CollectRetiredRecallGenerationsOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  generationRootDirectory: string;
  lockPath: string;
  retainedMarkerDirectory?: string;
  nowEpochMilliseconds?: () => number;
}

/** Generation IDs deleted only after validation, replay, retention, and pointer checks. */
export interface CollectRetiredRecallGenerationsResult {
  deletedGenerationIds: string[];
}

function isRecallGenerationCollectible(
  entry: RecallGenerationRegistryEntry,
  nowEpochMilliseconds: number,
): boolean {
  return (
    (entry.state === RecallGenerationCutoverState.ROLLBACK ||
      entry.state === RecallGenerationCutoverState.RETIRED) &&
    entry.validatedAtEpochMilliseconds !== undefined &&
    entry.validatedAtEpochMilliseconds !== null &&
    entry.retireAfterEpochMilliseconds !== undefined &&
    entry.retireAfterEpochMilliseconds !== null &&
    entry.retireAfterEpochMilliseconds <= nowEpochMilliseconds
  );
}

/** Collects only validated expired rollback material never selected by the active pointer. */
async function collectRetiredRecallGenerationsWithLock(
  options: CollectRetiredRecallGenerationsOptions,
): Promise<CollectRetiredRecallGenerationsResult> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(options.activeGenerationPointerPath),
    readRecallGenerationRegistry(options.generationRegistryPath),
  ]);
  if (!pointer || !registry) {
    throw new Error('Recall generation collection requires initialized pointer and registry');
  }
  if (
    registry.activeGenerationId !== pointer.activeGenerationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error('Recall generation collection found pointer and registry disagreement');
  }
  const activeEntry = registry.generations.find(
    ({ generationId }) => generationId === pointer.activeGenerationId,
  );
  if (
    activeEntry?.state !== RecallGenerationCutoverState.ACTIVE ||
    registry.buildingGenerationId !== null
  ) {
    return { deletedGenerationIds: [] };
  }
  const now = options.nowEpochMilliseconds?.() ?? Date.now();
  const candidates = registry.generations.filter(
    (entry) =>
      entry.generationId !== pointer.activeGenerationId &&
      entry.generationId !== registry.buildingGenerationId &&
      entry.state !== RecallGenerationCutoverState.BUILDING &&
      entry.state !== RecallGenerationCutoverState.READY &&
      entry.state !== RecallGenerationCutoverState.ACTIVE &&
      entry.state !== RecallGenerationCutoverState.REPLAY_PENDING &&
      entry.state !== RecallGenerationCutoverState.LEGACY_READ_ONLY &&
      isRecallGenerationCollectible(entry, now),
  );
  if (candidates.length === 0) {
    return { deletedGenerationIds: [] };
  }
  const candidateIds = new Set(candidates.map(({ generationId }) => generationId));
  const collectionRegistry = {
    ...registry,
    rollbackGenerationId:
      registry.rollbackGenerationId !== null && candidateIds.has(registry.rollbackGenerationId)
        ? null
        : registry.rollbackGenerationId,
    generations: registry.generations.map(
      (entry): RecallGenerationRegistryEntry =>
        candidateIds.has(entry.generationId)
          ? { ...entry, state: RecallGenerationCutoverState.RETIRED }
          : entry,
    ),
  };
  await writeRecallGenerationRegistry(options.generationRegistryPath, collectionRegistry);

  const deletedGenerationIds: string[] = [];
  for (const candidate of candidates) {
    createRecallActiveGenerationPointer(candidate.generationId);
    const candidatePath = join(options.generationRootDirectory, candidate.generationId);
    try {
      await access(candidatePath);
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        deletedGenerationIds.push(candidate.generationId);
        continue;
      }
      throw error;
    }
    const generationDirectory = await resolveRecallGenerationDirectory(
      options.generationRootDirectory,
      candidate.generationId,
    );
    await rm(generationDirectory, { recursive: true });
    deletedGenerationIds.push(candidate.generationId);
  }
  await writeRecallGenerationRegistry(options.generationRegistryPath, {
    ...collectionRegistry,
    generations: collectionRegistry.generations.filter(
      ({ generationId }) => !deletedGenerationIds.includes(generationId),
    ),
  });
  if (collectionRegistry.rollbackGenerationId === null && options.retainedMarkerDirectory) {
    await rm(options.retainedMarkerDirectory, { recursive: true, force: true });
  }
  return { deletedGenerationIds: deletedGenerationIds.toSorted() };
}

/** Collects expired generations under the operation lock so rollback cannot race deletion. */
export async function collectRetiredRecallGenerations(
  options: CollectRetiredRecallGenerationsOptions,
): Promise<CollectRetiredRecallGenerationsResult> {
  return coordinateRecallWriteWindow({ lockPath: options.lockPath, allowRecovery: false }, () =>
    collectRetiredRecallGenerationsWithLock(options),
  );
}

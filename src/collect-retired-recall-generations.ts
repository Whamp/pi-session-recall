import { access, rm } from 'node:fs/promises';
import { join } from 'node:path';

import { coordinateRecallWriteWindow } from './coordinate-recall-write-window.js';
import {
  createRecallActiveGenerationPointer,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
import {
  completeRetiredRecallGenerationCollectionTransition,
  prepareRetiredRecallGenerationCollectionTransition,
} from './recall-generation-transitions.js';
import { readNodeErrorCode } from './read-node-error-code.js';

/** Conservative generation garbage-collection inputs and deterministic clock. */
export interface CollectRetiredRecallGenerationsOptions {
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  generationRootDirectory: string;
  lockPath: string;
  retainedMarkerDirectory?: string;
  generationCollectionFault?: (
    stage: 'after_generation_directory_delete',
    generationId: string,
  ) => void | Promise<void>;
  nowEpochMilliseconds?: () => number;
}

/** Generation IDs deleted only after validation, replay, retention, and pointer checks. */
export interface CollectRetiredRecallGenerationsResult {
  deletedGenerationIds: string[];
}

/** Collects only validated expired rollback material never selected by the active pointer. */
async function collectRetiredRecallGenerationsWithLock(
  options: CollectRetiredRecallGenerationsOptions,
): Promise<CollectRetiredRecallGenerationsResult> {
  const collection = await prepareRetiredRecallGenerationCollectionTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    ...(options.nowEpochMilliseconds ? { nowEpochMilliseconds: options.nowEpochMilliseconds } : {}),
  });
  if (collection.candidateGenerationIds.length === 0) {
    return { deletedGenerationIds: [] };
  }

  const deletedGenerationIds: string[] = [];
  for (const generationId of collection.candidateGenerationIds) {
    createRecallActiveGenerationPointer(generationId);
    const candidatePath = join(options.generationRootDirectory, generationId);
    try {
      await access(candidatePath);
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        deletedGenerationIds.push(generationId);
        continue;
      }
      throw error;
    }
    const generationDirectory = await resolveRecallGenerationDirectory(
      options.generationRootDirectory,
      generationId,
    );
    await rm(generationDirectory, { recursive: true });
    deletedGenerationIds.push(generationId);
    await options.generationCollectionFault?.('after_generation_directory_delete', generationId);
  }
  const completion = await completeRetiredRecallGenerationCollectionTransition({
    activeGenerationPointerPath: options.activeGenerationPointerPath,
    generationRegistryPath: options.generationRegistryPath,
    deletedGenerationIds,
  });
  if (completion.removeRetainedMarkers && options.retainedMarkerDirectory) {
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

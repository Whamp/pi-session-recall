import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';

/** Verifies pointer and registry agreement before any active target-generation read. */
export async function readActiveTargetRecallManifestFingerprint(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
): Promise<string> {
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(config.activeGenerationPointerPath),
    readRecallGenerationRegistry(config.generationRegistryPath),
  ]);
  if (pointer === null) {
    throw new Error('Recall target generation active pointer missing');
  }
  if (registry === null) {
    throw new Error('Recall target generation registry missing');
  }
  if (
    pointer.activeGenerationId !== generationId ||
    registry.activeGenerationId !== generationId ||
    registry.activePointerChecksum !== pointer.checksum
  ) {
    throw new Error(
      `Recall target generation pointer and registry disagree for selected generation ${generationId}`,
    );
  }
  const activeEntry = registry.generations.find((entry) => entry.generationId === generationId);
  if (activeEntry === undefined) {
    throw new Error(`Recall target generation active registry entry missing for ${generationId}`);
  }
  if (
    activeEntry.embeddingProfileId !== undefined &&
    activeEntry.embeddingProfileId !== config.embeddingProfileId
  ) {
    throw new Error(
      `Recall target generation active embedding profile mismatch: expected ${config.embeddingProfileId}, received ${activeEntry.embeddingProfileId}`,
    );
  }
  return activeEntry.indexManifestFingerprint;
}

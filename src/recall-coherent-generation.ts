import { existsSync } from 'node:fs';
import { mkdir, rm } from 'node:fs/promises';
import { join } from 'node:path';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  assertRecallGenerationManifestCompatible,
  createRecallGenerationManifest,
  readRecallGenerationManifest,
  writeRecallGenerationManifest,
  type RecallGenerationManifest,
} from './recall-generation-manifest.js';
import {
  createEmptyRecallGenerationStores,
  createRecallGenerationComponentPaths,
  createRecallGenerationStoreContracts,
  readRecallGenerationStoreRecordMembership,
  validateEmptyRecallGenerationStores,
  validateRecallGenerationStores,
  type RecallGenerationComponentPaths,
  type RecallGenerationStoreCounts,
} from './recall-generation-stores.js';
import {
  assertRecallGenerationValidationReceipt,
  createEmptyRecallGenerationValidationReceipt,
  createRecallGenerationValidationReceipt,
  readRecallGenerationValidationReceipt,
  writeRecallGenerationValidationReceipt,
} from './recall-generation-validation-receipt.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  resolveRecallGenerationDirectory,
} from './recall-generation-state.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';
import type { RecallProjectLineages } from './resolve-project-identity.js';

/** Explicit identifier for creating one inactive empty coherent recall generation. */
export interface CreateEmptyRecallGenerationOptions {
  generationId: string;
}

/** Validated read-only identity and exact membership of one coherent recall generation. */
export interface OpenedValidatedRecallGeneration {
  generationId: string;
  generationDirectory: string;
  manifestPath: string;
  validationReceiptPath: string;
  manifestFingerprint: string;
  startingSnapshotFingerprint: string;
  storeCounts: RecallGenerationStoreCounts;
}

/** Configured semantic and coordination inputs for inactive coherent generation operations. */
export interface RecallCoherentGenerationConfig {
  sessionsDirectory: string;
  generationRootDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  embeddingProfileId: string;
  embeddingProfile: Readonly<RecallEmbeddingModelProfile>;
  projectLineages: RecallProjectLineages;
  chunkPolicy?: Readonly<RecallChunkPolicy>;
  nowEpochMilliseconds?: () => number;
}

function createExpectedRecallGenerationManifest(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
): RecallGenerationManifest {
  return createRecallGenerationManifest({
    generationId,
    embeddingProfileId: config.embeddingProfileId,
    embeddingProfile: config.embeddingProfile,
    projectLineages: config.projectLineages,
    ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
  });
}

async function resolveCoherentRecallGenerationPaths(
  generationRootDirectory: string,
  generationId: string,
): Promise<RecallGenerationComponentPaths> {
  let generationDirectory: string;
  try {
    generationDirectory = await resolveRecallGenerationDirectory(
      generationRootDirectory,
      generationId,
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation directory invalid for ${generationId}: ${message}`,
      { cause: error },
    );
  }
  return createRecallGenerationComponentPaths(generationDirectory);
}

/** Opens and completely validates one empty generation without session or cache access. */
export async function openValidatedRecallGeneration(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
): Promise<OpenedValidatedRecallGeneration> {
  const paths = await resolveCoherentRecallGenerationPaths(
    config.generationRootDirectory,
    generationId,
  );
  if (existsSync(paths.recoveryRecordPath)) {
    throw new Error(
      `Recall coherent generation recovery required for ${generationId}: ${paths.recoveryRecordPath}`,
    );
  }
  const expectedManifest = createExpectedRecallGenerationManifest(config, generationId);
  const { manifest, fingerprint } = await readRecallGenerationManifest(paths.manifestPath);
  assertRecallGenerationManifestCompatible(manifest, expectedManifest, paths.manifestPath);
  const receipt = await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  const contracts = createRecallGenerationStoreContracts(
    generationId,
    expectedManifest.embeddingProfile.storedDimensions,
  );
  const recordIds = await readRecallGenerationStoreRecordMembership(paths);
  const receiptCounts = receipt.exactMembership;
  for (const [responsibility, expectedCount, actualCount] of [
    ['lexical-source', receiptCounts.lexicalSource.count, recordIds.lexicalSource.length],
    ['dense-evidence', receiptCounts.dense.count, recordIds.dense.length],
    [
      'session-projection',
      receiptCounts.sessionProjection.count,
      recordIds.sessionProjection.length,
    ],
  ] as const) {
    if (actualCount !== expectedCount) {
      throw new Error(
        `Recall coherent generation ${responsibility} membership mismatch: expected ${expectedCount} rows, received ${actualCount}`,
      );
    }
  }
  const storeCounts = validateRecallGenerationStores(paths, contracts, generationId, recordIds);
  const expectedReceipt = createRecallGenerationValidationReceipt({
    generationId,
    manifestFingerprint: fingerprint,
    membership: {
      startingSnapshotFingerprint: receipt.startingSnapshot.fingerprint,
      physicalSourceCount: receipt.startingSnapshot.physicalSourceCount,
      logicalSessionOccurrenceCount: receipt.startingSnapshot.logicalSessionOccurrenceCount,
      lexicalSourceRecordIds: recordIds.lexicalSource,
      denseRecordIds: recordIds.dense,
      sessionProjectionRecordIds: recordIds.sessionProjection,
    },
    validatedAtEpochMilliseconds: receipt.validatedAtEpochMilliseconds,
  });
  assertRecallGenerationValidationReceipt(receipt, expectedReceipt, paths.validationReceiptPath);
  return {
    generationId,
    generationDirectory: paths.generationDirectory,
    manifestPath: paths.manifestPath,
    validationReceiptPath: paths.validationReceiptPath,
    manifestFingerprint: fingerprint,
    startingSnapshotFingerprint: receipt.startingSnapshot.fingerprint,
    storeCounts,
  };
}

/** Creates, closes, reopens, validates, receipts, and reopens one inactive empty generation. */
export async function createEmptyRecallGeneration(
  config: Readonly<RecallCoherentGenerationConfig>,
  options: Readonly<CreateEmptyRecallGenerationOptions>,
): Promise<OpenedValidatedRecallGeneration> {
  createRecallActiveGenerationPointer(options.generationId);
  await mkdir(config.generationRootDirectory, { recursive: true });
  const generationDirectory = join(config.generationRootDirectory, options.generationId);
  try {
    await mkdir(generationDirectory);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation creation refused for ${options.generationId}: ${message}`,
      { cause: error },
    );
  }
  const paths = await resolveCoherentRecallGenerationPaths(
    config.generationRootDirectory,
    options.generationId,
  );
  const manifest = createExpectedRecallGenerationManifest(config, options.generationId);
  const manifestFingerprint = await writeRecallGenerationManifest(paths.manifestPath, manifest);
  const contracts = createRecallGenerationStoreContracts(
    options.generationId,
    manifest.embeddingProfile.storedDimensions,
  );
  createEmptyRecallGenerationStores(paths, contracts);
  const storeCounts = validateEmptyRecallGenerationStores(paths, contracts, options.generationId);
  const receipt = createEmptyRecallGenerationValidationReceipt({
    generationId: options.generationId,
    manifestFingerprint,
    storeCounts,
    validatedAtEpochMilliseconds: (config.nowEpochMilliseconds ?? Date.now)(),
  });
  await writeRecallGenerationValidationReceipt(paths.validationReceiptPath, receipt);
  return openValidatedRecallGeneration(config, options.generationId);
}

/** Deletes one generation only when neither pointer nor any registry role protects it. */
export async function deleteUnprotectedRecallGeneration(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
): Promise<void> {
  const paths = await resolveCoherentRecallGenerationPaths(
    config.generationRootDirectory,
    generationId,
  );
  const [pointer, registry] = await Promise.all([
    readRecallActiveGenerationPointer(config.activeGenerationPointerPath),
    readRecallGenerationRegistry(config.generationRegistryPath),
  ]);
  if (pointer?.activeGenerationId === generationId) {
    throw new Error(
      `Recall coherent generation deletion refused for protected active generation ${generationId}`,
    );
  }
  if (registry?.generations.some((generation) => generation.generationId === generationId)) {
    throw new Error(
      `Recall coherent generation deletion refused for registry-protected generation ${generationId}`,
    );
  }
  await rm(paths.generationDirectory, { recursive: true, force: false });
}

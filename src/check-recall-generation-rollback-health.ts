import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';

import type { ZVecCollection } from '@zvec/zvec';

import { RecallSessionProjectionKind } from './enums.js';
import {
  parseRecallGenerationPhysicalProjectionArtifact,
  type RecallGenerationPhysicalProjectionArtifact,
  type RecallPhysicalSourceExpectedMembership,
} from './recall-generation-physical-projection.js';
import {
  createRecallGenerationComponentPaths,
  openRecallGenerationStoreForBoundedCheck,
  readRecallGenerationVectorValues,
} from './recall-generation-stores.js';
import { readRecallGenerationManifest } from './recall-generation-manifest.js';
import { resolveRecallGenerationDirectory } from './recall-generation-state.js';
import { readRecallGenerationValidationReceipt } from './recall-generation-validation-receipt.js';
import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';

/** Artifact and registry identity required for one bounded rollback health check. */
export interface CheckRecallGenerationRollbackHealthOptions {
  generationRootDirectory: string;
  generationId: string;
  expectedManifestFingerprint: string;
}

interface ProjectionSelectedCanary {
  recordId: string;
  physicalSourceIdentity: string;
}

function wrapRollbackHealthError(responsibility: string, error: unknown): Error {
  const message = error instanceof Error ? error.message : String(error);
  return new Error(`Recall rollback health ${responsibility}: ${message}`, { cause: error });
}

function sumExpectedMembershipCounts(
  artifacts: readonly RecallGenerationPhysicalProjectionArtifact[],
  responsibility: keyof RecallPhysicalSourceExpectedMembership,
): number {
  return artifacts.reduce(
    (total, artifact) => total + artifact.expectedMembership[responsibility].count,
    0,
  );
}

function selectProjectionCanary(
  artifacts: readonly RecallGenerationPhysicalProjectionArtifact[],
  responsibility: keyof RecallPhysicalSourceExpectedMembership,
): ProjectionSelectedCanary | null {
  for (const artifact of artifacts) {
    const recordId = artifact.expectedMembership[responsibility].canaryRecordId;
    if (recordId !== null) {
      return { recordId, physicalSourceIdentity: artifact.physicalSourceIdentity };
    }
  }
  return null;
}

function assertStoreCount(
  responsibility: 'lexical-source' | 'dense-evidence' | 'session-projection',
  actualCount: number,
  expectedCount: number,
): void {
  if (actualCount !== expectedCount) {
    throw new Error(
      `Recall rollback health ${responsibility} count mismatch: expected ${expectedCount}, received ${actualCount}`,
    );
  }
}

function assertFetchedCanary(
  responsibility: 'lexical-source' | 'dense-evidence' | 'session-projection',
  collection: ZVecCollection,
  lexicalSourceCollection: ZVecCollection,
  canary: ProjectionSelectedCanary | null,
  generationId: string,
  embeddingProfileId: string,
  storedDimensions: number,
): void {
  if (canary === null) {
    return;
  }
  const outputFields =
    responsibility === 'dense-evidence'
      ? [
          'generationId',
          'physicalSourceIdentity',
          'evidenceOccurrenceId',
          'embeddingProfileId',
          'storedDimensions',
          'evidenceChecksum',
          'embeddingInputChecksum',
          'vectorChecksum',
        ]
      : responsibility === 'lexical-source'
        ? [
            'generationId',
            'physicalSourceIdentity',
            'recordKind',
            'evidenceOccurrenceId',
            'evidenceChecksum',
          ]
        : ['generationId', 'physicalSourceIdentity', 'projectionKind', 'projectionJson'];
  const record = collection.fetchSync({
    ids: [canary.recordId],
    outputFields,
    includeVector: responsibility === 'dense-evidence',
  })[canary.recordId];
  if (
    record === undefined ||
    record.id !== canary.recordId ||
    record.fields.generationId !== generationId ||
    record.fields.physicalSourceIdentity !== canary.physicalSourceIdentity
  ) {
    throw new Error(`Recall rollback health ${responsibility} canary mismatch: ${canary.recordId}`);
  }
  if (responsibility === 'dense-evidence') {
    const lexicalRecord = lexicalSourceCollection.fetchSync({
      ids: [canary.recordId],
      outputFields: [
        'generationId',
        'physicalSourceIdentity',
        'recordKind',
        'isDenseSearchable',
        'evidenceOccurrenceId',
        'evidenceChecksum',
      ],
      includeVector: false,
    })[canary.recordId];
    const vector = readRecallGenerationVectorValues(record.vectors.embedding);
    const vectorChecksum = createHash('sha256')
      .update(Buffer.from(new Float32Array(vector).buffer))
      .digest('hex');
    if (
      record.fields.evidenceOccurrenceId !== canary.recordId ||
      record.fields.embeddingProfileId !== embeddingProfileId ||
      record.fields.storedDimensions !== storedDimensions ||
      !/^[a-f0-9]{64}$/u.test(String(record.fields.evidenceChecksum)) ||
      !/^[a-f0-9]{64}$/u.test(String(record.fields.embeddingInputChecksum)) ||
      record.fields.vectorChecksum !== vectorChecksum ||
      vector.length !== storedDimensions ||
      vector.some((value) => !Number.isFinite(value)) ||
      lexicalRecord === undefined ||
      lexicalRecord.fields.generationId !== generationId ||
      lexicalRecord.fields.physicalSourceIdentity !== canary.physicalSourceIdentity ||
      lexicalRecord.fields.recordKind !== 'evidence' ||
      lexicalRecord.fields.isDenseSearchable !== true ||
      lexicalRecord.fields.evidenceOccurrenceId !== canary.recordId ||
      lexicalRecord.fields.evidenceChecksum !== record.fields.evidenceChecksum
    ) {
      throw new Error(`Recall rollback health dense-evidence canary mismatch: ${canary.recordId}`);
    }
  }
  if (
    responsibility === 'lexical-source' &&
    record.fields.recordKind === 'evidence' &&
    (record.fields.evidenceOccurrenceId !== canary.recordId ||
      !/^[a-f0-9]{64}$/u.test(String(record.fields.evidenceChecksum)))
  ) {
    throw new Error(`Recall rollback health lexical-source canary mismatch: ${canary.recordId}`);
  }
  if (responsibility === 'session-projection') {
    if (record.fields.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION) {
      const artifact = parseRecallGenerationPhysicalProjectionArtifact(
        record.fields.projectionJson,
        generationId,
      );
      if (artifact.physicalSourceIdentity !== canary.physicalSourceIdentity) {
        throw new Error(
          `Recall rollback health session-projection canary mismatch: ${canary.recordId}`,
        );
      }
    } else if (record.fields.projectionKind !== RecallSessionProjectionKind.LOGICAL_SESSION) {
      throw new Error(
        `Recall rollback health session-projection canary mismatch: ${canary.recordId}`,
      );
    }
  }
}

/**
 * Checks rollback artifacts, schemas, projection-derived counts, and fixed canaries only.
 * It never reads session sources, embeds, rewrites receipts, or enumerates evidence rows.
 */
export async function checkRecallGenerationRollbackHealth(
  options: Readonly<CheckRecallGenerationRollbackHealthOptions>,
): Promise<void> {
  const generationDirectory = await resolveRecallGenerationDirectory(
    options.generationRootDirectory,
    options.generationId,
  );
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  if (existsSync(paths.recoveryRecordPath)) {
    throw new Error(
      `Recall rollback health recovery required for ${options.generationId}: ${paths.recoveryRecordPath}`,
    );
  }
  let manifestResult: Awaited<ReturnType<typeof readRecallGenerationManifest>>;
  try {
    manifestResult = await readRecallGenerationManifest(paths.manifestPath);
  } catch (error) {
    throw wrapRollbackHealthError('manifest unreadable', error);
  }
  if (
    manifestResult.manifest.generationId !== options.generationId ||
    manifestResult.fingerprint !== options.expectedManifestFingerprint
  ) {
    throw new Error(
      `Recall rollback health manifest fingerprint mismatch for ${options.generationId}`,
    );
  }
  let receipt: Awaited<ReturnType<typeof readRecallGenerationValidationReceipt>>;
  try {
    receipt = await readRecallGenerationValidationReceipt(paths.validationReceiptPath);
  } catch (error) {
    throw wrapRollbackHealthError('validation receipt unreadable', error);
  }
  if (
    receipt.generationId !== options.generationId ||
    receipt.manifestFingerprint !== manifestResult.fingerprint
  ) {
    throw new Error(
      `Recall rollback health validation receipt fingerprint mismatch for ${options.generationId}`,
    );
  }

  const contracts = manifestResult.manifest.stores;
  const collections: ZVecCollection[] = [];
  let lexicalSource: ZVecCollection;
  let dense: ZVecCollection;
  let sessionProjection: ZVecCollection;
  try {
    try {
      lexicalSource = openRecallGenerationStoreForBoundedCheck(
        paths.lexicalSourceStorePath,
        contracts.lexicalSource,
      );
      collections.push(lexicalSource);
    } catch (error) {
      throw wrapRollbackHealthError('lexical-source store', error);
    }
    try {
      dense = openRecallGenerationStoreForBoundedCheck(paths.denseStorePath, contracts.dense);
      collections.push(dense);
    } catch (error) {
      throw wrapRollbackHealthError('dense-evidence store', error);
    }
    try {
      sessionProjection = openRecallGenerationStoreForBoundedCheck(
        paths.sessionProjectionStorePath,
        contracts.sessionProjection,
      );
      collections.push(sessionProjection);
    } catch (error) {
      throw wrapRollbackHealthError('session-projection store', error);
    }

    const physicalArtifacts: RecallGenerationPhysicalProjectionArtifact[] = [];
    visitExactZvecDocuments(
      sessionProjection,
      {
        filter: `projectionKind = '${RecallSessionProjectionKind.PHYSICAL_SESSION}'`,
        uniquePartitionField: 'physicalSourceIdentity',
        outputFields: ['projectionJson'],
      },
      ({ fields }) => {
        physicalArtifacts.push(
          parseRecallGenerationPhysicalProjectionArtifact(
            fields.projectionJson,
            options.generationId,
          ),
        );
      },
    );
    physicalArtifacts.sort((left, right) =>
      left.physicalSourceIdentity.localeCompare(right.physicalSourceIdentity),
    );
    const expectedLexicalSourceCount = sumExpectedMembershipCounts(
      physicalArtifacts,
      'lexicalSource',
    );
    const expectedDenseCount = sumExpectedMembershipCounts(physicalArtifacts, 'dense');
    const expectedSessionProjectionCount = sumExpectedMembershipCounts(
      physicalArtifacts,
      'sessionProjection',
    );
    assertStoreCount('lexical-source', lexicalSource.stats.docCount, expectedLexicalSourceCount);
    assertStoreCount('dense-evidence', dense.stats.docCount, expectedDenseCount);
    assertStoreCount(
      'session-projection',
      sessionProjection.stats.docCount,
      expectedSessionProjectionCount,
    );
    assertFetchedCanary(
      'lexical-source',
      lexicalSource,
      lexicalSource,
      selectProjectionCanary(physicalArtifacts, 'lexicalSource'),
      options.generationId,
      manifestResult.manifest.embeddingProfile.profileId,
      manifestResult.manifest.embeddingProfile.storedDimensions,
    );
    assertFetchedCanary(
      'dense-evidence',
      dense,
      lexicalSource,
      selectProjectionCanary(physicalArtifacts, 'dense'),
      options.generationId,
      manifestResult.manifest.embeddingProfile.profileId,
      manifestResult.manifest.embeddingProfile.storedDimensions,
    );
    assertFetchedCanary(
      'session-projection',
      sessionProjection,
      lexicalSource,
      selectProjectionCanary(physicalArtifacts, 'sessionProjection'),
      options.generationId,
      manifestResult.manifest.embeddingProfile.profileId,
      manifestResult.manifest.embeddingProfile.storedDimensions,
    );
  } finally {
    for (const collection of collections.reverse()) {
      collection.closeSync();
    }
  }
}

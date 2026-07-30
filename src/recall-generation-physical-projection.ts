import { createHash } from 'node:crypto';

import { RecallSessionProjectionKind } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';

/** Count, digest, and deterministic canary for one physical source's store membership. */
export interface RecallPhysicalSourceStoreMembership {
  count: number;
  digest: string;
  canaryRecordId: string | null;
}

/** Projection-owned totals and canaries used by bounded rollback health checks. */
export interface RecallPhysicalSourceExpectedMembership {
  lexicalSource: RecallPhysicalSourceStoreMembership;
  dense: RecallPhysicalSourceStoreMembership;
  sessionProjection: RecallPhysicalSourceStoreMembership;
}

/** Minimal physical projection artifact needed without source reads or evidence scans. */
export interface RecallGenerationPhysicalProjectionArtifact {
  generationId: string;
  physicalSourceIdentity: string;
  expectedMembership: RecallPhysicalSourceExpectedMembership;
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/u;

/** Creates a stable physical-source membership fingerprint and canary from exact record IDs. */
export function createRecallPhysicalSourceStoreMembership(
  recordIds: readonly string[],
): RecallPhysicalSourceStoreMembership {
  const sortedRecordIds = [...recordIds].toSorted();
  return {
    count: sortedRecordIds.length,
    digest: createHash('sha256').update(JSON.stringify(sortedRecordIds)).digest('hex'),
    canaryRecordId: sortedRecordIds[0] ?? null,
  };
}

function parseStoreMembership(
  value: unknown,
  responsibility: keyof RecallPhysicalSourceExpectedMembership,
): RecallPhysicalSourceStoreMembership {
  if (
    !isUnknownRecord(value) ||
    !Number.isInteger(value.count) ||
    Number(value.count) < 0 ||
    typeof value.digest !== 'string' ||
    !SHA256_PATTERN.test(value.digest) ||
    !(
      value.canaryRecordId === null ||
      (typeof value.canaryRecordId === 'string' && value.canaryRecordId.length > 0)
    )
  ) {
    throw new Error(
      `Recall rollback health physical projection ${responsibility} membership invalid`,
    );
  }
  if ((value.count === 0) !== (value.canaryRecordId === null)) {
    throw new Error(`Recall rollback health physical projection ${responsibility} canary invalid`);
  }
  return {
    count: Number(value.count),
    digest: value.digest,
    canaryRecordId: value.canaryRecordId,
  };
}

/** Parses one physical projection's bounded rollback totals without decoding source progress. */
export function parseRecallGenerationPhysicalProjectionArtifact(
  projectionJson: unknown,
  expectedGenerationId: string,
): RecallGenerationPhysicalProjectionArtifact {
  if (typeof projectionJson !== 'string') {
    throw new Error('Recall rollback health physical projection JSON missing');
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(projectionJson);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall rollback health physical projection JSON invalid: ${message}`, {
      cause: error,
    });
  }
  if (
    !isUnknownRecord(parsed) ||
    parsed.schemaVersion !== 1 ||
    parsed.generationId !== expectedGenerationId ||
    parsed.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION ||
    !isUnknownRecord(parsed.physicalSource) ||
    typeof parsed.physicalSource.physicalSourceIdentity !== 'string' ||
    !isUnknownRecord(parsed.expectedMembership)
  ) {
    throw new Error(
      `Recall rollback health physical projection artifact invalid for ${expectedGenerationId}`,
    );
  }
  return {
    generationId: expectedGenerationId,
    physicalSourceIdentity: parsed.physicalSource.physicalSourceIdentity,
    expectedMembership: {
      lexicalSource: parseStoreMembership(parsed.expectedMembership.lexicalSource, 'lexicalSource'),
      dense: parseStoreMembership(parsed.expectedMembership.dense, 'dense'),
      sessionProjection: parseStoreMembership(
        parsed.expectedMembership.sessionProjection,
        'sessionProjection',
      ),
    },
  };
}

import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallGenerationStoreCounts } from './recall-generation-stores.js';

/** Current immutable success-receipt format for complete recall generation validation. */
export const RECALL_GENERATION_VALIDATION_RECEIPT_VERSION = 1;

/** Complete success evidence for reopened schemas and exact empty generation membership. */
export interface RecallGenerationValidationReceipt {
  receiptVersion: 1;
  generationId: string;
  successful: true;
  manifestFingerprint: string;
  startingSnapshot: Readonly<{
    version: 1;
    fingerprint: string;
    physicalSourceCount: 0;
    logicalSessionOccurrenceCount: 0;
  }>;
  exactMembership: Readonly<{
    lexicalSource: Readonly<{ count: 0; digest: string }>;
    dense: Readonly<{ count: 0; digest: string }>;
    sessionProjection: Readonly<{ count: 0; digest: string }>;
  }>;
  validationPolicyVersion: 1;
  canaryResults: Readonly<{
    storeSchemas: 'passed';
    storeIdentities: 'passed';
    exactMembership: 'passed';
  }>;
  validatedAtEpochMilliseconds: number;
}

const checksumSchema = Type.String({ pattern: '^[a-f0-9]{64}$' });
const emptyMembershipSchema = Type.Object(
  { count: Type.Literal(0), digest: checksumSchema },
  { additionalProperties: false },
);
const recallGenerationValidationReceiptSchema = Type.Object(
  {
    receiptVersion: Type.Literal(RECALL_GENERATION_VALIDATION_RECEIPT_VERSION),
    generationId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    successful: Type.Literal(true),
    manifestFingerprint: checksumSchema,
    startingSnapshot: Type.Object(
      {
        version: Type.Literal(1),
        fingerprint: checksumSchema,
        physicalSourceCount: Type.Literal(0),
        logicalSessionOccurrenceCount: Type.Literal(0),
      },
      { additionalProperties: false },
    ),
    exactMembership: Type.Object(
      {
        lexicalSource: emptyMembershipSchema,
        dense: emptyMembershipSchema,
        sessionProjection: emptyMembershipSchema,
      },
      { additionalProperties: false },
    ),
    validationPolicyVersion: Type.Literal(1),
    canaryResults: Type.Object(
      {
        storeSchemas: Type.Literal('passed'),
        storeIdentities: Type.Literal('passed'),
        exactMembership: Type.Literal('passed'),
      },
      { additionalProperties: false },
    ),
    validatedAtEpochMilliseconds: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

function calculateSha256(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function createEmptyMembershipDigest(storeName: string): string {
  return calculateSha256(JSON.stringify({ store: storeName, sortedRecordIds: [] }));
}

/** Canonical identity of the no-source snapshot certified by an empty generation. */
export const EMPTY_RECALL_GENERATION_STARTING_SNAPSHOT_FINGERPRINT = calculateSha256(
  JSON.stringify({
    version: 1,
    physicalSources: [],
    logicalSessionOccurrences: [],
    evidenceOccurrences: [],
    entryAnchors: [],
    projections: [],
  }),
);

/** Builds a success receipt only from reopened stores whose exact memberships are empty. */
export function createEmptyRecallGenerationValidationReceipt(options: {
  generationId: string;
  manifestFingerprint: string;
  storeCounts: Readonly<RecallGenerationStoreCounts>;
  validatedAtEpochMilliseconds: number;
}): RecallGenerationValidationReceipt {
  if (
    options.storeCounts.lexicalSource !== 0 ||
    options.storeCounts.dense !== 0 ||
    options.storeCounts.sessionProjection !== 0
  ) {
    throw new Error(
      'Recall coherent generation validation receipt requires exact empty membership',
    );
  }
  const receipt: RecallGenerationValidationReceipt = {
    receiptVersion: RECALL_GENERATION_VALIDATION_RECEIPT_VERSION,
    generationId: options.generationId,
    successful: true,
    manifestFingerprint: options.manifestFingerprint,
    startingSnapshot: {
      version: 1,
      fingerprint: EMPTY_RECALL_GENERATION_STARTING_SNAPSHOT_FINGERPRINT,
      physicalSourceCount: 0,
      logicalSessionOccurrenceCount: 0,
    },
    exactMembership: {
      lexicalSource: {
        count: 0,
        digest: createEmptyMembershipDigest('lexical-source'),
      },
      dense: { count: 0, digest: createEmptyMembershipDigest('dense') },
      sessionProjection: {
        count: 0,
        digest: createEmptyMembershipDigest('session-projection'),
      },
    },
    validationPolicyVersion: 1,
    canaryResults: {
      storeSchemas: 'passed',
      storeIdentities: 'passed',
      exactMembership: 'passed',
    },
    validatedAtEpochMilliseconds: options.validatedAtEpochMilliseconds,
  };
  return Value.Parse(recallGenerationValidationReceiptSchema, receipt);
}

/** Writes one immutable successful validation receipt and refuses replacement. */
export async function writeRecallGenerationValidationReceipt(
  receiptPath: string,
  receipt: RecallGenerationValidationReceipt,
): Promise<void> {
  const validated = Value.Parse(recallGenerationValidationReceiptSchema, receipt);
  await writeFile(receiptPath, `${JSON.stringify(validated, null, 2)}\n`, {
    encoding: 'utf8',
    flag: 'wx',
  });
}

/** Reads one immutable validation receipt through its strict success-only contract. */
export async function readRecallGenerationValidationReceipt(
  receiptPath: string,
): Promise<RecallGenerationValidationReceipt> {
  let content: string;
  try {
    content = await readFile(receiptPath, 'utf8');
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation validation receipt unreadable at ${receiptPath}: ${message}`,
      { cause: error },
    );
  }
  try {
    const parsed: unknown = JSON.parse(content);
    return Value.Parse(recallGenerationValidationReceiptSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation validation receipt invalid at ${receiptPath}: ${message}`,
      { cause: error },
    );
  }
}

/** Rejects a receipt not bound to this manifest, generation, empty snapshot, and membership. */
export function assertEmptyRecallGenerationValidationReceipt(
  actual: RecallGenerationValidationReceipt,
  expected: RecallGenerationValidationReceipt,
  receiptPath: string,
): void {
  const comparableActual = { ...actual, validatedAtEpochMilliseconds: 0 };
  const comparableExpected = { ...expected, validatedAtEpochMilliseconds: 0 };
  if (JSON.stringify(comparableActual) !== JSON.stringify(comparableExpected)) {
    throw new Error(`Recall coherent generation validation receipt mismatch at ${receiptPath}`);
  }
}

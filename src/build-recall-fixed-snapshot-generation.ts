import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, open, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecOpen, type ZVecCollection, type ZVecStatus } from '@zvec/zvec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallFixedSnapshotBuildFaultStage } from './enums.js';
import { InvalidRecallSessionSourceError } from './errors.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { openRecallZvecValidationStore } from './open-recall-zvec-validation-store.js';
import {
  openValidatedRecallGeneration,
  type OpenedValidatedRecallGeneration,
  type RecallCoherentGenerationConfig,
} from './recall-coherent-generation.js';
import {
  assertRecallGenerationManifestCompatible,
  readRecallGenerationManifest,
  writeRecallGenerationManifest,
} from './recall-generation-manifest.js';
import {
  createRecallGenerationComponentPaths,
  createRecallGenerationStoreContracts,
  readRecallGenerationStoreRecordMembership,
  resumeEmptyRecallGenerationStores,
  readRecallGenerationVectorValues,
  validateRecallGenerationDenseSubset,
  validateRecallGenerationStores,
} from './recall-generation-stores.js';
import {
  createRecallGenerationValidationReceipt,
  writeRecallGenerationValidationReceipt,
} from './recall-generation-validation-receipt.js';
import { createRecallActiveGenerationPointer } from './recall-generation-state.js';
import {
  createExpectedRecallPhysicalSourceManifest,
  materializeRecallPhysicalSourceGeneration,
  type CapturedRecallPhysicalSource,
  type CreateRecallGenerationFromPhysicalSourcesOptions,
  type MaterializedRecallPhysicalSourceGeneration,
  type RecallFixedSnapshotPhysicalSourceCheckpoint,
  type RecallGenerationDenseExpectation,
  type RecallPhysicalSourceGenerationDependencies,
} from './recall-physical-source-generation.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import {
  assertRepeatableStoredRecallEmbeddings,
  createStoredRecallEmbedding,
} from './recall-stored-embedding.js';

const FIXED_SNAPSHOT_BUILD_VERSION = 1;
const MAXIMUM_BUILD_WRITE_RECORDS = 32;
const FIXED_SNAPSHOT_STORE_SESSION_RECORD_LIMIT = 2_048;
const BOOTSTRAP_STATE_FILE = 'build-bootstrap.json';
const SNAPSHOT_DESCRIPTOR_FILE = 'build-snapshot.json';
const SNAPSHOT_SOURCE_DIRECTORY = 'build-sources';
const EXPECTED_SOURCE_DIRECTORY = 'expected-sources';

interface RecallGenerationScalarRow {
  id: string;
  fields: Record<string, unknown>;
}

interface RecallGenerationDenseRow extends RecallGenerationScalarRow {
  vectors: { embedding: number[] };
}

interface RecallFixedSnapshotStoreSession {
  lexicalSource: ZVecCollection;
  dense: ZVecCollection;
  sessionProjection: ZVecCollection;
}

interface RecallFixedSnapshotBootstrapSource {
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
}

interface RecallFixedSnapshotBootstrapState {
  version: 1;
  generationId: string;
  sources: RecallFixedSnapshotBootstrapSource[];
}

interface RecallFixedSnapshotSourceDescriptor {
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  sourceByteSize: number;
  sourceChecksum: string;
  sourceDevice: string;
  sourceInode: string;
  snapshotFileName: string;
}

interface RecallFixedSnapshotDescriptor {
  version: 1;
  generationId: string;
  manifestFingerprint: string;
  sources: RecallFixedSnapshotSourceDescriptor[];
}

interface RecallMaterializedExpectedPhysicalSourceArtifact extends MaterializedRecallPhysicalSourceGeneration {
  version: 1;
  generationId: string;
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  sourceByteSize: number;
  sourceChecksum: string;
}

interface RecallSkippedExpectedPhysicalSourceArtifact {
  version: 1;
  generationId: string;
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  sourceByteSize: number;
  sourceChecksum: string;
  skipReason: 'invalid-session-source';
  skipMessage: string;
}

type RecallExpectedPhysicalSourceArtifact =
  | RecallMaterializedExpectedPhysicalSourceArtifact
  | RecallSkippedExpectedPhysicalSourceArtifact;

const checksumSchema = Type.String({ pattern: '^[a-f0-9]{64}$' });
const scalarRowSchema = Type.Object(
  {
    id: Type.String({ minLength: 1 }),
    fields: Type.Record(Type.String(), Type.Unknown()),
  },
  { additionalProperties: false },
);
const denseExpectationSchema = Type.Object(
  {
    id: Type.String({ minLength: 1 }),
    fields: Type.Record(Type.String(), Type.Unknown()),
    embeddingInput: Type.String(),
  },
  { additionalProperties: false },
);
const fixedSnapshotBootstrapStateSchema = Type.Object(
  {
    version: Type.Literal(FIXED_SNAPSHOT_BUILD_VERSION),
    generationId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    sources: Type.Array(
      Type.Object(
        {
          physicalSourceIdentity: Type.String({ minLength: 1 }),
          sessionsRootRelativePath: Type.String({ minLength: 1 }),
        },
        { additionalProperties: false },
      ),
      { minItems: 1 },
    ),
  },
  { additionalProperties: false },
);
const fixedSnapshotDescriptorSchema = Type.Object(
  {
    version: Type.Literal(FIXED_SNAPSHOT_BUILD_VERSION),
    generationId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    manifestFingerprint: checksumSchema,
    sources: Type.Array(
      Type.Object(
        {
          physicalSourceIdentity: Type.String({ minLength: 1 }),
          sessionsRootRelativePath: Type.String({ minLength: 1 }),
          sourceByteSize: Type.Integer({ minimum: 0 }),
          sourceChecksum: checksumSchema,
          sourceDevice: Type.String({ minLength: 1 }),
          sourceInode: Type.String({ minLength: 1 }),
          snapshotFileName: Type.String({ pattern: '^[0-9]+-[a-f0-9]{64}\\.jsonl$' }),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);
const materializedExpectedPhysicalSourceArtifactSchema = Type.Object(
  {
    version: Type.Literal(FIXED_SNAPSHOT_BUILD_VERSION),
    generationId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    physicalSourceIdentity: Type.String({ minLength: 1 }),
    sessionsRootRelativePath: Type.String({ minLength: 1 }),
    sourceByteSize: Type.Integer({ minimum: 0 }),
    sourceChecksum: checksumSchema,
    lexicalSource: Type.Array(scalarRowSchema),
    dense: Type.Array(denseExpectationSchema),
    logicalSessionProjections: Type.Array(scalarRowSchema),
    physicalSessionProjection: scalarRowSchema,
    physicalSourceIdentities: Type.Array(Type.String({ minLength: 1 }), {
      minItems: 1,
      maxItems: 1,
    }),
    logicalSessionOccurrenceIds: Type.Array(Type.String({ minLength: 1 })),
  },
  { additionalProperties: false },
);
const skippedExpectedPhysicalSourceArtifactSchema = Type.Object(
  {
    version: Type.Literal(FIXED_SNAPSHOT_BUILD_VERSION),
    generationId: Type.String({ pattern: '^[A-Za-z0-9_-]+$' }),
    physicalSourceIdentity: Type.String({ minLength: 1 }),
    sessionsRootRelativePath: Type.String({ minLength: 1 }),
    sourceByteSize: Type.Integer({ minimum: 0 }),
    sourceChecksum: checksumSchema,
    skipReason: Type.Literal('invalid-session-source'),
    skipMessage: Type.String({ minLength: 1 }),
  },
  { additionalProperties: false },
);
const expectedPhysicalSourceArtifactSchema = Type.Union([
  materializedExpectedPhysicalSourceArtifactSchema,
  skippedExpectedPhysicalSourceArtifactSchema,
]);

function isSkippedExpectedPhysicalSourceArtifact(
  artifact: Readonly<RecallExpectedPhysicalSourceArtifact>,
): artifact is RecallSkippedExpectedPhysicalSourceArtifact {
  return 'skipReason' in artifact;
}

function countFixedSnapshotStoreSessionRecords(
  artifact: Readonly<RecallExpectedPhysicalSourceArtifact>,
): number {
  if (!('lexicalSource' in artifact)) {
    return 0;
  }
  return (
    artifact.lexicalSource.length +
    artifact.dense.length +
    artifact.logicalSessionProjections.length +
    1
  );
}

function calculateSha256(value: string | Uint8Array): string {
  return createHash('sha256').update(value).digest('hex');
}

function encodeStrictJson(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

function throwIfFixedSnapshotBuildCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error('Recall fixed snapshot generation build cancelled');
  }
}

async function invokeFixedSnapshotBuildFault(
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  stage: RecallFixedSnapshotBuildFaultStage,
  generationDirectory: string,
  physicalSourceIdentity?: string,
): Promise<void> {
  await dependencies.fixedSnapshotBuildFault?.(stage, {
    generationDirectory,
    ...(physicalSourceIdentity ? { physicalSourceIdentity } : {}),
  });
}

function assertCheckedBuildStatuses(
  operation: string,
  recordIds: readonly string[],
  statuses: readonly ZVecStatus[],
): void {
  if (statuses.length !== recordIds.length) {
    throw new Error(
      `Recall fixed snapshot generation ${operation} status mismatch: expected ${recordIds.length}, received ${statuses.length}`,
    );
  }
  for (const [index, status] of statuses.entries()) {
    if (!status.ok) {
      throw new Error(
        `Recall fixed snapshot generation ${operation} failed for ${recordIds[index] ?? 'unknown record'}: ${status.message}`,
      );
    }
  }
}

function fieldsMatch(
  actual: Readonly<Record<string, unknown>>,
  expected: Readonly<Record<string, unknown>>,
  ignoredFields: ReadonlySet<string> = new Set(),
): boolean {
  return Object.entries(expected).every(
    ([name, expectedValue]) =>
      ignoredFields.has(name) || JSON.stringify(actual[name]) === JSON.stringify(expectedValue),
  );
}

function createDenseReuseKey(expectation: Readonly<RecallGenerationDenseExpectation>): string {
  return `${String(expectation.fields.embeddingProfileId)}\0${String(
    expectation.fields.storedDimensions,
  )}\0${String(expectation.fields.embeddingInputChecksum)}\0${String(
    expectation.fields.evidenceChecksum,
  )}`;
}

function verifyDenseCandidate(
  candidate: ReturnType<ZVecCollection['fetchSync']>[string] | undefined,
  expectation: Readonly<RecallGenerationDenseExpectation>,
  permitCompatibleSourceGeneration = false,
): RecallGenerationDenseRow | null {
  const ignoredFields = new Set(['vectorChecksum']);
  if (permitCompatibleSourceGeneration) {
    ignoredFields.add('generationId');
  }
  if (
    candidate === undefined ||
    !fieldsMatch(candidate.fields, expectation.fields, ignoredFields)
  ) {
    return null;
  }
  const embedding = readRecallGenerationVectorValues(candidate.vectors.embedding);
  const expectedDimensions = expectation.fields.storedDimensions;
  if (
    typeof expectedDimensions !== 'number' ||
    embedding.length !== expectedDimensions ||
    embedding.some((value) => !Number.isFinite(value))
  ) {
    return null;
  }
  const vectorChecksum = calculateSha256(Buffer.from(new Float32Array(embedding).buffer));
  if (candidate.fields.vectorChecksum !== vectorChecksum) {
    return null;
  }
  return {
    id: expectation.id,
    fields: { ...expectation.fields, vectorChecksum },
    vectors: { embedding },
  };
}

async function writeFixedSnapshotBootstrapState(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  physicalSessionPaths: readonly string[],
  bootstrapStatePath: string,
): Promise<RecallFixedSnapshotBootstrapState> {
  const sources = physicalSessionPaths.map((physicalSessionPath) => {
    const identity = resolveRecallPhysicalSourceIdentity(
      config.sessionsDirectory,
      physicalSessionPath,
    );
    return {
      physicalSourceIdentity: identity.physicalSourceIdentity,
      sessionsRootRelativePath: identity.sessionsRootRelativePath,
    };
  });
  if (
    new Set(sources.map(({ physicalSourceIdentity }) => physicalSourceIdentity)).size !==
    sources.length
  ) {
    throw new Error(
      'Recall fixed snapshot generation bootstrap contains duplicate physical sources',
    );
  }
  const bootstrapState = Value.Parse(fixedSnapshotBootstrapStateSchema, {
    version: FIXED_SNAPSHOT_BUILD_VERSION,
    generationId,
    sources,
  });
  await writeFile(bootstrapStatePath, encodeStrictJson(bootstrapState), {
    encoding: 'utf8',
    flag: 'wx',
  });
  return bootstrapState;
}

async function readFixedSnapshotBootstrapState(
  bootstrapStatePath: string,
  generationId: string,
): Promise<RecallFixedSnapshotBootstrapState> {
  try {
    const parsed: unknown = JSON.parse(await readFile(bootstrapStatePath, 'utf8'));
    const bootstrapState = Value.Parse(fixedSnapshotBootstrapStateSchema, parsed);
    if (bootstrapState.generationId !== generationId) {
      throw new Error(`generation identity mismatch: ${bootstrapState.generationId}`);
    }
    if (
      new Set(bootstrapState.sources.map(({ physicalSourceIdentity }) => physicalSourceIdentity))
        .size !== bootstrapState.sources.length
    ) {
      throw new Error('duplicate physical sources');
    }
    return bootstrapState;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall fixed snapshot generation bootstrap state invalid for ${generationId}; discard this staging generation: ${message}`,
      { cause: error },
    );
  }
}

function resolveBootstrapPhysicalSessionPaths(
  config: Readonly<RecallCoherentGenerationConfig>,
  bootstrapState: Readonly<RecallFixedSnapshotBootstrapState>,
): string[] {
  try {
    return bootstrapState.sources.map((source) => {
      const physicalSessionPath = join(config.sessionsDirectory, source.sessionsRootRelativePath);
      const resolved = resolveRecallPhysicalSourceIdentity(
        config.sessionsDirectory,
        physicalSessionPath,
      );
      if (
        resolved.physicalSourceIdentity !== source.physicalSourceIdentity ||
        resolved.sessionsRootRelativePath !== source.sessionsRootRelativePath
      ) {
        throw new Error(`physical source identity mismatch: ${source.physicalSourceIdentity}`);
      }
      return physicalSessionPath;
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall fixed snapshot generation bootstrap sources invalid for ${bootstrapState.generationId}; discard this staging generation: ${message}`,
      { cause: error },
    );
  }
}

function fixedSnapshotBootstrapFaultStageForStore(
  responsibility: 'lexical-source' | 'dense-evidence' | 'session-projection',
): RecallFixedSnapshotBuildFaultStage {
  switch (responsibility) {
    case 'lexical-source':
      return RecallFixedSnapshotBuildFaultStage.AFTER_LEXICAL_SOURCE_STORE_CREATION;
    case 'dense-evidence':
      return RecallFixedSnapshotBuildFaultStage.AFTER_DENSE_STORE_CREATION;
    case 'session-projection':
      return RecallFixedSnapshotBuildFaultStage.AFTER_SESSION_PROJECTION_STORE_CREATION;
    default:
      throw new Error('Recall fixed snapshot generation bootstrap store unsupported');
  }
}

async function readStableFixedSnapshotSource(
  physicalSessionPath: string,
  generationDirectory: string,
  physicalSourceIdentity: string,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<{ sourceBytes: Buffer; sourceDevice: string; sourceInode: string }> {
  const sourceHandle = await open(physicalSessionPath, 'r');
  try {
    const metadataBeforeRead = await sourceHandle.stat({ bigint: true });
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_OPEN,
      generationDirectory,
      physicalSourceIdentity,
    );
    if (metadataBeforeRead.size > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new Error(
        `Recall fixed snapshot source exceeds the safe bounded-read size: ${physicalSessionPath}`,
      );
    }
    const sourceBytes = Buffer.alloc(Number(metadataBeforeRead.size));
    let readOffset = 0;
    while (readOffset < sourceBytes.length) {
      const { bytesRead } = await sourceHandle.read(
        sourceBytes,
        readOffset,
        sourceBytes.length - readOffset,
        readOffset,
      );
      if (bytesRead === 0) {
        throw new Error(
          `Recall fixed snapshot source ended during bounded descriptor read: ${physicalSessionPath}`,
        );
      }
      readOffset += bytesRead;
    }
    const metadataAfterRead = await sourceHandle.stat({ bigint: true });
    if (
      metadataBeforeRead.dev !== metadataAfterRead.dev ||
      metadataBeforeRead.ino !== metadataAfterRead.ino ||
      metadataBeforeRead.size !== metadataAfterRead.size ||
      metadataBeforeRead.mtimeNs !== metadataAfterRead.mtimeNs
    ) {
      throw new Error(
        `Recall fixed snapshot source changed during bounded descriptor read: ${physicalSessionPath}`,
      );
    }
    return {
      sourceBytes,
      sourceDevice: metadataAfterRead.dev.toString(),
      sourceInode: metadataAfterRead.ino.toString(),
    };
  } finally {
    await sourceHandle.close();
  }
}

async function captureFixedSourceSnapshot(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  physicalSessionPaths: readonly string[],
  signal: AbortSignal | undefined,
  generationDirectory: string,
  manifestFingerprint: string,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<RecallFixedSnapshotDescriptor> {
  const snapshotSourceDirectory = join(generationDirectory, SNAPSHOT_SOURCE_DIRECTORY);
  await mkdir(snapshotSourceDirectory);
  await invokeFixedSnapshotBuildFault(
    dependencies,
    RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_DIRECTORY_CREATION,
    generationDirectory,
  );
  await mkdir(join(generationDirectory, EXPECTED_SOURCE_DIRECTORY));
  await invokeFixedSnapshotBuildFault(
    dependencies,
    RecallFixedSnapshotBuildFaultStage.AFTER_EXPECTED_SOURCE_DIRECTORY_CREATION,
    generationDirectory,
  );
  const sources: RecallFixedSnapshotSourceDescriptor[] = [];
  for (const [index, physicalSessionPath] of physicalSessionPaths.entries()) {
    throwIfFixedSnapshotBuildCancelled(signal);
    const identity = resolveRecallPhysicalSourceIdentity(
      config.sessionsDirectory,
      physicalSessionPath,
    );
    const { sourceBytes, sourceDevice, sourceInode } = await readStableFixedSnapshotSource(
      physicalSessionPath,
      generationDirectory,
      identity.physicalSourceIdentity,
      dependencies,
    );
    const sourceChecksum = calculateSha256(sourceBytes);
    const snapshotFileName = `${index}-${sourceChecksum}.jsonl`;
    await writeFile(join(snapshotSourceDirectory, snapshotFileName), sourceBytes, { flag: 'wx' });
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_SOURCE_WRITE,
      generationDirectory,
      identity.physicalSourceIdentity,
    );
    sources.push({
      physicalSourceIdentity: identity.physicalSourceIdentity,
      sessionsRootRelativePath: identity.sessionsRootRelativePath,
      sourceByteSize: sourceBytes.length,
      sourceChecksum,
      sourceDevice,
      sourceInode,
      snapshotFileName,
    });
  }
  if (
    new Set(sources.map(({ physicalSourceIdentity }) => physicalSourceIdentity)).size !==
    sources.length
  ) {
    throw new Error('Recall fixed snapshot generation contains duplicate physical sources');
  }
  const descriptor = Value.Parse(fixedSnapshotDescriptorSchema, {
    version: FIXED_SNAPSHOT_BUILD_VERSION,
    generationId,
    manifestFingerprint,
    sources,
  });
  await writeFile(
    join(generationDirectory, SNAPSHOT_DESCRIPTOR_FILE),
    encodeStrictJson(descriptor),
    {
      encoding: 'utf8',
      flag: 'wx',
    },
  );
  return descriptor;
}

async function readFixedSnapshotDescriptor(
  descriptorPath: string,
): Promise<RecallFixedSnapshotDescriptor> {
  try {
    const parsed: unknown = JSON.parse(await readFile(descriptorPath, 'utf8'));
    return Value.Parse(fixedSnapshotDescriptorSchema, parsed);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall fixed snapshot generation descriptor invalid at ${descriptorPath}; discard this staging generation: ${message}`,
      {
        cause: error,
      },
    );
  }
}

async function readFixedSnapshotRecoveryPhysicalSourceIdentity(
  recoveryRecordPath: string,
  generationId: string,
): Promise<string> {
  try {
    const recovery: unknown = JSON.parse(await readFile(recoveryRecordPath, 'utf8'));
    if (!isUnknownRecord(recovery) || typeof recovery.physicalSourceIdentity !== 'string') {
      throw new Error('physical source identity missing');
    }
    return recovery.physicalSourceIdentity;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall fixed snapshot generation recovery record invalid at ${recoveryRecordPath} for ${generationId}: ${message}`,
      { cause: error },
    );
  }
}

function assertRequestedSourcesMatchSnapshot(
  config: Readonly<RecallCoherentGenerationConfig>,
  physicalSessionPaths: readonly string[],
  snapshot: Readonly<{
    generationId: string;
    sources: readonly RecallFixedSnapshotBootstrapSource[];
  }>,
): void {
  const requested = physicalSessionPaths.map((physicalSessionPath) =>
    resolveRecallPhysicalSourceIdentity(config.sessionsDirectory, physicalSessionPath),
  );
  const requestedIdentity = requested.map(
    ({ physicalSourceIdentity, sessionsRootRelativePath }) => ({
      physicalSourceIdentity,
      sessionsRootRelativePath,
    }),
  );
  const snapshotIdentity = snapshot.sources.map(
    ({ physicalSourceIdentity, sessionsRootRelativePath }) => ({
      physicalSourceIdentity,
      sessionsRootRelativePath,
    }),
  );
  if (JSON.stringify(requestedIdentity) !== JSON.stringify(snapshotIdentity)) {
    throw new Error(
      `Recall fixed snapshot generation resume source snapshot mismatch for ${snapshot.generationId}; discard this staging generation`,
    );
  }
}

async function readExpectedPhysicalSourceArtifact(
  artifactPath: string,
): Promise<{ artifact: RecallExpectedPhysicalSourceArtifact; fingerprint: string }> {
  try {
    const content = await readFile(artifactPath, 'utf8');
    const parsed: unknown = JSON.parse(content);
    return {
      artifact: Value.Parse(expectedPhysicalSourceArtifactSchema, parsed),
      fingerprint: calculateSha256(content),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall fixed snapshot generation expected source invalid at ${artifactPath}: ${message}`,
      { cause: error },
    );
  }
}

async function materializeExpectedPhysicalSource(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  generationDirectory: string,
  source: Readonly<RecallFixedSnapshotSourceDescriptor>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<{ artifact: RecallExpectedPhysicalSourceArtifact; fingerprint: string }> {
  const artifactPath = join(
    generationDirectory,
    EXPECTED_SOURCE_DIRECTORY,
    `${source.physicalSourceIdentity}.json`,
  );
  if (existsSync(artifactPath)) {
    return readExpectedPhysicalSourceArtifact(artifactPath);
  }
  const physicalSessionPath = join(config.sessionsDirectory, source.sessionsRootRelativePath);
  const snapshotPath = join(
    generationDirectory,
    SNAPSHOT_SOURCE_DIRECTORY,
    source.snapshotFileName,
  );
  const snapshotBytes = await readFile(snapshotPath);
  if (
    snapshotBytes.length !== source.sourceByteSize ||
    calculateSha256(snapshotBytes) !== source.sourceChecksum
  ) {
    throw new Error(
      `Recall fixed snapshot generation captured source mismatch for ${source.physicalSourceIdentity}; discard this staging generation`,
    );
  }
  let artifact: RecallExpectedPhysicalSourceArtifact;
  try {
    const capturedSource: CapturedRecallPhysicalSource = {
      sourceReadPath: snapshotPath,
      sourceByteSize: source.sourceByteSize,
      sourceDevice: source.sourceDevice,
      sourceInode: source.sourceInode,
    };
    const materialized = await materializeRecallPhysicalSourceGeneration(
      config,
      generationId,
      physicalSessionPath,
      capturedSource,
      dependencies,
    );
    artifact = Value.Parse(expectedPhysicalSourceArtifactSchema, {
      version: FIXED_SNAPSHOT_BUILD_VERSION,
      generationId,
      physicalSourceIdentity: source.physicalSourceIdentity,
      sessionsRootRelativePath: source.sessionsRootRelativePath,
      sourceByteSize: source.sourceByteSize,
      sourceChecksum: source.sourceChecksum,
      ...materialized,
    });
  } catch (error) {
    if (!(error instanceof InvalidRecallSessionSourceError)) {
      throw error;
    }
    artifact = Value.Parse(expectedPhysicalSourceArtifactSchema, {
      version: FIXED_SNAPSHOT_BUILD_VERSION,
      generationId,
      physicalSourceIdentity: source.physicalSourceIdentity,
      sessionsRootRelativePath: source.sessionsRootRelativePath,
      sourceByteSize: source.sourceByteSize,
      sourceChecksum: source.sourceChecksum,
      skipReason: 'invalid-session-source',
      skipMessage: error.message,
    });
  }
  const content = encodeStrictJson(artifact);
  await writeFile(artifactPath, content, { encoding: 'utf8', flag: 'wx' });
  return { artifact, fingerprint: calculateSha256(content) };
}

function upsertRowsInBoundedBatches(
  collection: ZVecCollection,
  operation: string,
  rows: readonly (RecallGenerationScalarRow | RecallGenerationDenseRow)[],
): void {
  if (rows.length === 0) {
    return;
  }
  const includeVector = rows.some((row) => 'vectors' in row);
  const outputFields = [...new Set(rows.flatMap(({ fields }) => Object.keys(fields)))];
  const fetched = collection.fetchSync({
    ids: rows.map(({ id }) => id),
    outputFields,
    includeVector,
  });
  const rowsRequiringWrite = rows.filter((row) => {
    const actual = fetched[row.id];
    if (actual === undefined || !fieldsMatch(actual.fields, row.fields)) {
      return true;
    }
    if (!('vectors' in row)) {
      return false;
    }
    const actualVectorChecksum = calculateSha256(
      Buffer.from(
        new Float32Array(readRecallGenerationVectorValues(actual.vectors.embedding)).buffer,
      ),
    );
    return actualVectorChecksum !== row.fields.vectorChecksum;
  });
  for (let offset = 0; offset < rowsRequiringWrite.length; offset += MAXIMUM_BUILD_WRITE_RECORDS) {
    const batch = rowsRequiringWrite.slice(offset, offset + MAXIMUM_BUILD_WRITE_RECORDS);
    assertCheckedBuildStatuses(
      operation,
      batch.map(({ id }) => id),
      collection.upsertSync(batch),
    );
  }
}

function verifyExpectedScalarRows(
  collection: ZVecCollection,
  responsibility: string,
  rows: readonly RecallGenerationScalarRow[],
): void {
  if (rows.length === 0) {
    return;
  }
  const fieldNames = [...new Set(rows.flatMap(({ fields }) => Object.keys(fields)))];
  const fetched = collection.fetchSync({
    ids: rows.map(({ id }) => id),
    outputFields: fieldNames,
    includeVector: false,
  });
  for (const row of rows) {
    const actual = fetched[row.id];
    if (actual === undefined || !fieldsMatch(actual.fields, row.fields)) {
      throw new Error(`Recall fixed snapshot generation ${responsibility} row mismatch: ${row.id}`);
    }
  }
}

async function resolveDenseRows(
  config: Readonly<RecallCoherentGenerationConfig>,
  destination: ZVecCollection,
  expectations: readonly RecallGenerationDenseExpectation[],
  buildVectorsByReuseKey: Map<string, number[]>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  validatedVectorSource: ZVecCollection | null,
  signal?: AbortSignal,
): Promise<RecallGenerationDenseRow[]> {
  const fetched =
    expectations.length === 0
      ? {}
      : destination.fetchSync({
          ids: expectations.map(({ id }) => id),
          outputFields: [
            'schemaVersion',
            'generationId',
            'evidenceOccurrenceId',
            'physicalSourceIdentity',
            'logicalSessionOccurrenceId',
            'embeddingProfileId',
            'storedDimensions',
            'evidenceChecksum',
            'embeddingInputChecksum',
            'vectorChecksum',
            'projectIdentity',
            'projectIdentityDigest',
          ],
          includeVector: true,
        });
  const compatibleSourceRows =
    validatedVectorSource === null || expectations.length === 0
      ? {}
      : validatedVectorSource.fetchSync({
          ids: expectations.map(({ id }) => id),
          outputFields: [
            'schemaVersion',
            'generationId',
            'evidenceOccurrenceId',
            'physicalSourceIdentity',
            'logicalSessionOccurrenceId',
            'embeddingProfileId',
            'storedDimensions',
            'evidenceChecksum',
            'embeddingInputChecksum',
            'vectorChecksum',
            'projectIdentity',
            'projectIdentityDigest',
          ],
          includeVector: true,
        });
  const rowsById = new Map<string, RecallGenerationDenseRow>();
  const unresolvedByReuseKey = new Map<string, RecallGenerationDenseExpectation[]>();
  for (const expectation of expectations) {
    const destinationRow = verifyDenseCandidate(fetched[expectation.id], expectation);
    if (destinationRow !== null) {
      rowsById.set(expectation.id, destinationRow);
      buildVectorsByReuseKey.set(createDenseReuseKey(expectation), [
        ...destinationRow.vectors.embedding,
      ]);
      continue;
    }
    const compatibleSourceRow = verifyDenseCandidate(
      compatibleSourceRows[expectation.id],
      expectation,
      true,
    );
    if (compatibleSourceRow !== null) {
      rowsById.set(expectation.id, compatibleSourceRow);
      buildVectorsByReuseKey.set(createDenseReuseKey(expectation), [
        ...compatibleSourceRow.vectors.embedding,
      ]);
      continue;
    }
    const reuseKey = createDenseReuseKey(expectation);
    const buildVector = buildVectorsByReuseKey.get(reuseKey);
    if (buildVector !== undefined) {
      const vectorChecksum = calculateSha256(Buffer.from(new Float32Array(buildVector).buffer));
      rowsById.set(expectation.id, {
        id: expectation.id,
        fields: { ...expectation.fields, vectorChecksum },
        vectors: { embedding: [...buildVector] },
      });
      continue;
    }
    const duplicateInputs = unresolvedByReuseKey.get(reuseKey) ?? [];
    duplicateInputs.push(expectation);
    unresolvedByReuseKey.set(reuseKey, duplicateInputs);
  }
  const unresolvedRepresentatives = [...unresolvedByReuseKey.values()].map(
    (duplicates) => duplicates[0],
  );
  for (
    let offset = 0;
    offset < unresolvedRepresentatives.length;
    offset += MAXIMUM_BUILD_WRITE_RECORDS
  ) {
    throwIfFixedSnapshotBuildCancelled(signal);
    const batch = unresolvedRepresentatives
      .slice(offset, offset + MAXIMUM_BUILD_WRITE_RECORDS)
      .filter((expectation): expectation is RecallGenerationDenseExpectation =>
        Boolean(expectation),
      );
    const nativeVectors = await dependencies.embeddingProvider.embedDocuments(
      batch.map(({ embeddingInput }) => embeddingInput),
      signal,
    );
    if (nativeVectors.length !== batch.length) {
      throw new Error(
        `Recall fixed snapshot generation document embedding count mismatch: expected ${batch.length}, received ${nativeVectors.length}`,
      );
    }
    for (const [index, representative] of batch.entries()) {
      const nativeVector = nativeVectors[index];
      if (nativeVector === undefined) {
        throw new Error(
          `Recall fixed snapshot generation document embedding missing for ${representative.id}`,
        );
      }
      const embedding = createStoredRecallEmbedding(nativeVector, {
        nativeDimensions: config.embeddingProfile.identity.dimensions,
        storedDimensions:
          config.embeddingProfile.storedDimensions ?? config.embeddingProfile.identity.dimensions,
        source: `generation ${String(representative.fields.generationId)}:${representative.id}`,
      });
      const reuseKey = createDenseReuseKey(representative);
      buildVectorsByReuseKey.set(reuseKey, [...embedding]);
      const vectorChecksum = calculateSha256(Buffer.from(new Float32Array(embedding).buffer));
      for (const expectation of unresolvedByReuseKey.get(reuseKey) ?? []) {
        rowsById.set(expectation.id, {
          id: expectation.id,
          fields: { ...expectation.fields, vectorChecksum },
          vectors: { embedding: [...embedding] },
        });
      }
    }
  }
  return expectations.map((expectation) => {
    const row = rowsById.get(expectation.id);
    if (row === undefined) {
      throw new Error(
        `Recall fixed snapshot generation dense vector resolution incomplete: ${expectation.id}`,
      );
    }
    return row;
  });
}

function openFixedSnapshotStoreSession(
  paths: Readonly<ReturnType<typeof createRecallGenerationComponentPaths>>,
): RecallFixedSnapshotStoreSession {
  const opened: ZVecCollection[] = [];
  try {
    const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
    opened.push(lexicalSource);
    const dense = ZVecOpen(paths.denseStorePath);
    opened.push(dense);
    const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
    opened.push(sessionProjection);
    return { lexicalSource, dense, sessionProjection };
  } catch (error) {
    for (const collection of opened.reverse()) {
      collection.closeSync();
    }
    throw error;
  }
}

function closeFixedSnapshotStoreSession(stores: RecallFixedSnapshotStoreSession): void {
  const errors: unknown[] = [];
  for (const collection of [stores.sessionProjection, stores.dense, stores.lexicalSource]) {
    try {
      collection.closeSync();
    } catch (error) {
      errors.push(error);
    }
  }
  if (errors.length > 0) {
    throw new AggregateError(errors, 'Recall fixed snapshot store session close failed');
  }
}

async function openFixedSnapshotValidationStoreSession(
  paths: Readonly<ReturnType<typeof createRecallGenerationComponentPaths>>,
  signal?: AbortSignal,
): Promise<RecallFixedSnapshotStoreSession> {
  const opened: ZVecCollection[] = [];
  try {
    const lexicalSource = await openRecallZvecValidationStore(
      () => ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true }),
      signal,
    );
    opened.push(lexicalSource);
    const dense = await openRecallZvecValidationStore(
      () => ZVecOpen(paths.denseStorePath, { readOnly: true }),
      signal,
    );
    opened.push(dense);
    const sessionProjection = await openRecallZvecValidationStore(
      () => ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true }),
      signal,
    );
    opened.push(sessionProjection);
    return { lexicalSource, dense, sessionProjection };
  } catch (error) {
    for (const collection of opened.reverse()) {
      collection.closeSync();
    }
    throw error;
  }
}

function verifyExpectedPhysicalSourceRows(
  stores: Readonly<RecallFixedSnapshotStoreSession>,
  artifact: Readonly<RecallMaterializedExpectedPhysicalSourceArtifact>,
  responsibility: 'checkpoint' | 'validation',
): void {
  verifyExpectedScalarRows(
    stores.lexicalSource,
    `lexical/source ${responsibility}`,
    artifact.lexicalSource,
  );
  verifyExpectedScalarRows(
    stores.sessionProjection,
    `logical projection ${responsibility}`,
    artifact.logicalSessionProjections,
  );
  verifyExpectedScalarRows(stores.sessionProjection, `physical projection ${responsibility}`, [
    artifact.physicalSessionProjection,
  ]);
  const fetchedDense =
    artifact.dense.length === 0
      ? {}
      : stores.dense.fetchSync({
          ids: artifact.dense.map(({ id }) => id),
          outputFields: [
            'schemaVersion',
            'generationId',
            'evidenceOccurrenceId',
            'physicalSourceIdentity',
            'logicalSessionOccurrenceId',
            'embeddingProfileId',
            'storedDimensions',
            'evidenceChecksum',
            'embeddingInputChecksum',
            'vectorChecksum',
            'projectIdentity',
            'projectIdentityDigest',
          ],
          includeVector: true,
        });
  for (const expectation of artifact.dense) {
    if (verifyDenseCandidate(fetchedDense[expectation.id], expectation) === null) {
      throw new Error(
        `Recall fixed snapshot generation dense ${responsibility} mismatch: ${expectation.id}`,
      );
    }
  }
}

async function writeExpectedPhysicalSource(
  config: Readonly<RecallCoherentGenerationConfig>,
  paths: Readonly<ReturnType<typeof createRecallGenerationComponentPaths>>,
  stores: Readonly<RecallFixedSnapshotStoreSession>,
  artifact: Readonly<RecallMaterializedExpectedPhysicalSourceArtifact>,
  expectedArtifactFingerprint: string,
  buildVectorsByReuseKey: Map<string, number[]>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  validatedVectorSource: ZVecCollection | null,
  signal?: AbortSignal,
): Promise<void> {
  const recovery = {
    version: FIXED_SNAPSHOT_BUILD_VERSION,
    generationId: artifact.generationId,
    operation: 'fixed-snapshot-source',
    physicalSourceIdentity: artifact.physicalSourceIdentity,
    expectedArtifactFingerprint,
  };
  if (existsSync(paths.recoveryRecordPath)) {
    const actual: unknown = JSON.parse(await readFile(paths.recoveryRecordPath, 'utf8'));
    if (JSON.stringify(actual) !== JSON.stringify(recovery)) {
      throw new Error(
        `Recall fixed snapshot generation recovery mismatch for ${artifact.physicalSourceIdentity}`,
      );
    }
  } else {
    await writeFile(paths.recoveryRecordPath, encodeStrictJson(recovery), {
      encoding: 'utf8',
      flag: 'wx',
    });
  }
  throwIfFixedSnapshotBuildCancelled(signal);
  const denseRows = await resolveDenseRows(
    config,
    stores.dense,
    artifact.dense,
    buildVectorsByReuseKey,
    dependencies,
    validatedVectorSource,
    signal,
  );
  upsertRowsInBoundedBatches(stores.lexicalSource, 'lexical/source write', artifact.lexicalSource);
  upsertRowsInBoundedBatches(stores.dense, 'dense write', denseRows);
  await invokeFixedSnapshotBuildFault(
    dependencies,
    RecallFixedSnapshotBuildFaultStage.AFTER_DENSE_WRITE,
    paths.generationDirectory,
    artifact.physicalSourceIdentity,
  );
  upsertRowsInBoundedBatches(
    stores.sessionProjection,
    'logical session projection write',
    artifact.logicalSessionProjections,
  );
  upsertRowsInBoundedBatches(stores.sessionProjection, 'physical session projection write', [
    artifact.physicalSessionProjection,
  ]);
  verifyExpectedPhysicalSourceRows(stores, artifact, 'checkpoint');
  await rm(paths.recoveryRecordPath);
}

async function validateExpectedFixedSnapshotArtifacts(
  generationDirectory: string,
  artifacts: readonly Readonly<{
    artifact: RecallExpectedPhysicalSourceArtifact;
    fingerprint: string;
  }>[],
  signal?: AbortSignal,
): Promise<void> {
  for (const expected of artifacts) {
    const artifactPath = join(
      generationDirectory,
      EXPECTED_SOURCE_DIRECTORY,
      `${expected.artifact.physicalSourceIdentity}.json`,
    );
    const persisted = await readExpectedPhysicalSourceArtifact(artifactPath);
    if (persisted.fingerprint !== expected.fingerprint) {
      throw new Error(
        `Recall fixed snapshot generation expected source digest mismatch: ${expected.artifact.physicalSourceIdentity}`,
      );
    }
    if (JSON.stringify(persisted.artifact) !== JSON.stringify(expected.artifact)) {
      throw new Error(
        `Recall fixed snapshot generation expected source content mismatch: ${expected.artifact.physicalSourceIdentity}`,
      );
    }
  }

  const stores = await openFixedSnapshotValidationStoreSession(
    createRecallGenerationComponentPaths(generationDirectory),
    signal,
  );
  try {
    for (const { artifact } of artifacts) {
      if (!isSkippedExpectedPhysicalSourceArtifact(artifact)) {
        verifyExpectedPhysicalSourceRows(stores, artifact, 'validation');
      }
    }
  } finally {
    closeFixedSnapshotStoreSession(stores);
  }
}

async function verifyFixedSnapshotCanary(
  config: Readonly<RecallCoherentGenerationConfig>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  generationId: string,
): Promise<void> {
  const canary = config.embeddingProfile.canary;
  if (!canary) {
    return;
  }
  const [firstNativeCanary, repeatedNativeCanary] = await Promise.all([
    dependencies.embeddingProvider.embedQuery(canary.query),
    dependencies.embeddingProvider.embedQuery(canary.query),
  ]);
  const options = {
    nativeDimensions: config.embeddingProfile.identity.dimensions,
    storedDimensions:
      config.embeddingProfile.storedDimensions ?? config.embeddingProfile.identity.dimensions,
    source: `generation ${generationId} query canary`,
  };
  const firstStoredCanary = createStoredRecallEmbedding(firstNativeCanary, options);
  const repeatedStoredCanary = createStoredRecallEmbedding(repeatedNativeCanary, options);
  assertRepeatableStoredRecallEmbeddings(firstStoredCanary, repeatedStoredCanary, {
    minimumCosineSimilarity: canary.minimumRepeatCosineSimilarity,
    source: `generation ${generationId} query canary`,
  });
}

/** Builds or resumes one inactive generation bound to an immutable physical-source snapshot. */
export async function buildRecallFixedSnapshotGeneration(
  config: Readonly<RecallCoherentGenerationConfig>,
  options: Readonly<CreateRecallGenerationFromPhysicalSourcesOptions>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
): Promise<OpenedValidatedRecallGeneration> {
  createRecallActiveGenerationPointer(options.generationId);
  if (!options.resumeExistingGeneration && options.physicalSessionPaths.length === 0) {
    throw new Error('Recall fixed snapshot generation requires at least one physical session');
  }
  throwIfFixedSnapshotBuildCancelled(options.signal);
  await mkdir(config.generationRootDirectory, { recursive: true });
  const generationDirectory = join(config.generationRootDirectory, options.generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  const expectedManifest = createExpectedRecallPhysicalSourceManifest(config, options.generationId);
  const bootstrapStatePath = join(generationDirectory, BOOTSTRAP_STATE_FILE);
  const snapshotDescriptorPath = join(generationDirectory, SNAPSHOT_DESCRIPTOR_FILE);
  let bootstrapState: RecallFixedSnapshotBootstrapState;
  const isNewGeneration = !existsSync(generationDirectory);
  if (isNewGeneration) {
    await mkdir(generationDirectory);
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_GENERATION_DIRECTORY_CREATION,
      generationDirectory,
    );
    bootstrapState = await writeFixedSnapshotBootstrapState(
      config,
      options.generationId,
      options.physicalSessionPaths,
      bootstrapStatePath,
    );
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_BOOTSTRAP_STATE_WRITE,
      generationDirectory,
    );
  } else {
    if (existsSync(paths.validationReceiptPath)) {
      return openValidatedRecallGeneration(config, options.generationId);
    }
    if (!existsSync(bootstrapStatePath)) {
      throw new Error(
        `Recall fixed snapshot generation bootstrap state missing for ${options.generationId}; discard this staging generation`,
      );
    }
    bootstrapState = await readFixedSnapshotBootstrapState(
      bootstrapStatePath,
      options.generationId,
    );
    if (options.physicalSessionPaths.length > 0) {
      assertRequestedSourcesMatchSnapshot(config, options.physicalSessionPaths, bootstrapState);
    }
  }

  const physicalSessionPaths = resolveBootstrapPhysicalSessionPaths(config, bootstrapState);
  let manifestFingerprint: string;
  if (existsSync(paths.manifestPath)) {
    try {
      const actualManifest = await readRecallGenerationManifest(paths.manifestPath);
      assertRecallGenerationManifestCompatible(
        actualManifest.manifest,
        expectedManifest,
        paths.manifestPath,
      );
      manifestFingerprint = actualManifest.fingerprint;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall fixed snapshot generation bootstrap manifest incompatible for ${options.generationId}; discard this staging generation: ${message}`,
        { cause: error },
      );
    }
  } else {
    if (!isNewGeneration) {
      throw new Error(
        `Recall fixed snapshot generation bootstrap manifest missing for ${options.generationId}; discard this staging generation`,
      );
    }
    manifestFingerprint = await writeRecallGenerationManifest(paths.manifestPath, expectedManifest);
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_MANIFEST_WRITE,
      generationDirectory,
    );
  }

  let snapshot: RecallFixedSnapshotDescriptor;
  if (existsSync(snapshotDescriptorPath)) {
    snapshot = await readFixedSnapshotDescriptor(snapshotDescriptorPath);
  } else {
    if (!isNewGeneration) {
      throw new Error(
        `Recall fixed snapshot generation snapshot capture incomplete for ${options.generationId}; discard this staging generation`,
      );
    }
    snapshot = await captureFixedSourceSnapshot(
      config,
      options.generationId,
      physicalSessionPaths,
      options.signal,
      generationDirectory,
      manifestFingerprint,
      dependencies,
    );
    await invokeFixedSnapshotBuildFault(
      dependencies,
      RecallFixedSnapshotBuildFaultStage.AFTER_SNAPSHOT_CAPTURE,
      generationDirectory,
    );
  }
  if (
    snapshot.generationId !== options.generationId ||
    snapshot.manifestFingerprint !== manifestFingerprint
  ) {
    throw new Error(
      `Recall fixed snapshot generation resume identity mismatch for ${options.generationId}; discard this staging generation`,
    );
  }
  assertRequestedSourcesMatchSnapshot(config, physicalSessionPaths, snapshot);

  const contracts = createRecallGenerationStoreContracts(
    options.generationId,
    expectedManifest.embeddingProfile.storedDimensions,
  );
  const allBootstrapStoresExist = [
    paths.lexicalSourceStorePath,
    paths.denseStorePath,
    paths.sessionProjectionStorePath,
  ].every((storePath) => existsSync(storePath));
  if (allBootstrapStoresExist) {
    try {
      const existingMembership = await readRecallGenerationStoreRecordMembership(paths);
      validateRecallGenerationStores(paths, contracts, options.generationId, existingMembership);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall fixed snapshot generation stores incompatible for ${options.generationId}; discard this staging generation: ${message}`,
        { cause: error },
      );
    }
  } else {
    await resumeEmptyRecallGenerationStores(paths, contracts, async (responsibility) => {
      await invokeFixedSnapshotBuildFault(
        dependencies,
        fixedSnapshotBootstrapFaultStageForStore(responsibility),
        generationDirectory,
      );
    });
  }

  await verifyFixedSnapshotCanary(config, dependencies, options.generationId);
  const artifacts: Array<{
    artifact: RecallExpectedPhysicalSourceArtifact;
    fingerprint: string;
  }> = [];
  const projectIdentityBySessionOrigin = new Map<
    string,
    ReturnType<RecallPhysicalSourceGenerationDependencies['resolveProjectIdentity']>
  >();
  const buildDependencies: RecallPhysicalSourceGenerationDependencies = {
    ...dependencies,
    resolveProjectIdentity(sessionOrigin) {
      const existing = projectIdentityBySessionOrigin.get(sessionOrigin);
      if (existing !== undefined) {
        return existing;
      }
      const resolution = dependencies.resolveProjectIdentity(sessionOrigin);
      projectIdentityBySessionOrigin.set(sessionOrigin, resolution);
      return resolution;
    },
  };
  const buildVectorsByReuseKey = new Map<string, number[]>();
  let validatedVectorSource: ZVecCollection | null = null;
  if (options.validatedVectorSourceGenerationId !== undefined) {
    if (options.validatedVectorSourceGenerationId === options.generationId) {
      throw new Error(
        `Recall fixed snapshot generation vector source must differ from ${options.generationId}`,
      );
    }
    const openedVectorSource = await openValidatedRecallGeneration(
      config,
      options.validatedVectorSourceGenerationId,
    );
    validatedVectorSource = ZVecOpen(
      createRecallGenerationComponentPaths(openedVectorSource.generationDirectory).denseStorePath,
      { readOnly: true },
    );
  }
  let storeSession: RecallFixedSnapshotStoreSession | null = null;
  let pendingPhysicalSourceCheckpoints: RecallFixedSnapshotPhysicalSourceCheckpoint[] = [];
  try {
    let sourcesInBuildOrder = snapshot.sources;
    if (options.resumeExistingGeneration && existsSync(paths.recoveryRecordPath)) {
      const recoveryPhysicalSourceIdentity = await readFixedSnapshotRecoveryPhysicalSourceIdentity(
        paths.recoveryRecordPath,
        options.generationId,
      );
      const recoveringSource = snapshot.sources.find(
        ({ physicalSourceIdentity }) => physicalSourceIdentity === recoveryPhysicalSourceIdentity,
      );
      if (recoveringSource === undefined) {
        throw new Error(
          `Recall fixed snapshot generation recovery source missing for ${options.generationId}: ${recoveryPhysicalSourceIdentity}`,
        );
      }
      sourcesInBuildOrder = [
        recoveringSource,
        ...snapshot.sources.filter(
          ({ physicalSourceIdentity }) =>
            physicalSourceIdentity !== recoveringSource.physicalSourceIdentity,
        ),
      ];
    }
    storeSession = openFixedSnapshotStoreSession(paths);
    let checkpointedPhysicalSourceCount = 0;
    let storeSessionRecordCount = 0;
    for (const [sourceIndex, source] of sourcesInBuildOrder.entries()) {
      throwIfFixedSnapshotBuildCancelled(options.signal);
      if (storeSession === null) {
        throw new Error('Recall fixed snapshot writable store session missing');
      }
      const activeStoreSession = storeSession;
      const expected = await materializeExpectedPhysicalSource(
        config,
        options.generationId,
        generationDirectory,
        source,
        buildDependencies,
      );
      artifacts.push(expected);
      if (!isSkippedExpectedPhysicalSourceArtifact(expected.artifact)) {
        await writeExpectedPhysicalSource(
          config,
          paths,
          activeStoreSession,
          expected.artifact,
          expected.fingerprint,
          buildVectorsByReuseKey,
          buildDependencies,
          validatedVectorSource,
          options.signal,
        );
      }
      checkpointedPhysicalSourceCount += 1;
      storeSessionRecordCount += countFixedSnapshotStoreSessionRecords(expected.artifact);
      pendingPhysicalSourceCheckpoints.push({
        physicalSourceIdentity: expected.artifact.physicalSourceIdentity,
        sessionsRootRelativePath: expected.artifact.sessionsRootRelativePath,
        completedPhysicalSourceCount: checkpointedPhysicalSourceCount,
        totalPhysicalSourceCount: snapshot.sources.length,
      });
      const atStoreSessionBoundary =
        storeSessionRecordCount >= FIXED_SNAPSHOT_STORE_SESSION_RECORD_LIMIT ||
        sourceIndex === sourcesInBuildOrder.length - 1;
      if (!atStoreSessionBoundary) {
        continue;
      }

      storeSession = null;
      closeFixedSnapshotStoreSession(activeStoreSession);
      await invokeFixedSnapshotBuildFault(
        dependencies,
        RecallFixedSnapshotBuildFaultStage.AFTER_STORE_CLOSE,
        generationDirectory,
        source.physicalSourceIdentity,
      );
      for (const checkpoint of pendingPhysicalSourceCheckpoints) {
        options.onPhysicalSourceCheckpoint?.(checkpoint);
      }
      pendingPhysicalSourceCheckpoints = [];
      storeSessionRecordCount = 0;
      if (sourceIndex < sourcesInBuildOrder.length - 1) {
        storeSession = openFixedSnapshotStoreSession(paths);
      }
    }
  } finally {
    if (storeSession !== null) {
      closeFixedSnapshotStoreSession(storeSession);
      for (const checkpoint of pendingPhysicalSourceCheckpoints) {
        options.onPhysicalSourceCheckpoint?.(checkpoint);
      }
    }
    validatedVectorSource?.closeSync();
  }
  const snapshotSourceOrder = new Map(
    snapshot.sources.map(({ physicalSourceIdentity }, index) => [physicalSourceIdentity, index]),
  );
  artifacts.sort(
    (left, right) =>
      (snapshotSourceOrder.get(left.artifact.physicalSourceIdentity) ?? 0) -
      (snapshotSourceOrder.get(right.artifact.physicalSourceIdentity) ?? 0),
  );

  const expectedRecordIds = {
    lexicalSource: artifacts
      .flatMap(({ artifact }) =>
        isSkippedExpectedPhysicalSourceArtifact(artifact)
          ? []
          : artifact.lexicalSource.map(({ id }) => id),
      )
      .toSorted(),
    dense: artifacts
      .flatMap(({ artifact }) =>
        isSkippedExpectedPhysicalSourceArtifact(artifact) ? [] : artifact.dense.map(({ id }) => id),
      )
      .toSorted(),
    sessionProjection: artifacts
      .flatMap(({ artifact }) =>
        isSkippedExpectedPhysicalSourceArtifact(artifact)
          ? []
          : [
              ...artifact.logicalSessionProjections.map(({ id }) => id),
              artifact.physicalSessionProjection.id,
            ],
      )
      .toSorted(),
  };
  if (
    new Set(expectedRecordIds.lexicalSource).size !== expectedRecordIds.lexicalSource.length ||
    new Set(expectedRecordIds.dense).size !== expectedRecordIds.dense.length ||
    new Set(expectedRecordIds.sessionProjection).size !== expectedRecordIds.sessionProjection.length
  ) {
    throw new Error('Recall fixed snapshot generation expected membership contains duplicate IDs');
  }
  await invokeFixedSnapshotBuildFault(
    dependencies,
    RecallFixedSnapshotBuildFaultStage.BEFORE_VALIDATION_RECEIPT,
    generationDirectory,
  );
  throwIfFixedSnapshotBuildCancelled(options.signal);
  await validateExpectedFixedSnapshotArtifacts(generationDirectory, artifacts, options.signal);
  validateRecallGenerationStores(paths, contracts, options.generationId, expectedRecordIds);
  validateRecallGenerationDenseSubset(
    paths,
    options.generationId,
    expectedManifest.embeddingProfile.profileId,
    expectedManifest.embeddingProfile.storedDimensions,
    expectedRecordIds,
  );
  const startingSnapshotFingerprint = calculateSha256(
    encodeStrictJson({
      snapshot,
      expectedSourceFingerprints: artifacts.map(({ fingerprint }) => fingerprint),
      expectedRecordIds,
    }),
  );
  throwIfFixedSnapshotBuildCancelled(options.signal);
  const receipt = createRecallGenerationValidationReceipt({
    generationId: options.generationId,
    manifestFingerprint,
    membership: {
      startingSnapshotFingerprint,
      physicalSourceCount: snapshot.sources.length,
      logicalSessionOccurrenceCount: artifacts.reduce(
        (count, { artifact }) =>
          count +
          (isSkippedExpectedPhysicalSourceArtifact(artifact)
            ? 0
            : artifact.logicalSessionOccurrenceIds.length),
        0,
      ),
      lexicalSourceRecordIds: expectedRecordIds.lexicalSource,
      denseRecordIds: expectedRecordIds.dense,
      sessionProjectionRecordIds: expectedRecordIds.sessionProjection,
    },
    validatedAtEpochMilliseconds: (config.nowEpochMilliseconds ?? Date.now)(),
  });
  await writeRecallGenerationValidationReceipt(paths.validationReceiptPath, receipt);
  return openValidatedRecallGeneration(config, options.generationId);
}

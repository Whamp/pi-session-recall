import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import { ZVecOpen, type ZVecCollection, type ZVecStatus, type ZVecVector } from '@zvec/zvec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

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
  createEmptyRecallGenerationStores,
  createRecallGenerationComponentPaths,
  createRecallGenerationStoreContracts,
  readRecallGenerationStoreRecordMembership,
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
  type CreateRecallGenerationFromPhysicalSourcesOptions,
  type MaterializedRecallPhysicalSourceGeneration,
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

interface RecallFixedSnapshotSourceDescriptor {
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  sourceByteSize: number;
  sourceChecksum: string;
  snapshotFileName: string;
}

interface RecallFixedSnapshotDescriptor {
  version: 1;
  generationId: string;
  manifestFingerprint: string;
  sources: RecallFixedSnapshotSourceDescriptor[];
}

interface RecallExpectedPhysicalSourceArtifact extends MaterializedRecallPhysicalSourceGeneration {
  version: 1;
  generationId: string;
  physicalSourceIdentity: string;
  sessionsRootRelativePath: string;
  sourceByteSize: number;
  sourceChecksum: string;
}

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
          snapshotFileName: Type.String({ pattern: '^[0-9]+-[a-f0-9]{64}\\.jsonl$' }),
        },
        { additionalProperties: false },
      ),
    ),
  },
  { additionalProperties: false },
);
const expectedPhysicalSourceArtifactSchema = Type.Object(
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
  stage: 'after-snapshot-capture' | 'after-dense-write' | 'before-validation-receipt',
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

function readVectorValues(vector: ZVecVector | undefined): number[] {
  if (vector === undefined) {
    return [];
  }
  if (Array.isArray(vector)) {
    return [...vector];
  }
  if (vector instanceof Float32Array || vector instanceof Int8Array) {
    return Array.from(vector);
  }
  return Object.values(vector);
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
  const embedding = readVectorValues(candidate.vectors.embedding);
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

async function captureFixedSourceSnapshot(
  config: Readonly<RecallCoherentGenerationConfig>,
  options: Readonly<CreateRecallGenerationFromPhysicalSourcesOptions>,
  generationDirectory: string,
  manifestFingerprint: string,
): Promise<RecallFixedSnapshotDescriptor> {
  const snapshotSourceDirectory = join(generationDirectory, SNAPSHOT_SOURCE_DIRECTORY);
  await Promise.all([
    mkdir(snapshotSourceDirectory),
    mkdir(join(generationDirectory, EXPECTED_SOURCE_DIRECTORY)),
  ]);
  const sources: RecallFixedSnapshotSourceDescriptor[] = [];
  for (const [index, physicalSessionPath] of options.physicalSessionPaths.entries()) {
    throwIfFixedSnapshotBuildCancelled(options.signal);
    const identity = resolveRecallPhysicalSourceIdentity(
      config.sessionsDirectory,
      physicalSessionPath,
    );
    const sourceBytes = await readFile(physicalSessionPath);
    const sourceChecksum = calculateSha256(sourceBytes);
    const snapshotFileName = `${index}-${sourceChecksum}.jsonl`;
    await writeFile(join(snapshotSourceDirectory, snapshotFileName), sourceBytes, { flag: 'wx' });
    sources.push({
      physicalSourceIdentity: identity.physicalSourceIdentity,
      sessionsRootRelativePath: identity.sessionsRootRelativePath,
      sourceByteSize: sourceBytes.length,
      sourceChecksum,
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
    generationId: options.generationId,
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
      `Recall fixed snapshot generation descriptor invalid at ${descriptorPath}: ${message}`,
      {
        cause: error,
      },
    );
  }
}

function assertRequestedSourcesMatchSnapshot(
  config: Readonly<RecallCoherentGenerationConfig>,
  physicalSessionPaths: readonly string[],
  snapshot: Readonly<RecallFixedSnapshotDescriptor>,
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
      `Recall fixed snapshot generation resume source snapshot mismatch for ${snapshot.generationId}`,
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
      `Recall fixed snapshot generation captured source mismatch for ${source.physicalSourceIdentity}`,
    );
  }
  const materialized = await materializeRecallPhysicalSourceGeneration(
    config,
    generationId,
    physicalSessionPath,
    dependencies,
    snapshotPath,
  );
  const artifact = Value.Parse(expectedPhysicalSourceArtifactSchema, {
    version: FIXED_SNAPSHOT_BUILD_VERSION,
    generationId,
    physicalSourceIdentity: source.physicalSourceIdentity,
    sessionsRootRelativePath: source.sessionsRootRelativePath,
    sourceByteSize: source.sourceByteSize,
    sourceChecksum: source.sourceChecksum,
    ...materialized,
  });
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
      Buffer.from(new Float32Array(readVectorValues(actual.vectors.embedding)).buffer),
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

async function writeExpectedPhysicalSource(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationDirectory: string,
  artifact: Readonly<RecallExpectedPhysicalSourceArtifact>,
  expectedArtifactFingerprint: string,
  buildVectorsByReuseKey: Map<string, number[]>,
  dependencies: Readonly<RecallPhysicalSourceGenerationDependencies>,
  validatedVectorSource: ZVecCollection | null,
  signal?: AbortSignal,
): Promise<void> {
  const paths = createRecallGenerationComponentPaths(generationDirectory);
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
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath);
  const dense = ZVecOpen(paths.denseStorePath);
  const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath);
  try {
    throwIfFixedSnapshotBuildCancelled(signal);
    const denseRows = await resolveDenseRows(
      config,
      dense,
      artifact.dense,
      buildVectorsByReuseKey,
      dependencies,
      validatedVectorSource,
      signal,
    );
    upsertRowsInBoundedBatches(lexicalSource, 'lexical/source write', artifact.lexicalSource);
    upsertRowsInBoundedBatches(dense, 'dense write', denseRows);
    await invokeFixedSnapshotBuildFault(
      dependencies,
      'after-dense-write',
      generationDirectory,
      artifact.physicalSourceIdentity,
    );
    upsertRowsInBoundedBatches(
      sessionProjection,
      'logical session projection write',
      artifact.logicalSessionProjections,
    );
    upsertRowsInBoundedBatches(sessionProjection, 'physical session projection write', [
      artifact.physicalSessionProjection,
    ]);
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
    sessionProjection.closeSync();
  }
  const reopenedLexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  const reopenedDense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  const reopenedSessionProjection = ZVecOpen(paths.sessionProjectionStorePath, {
    readOnly: true,
  });
  try {
    verifyExpectedScalarRows(
      reopenedLexicalSource,
      'lexical/source checkpoint',
      artifact.lexicalSource,
    );
    verifyExpectedScalarRows(
      reopenedSessionProjection,
      'logical projection checkpoint',
      artifact.logicalSessionProjections,
    );
    verifyExpectedScalarRows(reopenedSessionProjection, 'physical projection checkpoint', [
      artifact.physicalSessionProjection,
    ]);
    const fetchedDense =
      artifact.dense.length === 0
        ? {}
        : reopenedDense.fetchSync({
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
          `Recall fixed snapshot generation dense checkpoint mismatch: ${expectation.id}`,
        );
      }
    }
  } finally {
    reopenedLexicalSource.closeSync();
    reopenedDense.closeSync();
    reopenedSessionProjection.closeSync();
  }
  await rm(paths.recoveryRecordPath);
}

async function validateExpectedFixedSnapshotArtifacts(
  generationDirectory: string,
  artifacts: readonly Readonly<{
    artifact: RecallExpectedPhysicalSourceArtifact;
    fingerprint: string;
  }>[],
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

  const paths = createRecallGenerationComponentPaths(generationDirectory);
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  const dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  const sessionProjection = ZVecOpen(paths.sessionProjectionStorePath, { readOnly: true });
  try {
    for (const { artifact } of artifacts) {
      verifyExpectedScalarRows(lexicalSource, 'lexical/source validation', artifact.lexicalSource);
      verifyExpectedScalarRows(
        sessionProjection,
        'logical projection validation',
        artifact.logicalSessionProjections,
      );
      verifyExpectedScalarRows(sessionProjection, 'physical projection validation', [
        artifact.physicalSessionProjection,
      ]);
      const fetchedDense =
        artifact.dense.length === 0
          ? {}
          : dense.fetchSync({
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
            `Recall fixed snapshot generation dense validation row mismatch: ${expectation.id}`,
          );
        }
      }
    }
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
    sessionProjection.closeSync();
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
  if (options.physicalSessionPaths.length === 0) {
    throw new Error('Recall fixed snapshot generation requires at least one physical session');
  }
  throwIfFixedSnapshotBuildCancelled(options.signal);
  await mkdir(config.generationRootDirectory, { recursive: true });
  const generationDirectory = join(config.generationRootDirectory, options.generationId);
  const paths = createRecallGenerationComponentPaths(generationDirectory);
  const expectedManifest = createExpectedRecallPhysicalSourceManifest(config, options.generationId);
  let snapshot: RecallFixedSnapshotDescriptor;
  let manifestFingerprint: string;
  if (!existsSync(generationDirectory)) {
    await mkdir(generationDirectory);
    manifestFingerprint = await writeRecallGenerationManifest(paths.manifestPath, expectedManifest);
    const contracts = createRecallGenerationStoreContracts(
      options.generationId,
      expectedManifest.embeddingProfile.storedDimensions,
    );
    createEmptyRecallGenerationStores(paths, contracts);
    snapshot = await captureFixedSourceSnapshot(
      config,
      options,
      generationDirectory,
      manifestFingerprint,
    );
    await invokeFixedSnapshotBuildFault(
      dependencies,
      'after-snapshot-capture',
      generationDirectory,
    );
  } else {
    if (existsSync(paths.validationReceiptPath)) {
      return openValidatedRecallGeneration(config, options.generationId);
    }
    const actualManifest = await readRecallGenerationManifest(paths.manifestPath);
    assertRecallGenerationManifestCompatible(
      actualManifest.manifest,
      expectedManifest,
      paths.manifestPath,
    );
    manifestFingerprint = actualManifest.fingerprint;
    snapshot = await readFixedSnapshotDescriptor(
      join(generationDirectory, SNAPSHOT_DESCRIPTOR_FILE),
    );
    if (
      snapshot.generationId !== options.generationId ||
      snapshot.manifestFingerprint !== manifestFingerprint
    ) {
      throw new Error(
        `Recall fixed snapshot generation resume identity mismatch for ${options.generationId}`,
      );
    }
    assertRequestedSourcesMatchSnapshot(config, options.physicalSessionPaths, snapshot);
    const contracts = createRecallGenerationStoreContracts(
      options.generationId,
      expectedManifest.embeddingProfile.storedDimensions,
    );
    const existingMembership = await readRecallGenerationStoreRecordMembership(paths);
    validateRecallGenerationStores(paths, contracts, options.generationId, existingMembership);
  }

  await verifyFixedSnapshotCanary(config, dependencies, options.generationId);
  const artifacts: Array<{
    artifact: RecallExpectedPhysicalSourceArtifact;
    fingerprint: string;
  }> = [];
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
  try {
    for (const source of snapshot.sources) {
      throwIfFixedSnapshotBuildCancelled(options.signal);
      const expected = await materializeExpectedPhysicalSource(
        config,
        options.generationId,
        generationDirectory,
        source,
        dependencies,
      );
      artifacts.push(expected);
      await writeExpectedPhysicalSource(
        config,
        generationDirectory,
        expected.artifact,
        expected.fingerprint,
        buildVectorsByReuseKey,
        dependencies,
        validatedVectorSource,
        options.signal,
      );
    }
  } finally {
    validatedVectorSource?.closeSync();
  }

  const expectedRecordIds = {
    lexicalSource: artifacts
      .flatMap(({ artifact }) => artifact.lexicalSource.map(({ id }) => id))
      .toSorted(),
    dense: artifacts.flatMap(({ artifact }) => artifact.dense.map(({ id }) => id)).toSorted(),
    sessionProjection: artifacts
      .flatMap(({ artifact }) => [
        ...artifact.logicalSessionProjections.map(({ id }) => id),
        artifact.physicalSessionProjection.id,
      ])
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
    'before-validation-receipt',
    generationDirectory,
  );
  throwIfFixedSnapshotBuildCancelled(options.signal);
  await validateExpectedFixedSnapshotArtifacts(generationDirectory, artifacts);
  const contracts = createRecallGenerationStoreContracts(
    options.generationId,
    expectedManifest.embeddingProfile.storedDimensions,
  );
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
        (count, { artifact }) => count + artifact.logicalSessionOccurrenceIds.length,
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

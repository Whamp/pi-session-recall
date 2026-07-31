import { createHash } from 'node:crypto';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecFieldSchema,
  type ZVecVector,
  type ZVecVectorSchema,
} from '@zvec/zvec';

import { RecallSessionProjectionKind } from './enums.js';
import {
  type ExactZvecDocumentEnumeration,
  visitExactZvecDocuments,
} from './visit-exact-zvec-documents.js';

/** Current schema version shared by the three target recall generation stores. */
export const RECALL_GENERATION_STORE_FORMAT_VERSION = 1;

/** Fixed directory names for the three independent stores inside one recall generation. */
export const RECALL_GENERATION_STORE_DIRECTORIES = Object.freeze({
  lexicalSource: 'lexical-source',
  dense: 'dense',
  sessionProjection: 'session-projections',
});

const DENSE_HNSW_M = 16;
const DENSE_HNSW_EF_CONSTRUCTION = 200;

type RecallGenerationScalarType = 'string' | 'boolean' | 'int32' | 'int64' | 'array-string';
type RecallGenerationVectorType = 'vector-fp32';

/** One scalar field and optional lexical index declared by a generation store contract. */
export interface RecallGenerationScalarFieldContract {
  name: string;
  type: RecallGenerationScalarType;
  index?: Readonly<{
    kind: 'full-text';
    tokenizer: 'standard';
    filters: readonly string[];
  }>;
}

/** One dense vector field and its fixed local index declared by a generation store contract. */
export interface RecallGenerationVectorFieldContract {
  name: string;
  type: RecallGenerationVectorType;
  dimensions: number;
  index: Readonly<{
    kind: 'hnsw';
    metric: 'cosine';
    m: number;
    efConstruction: number;
  }>;
}

/** Immutable schema, identity, and responsibility for one generation-owned zvec store. */
export interface RecallGenerationStoreContract {
  formatVersion: 1;
  directory: string;
  collectionName: string;
  responsibility: 'lexical-source' | 'dense-evidence' | 'session-projection';
  scalarFields: readonly Readonly<RecallGenerationScalarFieldContract>[];
  vectorFields: readonly Readonly<RecallGenerationVectorFieldContract>[];
}

/** Paths to every component owned by one coherent recall generation. */
export interface RecallGenerationComponentPaths {
  generationDirectory: string;
  manifestPath: string;
  validationReceiptPath: string;
  recoveryRecordPath: string;
  lexicalSourceStorePath: string;
  denseStorePath: string;
  sessionProjectionStorePath: string;
}

/** Exact reopened membership counts for all three generation stores. */
export interface RecallGenerationStoreCounts {
  lexicalSource: number;
  dense: number;
  sessionProjection: number;
}

const LEXICAL_SOURCE_SCALAR_FIELDS = Object.freeze([
  { name: 'schemaVersion', type: 'int32' },
  { name: 'generationId', type: 'string' },
  { name: 'recordKind', type: 'string' },
  { name: 'physicalSourceIdentity', type: 'string' },
  { name: 'sessionsRootRelativePath', type: 'string' },
  { name: 'logicalSessionOccurrenceId', type: 'string' },
  { name: 'rawSessionId', type: 'string' },
  { name: 'headerSourceLine', type: 'int32' },
  { name: 'entryAnchorId', type: 'string' },
  { name: 'entryId', type: 'string' },
  { name: 'parentEntryId', type: 'string' },
  { name: 'childEntryIds', type: 'array-string' },
  { name: 'branchPathLeafIds', type: 'array-string' },
  { name: 'evidenceOccurrenceIds', type: 'array-string' },
  { name: 'sourceOrder', type: 'int64' },
  { name: 'entryType', type: 'string' },
  { name: 'timestamp', type: 'string' },
  { name: 'entryStartByte', type: 'int64' },
  { name: 'entryEndByte', type: 'int64' },
  { name: 'evidenceOccurrenceId', type: 'string' },
  { name: 'documentKind', type: 'string' },
  { name: 'evidenceKind', type: 'string' },
  { name: 'evidencePart', type: 'string' },
  { name: 'isDenseSearchable', type: 'boolean' },
  { name: 'evidenceChecksum', type: 'string' },
  { name: 'projectIdentity', type: 'string' },
  { name: 'projectIdentityDigest', type: 'string' },
  { name: 'sourceLineStart', type: 'int32' },
  { name: 'sourceLineEnd', type: 'int32' },
  { name: 'sourceBlockStart', type: 'int32' },
  { name: 'sourceBlockEnd', type: 'int32' },
  { name: 'characterStart', type: 'int32' },
  { name: 'characterEnd', type: 'int32' },
  { name: 'tokenStart', type: 'int32' },
  { name: 'tokenEnd', type: 'int32' },
  { name: 'textRunIndex', type: 'int32' },
  { name: 'chunkIndex', type: 'int32' },
  { name: 'recordJson', type: 'string' },
  {
    name: 'content',
    type: 'string',
    index: { kind: 'full-text', tokenizer: 'standard', filters: ['lowercase'] },
  },
  {
    name: 'identifierContent',
    type: 'string',
    index: { kind: 'full-text', tokenizer: 'standard', filters: [] },
  },
] satisfies readonly RecallGenerationScalarFieldContract[]);

const DENSE_SCALAR_FIELDS = Object.freeze([
  { name: 'schemaVersion', type: 'int32' },
  { name: 'generationId', type: 'string' },
  { name: 'evidenceOccurrenceId', type: 'string' },
  { name: 'physicalSourceIdentity', type: 'string' },
  { name: 'logicalSessionOccurrenceId', type: 'string' },
  { name: 'embeddingProfileId', type: 'string' },
  { name: 'storedDimensions', type: 'int32' },
  { name: 'evidenceChecksum', type: 'string' },
  { name: 'embeddingInputChecksum', type: 'string' },
  { name: 'vectorChecksum', type: 'string' },
  { name: 'projectIdentity', type: 'string' },
  { name: 'projectIdentityDigest', type: 'string' },
] satisfies readonly RecallGenerationScalarFieldContract[]);

const SESSION_PROJECTION_SCALAR_FIELDS = Object.freeze([
  { name: 'schemaVersion', type: 'int32' },
  { name: 'generationId', type: 'string' },
  { name: 'projectionKind', type: 'string' },
  { name: 'physicalSourceIdentity', type: 'string' },
  { name: 'logicalSessionOccurrenceId', type: 'string' },
  { name: 'projectionJson', type: 'string' },
] satisfies readonly RecallGenerationScalarFieldContract[]);

const ZVEC_SCALAR_TYPES: Readonly<Record<RecallGenerationScalarType, ZVecDataType>> = {
  string: ZVecDataType.STRING,
  boolean: ZVecDataType.BOOL,
  int32: ZVecDataType.INT32,
  int64: ZVecDataType.INT64,
  'array-string': ZVecDataType.ARRAY_STRING,
};

/** Resolves fixed component paths without sharing a store file across generations. */
export function createRecallGenerationComponentPaths(
  generationDirectory: string,
): RecallGenerationComponentPaths {
  return {
    generationDirectory,
    manifestPath: join(generationDirectory, 'index-manifest.json'),
    validationReceiptPath: join(generationDirectory, 'validation-receipt.json'),
    recoveryRecordPath: join(generationDirectory, 'write-recovery.json'),
    lexicalSourceStorePath: join(
      generationDirectory,
      RECALL_GENERATION_STORE_DIRECTORIES.lexicalSource,
    ),
    denseStorePath: join(generationDirectory, RECALL_GENERATION_STORE_DIRECTORIES.dense),
    sessionProjectionStorePath: join(
      generationDirectory,
      RECALL_GENERATION_STORE_DIRECTORIES.sessionProjection,
    ),
  };
}

function createRecallGenerationCollectionName(
  responsibility: 'lexical_source' | 'dense' | 'projection',
  generationId: string,
): string {
  const generationIdentityDigest = createHash('sha256')
    .update(generationId)
    .digest('hex')
    .slice(0, 32);
  return `pi_recall_v1_${responsibility}_${generationIdentityDigest}`;
}

/** Creates generation-bound collection names and final store responsibility contracts. */
export function createRecallGenerationStoreContracts(
  generationId: string,
  storedDimensions: number,
): Readonly<{
  lexicalSource: RecallGenerationStoreContract;
  dense: RecallGenerationStoreContract;
  sessionProjection: RecallGenerationStoreContract;
}> {
  return Object.freeze({
    lexicalSource: Object.freeze({
      formatVersion: RECALL_GENERATION_STORE_FORMAT_VERSION,
      directory: RECALL_GENERATION_STORE_DIRECTORIES.lexicalSource,
      collectionName: createRecallGenerationCollectionName('lexical_source', generationId),
      responsibility: 'lexical-source',
      scalarFields: LEXICAL_SOURCE_SCALAR_FIELDS,
      vectorFields: Object.freeze([]),
    }),
    dense: Object.freeze({
      formatVersion: RECALL_GENERATION_STORE_FORMAT_VERSION,
      directory: RECALL_GENERATION_STORE_DIRECTORIES.dense,
      collectionName: createRecallGenerationCollectionName('dense', generationId),
      responsibility: 'dense-evidence',
      scalarFields: DENSE_SCALAR_FIELDS,
      vectorFields: Object.freeze([
        Object.freeze({
          name: 'embedding',
          type: 'vector-fp32',
          dimensions: storedDimensions,
          index: Object.freeze({
            kind: 'hnsw',
            metric: 'cosine',
            m: DENSE_HNSW_M,
            efConstruction: DENSE_HNSW_EF_CONSTRUCTION,
          }),
        }),
      ]),
    }),
    sessionProjection: Object.freeze({
      formatVersion: RECALL_GENERATION_STORE_FORMAT_VERSION,
      directory: RECALL_GENERATION_STORE_DIRECTORIES.sessionProjection,
      collectionName: createRecallGenerationCollectionName('projection', generationId),
      responsibility: 'session-projection',
      scalarFields: SESSION_PROJECTION_SCALAR_FIELDS,
      vectorFields: Object.freeze([]),
    }),
  });
}

function createZvecScalarFieldSchema(
  field: Readonly<RecallGenerationScalarFieldContract>,
): ZVecFieldSchema {
  return {
    name: field.name,
    dataType: ZVEC_SCALAR_TYPES[field.type],
    ...(field.index
      ? {
          indexParams: {
            indexType: ZVecIndexType.FTS,
            tokenizerName: field.index.tokenizer,
            filters: [...field.index.filters],
          },
        }
      : {}),
  };
}

function createZvecVectorFieldSchema(
  field: Readonly<RecallGenerationVectorFieldContract>,
): ZVecVectorSchema {
  return {
    name: field.name,
    dataType: ZVecDataType.VECTOR_FP32,
    dimension: field.dimensions,
    indexParams: {
      indexType: ZVecIndexType.HNSW,
      metricType: ZVecMetricType.COSINE,
      m: field.index.m,
      efConstruction: field.index.efConstruction,
    },
  };
}

function createEmptyRecallGenerationStore(
  storePath: string,
  contract: Readonly<RecallGenerationStoreContract>,
): void {
  if (existsSync(storePath)) {
    throw new Error(`Recall coherent generation store already exists at ${storePath}`);
  }
  mkdirSync(dirname(storePath), { recursive: true });
  let collection: ZVecCollection;
  try {
    collection = ZVecCreateAndOpen(
      storePath,
      new ZVecCollectionSchema({
        name: contract.collectionName,
        fields: contract.scalarFields.map(createZvecScalarFieldSchema),
        vectors: contract.vectorFields.map(createZvecVectorFieldSchema),
      }),
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation ${contract.responsibility} store creation failed at ${storePath}: ${message}`,
      { cause: error },
    );
  }
  collection.closeSync();
}

function assertRecallGenerationScalarSchema(
  collection: ZVecCollection,
  contract: Readonly<RecallGenerationStoreContract>,
): void {
  const storedFields = collection.schema.fields();
  if (storedFields.length !== contract.scalarFields.length) {
    throw new Error(
      `Recall coherent generation ${contract.responsibility} scalar schema mismatch: expected ${contract.scalarFields.length} fields, received ${storedFields.length}`,
    );
  }
  for (const expected of contract.scalarFields) {
    const actual = storedFields.find(({ name }) => name === expected.name);
    if (!actual || actual.dataType !== ZVEC_SCALAR_TYPES[expected.type]) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} scalar field mismatch: ${expected.name}`,
      );
    }
    if (!expected.index && actual.indexParams !== undefined) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} unexpected scalar index: ${expected.name}`,
      );
    }
    if (
      expected.index &&
      (actual.indexParams?.indexType !== ZVecIndexType.FTS ||
        actual.indexParams.tokenizerName !== expected.index.tokenizer ||
        JSON.stringify(actual.indexParams.filters) !== JSON.stringify(expected.index.filters))
    ) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} full-text index mismatch: ${expected.name}`,
      );
    }
  }
}

function assertRecallGenerationVectorSchema(
  collection: ZVecCollection,
  contract: Readonly<RecallGenerationStoreContract>,
): void {
  const storedVectors = collection.schema.vectors();
  if (storedVectors.length !== contract.vectorFields.length) {
    throw new Error(
      `Recall coherent generation ${contract.responsibility} vector schema mismatch: expected ${contract.vectorFields.length}, received ${storedVectors.length}`,
    );
  }
  for (const expected of contract.vectorFields) {
    const actual = storedVectors.find(({ name }) => name === expected.name);
    if (
      !actual ||
      actual.dataType !== ZVecDataType.VECTOR_FP32 ||
      actual.dimension !== expected.dimensions ||
      actual.indexParams?.indexType !== ZVecIndexType.HNSW ||
      actual.indexParams.metricType !== ZVecMetricType.COSINE ||
      actual.indexParams.m !== expected.index.m ||
      actual.indexParams.efConstruction !== expected.index.efConstruction
    ) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} dense vector schema mismatch: ${expected.name}`,
      );
    }
  }
}

/** Opens one store read-only and checks only declared identity, schema, and indexes. */
export function openRecallGenerationStoreForBoundedCheck(
  storePath: string,
  contract: Readonly<RecallGenerationStoreContract>,
): ZVecCollection {
  if (!existsSync(storePath)) {
    throw new Error(
      `Recall coherent generation ${contract.responsibility} store missing at ${storePath}`,
    );
  }
  let collection: ZVecCollection;
  try {
    collection = ZVecOpen(storePath, { readOnly: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation ${contract.responsibility} store open failed at ${storePath}: ${message}`,
      { cause: error },
    );
  }
  try {
    if (collection.schema.name !== contract.collectionName) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} identity mismatch: expected ${contract.collectionName}, received ${collection.schema.name}`,
      );
    }
    assertRecallGenerationScalarSchema(collection, contract);
    assertRecallGenerationVectorSchema(collection, contract);
    return collection;
  } catch (error) {
    collection.closeSync();
    throw error;
  }
}

function openAndValidateRecallGenerationStore(
  storePath: string,
  contract: Readonly<RecallGenerationStoreContract>,
  expectedGenerationId: string,
  expectedRecordIds: readonly string[],
): ZVecCollection {
  const collection = openRecallGenerationStoreForBoundedCheck(storePath, contract);
  try {
    if (collection.stats.docCount !== expectedRecordIds.length) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} membership mismatch: expected ${expectedRecordIds.length} rows, received ${collection.stats.docCount}`,
      );
    }
    if (expectedRecordIds.length > 0) {
      const fetched = collection.fetchSync({
        ids: [...expectedRecordIds],
        outputFields: ['generationId'],
        includeVector: false,
      });
      if (Object.keys(fetched).length !== expectedRecordIds.length) {
        throw new Error(
          `Recall coherent generation ${contract.responsibility} exact membership mismatch`,
        );
      }
      for (const recordId of expectedRecordIds) {
        const generationId: unknown = fetched[recordId]?.fields.generationId;
        if (generationId !== expectedGenerationId) {
          throw new Error(
            `Recall coherent generation ${contract.responsibility} row generation identity mismatch: ${recordId}`,
          );
        }
      }
    }
    return collection;
  } catch (error) {
    collection.closeSync();
    throw error;
  }
}

/** Creates and closes all three independent empty zvec stores in manifest order. */
export function createEmptyRecallGenerationStores(
  paths: Readonly<RecallGenerationComponentPaths>,
  contracts: ReturnType<typeof createRecallGenerationStoreContracts>,
): void {
  createEmptyRecallGenerationStore(paths.lexicalSourceStorePath, contracts.lexicalSource);
  createEmptyRecallGenerationStore(paths.denseStorePath, contracts.dense);
  createEmptyRecallGenerationStore(paths.sessionProjectionStorePath, contracts.sessionProjection);
}

/** Creates missing bootstrap stores and accepts only compatible empty stores already created. */
export async function resumeEmptyRecallGenerationStores(
  paths: Readonly<RecallGenerationComponentPaths>,
  contracts: ReturnType<typeof createRecallGenerationStoreContracts>,
  onStoreCreated?: (
    responsibility: RecallGenerationStoreContract['responsibility'],
  ) => void | Promise<void>,
): Promise<void> {
  const stores = [
    [paths.lexicalSourceStorePath, contracts.lexicalSource],
    [paths.denseStorePath, contracts.dense],
    [paths.sessionProjectionStorePath, contracts.sessionProjection],
  ] as const;
  for (const [storePath, contract] of stores) {
    if (existsSync(storePath)) {
      let collection: ZVecCollection | undefined;
      try {
        collection = openRecallGenerationStoreForBoundedCheck(storePath, contract);
        if (collection.stats.docCount !== 0) {
          throw new Error('store is not empty');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Recall fixed snapshot generation bootstrap ${contract.responsibility} store incompatible; discard this staging generation: ${message}`,
          { cause: error },
        );
      } finally {
        collection?.closeSync();
      }
      continue;
    }
    try {
      createEmptyRecallGenerationStore(storePath, contract);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Recall fixed snapshot generation bootstrap ${contract.responsibility} store creation incomplete; discard this staging generation: ${message}`,
        { cause: error },
      );
    }
    await onStoreCreated?.(contract.responsibility);
  }
}

function createRecallGenerationStoreEnumerations(
  responsibility: RecallGenerationStoreContract['responsibility'],
): readonly ExactZvecDocumentEnumeration[] {
  switch (responsibility) {
    case 'lexical-source':
      return [
        {
          filter: "recordKind = 'entry-anchor'",
          uniquePartitionField: 'entryAnchorId',
          outputFields: [],
        },
        {
          filter: "recordKind = 'evidence'",
          uniquePartitionField: 'evidenceOccurrenceId',
          outputFields: [],
        },
      ];
    case 'dense-evidence':
      return [{ uniquePartitionField: 'evidenceOccurrenceId', outputFields: [] }];
    case 'session-projection':
      return [
        {
          filter: `projectionKind = '${RecallSessionProjectionKind.PHYSICAL_SESSION}'`,
          uniquePartitionField: 'physicalSourceIdentity',
          outputFields: [],
        },
        {
          filter: `projectionKind = '${RecallSessionProjectionKind.LOGICAL_SESSION}'`,
          uniquePartitionField: 'logicalSessionOccurrenceId',
          outputFields: [],
        },
      ];
    default:
      throw new Error('Recall coherent generation store responsibility unsupported');
  }
}

async function readRecallGenerationStoreRecordIds(
  storePath: string,
  responsibility: RecallGenerationStoreContract['responsibility'],
): Promise<string[]> {
  if (!existsSync(storePath)) {
    throw new Error(`Recall coherent generation ${responsibility} store missing at ${storePath}`);
  }
  let collection: ZVecCollection;
  try {
    collection = ZVecOpen(storePath, { readOnly: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall coherent generation ${responsibility} store open failed at ${storePath}: ${message}`,
      { cause: error },
    );
  }
  try {
    if (collection.stats.docCount === 0) {
      return [];
    }
    const recordIds: string[] = [];
    for (const enumeration of createRecallGenerationStoreEnumerations(responsibility)) {
      visitExactZvecDocuments(collection, enumeration, ({ id }) => recordIds.push(id));
    }
    if (recordIds.length !== collection.stats.docCount) {
      throw new Error(
        `Recall coherent generation ${responsibility} exact membership classification mismatch: expected ${collection.stats.docCount} rows, received ${recordIds.length}`,
      );
    }
    return recordIds.toSorted();
  } finally {
    collection.closeSync();
  }
}

/** Enumerates exact closed-store membership for source-free validation receipt checks. */
export async function readRecallGenerationStoreRecordMembership(
  paths: Readonly<RecallGenerationComponentPaths>,
): Promise<
  Readonly<{
    lexicalSource: string[];
    dense: string[];
    sessionProjection: string[];
  }>
> {
  const [lexicalSource, dense, sessionProjection] = await Promise.all([
    readRecallGenerationStoreRecordIds(paths.lexicalSourceStorePath, 'lexical-source'),
    readRecallGenerationStoreRecordIds(paths.denseStorePath, 'dense-evidence'),
    readRecallGenerationStoreRecordIds(paths.sessionProjectionStorePath, 'session-projection'),
  ]);
  return { lexicalSource, dense, sessionProjection };
}

/** Reopens every store read-only and validates schemas, identities, and exact expected membership. */
export function validateRecallGenerationStores(
  paths: Readonly<RecallGenerationComponentPaths>,
  contracts: ReturnType<typeof createRecallGenerationStoreContracts>,
  expectedGenerationId: string,
  expectedRecordIds: Readonly<{
    lexicalSource: readonly string[];
    dense: readonly string[];
    sessionProjection: readonly string[];
  }>,
): RecallGenerationStoreCounts {
  const collections: ZVecCollection[] = [];
  try {
    collections.push(
      openAndValidateRecallGenerationStore(
        paths.lexicalSourceStorePath,
        contracts.lexicalSource,
        expectedGenerationId,
        expectedRecordIds.lexicalSource,
      ),
      openAndValidateRecallGenerationStore(
        paths.denseStorePath,
        contracts.dense,
        expectedGenerationId,
        expectedRecordIds.dense,
      ),
      openAndValidateRecallGenerationStore(
        paths.sessionProjectionStorePath,
        contracts.sessionProjection,
        expectedGenerationId,
        expectedRecordIds.sessionProjection,
      ),
    );
    return {
      lexicalSource: expectedRecordIds.lexicalSource.length,
      dense: expectedRecordIds.dense.length,
      sessionProjection: expectedRecordIds.sessionProjection.length,
    };
  } finally {
    for (const collection of collections) {
      collection.closeSync();
    }
  }
}

/** Normalizes any zvec vector representation for checksum and width verification. */
export function readRecallGenerationVectorValues(vector: ZVecVector | undefined): number[] {
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

/** Validates the reopened dense store as the exact profile-bound lexical/source subset. */
export function validateRecallGenerationDenseSubset(
  paths: Readonly<RecallGenerationComponentPaths>,
  expectedGenerationId: string,
  expectedEmbeddingProfileId: string,
  expectedStoredDimensions: number,
  recordIds: Readonly<{
    lexicalSource: readonly string[];
    dense: readonly string[];
  }>,
): void {
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  const dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  try {
    const lexicalRecords = lexicalSource.fetchSync({
      ids: [...recordIds.lexicalSource],
      outputFields: ['recordKind', 'isDenseSearchable', 'evidenceOccurrenceId', 'evidenceChecksum'],
      includeVector: false,
    });
    const denseEvidenceChecksums = new Map<string, unknown>();
    for (const record of Object.values(lexicalRecords)) {
      if (record.fields.recordKind === 'evidence' && record.fields.isDenseSearchable === true) {
        denseEvidenceChecksums.set(record.id, record.fields.evidenceChecksum);
      }
    }
    const expectedDenseRecordIds = [...denseEvidenceChecksums.keys()].toSorted();
    if (
      JSON.stringify(expectedDenseRecordIds) !== JSON.stringify([...recordIds.dense].toSorted())
    ) {
      throw new Error(
        `Recall coherent generation dense subset membership mismatch for ${expectedGenerationId}`,
      );
    }
    if (recordIds.dense.length === 0) {
      return;
    }
    const denseRecords = dense.fetchSync({
      ids: [...recordIds.dense],
      outputFields: [
        'generationId',
        'evidenceOccurrenceId',
        'embeddingProfileId',
        'storedDimensions',
        'evidenceChecksum',
        'embeddingInputChecksum',
        'vectorChecksum',
      ],
      includeVector: true,
    });
    for (const recordId of recordIds.dense) {
      const record = denseRecords[recordId];
      if (record === undefined) {
        throw new Error(
          `Recall coherent generation dense subset row missing for ${expectedGenerationId}: ${recordId}`,
        );
      }
      if (
        record.fields.generationId !== expectedGenerationId ||
        record.fields.evidenceOccurrenceId !== recordId
      ) {
        throw new Error(
          `Recall coherent generation dense occurrence identity mismatch for ${recordId}`,
        );
      }
      if (record.fields.embeddingProfileId !== expectedEmbeddingProfileId) {
        throw new Error(
          `Recall coherent generation dense embedding profile mismatch for ${recordId}`,
        );
      }
      if (record.fields.storedDimensions !== expectedStoredDimensions) {
        throw new Error(`Recall coherent generation dense stored width mismatch for ${recordId}`);
      }
      if (record.fields.evidenceChecksum !== denseEvidenceChecksums.get(recordId)) {
        throw new Error(
          `Recall coherent generation dense evidence checksum mismatch for ${recordId}`,
        );
      }
      if (!/^[a-f0-9]{64}$/u.test(String(record.fields.embeddingInputChecksum))) {
        throw new Error(
          `Recall coherent generation dense embedding input checksum mismatch for ${recordId}`,
        );
      }
      const vector = readRecallGenerationVectorValues(record.vectors.embedding);
      if (
        vector.length !== expectedStoredDimensions ||
        vector.some((value) => !Number.isFinite(value))
      ) {
        throw new Error(`Recall coherent generation dense vector width mismatch for ${recordId}`);
      }
      const vectorChecksum = createHash('sha256')
        .update(Buffer.from(new Float32Array(vector).buffer))
        .digest('hex');
      if (record.fields.vectorChecksum !== vectorChecksum) {
        throw new Error(
          `Recall coherent generation dense vector checksum mismatch for ${recordId}`,
        );
      }
    }
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
  }
}

/** Reopens every store read-only and validates exact empty membership and generation identity. */
export function validateEmptyRecallGenerationStores(
  paths: Readonly<RecallGenerationComponentPaths>,
  contracts: ReturnType<typeof createRecallGenerationStoreContracts>,
  expectedGenerationId: string,
): RecallGenerationStoreCounts {
  return validateRecallGenerationStores(paths, contracts, expectedGenerationId, {
    lexicalSource: [],
    dense: [],
    sessionProjection: [],
  });
}

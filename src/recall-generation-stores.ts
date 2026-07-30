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
  type ZVecVectorSchema,
} from '@zvec/zvec';

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
  { name: 'logicalSessionOccurrenceId', type: 'string' },
  { name: 'entryId', type: 'string' },
  { name: 'parentEntryId', type: 'string' },
  { name: 'sourceOrder', type: 'int64' },
  { name: 'evidenceOccurrenceId', type: 'string' },
  { name: 'evidenceKind', type: 'string' },
  { name: 'evidencePart', type: 'string' },
  { name: 'projectIdentity', type: 'string' },
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
  { name: 'embeddingProfileId', type: 'string' },
  { name: 'storedDimensions', type: 'int32' },
  { name: 'evidenceChecksum', type: 'string' },
  { name: 'embeddingInputChecksum', type: 'string' },
  { name: 'vectorChecksum', type: 'string' },
  { name: 'projectIdentity', type: 'string' },
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

function openAndValidateEmptyRecallGenerationStore(
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
    if (collection.stats.docCount !== 0) {
      throw new Error(
        `Recall coherent generation ${contract.responsibility} membership mismatch: expected 0 rows, received ${collection.stats.docCount}`,
      );
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

/** Reopens every store read-only and validates exact empty membership and generation identity. */
export function validateEmptyRecallGenerationStores(
  paths: Readonly<RecallGenerationComponentPaths>,
  contracts: ReturnType<typeof createRecallGenerationStoreContracts>,
): RecallGenerationStoreCounts {
  const collections: ZVecCollection[] = [];
  try {
    collections.push(
      openAndValidateEmptyRecallGenerationStore(
        paths.lexicalSourceStorePath,
        contracts.lexicalSource,
      ),
      openAndValidateEmptyRecallGenerationStore(paths.denseStorePath, contracts.dense),
      openAndValidateEmptyRecallGenerationStore(
        paths.sessionProjectionStorePath,
        contracts.sessionProjection,
      ),
    );
    return { lexicalSource: 0, dense: 0, sessionProjection: 0 };
  } finally {
    for (const collection of collections) {
      collection.closeSync();
    }
  }
}

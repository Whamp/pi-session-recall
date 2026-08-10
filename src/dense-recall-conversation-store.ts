import { createHash } from 'node:crypto';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecVector,
  type ZVecVectorSchema,
} from '@zvec/zvec';

import type { RecallDenseCandidate } from './rank-recall-search-results.js';
import { convertNormalizedRecallInnerProductToCosineDistance } from './recall-stored-embedding.js';
import type { ProjectIdentity } from './resolve-project-identity.js';
import type { SessionConversationChunk } from './session-conversation-index.js';
import {
  DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS,
  deserializeDenseRecallDocumentFields,
  serializeDenseRecallDocumentFields,
} from './dense-recall-document-codec.js';

/** Fixed FP32 vector width established by the production dense recall prototype. */
export const DENSE_RECALL_EMBEDDING_DIMENSIONS = 1_024;

/** Version of the dense-only flat conversation store schema. */
export const DENSE_RECALL_CONVERSATION_SCHEMA_VERSION = 1;

/** Manifest identity that distinguishes the dense-only flat store from legacy zvec layouts. */
export interface DenseRecallConversationStoreIdentity {
  schemaVersion: number;
  layout: 'dense-only';
  embeddingDimensions: 1_024;
  vectorQuantization: 'fp32';
  metric: 'inner-product';
  vectorIndex: 'flat';
  fullTextIndexes: false;
}

/** Current manifest identity for the dense-only flat conversation store. */
export const DENSE_RECALL_CONVERSATION_STORE_IDENTITY: Readonly<DenseRecallConversationStoreIdentity> =
  Object.freeze({
    schemaVersion: DENSE_RECALL_CONVERSATION_SCHEMA_VERSION,
    layout: 'dense-only',
    embeddingDimensions: DENSE_RECALL_EMBEDDING_DIMENSIONS,
    vectorQuantization: 'fp32',
    metric: 'inner-product',
    vectorIndex: 'flat',
    fullTextIndexes: false,
  });

/** One conversation, summary, branch-summary, or turn-context document with a real embedding. */
export interface DenseRecallDocument extends SessionConversationChunk {
  embedding: number[];
}

/** Dense-only persistence and flat vector search operations for conversation recall. */
export interface DenseRecallConversationStore {
  upsertDocuments(documents: DenseRecallDocument[]): void;
  deleteDocuments(ids: string[]): void;
  fetchDocuments(this: void, ids: string[]): Map<string, SessionConversationChunk>;
  fetchVectors(ids: string[]): Map<string, number[]>;
  searchDenseCandidates(
    embedding: number[],
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallDenseCandidate[];
  countDocuments(): number;
  close(): void;
}

const DENSE_RECALL_OUTPUT_FIELDS = DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS.map((field) => field.name);

function convertDenseRecallVector(id: string, vector?: ZVecVector): number[] {
  if (Array.isArray(vector) || vector instanceof Float32Array || vector instanceof Int8Array) {
    return Array.from(vector, Number);
  }
  throw new Error(`Dense recall vector invalid for document ${id}: expected a dense vector`);
}

function assertDenseRecallEmbedding(embedding: readonly number[], subject: string): void {
  if (embedding.length !== DENSE_RECALL_EMBEDDING_DIMENSIONS) {
    throw new Error(
      `Dense recall embedding invalid for ${subject}: expected ${DENSE_RECALL_EMBEDDING_DIMENSIONS} dimensions, received ${embedding.length}`,
    );
  }
  let hasNonZeroFp32Component = false;
  for (const value of embedding) {
    const fp32Value = Math.fround(value);
    if (!Number.isFinite(value) || !Number.isFinite(fp32Value)) {
      throw new Error(`Dense recall embedding invalid for ${subject}: expected finite FP32 values`);
    }
    hasNonZeroFp32Component ||= fp32Value !== 0;
  }
  if (!hasNonZeroFp32Component) {
    throw new Error(`Dense recall embedding invalid for ${subject}: zero vectors are not allowed`);
  }
}

function assertDenseRecallDocument(document: DenseRecallDocument): void {
  if (
    document.documentKind !== 'conversation' &&
    document.documentKind !== 'turn_context' &&
    document.documentKind !== 'summary'
  ) {
    throw new Error(
      `Dense recall document invalid for ${document.id}: only conversation, summary, branch-summary, and turn-context documents are allowed`,
    );
  }
  assertDenseRecallEmbedding(document.embedding, `document ${document.id}`);
}

function assertDenseRecallCandidateLimit(limit: number): void {
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    throw new Error('Dense recall candidate limit invalid: expected an integer from 1 to 200');
  }
}

function createDenseRecallProjectFilter(projectIdentity?: ProjectIdentity): string | undefined {
  if (!projectIdentity) {
    return undefined;
  }
  const digest = createHash('sha256').update(projectIdentity).digest('hex');
  return `projectIdentityDigest = '${digest}'`;
}

function assertDenseRecallCollectionSchema(collection: ZVecCollection, databasePath: string): void {
  let vector: ZVecVectorSchema;
  try {
    vector = collection.schema.vector('embedding');
  } catch (error) {
    collection.closeSync();
    throw new Error(
      `Dense recall zvec schema incompatible at ${databasePath}; rebuild with psr index --rebuild`,
      { cause: error },
    );
  }
  const fields = collection.schema.fields();
  const fieldSchemasMatch =
    fields.length === DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS.length &&
    fields.every((field, index) => {
      const expected = DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS[index];
      return (
        expected !== undefined &&
        field.name === expected.name &&
        field.dataType === expected.dataType &&
        Boolean(field.nullable) === Boolean(expected.nullable) &&
        field.indexParams === undefined
      );
    });
  if (
    collection.schema.vectors().length !== 1 ||
    vector.dataType !== ZVecDataType.VECTOR_FP32 ||
    vector.dimension !== DENSE_RECALL_EMBEDDING_DIMENSIONS ||
    vector.indexParams?.indexType !== ZVecIndexType.FLAT ||
    vector.indexParams.metricType !== ZVecMetricType.IP ||
    !fieldSchemasMatch
  ) {
    collection.closeSync();
    throw new Error(
      `Dense recall zvec schema incompatible at ${databasePath}; rebuild with psr index --rebuild`,
    );
  }
}

/** Opens the additive dense-only FP32 FLAT conversation store. */
export function openDenseRecallConversationStore(config: {
  databasePath: string;
  createIfMissing?: boolean;
  readOnly?: boolean;
}): DenseRecallConversationStore {
  const databaseExists = existsSync(config.databasePath);
  if (!databaseExists && config.createIfMissing === false) {
    throw new Error(
      `Dense recall zvec collection missing at ${config.databasePath}; rebuild with psr index --rebuild`,
    );
  }
  if (!databaseExists) {
    mkdirSync(dirname(config.databasePath), { recursive: true });
  }
  const collection = databaseExists
    ? config.readOnly
      ? ZVecOpen(config.databasePath, { readOnly: true })
      : ZVecOpen(config.databasePath)
    : ZVecCreateAndOpen(
        config.databasePath,
        new ZVecCollectionSchema({
          name: 'pi_session_recall_dense',
          vectors: {
            name: 'embedding',
            dataType: ZVecDataType.VECTOR_FP32,
            dimension: DENSE_RECALL_EMBEDDING_DIMENSIONS,
            indexParams: {
              indexType: ZVecIndexType.FLAT,
              metricType: ZVecMetricType.IP,
            },
          },
          fields: [...DENSE_RECALL_DOCUMENT_FIELD_SCHEMAS],
        }),
      );
  assertDenseRecallCollectionSchema(collection, config.databasePath);

  return {
    upsertDocuments(documents) {
      if (documents.length === 0) {
        return;
      }
      for (const document of documents) {
        assertDenseRecallDocument(document);
      }
      collection.upsertSync(
        documents.map(({ embedding, ...document }) => ({
          id: document.id,
          vectors: { embedding: embedding.map(Math.fround) },
          fields: serializeDenseRecallDocumentFields(document),
        })),
      );
    },
    deleteDocuments(ids) {
      if (ids.length > 0) {
        collection.deleteSync(ids);
      }
    },
    fetchDocuments(ids) {
      if (ids.length === 0) {
        return new Map();
      }
      const documents = collection.fetchSync({
        ids,
        outputFields: DENSE_RECALL_OUTPUT_FIELDS,
        includeVector: false,
      });
      return new Map(
        Object.values(documents).map((document) => [
          document.id,
          deserializeDenseRecallDocumentFields(document),
        ]),
      );
    },
    fetchVectors(ids) {
      if (ids.length === 0) {
        return new Map();
      }
      const documents = collection.fetchSync({ ids, outputFields: [], includeVector: true });
      return new Map(
        Object.values(documents).map((document) => [
          document.id,
          convertDenseRecallVector(document.id, document.vectors.embedding),
        ]),
      );
    },
    searchDenseCandidates(embedding, limit, projectIdentity) {
      assertDenseRecallEmbedding(embedding, 'query');
      assertDenseRecallCandidateLimit(limit);
      const projectFilter = createDenseRecallProjectFilter(projectIdentity);
      return collection
        .querySync({
          fieldName: 'embedding',
          vector: embedding.map(Math.fround),
          topk: limit,
          outputFields: DENSE_RECALL_OUTPUT_FIELDS,
          includeVector: false,
          ...(projectFilter ? { filter: projectFilter } : {}),
        })
        .map((document) => ({
          ...deserializeDenseRecallDocumentFields(document),
          cosineDistance: convertNormalizedRecallInnerProductToCosineDistance(document.score),
        }));
    },
    countDocuments() {
      return collection.stats.docCount;
    },
    close() {
      collection.closeSync();
    },
  };
}

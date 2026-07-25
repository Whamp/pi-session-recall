import { existsSync } from 'node:fs';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
} from '@zvec/zvec';

import type {
  PiSessionEntryId,
  PiSessionId,
  SessionConversationChunk,
} from './session-conversation-index.js';

/** A conversation chunk paired with the local embedding persisted in zvec. */
export interface EmbeddedSessionConversationChunk extends SessionConversationChunk {
  embedding: number[];
}

/** A ranked semantic match with exact Pi session provenance. */
export interface RecallSearchResult extends SessionConversationChunk {
  score: number;
}

/** The narrow zvec persistence contract used by incremental indexing and recall search. */
export interface ZvecConversationStore {
  upsertChunks(chunks: EmbeddedSessionConversationChunk[]): void;
  deleteChunks(ids: string[]): void;
  search(embedding: number[], limit: number): RecallSearchResult[];
  fetchChecksums(ids: string[]): Map<string, string>;
  optimize(): Promise<void>;
  close(): void;
  count(): number;
}

const RECALL_OUTPUT_FIELDS = [
  'checksum',
  'sessionId',
  'sessionPath',
  'cwd',
  'sessionName',
  'entryId',
  'role',
  'timestamp',
  'chunkIndex',
  'content',
];

function parseRecallConversationRole(value: unknown): SessionConversationChunk['role'] {
  if (value === 'user' || value === 'assistant' || value === 'summary' || value === 'custom') {
    return value;
  }
  throw new Error(`Recall zvec role invalid: ${String(value)}`);
}

/** Opens the durable zvec collection that stores embedded Pi conversation chunks. */
export function openZvecConversationStore(config: {
  databasePath: string;
  dimensions: number;
}): ZvecConversationStore {
  mkdirSync(dirname(config.databasePath), { recursive: true });
  const collection: ZVecCollection = existsSync(config.databasePath)
    ? ZVecOpen(config.databasePath)
    : ZVecCreateAndOpen(
        config.databasePath,
        new ZVecCollectionSchema({
          name: 'pi_session_recall',
          vectors: {
            name: 'embedding',
            dataType: ZVecDataType.VECTOR_FP32,
            dimension: config.dimensions,
            indexParams: {
              indexType: ZVecIndexType.HNSW,
              metricType: ZVecMetricType.COSINE,
            },
          },
          fields: RECALL_OUTPUT_FIELDS.map((name) => ({
            name,
            dataType: name === 'chunkIndex' ? ZVecDataType.INT32 : ZVecDataType.STRING,
          })),
        }),
      );
  const storedDimensions = collection.schema.vector('embedding').dimension;
  if (storedDimensions !== config.dimensions) {
    collection.closeSync();
    throw new Error(
      `Recall zvec dimension mismatch: collection uses ${storedDimensions}, configured model uses ${config.dimensions}; remove the recall data directory and reindex`,
    );
  }

  return {
    upsertChunks(chunks) {
      if (chunks.length === 0) {
        return;
      }
      collection.upsertSync(
        chunks.map(({ embedding, ...chunk }) => ({
          id: chunk.id,
          vectors: { embedding },
          fields: {
            checksum: chunk.checksum,
            sessionId: chunk.sessionId.value,
            sessionPath: chunk.sessionPath,
            cwd: chunk.cwd,
            sessionName: chunk.sessionName,
            entryId: chunk.entryId.value,
            role: chunk.role,
            timestamp: chunk.timestamp,
            chunkIndex: chunk.chunkIndex,
            content: chunk.content,
          },
        })),
      );
    },
    deleteChunks(ids) {
      if (ids.length > 0) {
        collection.deleteSync(ids);
      }
    },
    search(embedding, limit) {
      return collection
        .querySync({
          fieldName: 'embedding',
          vector: embedding,
          topk: limit,
          outputFields: RECALL_OUTPUT_FIELDS,
          includeVector: false,
        })
        .map((doc) => ({
          id: doc.id,
          checksum: String(doc.fields.checksum),
          sessionId: { value: String(doc.fields.sessionId) } satisfies PiSessionId,
          sessionPath: String(doc.fields.sessionPath),
          cwd: String(doc.fields.cwd),
          sessionName: String(doc.fields.sessionName),
          entryId: { value: String(doc.fields.entryId) } satisfies PiSessionEntryId,
          role: parseRecallConversationRole(doc.fields.role),
          timestamp: String(doc.fields.timestamp),
          chunkIndex: Number(doc.fields.chunkIndex),
          content: String(doc.fields.content),
          score: doc.score,
        }));
    },
    fetchChecksums(ids) {
      if (ids.length === 0) {
        return new Map();
      }
      const docs = collection.fetchSync({ ids, outputFields: ['checksum'], includeVector: false });
      return new Map(Object.values(docs).map((doc) => [doc.id, String(doc.fields.checksum)]));
    },
    optimize() {
      return collection.optimize();
    },
    close() {
      collection.closeSync();
    },
    count() {
      return collection.stats.docCount;
    },
  };
}

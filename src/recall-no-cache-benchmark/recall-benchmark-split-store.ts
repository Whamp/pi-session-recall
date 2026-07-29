import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
  ZVecOpen,
  type ZVecCollection,
  type ZVecDoc,
  type ZVecDocInput,
  type ZVecStatus,
} from '@zvec/zvec';

import type { SessionConversationChunk } from '../session-conversation-index.js';

/** Maximum immutable write window used by the issue 112 benchmark stores. */
export const RECALL_BENCHMARK_WRITE_BATCH_SIZE = 32;

/** One source entry anchor used for index-only conversation-neighborhood traversal. */
export interface RecallBenchmarkEntryAnchor {
  anchorId: string;
  checksum: string;
  physicalSessionProjectionId: string;
  logicalSessionId: string;
  rawSessionId: string;
  sessionPath: string;
  entryId: string;
  parentEntryId: string;
  childEntryIds: string[];
  branchPathLeafIds: string[];
  occurrenceIds: string[];
  sourceLineStart: number;
  sourceLineEnd: number;
}

/** One vector-free searchable occurrence with its stable split-store identity. */
export interface RecallBenchmarkEvidenceOccurrence {
  occurrenceId: string;
  logicalSessionId: string;
  physicalSessionProjectionId: string;
  embeddingInputChecksum: string;
  chunk: SessionConversationChunk;
}

/** One dense occurrence vector ready for checked insertion. */
export interface RecallBenchmarkDenseOccurrence {
  occurrence: RecallBenchmarkEvidenceOccurrence;
  embeddingProfileId: string;
  embedding: number[];
}

/** Exact dense row recovered from one previous scratch generation. */
export interface StoredRecallBenchmarkDenseOccurrence {
  occurrenceId: string;
  checksum: string;
  embeddingInputChecksum: string;
  embeddingProfileId: string;
  embedding: number[];
}

/** Aggregate split-store validation canaries that contain no conversation content. */
export interface RecallBenchmarkStoreValidation {
  lexicalSourceRows: number;
  denseRows: number;
  exactAnchorCanary: boolean;
  exactEvidenceCanary: boolean;
  lexicalFtsCanary: boolean;
  denseNearestNeighborCanary: boolean;
  denseMembershipCanary: boolean;
}

/** Open vector-free lexical/source and dense zvec stores for one scratch generation. */
export interface RecallBenchmarkSplitStores {
  lexicalSource: ZVecCollection;
  dense: ZVecCollection;
  insertEntryAnchors(anchors: readonly RecallBenchmarkEntryAnchor[]): number;
  insertLexicalEvidence(occurrences: readonly RecallBenchmarkEvidenceOccurrence[]): number;
  insertDenseEvidence(occurrences: readonly RecallBenchmarkDenseOccurrence[]): number;
  fetchDenseEvidence(
    occurrences: readonly Pick<RecallBenchmarkEvidenceOccurrence, 'occurrenceId'>[],
  ): Map<string, StoredRecallBenchmarkDenseOccurrence>;
  optimize(): Promise<{ lexicalSourceMilliseconds: number; denseMilliseconds: number }>;
  validate(options: {
    expectedLexicalSourceRows: number;
    expectedDenseRows: number;
    anchorCanaryId: string | null;
    evidenceCanary: RecallBenchmarkEvidenceOccurrence | null;
    denseCanary: RecallBenchmarkDenseOccurrence | null;
  }): RecallBenchmarkStoreValidation;
  close(): void;
}

const lexicalSourceFieldSchemas = [
  {
    name: 'rowKind',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'schemaVersion', dataType: ZVecDataType.INT32 },
  { name: 'documentKind', dataType: ZVecDataType.STRING },
  { name: 'summaryKind', dataType: ZVecDataType.STRING },
  { name: 'evidenceKind', dataType: ZVecDataType.STRING },
  { name: 'evidencePart', dataType: ZVecDataType.STRING },
  { name: 'isDenseSearchable', dataType: ZVecDataType.BOOL },
  { name: 'checksum', dataType: ZVecDataType.STRING },
  { name: 'embeddingInputChecksum', dataType: ZVecDataType.STRING },
  { name: 'sessionId', dataType: ZVecDataType.STRING },
  { name: 'sessionPath', dataType: ZVecDataType.STRING },
  {
    name: 'physicalSessionProjectionId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  {
    name: 'logicalSessionId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'parentSessionPath', dataType: ZVecDataType.STRING },
  { name: 'cwd', dataType: ZVecDataType.STRING },
  { name: 'projectPath', dataType: ZVecDataType.STRING },
  { name: 'projectIdentity', dataType: ZVecDataType.STRING },
  {
    name: 'projectIdentityDigest',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'projectIdentitySource', dataType: ZVecDataType.STRING },
  { name: 'sessionName', dataType: ZVecDataType.STRING },
  {
    name: 'entryId',
    dataType: ZVecDataType.STRING,
    indexParams: { indexType: ZVecIndexType.INVERT },
  },
  { name: 'parentEntryId', dataType: ZVecDataType.STRING },
  { name: 'childEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'contributingEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'currentLeafId', dataType: ZVecDataType.STRING },
  { name: 'branchPathLeafIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'occurrenceIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'isOnActiveBranch', dataType: ZVecDataType.BOOL },
  { name: 'isVisibleInActiveContext', dataType: ZVecDataType.BOOL },
  { name: 'compactedByEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'compactionFirstKeptEntryId', dataType: ZVecDataType.STRING },
  { name: 'branchSummaryFromEntryId', dataType: ZVecDataType.STRING },
  { name: 'role', dataType: ZVecDataType.STRING },
  { name: 'timestamp', dataType: ZVecDataType.STRING },
  { name: 'sourceLineStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceLineEnd', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockStart', dataType: ZVecDataType.INT32 },
  { name: 'sourceBlockEnd', dataType: ZVecDataType.INT32 },
  { name: 'characterStart', dataType: ZVecDataType.INT32 },
  { name: 'characterEnd', dataType: ZVecDataType.INT32 },
  { name: 'tokenStart', dataType: ZVecDataType.INT32 },
  { name: 'tokenEnd', dataType: ZVecDataType.INT32 },
  { name: 'tokenCount', dataType: ZVecDataType.INT32 },
  { name: 'overlapTokenCount', dataType: ZVecDataType.INT32 },
  { name: 'textRunId', dataType: ZVecDataType.STRING },
  { name: 'textRunIndex', dataType: ZVecDataType.INT32 },
  { name: 'chunkIndex', dataType: ZVecDataType.INT32 },
  { name: 'chunkCount', dataType: ZVecDataType.INT32 },
  { name: 'siblingIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'previousSiblingId', dataType: ZVecDataType.STRING },
  { name: 'nextSiblingId', dataType: ZVecDataType.STRING },
  { name: 'toolCallId', dataType: ZVecDataType.STRING },
  { name: 'toolName', dataType: ZVecDataType.STRING },
  { name: 'toolCallEntryId', dataType: ZVecDataType.STRING },
  { name: 'toolResultEntryId', dataType: ZVecDataType.STRING },
  { name: 'toolError', dataType: ZVecDataType.INT32 },
  {
    name: 'content',
    dataType: ZVecDataType.STRING,
    indexParams: {
      indexType: ZVecIndexType.FTS,
      tokenizerName: 'standard',
      filters: ['lowercase'],
    },
  },
  {
    name: 'identifierContent',
    dataType: ZVecDataType.STRING,
    indexParams: {
      indexType: ZVecIndexType.FTS,
      tokenizerName: 'standard',
      filters: [],
    },
  },
];

const denseOutputFields = ['checksum', 'embeddingInputChecksum', 'embeddingProfileId'] as const;

function digestHex(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function serializeNullableEntryId(value: { value: string } | null): string {
  return value?.value ?? '';
}

function serializeEvidenceOccurrence(occurrence: RecallBenchmarkEvidenceOccurrence): ZVecDocInput {
  const chunk = occurrence.chunk;
  return {
    id: occurrence.occurrenceId,
    fields: {
      rowKind: 'evidence',
      schemaVersion: chunk.schemaVersion,
      documentKind: chunk.documentKind,
      summaryKind: chunk.summaryKind ?? '',
      evidenceKind: chunk.evidenceKind,
      evidencePart: chunk.evidencePart,
      isDenseSearchable: chunk.isDenseSearchable,
      checksum: chunk.checksum,
      embeddingInputChecksum: occurrence.embeddingInputChecksum,
      sessionId: chunk.sessionId.value,
      sessionPath: chunk.sessionPath,
      physicalSessionProjectionId: occurrence.physicalSessionProjectionId,
      logicalSessionId: occurrence.logicalSessionId,
      parentSessionPath: chunk.parentSessionPath ?? '',
      cwd: chunk.cwd,
      projectPath: chunk.projectPath,
      projectIdentity: chunk.projectAttribution?.projectIdentity ?? '',
      projectIdentityDigest: chunk.projectAttribution
        ? digestHex(chunk.projectAttribution.projectIdentity)
        : '',
      projectIdentitySource: chunk.projectAttribution?.identitySource ?? '',
      sessionName: chunk.sessionName,
      entryId: chunk.entryId.value,
      parentEntryId: serializeNullableEntryId(chunk.parentEntryId),
      childEntryIds: chunk.childEntryIds.map((id) => id.value),
      contributingEntryIds: chunk.contributingEntryIds.map((id) => id.value),
      currentLeafId: serializeNullableEntryId(chunk.currentLeafId),
      branchPathLeafIds: chunk.branchPathLeafIds.map((id) => id.value),
      occurrenceIds: [],
      isOnActiveBranch: chunk.isOnActiveBranch,
      isVisibleInActiveContext: chunk.isVisibleInActiveContext,
      compactedByEntryIds: chunk.compactedByEntryIds.map((id) => id.value),
      compactionFirstKeptEntryId: serializeNullableEntryId(chunk.compactionFirstKeptEntryId),
      branchSummaryFromEntryId: serializeNullableEntryId(chunk.branchSummaryFromEntryId),
      role: chunk.role,
      timestamp: chunk.timestamp,
      sourceLineStart: chunk.sourceLineStart,
      sourceLineEnd: chunk.sourceLineEnd,
      sourceBlockStart: chunk.sourceBlockStart ?? -1,
      sourceBlockEnd: chunk.sourceBlockEnd ?? -1,
      characterStart: chunk.characterStart,
      characterEnd: chunk.characterEnd,
      tokenStart: chunk.tokenStart,
      tokenEnd: chunk.tokenEnd,
      tokenCount: chunk.tokenCount,
      overlapTokenCount: chunk.overlapTokenCount,
      textRunId: chunk.textRunId,
      textRunIndex: chunk.textRunIndex,
      chunkIndex: chunk.chunkIndex,
      chunkCount: chunk.chunkCount,
      siblingIds: chunk.siblingIds,
      previousSiblingId: chunk.previousSiblingId ?? '',
      nextSiblingId: chunk.nextSiblingId ?? '',
      toolCallId: chunk.toolCallId ?? '',
      toolName: chunk.toolName ?? '',
      toolCallEntryId: serializeNullableEntryId(chunk.toolCallEntryId),
      toolResultEntryId: serializeNullableEntryId(chunk.toolResultEntryId),
      toolError: chunk.toolError === null ? -1 : Number(chunk.toolError),
      content: chunk.content,
      identifierContent: chunk.content,
    },
  };
}

function serializeEntryAnchor(anchor: RecallBenchmarkEntryAnchor): ZVecDocInput {
  return {
    id: anchor.anchorId,
    fields: {
      rowKind: 'entry_anchor',
      schemaVersion: 1,
      documentKind: '',
      summaryKind: '',
      evidenceKind: '',
      evidencePart: '',
      isDenseSearchable: false,
      checksum: anchor.checksum,
      embeddingInputChecksum: '',
      sessionId: anchor.rawSessionId,
      sessionPath: anchor.sessionPath,
      physicalSessionProjectionId: anchor.physicalSessionProjectionId,
      logicalSessionId: anchor.logicalSessionId,
      parentSessionPath: '',
      cwd: '',
      projectPath: '',
      projectIdentity: '',
      projectIdentityDigest: '',
      projectIdentitySource: '',
      sessionName: '',
      entryId: anchor.entryId,
      parentEntryId: anchor.parentEntryId,
      childEntryIds: anchor.childEntryIds,
      contributingEntryIds: [],
      currentLeafId: '',
      branchPathLeafIds: anchor.branchPathLeafIds,
      occurrenceIds: anchor.occurrenceIds,
      isOnActiveBranch: false,
      isVisibleInActiveContext: false,
      compactedByEntryIds: [],
      compactionFirstKeptEntryId: '',
      branchSummaryFromEntryId: '',
      role: '',
      timestamp: '',
      sourceLineStart: anchor.sourceLineStart,
      sourceLineEnd: anchor.sourceLineEnd,
      sourceBlockStart: -1,
      sourceBlockEnd: -1,
      characterStart: 0,
      characterEnd: 0,
      tokenStart: 0,
      tokenEnd: 0,
      tokenCount: 0,
      overlapTokenCount: 0,
      textRunId: '',
      textRunIndex: 0,
      chunkIndex: 0,
      chunkCount: 0,
      siblingIds: [],
      previousSiblingId: '',
      nextSiblingId: '',
      toolCallId: '',
      toolName: '',
      toolCallEntryId: '',
      toolResultEntryId: '',
      toolError: -1,
      content: '',
      identifierContent: '',
    },
  };
}

function serializeDenseOccurrence(occurrence: RecallBenchmarkDenseOccurrence): ZVecDocInput {
  const chunk = occurrence.occurrence.chunk;
  return {
    id: occurrence.occurrence.occurrenceId,
    vectors: { embedding: occurrence.embedding },
    fields: {
      checksum: chunk.checksum,
      embeddingInputChecksum: occurrence.occurrence.embeddingInputChecksum,
      embeddingProfileId: occurrence.embeddingProfileId,
      physicalSessionProjectionId: occurrence.occurrence.physicalSessionProjectionId,
      logicalSessionId: occurrence.occurrence.logicalSessionId,
      entryId: chunk.entryId.value,
      projectIdentityDigest: chunk.projectAttribution
        ? digestHex(chunk.projectAttribution.projectIdentity)
        : '',
    },
  };
}

function normalizeStatuses(statuses: ZVecStatus | ZVecStatus[]): ZVecStatus[] {
  return Array.isArray(statuses) ? statuses : [statuses];
}

function assertCheckedInsertStatuses(operation: string, statuses: ZVecStatus | ZVecStatus[]): void {
  for (const [index, status] of normalizeStatuses(statuses).entries()) {
    if (!status.ok) {
      throw new Error(
        `Recall no-cache benchmark ${operation} failed at position ${index} [${status.code}]: ${status.message}`,
      );
    }
  }
}

function insertDocuments(collection: ZVecCollection, documents: readonly ZVecDocInput[]): number {
  let inserted = 0;
  for (let start = 0; start < documents.length; start += RECALL_BENCHMARK_WRITE_BATCH_SIZE) {
    const batch = documents.slice(start, start + RECALL_BENCHMARK_WRITE_BATCH_SIZE);
    assertCheckedInsertStatuses('insert', collection.insertSync(batch));
    inserted += batch.length;
  }
  return inserted;
}

function readRequiredStringField(document: ZVecDoc, name: string): string {
  const value: unknown = Reflect.get(document.fields, name);
  if (typeof value !== 'string') {
    throw new Error(`Recall no-cache benchmark dense field ${name} is not a string`);
  }
  return value;
}

function readDenseVector(document: ZVecDoc): number[] {
  const vector = document.vectors.embedding;
  if (Array.isArray(vector) || vector instanceof Float32Array || vector instanceof Int8Array) {
    return Array.from(vector, Number);
  }
  throw new Error(`Recall no-cache benchmark dense vector missing for ${document.id}`);
}

function fetchRequiredDocument(
  collection: ZVecCollection,
  id: string,
  outputFields: string[],
  includeVector: boolean,
): ZVecDoc | null {
  const documents = collection.fetchSync({ ids: [id], outputFields, includeVector });
  return documents[id] ?? null;
}

function selectFtsCanaryTerm(content: string): string | null {
  return content.match(/[\p{L}\p{N}_-]{4,}/u)?.[0] ?? null;
}

/** Creates the vector-free lexical/source and dense stores owned by one scratch generation. */
export function createRecallBenchmarkSplitStores(options: {
  generationDirectory: string;
  embeddingDimensions: number;
}): RecallBenchmarkSplitStores {
  mkdirSync(options.generationDirectory, { recursive: true });
  const lexicalSource = ZVecCreateAndOpen(
    join(options.generationDirectory, 'lexical-source'),
    new ZVecCollectionSchema({
      name: 'recall_no_cache_benchmark_lexical_source',
      fields: lexicalSourceFieldSchemas,
    }),
  );
  const dense = ZVecCreateAndOpen(
    join(options.generationDirectory, 'dense'),
    new ZVecCollectionSchema({
      name: 'recall_no_cache_benchmark_dense',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: options.embeddingDimensions,
        indexParams: {
          indexType: ZVecIndexType.HNSW,
          metricType: ZVecMetricType.COSINE,
          m: 16,
          efConstruction: 100,
        },
      },
      fields: [
        { name: 'checksum', dataType: ZVecDataType.STRING },
        {
          name: 'embeddingInputChecksum',
          dataType: ZVecDataType.STRING,
          indexParams: { indexType: ZVecIndexType.INVERT },
        },
        {
          name: 'embeddingProfileId',
          dataType: ZVecDataType.STRING,
          indexParams: { indexType: ZVecIndexType.INVERT },
        },
        {
          name: 'physicalSessionProjectionId',
          dataType: ZVecDataType.STRING,
          indexParams: { indexType: ZVecIndexType.INVERT },
        },
        { name: 'logicalSessionId', dataType: ZVecDataType.STRING },
        { name: 'entryId', dataType: ZVecDataType.STRING },
        {
          name: 'projectIdentityDigest',
          dataType: ZVecDataType.STRING,
          indexParams: { indexType: ZVecIndexType.INVERT },
        },
      ],
    }),
  );

  return {
    lexicalSource,
    dense,
    insertEntryAnchors(anchors) {
      return insertDocuments(lexicalSource, anchors.map(serializeEntryAnchor));
    },
    insertLexicalEvidence(occurrences) {
      return insertDocuments(lexicalSource, occurrences.map(serializeEvidenceOccurrence));
    },
    insertDenseEvidence(occurrences) {
      return insertDocuments(dense, occurrences.map(serializeDenseOccurrence));
    },
    fetchDenseEvidence(occurrences) {
      const recovered = new Map<string, StoredRecallBenchmarkDenseOccurrence>();
      for (let start = 0; start < occurrences.length; start += RECALL_BENCHMARK_WRITE_BATCH_SIZE) {
        const ids = occurrences
          .slice(start, start + RECALL_BENCHMARK_WRITE_BATCH_SIZE)
          .map(({ occurrenceId }) => occurrenceId);
        const documents = dense.fetchSync({
          ids,
          outputFields: [...denseOutputFields],
          includeVector: true,
        });
        for (const document of Object.values(documents)) {
          recovered.set(document.id, {
            occurrenceId: document.id,
            checksum: readRequiredStringField(document, 'checksum'),
            embeddingInputChecksum: readRequiredStringField(document, 'embeddingInputChecksum'),
            embeddingProfileId: readRequiredStringField(document, 'embeddingProfileId'),
            embedding: readDenseVector(document),
          });
        }
      }
      return recovered;
    },
    async optimize() {
      const lexicalStartedAt = performance.now();
      await lexicalSource.optimize();
      const lexicalSourceMilliseconds = performance.now() - lexicalStartedAt;
      const denseStartedAt = performance.now();
      if (dense.stats.docCount > 0) {
        await dense.optimize();
      }
      return {
        lexicalSourceMilliseconds,
        denseMilliseconds: performance.now() - denseStartedAt,
      };
    },
    validate(validationOptions) {
      const evidenceCanary = validationOptions.evidenceCanary;
      const denseCanary = validationOptions.denseCanary;
      const anchorCanary = validationOptions.anchorCanaryId
        ? fetchRequiredDocument(lexicalSource, validationOptions.anchorCanaryId, ['rowKind'], false)
        : null;
      const exactEvidence = evidenceCanary
        ? fetchRequiredDocument(
            lexicalSource,
            evidenceCanary.occurrenceId,
            ['rowKind', 'checksum'],
            false,
          )
        : null;
      const ftsTerm = evidenceCanary ? selectFtsCanaryTerm(evidenceCanary.chunk.content) : null;
      const ftsMatches = ftsTerm
        ? lexicalSource.querySync({
            fieldName: 'content',
            fts: { matchString: ftsTerm },
            filter: "rowKind = 'evidence'",
            topk: 5,
            outputFields: ['rowKind'],
            includeVector: false,
            params: { indexType: ZVecIndexType.FTS, defaultOperator: 'AND' },
          })
        : [];
      const denseNearest = denseCanary
        ? dense.querySync({
            fieldName: 'embedding',
            vector: denseCanary.embedding,
            topk: 1,
            outputFields: ['checksum'],
            includeVector: false,
            params: { indexType: ZVecIndexType.HNSW, ef: 64 },
          })
        : [];
      const denseMembership = denseCanary
        ? fetchRequiredDocument(
            lexicalSource,
            denseCanary.occurrence.occurrenceId,
            ['checksum'],
            false,
          )
        : null;
      return {
        lexicalSourceRows: lexicalSource.stats.docCount,
        denseRows: dense.stats.docCount,
        exactAnchorCanary:
          anchorCanary?.fields.rowKind === 'entry_anchor' &&
          lexicalSource.stats.docCount === validationOptions.expectedLexicalSourceRows,
        exactEvidenceCanary:
          exactEvidence?.fields.rowKind === 'evidence' &&
          exactEvidence.fields.checksum === evidenceCanary?.chunk.checksum,
        lexicalFtsCanary: ftsMatches.length > 0,
        denseNearestNeighborCanary:
          denseCanary === null ||
          (denseNearest[0]?.id === denseCanary.occurrence.occurrenceId &&
            dense.stats.docCount === validationOptions.expectedDenseRows),
        denseMembershipCanary:
          denseCanary === null ||
          denseMembership?.fields.checksum === denseCanary.occurrence.chunk.checksum,
      };
    },
    close() {
      lexicalSource.closeSync();
      dense.closeSync();
    },
  };
}

/** Reopens a completed scratch generation's dense store for exact vector transfer. */
export function openRecallBenchmarkDenseTransferSource(
  generationDirectory: string,
): Pick<RecallBenchmarkSplitStores, 'dense' | 'fetchDenseEvidence' | 'close'> {
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  return {
    dense,
    fetchDenseEvidence(occurrences) {
      const recovered = new Map<string, StoredRecallBenchmarkDenseOccurrence>();
      for (let start = 0; start < occurrences.length; start += RECALL_BENCHMARK_WRITE_BATCH_SIZE) {
        const ids = occurrences
          .slice(start, start + RECALL_BENCHMARK_WRITE_BATCH_SIZE)
          .map(({ occurrenceId }) => occurrenceId);
        const documents = dense.fetchSync({
          ids,
          outputFields: [...denseOutputFields],
          includeVector: true,
        });
        for (const document of Object.values(documents)) {
          recovered.set(document.id, {
            occurrenceId: document.id,
            checksum: readRequiredStringField(document, 'checksum'),
            embeddingInputChecksum: readRequiredStringField(document, 'embeddingInputChecksum'),
            embeddingProfileId: readRequiredStringField(document, 'embeddingProfileId'),
            embedding: readDenseVector(document),
          });
        }
      }
      return recovered;
    },
    close() {
      dense.closeSync();
    },
  };
}

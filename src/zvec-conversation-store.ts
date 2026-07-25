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
  type ZVecDoc,
  type ZVecFieldSchema,
  type ZVecFtsIndexParams,
  type ZVecFtsQuery,
  type ZVecGroupResult,
  type ZVecInvertIndexParams,
  type ZVecVector,
} from '@zvec/zvec';

import { RecallProjectIdentitySource } from './enums.js';
import type {
  RecallDenseCandidate,
  RecallFullTextCandidate,
} from './fuse-recall-search-candidates.js';
import {
  isCanonicalRepositoryIdentity,
  parseProjectIdentity,
  type ProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import type {
  PiSessionEntryId,
  PiSessionId,
  SessionConversationChunk,
} from './session-conversation-index.js';

/** Version of the scalar/vector schema persisted in the zvec collection. */
export const ZVEC_CONVERSATION_SCHEMA_VERSION = 7;

/** Version of ordinary and case-preserving full text search (FTS) fields in zvec. */
export const ZVEC_FTS_CONFIGURATION_VERSION = 2;

/** Pinned HNSW graph degree used to make index geometry reproducible. */
export const ZVEC_HNSW_M = 50;

/** Pinned HNSW construction candidate count used to make index geometry reproducible. */
export const ZVEC_HNSW_EF_CONSTRUCTION = 500;

/** Pinned HNSW query candidate count used by dense conversation search. */
export const ZVEC_HNSW_EF_SEARCH = 300;

/** A dense-searchable recall document paired with its local embedding. */
export interface EmbeddedSessionConversationChunk extends SessionConversationChunk {
  isDenseSearchable: true;
  embedding: number[];
}

/** A lexical-only recall document that must never carry an embedding. */
export interface LexicalSessionConversationChunk extends SessionConversationChunk {
  isDenseSearchable: false;
  embedding?: never;
}

/** One recall document ready for dense-and-lexical or lexical-only persistence. */
export type IndexedSessionConversationChunk =
  | EmbeddedSessionConversationChunk
  | LexicalSessionConversationChunk;

/** The narrow persistence contract required by incremental recall evidence indexing. */
export interface ConversationChunkStore {
  upsertChunks(chunks: IndexedSessionConversationChunk[]): void;
  deleteChunks(ids: string[]): void;
}

/** Durable conversation operations plus zvec evolution and bounded-query capabilities. */
export interface ZvecConversationStore extends ConversationChunkStore {
  searchDenseCandidates(
    embedding: number[],
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallDenseCandidate[];
  searchLexicalCandidates(
    query: string,
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallFullTextCandidate[];
  searchIdentifierCandidates(
    query: string,
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallFullTextCandidate[];
  fetchConversationChunks(this: void, ids: string[]): Map<string, SessionConversationChunk>;
  fetchVectors(ids: string[]): Map<string, number[]>;
  groupDenseCandidates(
    embedding: number[],
    groupByFieldName: string,
    groupCount: number,
    topkPerGroup: number,
  ): ZVecGroupResult[];
  addColumn(fieldSchema: ZVecFieldSchema, expression?: string): void;
  alterColumn(columnName: string, fieldSchema: ZVecFieldSchema): void;
  createIndex(fieldName: string, indexParams: ZVecInvertIndexParams | ZVecFtsIndexParams): void;
  optimize(): Promise<void>;
  close(): void;
  count(): number;
}

const RECALL_FIELD_SCHEMAS: ZVecFieldSchema[] = [
  { name: 'schemaVersion', dataType: ZVecDataType.INT32 },
  { name: 'documentKind', dataType: ZVecDataType.STRING },
  { name: 'summaryKind', dataType: ZVecDataType.STRING },
  { name: 'evidenceKind', dataType: ZVecDataType.STRING },
  { name: 'evidencePart', dataType: ZVecDataType.STRING },
  { name: 'isDenseSearchable', dataType: ZVecDataType.BOOL },
  { name: 'checksum', dataType: ZVecDataType.STRING },
  { name: 'sessionId', dataType: ZVecDataType.STRING },
  { name: 'sessionPath', dataType: ZVecDataType.STRING },
  { name: 'parentSessionPath', dataType: ZVecDataType.STRING },
  { name: 'cwd', dataType: ZVecDataType.STRING },
  { name: 'projectPath', dataType: ZVecDataType.STRING },
  { name: 'projectIdentity', dataType: ZVecDataType.STRING },
  { name: 'projectIdentityDigest', dataType: ZVecDataType.STRING },
  { name: 'projectIdentitySource', dataType: ZVecDataType.STRING },
  { name: 'sessionName', dataType: ZVecDataType.STRING },
  { name: 'entryId', dataType: ZVecDataType.STRING },
  { name: 'parentEntryId', dataType: ZVecDataType.STRING },
  { name: 'childEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'contributingEntryIds', dataType: ZVecDataType.ARRAY_STRING },
  { name: 'currentLeafId', dataType: ZVecDataType.STRING },
  { name: 'branchPathLeafIds', dataType: ZVecDataType.ARRAY_STRING },
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

const RECALL_OUTPUT_FIELDS = RECALL_FIELD_SCHEMAS.map((field) => field.name).filter(
  (name) => name !== 'identifierContent',
);

function serializeNullableEntryId(value: PiSessionEntryId | null): string {
  return value?.value ?? '';
}

function serializeConversationChunk(chunk: SessionConversationChunk): Record<string, unknown> {
  return {
    schemaVersion: chunk.schemaVersion,
    documentKind: chunk.documentKind,
    summaryKind: chunk.summaryKind ?? '',
    evidenceKind: chunk.evidenceKind,
    evidencePart: chunk.evidencePart,
    isDenseSearchable: chunk.isDenseSearchable,
    checksum: chunk.checksum,
    sessionId: chunk.sessionId.value,
    sessionPath: chunk.sessionPath,
    parentSessionPath: chunk.parentSessionPath ?? '',
    cwd: chunk.cwd,
    projectPath: chunk.projectPath,
    projectIdentity: chunk.projectAttribution?.projectIdentity ?? '',
    projectIdentityDigest: chunk.projectAttribution
      ? createHash('sha256').update(chunk.projectAttribution.projectIdentity).digest('hex')
      : '',
    projectIdentitySource: chunk.projectAttribution?.identitySource ?? '',
    sessionName: chunk.sessionName,
    entryId: chunk.entryId.value,
    parentEntryId: serializeNullableEntryId(chunk.parentEntryId),
    childEntryIds: chunk.childEntryIds.map((id) => id.value),
    contributingEntryIds: chunk.contributingEntryIds.map((id) => id.value),
    currentLeafId: serializeNullableEntryId(chunk.currentLeafId),
    branchPathLeafIds: chunk.branchPathLeafIds.map((id) => id.value),
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
  };
}

function readStringField(fields: Record<string, unknown>, name: string): string {
  const value = fields[name];
  if (typeof value !== 'string') {
    throw new Error(`Recall zvec field ${name} invalid: expected string`);
  }
  return value;
}

function readNumberField(fields: Record<string, unknown>, name: string): number {
  const value = fields[name];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`Recall zvec field ${name} invalid: expected finite number`);
  }
  return value;
}

function readBooleanField(fields: Record<string, unknown>, name: string): boolean {
  const value = fields[name];
  if (typeof value !== 'boolean') {
    throw new Error(`Recall zvec field ${name} invalid: expected boolean`);
  }
  return value;
}

function readStringArrayField(fields: Record<string, unknown>, name: string): string[] {
  const value = fields[name];
  if (!Array.isArray(value) || !value.every((item) => typeof item === 'string')) {
    throw new Error(`Recall zvec field ${name} invalid: expected string array`);
  }
  return value.map(String);
}

function parseNullableString(value: string): string | null {
  return value || null;
}

function parseNullableEntryId(value: string): PiSessionEntryId | null {
  return value ? { value } : null;
}

function parseRecallProjectIdentitySource(value: string): RecallProjectIdentitySource {
  if (value === 'git_origin') {
    return RecallProjectIdentitySource.GIT_ORIGIN;
  }
  if (value === 'git_common_directory') {
    return RecallProjectIdentitySource.GIT_COMMON_DIRECTORY;
  }
  if (value === 'non_git_session_origin') {
    return RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN;
  }
  if (value === 'configured_project_lineage') {
    return RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE;
  }
  throw new Error(`Recall zvec projectIdentitySource invalid: ${value}`);
}

function parseRecallDocumentKind(value: string): SessionConversationChunk['documentKind'] {
  if (
    value === 'conversation' ||
    value === 'turn_context' ||
    value === 'summary' ||
    value === 'tool'
  ) {
    return value;
  }
  throw new Error(`Recall zvec documentKind invalid: ${value}`);
}

function parseRecallSummaryKind(value: string): SessionConversationChunk['summaryKind'] {
  if (value === '') {
    return null;
  }
  if (value === 'compaction' || value === 'branch') {
    return value;
  }
  throw new Error(`Recall zvec summaryKind invalid: ${value}`);
}

function parseRecallEvidenceKind(value: string): SessionConversationChunk['evidenceKind'] {
  if (
    value === 'conversation' ||
    value === 'turn_context' ||
    value === 'compaction_summary' ||
    value === 'branch_summary' ||
    value === 'tool_call' ||
    value === 'tool_result' ||
    value === 'bash_execution'
  ) {
    return value;
  }
  throw new Error(`Recall zvec evidenceKind invalid: ${value}`);
}

function parseRecallEvidencePart(value: string): SessionConversationChunk['evidencePart'] {
  if (
    value === 'content' ||
    value === 'name' ||
    value === 'arguments' ||
    value === 'result' ||
    value === 'command' ||
    value === 'output'
  ) {
    return value;
  }
  throw new Error(`Recall zvec evidencePart invalid: ${value}`);
}

function parseRecallConversationRole(value: string): SessionConversationChunk['role'] {
  if (
    value === 'user' ||
    value === 'assistant' ||
    value === 'turn' ||
    value === 'summary' ||
    value === 'custom' ||
    value === 'tool'
  ) {
    return value;
  }
  throw new Error(`Recall zvec role invalid: ${value}`);
}

function parseNullableSourceBlock(value: number): number | null {
  return value < 0 ? null : value;
}

function parseNullableToolError(value: number): boolean | null {
  if (value === -1) {
    return null;
  }
  if (value === 0) {
    return false;
  }
  if (value === 1) {
    return true;
  }
  throw new Error(`Recall zvec toolError invalid: ${value}`);
}

function parseProjectAttribution(
  projectIdentity: string,
  identitySource: string,
): ResolvedProjectIdentity | null {
  if (!projectIdentity && !identitySource) {
    return null;
  }
  if (!projectIdentity || !identitySource) {
    throw new Error(
      'Recall zvec project attribution invalid: identity and source must both be present',
    );
  }
  const parsedIdentity = parseProjectIdentity(projectIdentity);
  const parsedSource = parseRecallProjectIdentitySource(identitySource);
  if (parsedSource === RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN) {
    return { projectIdentity: parsedIdentity, identitySource: parsedSource };
  }
  if (!isCanonicalRepositoryIdentity(parsedIdentity)) {
    throw new Error(
      `Recall zvec project attribution invalid: ${parsedSource} requires repository identity`,
    );
  }
  return { projectIdentity: parsedIdentity, identitySource: parsedSource };
}

function deserializeConversationChunk(doc: ZVecDoc): SessionConversationChunk {
  const fields: Record<string, unknown> = doc.fields;
  return {
    schemaVersion: readNumberField(fields, 'schemaVersion'),
    documentKind: parseRecallDocumentKind(readStringField(fields, 'documentKind')),
    summaryKind: parseRecallSummaryKind(readStringField(fields, 'summaryKind')),
    evidenceKind: parseRecallEvidenceKind(readStringField(fields, 'evidenceKind')),
    evidencePart: parseRecallEvidencePart(readStringField(fields, 'evidencePart')),
    isDenseSearchable: readBooleanField(fields, 'isDenseSearchable'),
    id: doc.id,
    checksum: readStringField(fields, 'checksum'),
    sessionId: { value: readStringField(fields, 'sessionId') } satisfies PiSessionId,
    sessionPath: readStringField(fields, 'sessionPath'),
    parentSessionPath: parseNullableString(readStringField(fields, 'parentSessionPath')),
    cwd: readStringField(fields, 'cwd'),
    projectPath: readStringField(fields, 'projectPath'),
    projectAttribution: parseProjectAttribution(
      readStringField(fields, 'projectIdentity'),
      readStringField(fields, 'projectIdentitySource'),
    ),
    sessionName: readStringField(fields, 'sessionName'),
    entryId: { value: readStringField(fields, 'entryId') } satisfies PiSessionEntryId,
    parentEntryId: parseNullableEntryId(readStringField(fields, 'parentEntryId')),
    childEntryIds: readStringArrayField(fields, 'childEntryIds').map((value) => ({ value })),
    contributingEntryIds: readStringArrayField(fields, 'contributingEntryIds').map((value) => ({
      value,
    })),
    currentLeafId: parseNullableEntryId(readStringField(fields, 'currentLeafId')),
    branchPathLeafIds: readStringArrayField(fields, 'branchPathLeafIds').map((value) => ({
      value,
    })),
    isOnActiveBranch: readBooleanField(fields, 'isOnActiveBranch'),
    isVisibleInActiveContext: readBooleanField(fields, 'isVisibleInActiveContext'),
    compactedByEntryIds: readStringArrayField(fields, 'compactedByEntryIds').map((value) => ({
      value,
    })),
    compactionFirstKeptEntryId: parseNullableEntryId(
      readStringField(fields, 'compactionFirstKeptEntryId'),
    ),
    branchSummaryFromEntryId: parseNullableEntryId(
      readStringField(fields, 'branchSummaryFromEntryId'),
    ),
    role: parseRecallConversationRole(readStringField(fields, 'role')),
    timestamp: readStringField(fields, 'timestamp'),
    sourceLineStart: readNumberField(fields, 'sourceLineStart'),
    sourceLineEnd: readNumberField(fields, 'sourceLineEnd'),
    sourceBlockStart: parseNullableSourceBlock(readNumberField(fields, 'sourceBlockStart')),
    sourceBlockEnd: parseNullableSourceBlock(readNumberField(fields, 'sourceBlockEnd')),
    characterStart: readNumberField(fields, 'characterStart'),
    characterEnd: readNumberField(fields, 'characterEnd'),
    tokenStart: readNumberField(fields, 'tokenStart'),
    tokenEnd: readNumberField(fields, 'tokenEnd'),
    tokenCount: readNumberField(fields, 'tokenCount'),
    overlapTokenCount: readNumberField(fields, 'overlapTokenCount'),
    textRunId: readStringField(fields, 'textRunId'),
    textRunIndex: readNumberField(fields, 'textRunIndex'),
    chunkIndex: readNumberField(fields, 'chunkIndex'),
    chunkCount: readNumberField(fields, 'chunkCount'),
    siblingIds: readStringArrayField(fields, 'siblingIds'),
    previousSiblingId: parseNullableString(readStringField(fields, 'previousSiblingId')),
    nextSiblingId: parseNullableString(readStringField(fields, 'nextSiblingId')),
    toolCallId: parseNullableString(readStringField(fields, 'toolCallId')),
    toolName: parseNullableString(readStringField(fields, 'toolName')),
    toolCallEntryId: parseNullableEntryId(readStringField(fields, 'toolCallEntryId')),
    toolResultEntryId: parseNullableEntryId(readStringField(fields, 'toolResultEntryId')),
    toolError: parseNullableToolError(readNumberField(fields, 'toolError')),
    content: readStringField(fields, 'content'),
  };
}

function convertDenseVector(id: string, vector?: ZVecVector): number[] {
  if (Array.isArray(vector) || vector instanceof Float32Array || vector instanceof Int8Array) {
    return Array.from(vector, Number);
  }
  throw new Error(`Recall zvec vector invalid for document ${id}: expected dense vector`);
}

function assertRecallCandidateLimit(limit: number, channelName: string): void {
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    throw new Error(
      `Recall candidate limit invalid (${channelName}): expected an integer from 1 to 200`,
    );
  }
}

function createRecallProjectIdentityFilter(projectIdentity?: ProjectIdentity): string | undefined {
  const identityDigest = projectIdentity
    ? createHash('sha256').update(projectIdentity).digest('hex')
    : undefined;
  return identityDigest ? `projectIdentityDigest = '${identityDigest}'` : undefined;
}

function createRecallFullTextQuery(query: string): ZVecFtsQuery {
  const openingQuote = query.indexOf('"');
  const closingQuote = query.indexOf('"', openingQuote + 1);
  const hasOneQuotePair =
    openingQuote >= 0 && closingQuote > openingQuote && query.indexOf('"', closingQuote + 1) < 0;
  if (hasOneQuotePair) {
    const quotedPhrase = query.slice(openingQuote + 1, closingQuote).trim();
    if (quotedPhrase && !quotedPhrase.includes('\\')) {
      return { queryString: `"${quotedPhrase}"` };
    }
  }
  return { matchString: query };
}

/** Opens the durable zvec collection for dense-and-lexical and lexical-only recall evidence. */
export function openZvecConversationStore(config: {
  databasePath: string;
  dimensions: number;
  createIfMissing?: boolean;
  readOnly?: boolean;
}): ZvecConversationStore {
  const databaseExists = existsSync(config.databasePath);
  if (!databaseExists && config.createIfMissing === false) {
    throw new Error(
      `Recall zvec collection missing at ${config.databasePath}; reindex with /pi-session-recall-index --rebuild`,
    );
  }
  if (!databaseExists) {
    mkdirSync(dirname(config.databasePath), { recursive: true });
  }
  const collection: ZVecCollection = databaseExists
    ? config.readOnly
      ? ZVecOpen(config.databasePath, { readOnly: true })
      : ZVecOpen(config.databasePath)
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
              m: ZVEC_HNSW_M,
              efConstruction: ZVEC_HNSW_EF_CONSTRUCTION,
            },
          },
          fields: RECALL_FIELD_SCHEMAS,
        }),
      );
  const storedDimensions = collection.schema.vector('embedding').dimension;
  if (storedDimensions !== config.dimensions) {
    collection.closeSync();
    throw new Error(
      `Recall zvec dimension mismatch: collection uses ${storedDimensions}, configured model uses ${config.dimensions}; reindex with /pi-session-recall-index --rebuild`,
    );
  }

  const searchFullTextCandidates = (
    fieldName: string,
    query: string,
    limit: number,
    channelName: string,
    defaultOperator: 'AND' | 'OR',
    projectIdentity?: ProjectIdentity,
  ): RecallFullTextCandidate[] => {
    assertRecallCandidateLimit(limit, channelName);
    const projectFilter = createRecallProjectIdentityFilter(projectIdentity);
    return collection
      .querySync({
        fieldName,
        fts: createRecallFullTextQuery(query),
        topk: limit,
        outputFields: RECALL_OUTPUT_FIELDS,
        includeVector: false,
        ...(projectFilter ? { filter: projectFilter } : {}),
        params: { indexType: ZVecIndexType.FTS, defaultOperator },
      })
      .map((doc) => ({
        ...deserializeConversationChunk(doc),
        fullTextScore: doc.score,
      }));
  };

  return {
    upsertChunks(chunks) {
      if (chunks.length === 0) {
        return;
      }
      collection.upsertSync(
        chunks.map((indexedChunk) => {
          if (indexedChunk.isDenseSearchable) {
            const { embedding, ...chunk } = indexedChunk;
            return {
              id: chunk.id,
              vectors: { embedding },
              fields: serializeConversationChunk(chunk),
            };
          }
          return {
            id: indexedChunk.id,
            vectors: { embedding: new Array<number>(storedDimensions).fill(0) },
            fields: serializeConversationChunk(indexedChunk),
          };
        }),
      );
    },
    deleteChunks(ids) {
      if (ids.length > 0) {
        collection.deleteSync(ids);
      }
    },
    searchDenseCandidates(embedding, limit, projectIdentity) {
      assertRecallCandidateLimit(limit, 'dense');
      const projectFilter = createRecallProjectIdentityFilter(projectIdentity);
      return collection
        .querySync({
          fieldName: 'embedding',
          vector: embedding,
          topk: limit,
          outputFields: RECALL_OUTPUT_FIELDS,
          includeVector: false,
          filter: projectFilter
            ? `isDenseSearchable = true AND ${projectFilter}`
            : 'isDenseSearchable = true',
          params: { indexType: ZVecIndexType.HNSW, ef: ZVEC_HNSW_EF_SEARCH },
        })
        .map((doc) => ({
          ...deserializeConversationChunk(doc),
          cosineDistance: doc.score,
        }));
    },
    searchLexicalCandidates(query, limit, projectIdentity) {
      return searchFullTextCandidates('content', query, limit, 'lexical', 'OR', projectIdentity);
    },
    searchIdentifierCandidates(query, limit, projectIdentity) {
      return searchFullTextCandidates(
        'identifierContent',
        query,
        limit,
        'identifier',
        'AND',
        projectIdentity,
      );
    },
    fetchConversationChunks(ids) {
      if (ids.length === 0) {
        return new Map();
      }
      const docs = collection.fetchSync({
        ids,
        outputFields: RECALL_OUTPUT_FIELDS,
        includeVector: false,
      });
      return new Map(Object.values(docs).map((doc) => [doc.id, deserializeConversationChunk(doc)]));
    },
    fetchVectors(ids) {
      if (ids.length === 0) {
        return new Map();
      }
      const docs = collection.fetchSync({ ids, outputFields: [], includeVector: true });
      return new Map(
        Object.values(docs).map((doc) => [
          doc.id,
          convertDenseVector(doc.id, doc.vectors.embedding),
        ]),
      );
    },
    groupDenseCandidates(embedding, groupByFieldName, groupCount, topkPerGroup) {
      if (
        !Number.isInteger(groupCount) ||
        groupCount < 1 ||
        groupCount > 200 ||
        !Number.isInteger(topkPerGroup) ||
        topkPerGroup < 1 ||
        topkPerGroup > 20
      ) {
        throw new Error(
          'Recall dense grouping limits invalid: expected 1..200 groups and 1..20 candidates per group',
        );
      }
      return collection.groupByQuerySync({
        fieldName: 'embedding',
        vector: embedding,
        groupByFieldName,
        groupCount,
        topkPerGroup,
        includeVector: false,
        outputFields: RECALL_OUTPUT_FIELDS,
        params: { indexType: ZVecIndexType.HNSW, ef: ZVEC_HNSW_EF_SEARCH },
      });
    },
    addColumn(fieldSchema, expression) {
      collection.addColumnSync({
        fieldSchema,
        ...(expression === undefined ? {} : { expression }),
      });
    },
    alterColumn(columnName, fieldSchema) {
      collection.alterColumnSync({ columnName, fieldSchema });
    },
    createIndex(fieldName, indexParams) {
      collection.createIndexSync({ fieldName, indexParams });
    },
    async optimize() {
      if (collection.stats.docCount > 0) {
        await collection.optimize();
      }
    },
    close() {
      collection.closeSync();
    },
    count() {
      return collection.stats.docCount;
    },
  };
}

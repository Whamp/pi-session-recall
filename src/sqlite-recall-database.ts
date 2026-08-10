import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { DatabaseSync, type SQLOutputValue } from 'node:sqlite';

import * as sqliteVec from 'sqlite-vec';
import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallProjectIdentitySource } from './enums.js';
import type { InvocationRecord } from './createSessionInvocationRecords.js';
import type { RecallDenseCandidate } from './fuse-recall-search-candidates.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  isCanonicalRepositoryIdentity,
  parseProjectIdentity,
  type ProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

const SQLITE_RECALL_SCHEMA_VERSION = 2;
const SQLITE_RECALL_STORAGE_LAYOUT = 'unified-sqlite-vec';
/** Production width of every FP32 vector stored in the SQLite Recall database. */
export const SQLITE_RECALL_EMBEDDING_DIMENSIONS = 1_024;
const SQLITE_RECALL_VECTOR_ENCODING = 'float32';
const SQLITE_RECALL_DISTANCE_METRIC = 'cosine';
const SQLITE_RECALL_PROJECT_BUCKET_COUNT = 16;
const SQLITE_RECALL_VEC_PACKAGE_VERSION = '0.1.9';
const SQLITE_RECALL_VEC_RUNTIME_VERSION = `v${SQLITE_RECALL_VEC_PACKAGE_VERSION}`;
const SQLITE_RECALL_BUSY_TIMEOUT_MILLISECONDS = 5_000;

/** Static schema identity stored in a version 8 index manifest, excluding connection runtime state. */
export const SQLITE_RECALL_DATABASE_MANIFEST_IDENTITY = Object.freeze({
  schemaVersion: SQLITE_RECALL_SCHEMA_VERSION,
  storageLayout: SQLITE_RECALL_STORAGE_LAYOUT,
  sqliteVecVersion: SQLITE_RECALL_VEC_PACKAGE_VERSION,
  embedding: Object.freeze({
    dimensions: SQLITE_RECALL_EMBEDDING_DIMENSIONS,
    encoding: 'fp32' as const,
    distanceMetric: SQLITE_RECALL_DISTANCE_METRIC,
  }),
  routing: Object.freeze({
    global: 'unpartitioned' as const,
    project: Object.freeze({
      bucketCount: SQLITE_RECALL_PROJECT_BUCKET_COUNT,
      bucketFunction: 'project-key-modulo-16' as const,
      exactProjectKey: true as const,
    }),
  }),
  fullTextSearch: Object.freeze({ engine: 'fts5' as const, tokenizer: 'unicode61' as const }),
});

/** Compatibility-bearing SQLite Recall database identity written to the index manifest. */
export type SqliteRecallDatabaseManifestIdentity = typeof SQLITE_RECALL_DATABASE_MANIFEST_IDENTITY;

type SqliteRow = Record<string, SQLOutputValue>;

const PI_SESSION_ID_SCHEMA = Type.Object({ value: Type.String() }, { additionalProperties: false });
const NULLABLE_PI_SESSION_ID_SCHEMA = Type.Union([PI_SESSION_ID_SCHEMA, Type.Null()]);
const PROJECT_IDENTITY_SOURCE_SCHEMA = Type.Union([
  Type.Literal('git_origin'),
  Type.Literal('git_common_directory'),
  Type.Literal('non_git_session_origin'),
  Type.Literal('configured_project_lineage'),
]);
const DENSE_DOCUMENT_METADATA_SCHEMA = Type.Object(
  {
    schemaVersion: Type.Integer(),
    documentKind: Type.Union([
      Type.Literal('conversation'),
      Type.Literal('turn_context'),
      Type.Literal('summary'),
      Type.Literal('tool'),
    ]),
    summaryKind: Type.Union([Type.Literal('compaction'), Type.Literal('branch'), Type.Null()]),
    evidenceKind: Type.Union([
      Type.Literal('conversation'),
      Type.Literal('turn_context'),
      Type.Literal('compaction_summary'),
      Type.Literal('branch_summary'),
      Type.Literal('tool_call'),
      Type.Literal('tool_result'),
      Type.Literal('bash_execution'),
    ]),
    evidencePart: Type.Union([
      Type.Literal('content'),
      Type.Literal('name'),
      Type.Literal('arguments'),
      Type.Literal('result'),
      Type.Literal('command'),
      Type.Literal('output'),
    ]),
    isDenseSearchable: Type.Boolean(),
    id: Type.String(),
    checksum: Type.String(),
    sessionId: PI_SESSION_ID_SCHEMA,
    sessionPath: Type.String(),
    parentSessionPath: Type.Union([Type.String(), Type.Null()]),
    cwd: Type.String(),
    projectPath: Type.String(),
    projectAttribution: Type.Union([
      Type.Object(
        {
          projectIdentity: Type.String(),
          identitySource: PROJECT_IDENTITY_SOURCE_SCHEMA,
        },
        { additionalProperties: false },
      ),
      Type.Null(),
    ]),
    sessionName: Type.String(),
    entryId: PI_SESSION_ID_SCHEMA,
    parentEntryId: NULLABLE_PI_SESSION_ID_SCHEMA,
    childEntryIds: Type.Array(PI_SESSION_ID_SCHEMA),
    contributingEntryIds: Type.Array(PI_SESSION_ID_SCHEMA),
    currentLeafId: NULLABLE_PI_SESSION_ID_SCHEMA,
    branchPathLeafIds: Type.Array(PI_SESSION_ID_SCHEMA),
    isOnActiveBranch: Type.Boolean(),
    isVisibleInActiveContext: Type.Boolean(),
    compactedByEntryIds: Type.Array(PI_SESSION_ID_SCHEMA),
    compactionFirstKeptEntryId: NULLABLE_PI_SESSION_ID_SCHEMA,
    branchSummaryFromEntryId: NULLABLE_PI_SESSION_ID_SCHEMA,
    role: Type.Union([
      Type.Literal('user'),
      Type.Literal('assistant'),
      Type.Literal('turn'),
      Type.Literal('summary'),
      Type.Literal('custom'),
      Type.Literal('tool'),
    ]),
    timestamp: Type.String(),
    sourceLineStart: Type.Integer(),
    sourceLineEnd: Type.Integer(),
    sourceBlockStart: Type.Union([Type.Integer(), Type.Null()]),
    sourceBlockEnd: Type.Union([Type.Integer(), Type.Null()]),
    characterStart: Type.Integer(),
    characterEnd: Type.Integer(),
    tokenStart: Type.Integer(),
    tokenEnd: Type.Integer(),
    tokenCount: Type.Integer(),
    overlapTokenCount: Type.Integer(),
    textRunId: Type.String(),
    textRunIndex: Type.Integer(),
    chunkIndex: Type.Integer(),
    chunkCount: Type.Integer(),
    siblingIds: Type.Array(Type.String()),
    previousSiblingId: Type.Union([Type.String(), Type.Null()]),
    nextSiblingId: Type.Union([Type.String(), Type.Null()]),
    toolCallId: Type.Union([Type.String(), Type.Null()]),
    toolName: Type.Union([Type.String(), Type.Null()]),
    toolCallEntryId: NULLABLE_PI_SESSION_ID_SCHEMA,
    toolResultEntryId: NULLABLE_PI_SESSION_ID_SCHEMA,
    toolError: Type.Union([Type.Boolean(), Type.Null()]),
    content: Type.String(),
  },
  { additionalProperties: false },
);

/** Options controlling write access and optional first-open legacy state import. */
export interface OpenSqliteRecallDatabaseOptions {
  legacyStatePath?: string;
  readOnly?: boolean;
}

/** Persisted storage identity plus the SQLite runtime serving this connection. */
export interface SqliteRecallDatabaseIdentity {
  schemaVersion: number;
  storageLayout: string;
  embeddingDimensions: number;
  vectorEncoding: string;
  distanceMetric: string;
  projectBucketCount: number;
  sqliteVecVersion: string;
  sqliteVersion: string;
  journalMode: string;
  queryOnly: boolean;
}

/** Incremental state retained for one physical session file. */
export interface SqliteRecallPhysicalSessionState {
  size: number;
  mtimeMs: number;
  documentIds: string[];
  denseDocumentIds: string[];
}

/** One compact Invocation search match with its SQLite FTS rank. */
export interface SqliteRecallInvocationSearchResult extends InvocationRecord {
  rank: number;
}

/** Row counts for every physical-session, dense, vector, and Invocation projection. */
export interface SqliteRecallDatabaseCounts {
  physicalSessions: number;
  sessionDocuments: number;
  invocations: number;
  denseDocuments: number;
  denseGlobalVectors: number;
  denseProjectVectors: number;
  denseProjects: number;
}

/** Parity diagnostics for dense metadata and its two required vec0 copies. */
export interface SqliteRecallVectorParityDiagnostics {
  denseDocuments: number;
  globalVectors: number;
  projectVectors: number;
  denseDocumentsMissingGlobalVector: number;
  globalVectorsMissingDenseDocument: number;
  denseDocumentsMissingProjectVector: number;
  projectVectorsMissingDenseDocument: number;
  projectMetadataMismatches: number;
  vectorValueMismatches: number;
  healthy: boolean;
}

/** Read-only SQLite, foreign-key, Invocation FTS, and vector-parity diagnostics. */
export interface SqliteRecallIntegrityDiagnostics {
  sqliteIntegrity: string[];
  foreignKeyViolations: number;
  invocationFtsIntegrityChecked: boolean;
  invocationFtsDocuments: number;
  invocationsMissingFts: number;
  ftsDocumentsMissingInvocation: number;
  vectorParity: SqliteRecallVectorParityDiagnostics;
  healthy: boolean;
}

/** Complete replacement of one physical session and every derived Recall projection it owns. */
export interface SqliteRecallPhysicalSessionReplacement {
  sessionPath: string;
  size: number;
  mtimeMs: number;
  documentIds: readonly string[];
  denseDocuments: readonly SessionConversationChunk[];
  denseEmbeddings: ReadonlyMap<string, readonly number[]>;
  invocations: readonly InvocationRecord[];
}

/** Deep persistence interface for one unified SQLite Recall database. */
export interface SqliteRecallDatabase {
  readonly identity: Readonly<SqliteRecallDatabaseIdentity>;
  readPhysicalSessionState(sessionPath: string): SqliteRecallPhysicalSessionState | null;
  listPhysicalSessionPaths(): string[];
  requiresInvocationBackfill(sessionPath: string): boolean;
  fetchDenseDocuments(ids: readonly string[]): Map<string, SessionConversationChunk>;
  fetchDenseVectors(ids: readonly string[]): Map<string, number[]>;
  searchDenseCandidates(
    embedding: readonly number[],
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallDenseCandidate[];
  searchInvocations(
    query: string,
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): SqliteRecallInvocationSearchResult[];
  readCounts(): SqliteRecallDatabaseCounts;
  checkIntegrity(): SqliteRecallIntegrityDiagnostics;
  replacePhysicalSession(replacement: SqliteRecallPhysicalSessionReplacement): void;
  deletePhysicalSession(sessionPath: string): boolean;
  close(): void;
}

interface PreparedDenseDocument {
  document: SessionConversationChunk;
  metadataJson: string;
  projectIdentity: string | null;
  vectorBlob: Uint8Array;
}

interface LegacyPhysicalSessionState extends SqliteRecallPhysicalSessionState {
  sessionPath: string;
}

function readRequiredString(row: SqliteRow | undefined, column: string): string {
  const value = row?.[column];
  if (typeof value !== 'string') {
    throw new Error(`SQLite Recall database row invalid: ${column} must be text`);
  }
  return value;
}

function readRequiredNumber(row: SqliteRow | undefined, column: string): number {
  const value = row?.[column];
  if (typeof value !== 'number') {
    throw new Error(`SQLite Recall database row invalid: ${column} must be numeric`);
  }
  return value;
}

function readRequiredInteger(row: SqliteRow | undefined, column: string): number {
  const value = row?.[column];
  const numberValue = typeof value === 'bigint' ? Number(value) : value;
  if (typeof numberValue !== 'number' || !Number.isSafeInteger(numberValue)) {
    throw new Error(`SQLite Recall database row invalid: ${column} must be a safe integer`);
  }
  return numberValue;
}

function convertSqliteInteger(value: number | bigint, subject: string): number {
  const converted = typeof value === 'bigint' ? Number(value) : value;
  if (!Number.isSafeInteger(converted)) {
    throw new Error(`SQLite Recall database ${subject} invalid: expected a safe integer`);
  }
  return converted;
}

function runSqliteRecallTransaction(database: DatabaseSync, operation: () => void): void {
  database.exec('BEGIN IMMEDIATE');
  try {
    operation();
    database.exec('COMMIT');
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function runSqliteRecallRead<T>(database: DatabaseSync, operation: () => T): T {
  if (database.isTransaction) {
    return operation();
  }
  database.exec('BEGIN');
  try {
    const result = operation();
    database.exec('COMMIT');
    return result;
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function loadPinnedSqliteVec(database: DatabaseSync): void {
  try {
    sqliteVec.load(database);
  } finally {
    database.enableLoadExtension(false);
  }
  const runtimeVersion = readRequiredString(
    database.prepare('SELECT vec_version() AS sqlite_vec_version').get(),
    'sqlite_vec_version',
  );
  if (runtimeVersion !== SQLITE_RECALL_VEC_RUNTIME_VERSION) {
    throw new Error(
      `SQLite Recall database sqlite-vec runtime incompatible: found ${runtimeVersion}, expected ${SQLITE_RECALL_VEC_RUNTIME_VERSION}`,
    );
  }
}

function configureSqliteRecallConnection(database: DatabaseSync, readOnly: boolean): void {
  database.exec(`
    PRAGMA foreign_keys = ON;
    PRAGMA busy_timeout = 5000;
    PRAGMA cache_size = -65536;
    PRAGMA mmap_size = 2147483648;
    PRAGMA temp_store = MEMORY;
  `);
  if (readOnly) {
    database.exec('PRAGMA query_only = ON');
    return;
  }
  database.exec('PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL');
}

function readLegacyPhysicalSessionStates(legacyStatePath: string): LegacyPhysicalSessionState[] {
  let decoded: unknown;
  try {
    decoded = JSON.parse(readFileSync(legacyStatePath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`SQLite Recall legacy state invalid at ${legacyStatePath}: ${message}`, {
      cause: error,
    });
  }
  if (
    !isUnknownRecord(decoded) ||
    decoded.version !== 3 ||
    decoded.importPolicyVersion !== SESSION_IMPORT_POLICY_VERSION ||
    !isUnknownRecord(decoded.sessions)
  ) {
    throw new Error(
      `SQLite Recall legacy state invalid at ${legacyStatePath}: incompatible state schema`,
    );
  }
  return Object.entries(decoded.sessions).map(([sessionPath, value]) => {
    if (
      !isUnknownRecord(value) ||
      typeof value.size !== 'number' ||
      !Number.isSafeInteger(value.size) ||
      value.size < 0 ||
      typeof value.mtimeMs !== 'number' ||
      !Number.isFinite(value.mtimeMs) ||
      value.mtimeMs < 0 ||
      !Array.isArray(value.chunks)
    ) {
      throw new Error(
        `SQLite Recall legacy state invalid at ${legacyStatePath}: invalid physical session ${sessionPath}`,
      );
    }
    const documentIds: string[] = [];
    for (const chunk of value.chunks) {
      if (!isUnknownRecord(chunk) || typeof chunk.id !== 'string' || !chunk.id) {
        throw new Error(
          `SQLite Recall legacy state invalid at ${legacyStatePath}: invalid document identity for ${sessionPath}`,
        );
      }
      documentIds.push(chunk.id);
    }
    return {
      sessionPath,
      size: value.size,
      mtimeMs: value.mtimeMs,
      documentIds,
      denseDocumentIds: [],
    };
  });
}

function importLegacyPhysicalSessionStates(
  database: DatabaseSync,
  states: readonly LegacyPhysicalSessionState[],
): void {
  const insertSession = database.prepare(`
    INSERT INTO physical_sessions(session_path, size, mtime_ms, invocations_indexed)
    VALUES (?, ?, ?, 0)
  `);
  const insertDocument = database.prepare(`
    INSERT INTO session_documents(session_path, document_id, is_dense)
    VALUES (?, ?, 0)
  `);
  for (const state of states) {
    insertSession.run(state.sessionPath, state.size, state.mtimeMs);
    for (const documentId of state.documentIds) {
      insertDocument.run(state.sessionPath, documentId);
    }
  }
}

function createSqliteRecallSchema(database: DatabaseSync): void {
  database.exec(`
    CREATE TABLE recall_database_identity (
      singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
      storage_layout TEXT NOT NULL,
      embedding_dimensions INTEGER NOT NULL,
      vector_encoding TEXT NOT NULL,
      distance_metric TEXT NOT NULL,
      project_bucket_count INTEGER NOT NULL,
      sqlite_vec_version TEXT NOT NULL
    ) STRICT;

    INSERT INTO recall_database_identity(
      singleton,
      storage_layout,
      embedding_dimensions,
      vector_encoding,
      distance_metric,
      project_bucket_count,
      sqlite_vec_version
    ) VALUES (
      1,
      'unified-sqlite-vec',
      1024,
      'float32',
      'cosine',
      16,
      'v0.1.9'
    );

    CREATE TABLE physical_sessions (
      session_path TEXT PRIMARY KEY,
      size INTEGER NOT NULL CHECK (size >= 0),
      mtime_ms REAL NOT NULL CHECK (mtime_ms >= 0),
      invocations_indexed INTEGER NOT NULL CHECK (invocations_indexed IN (0, 1))
    ) STRICT;

    CREATE TABLE session_documents (
      session_path TEXT NOT NULL REFERENCES physical_sessions(session_path) ON DELETE CASCADE,
      document_id TEXT NOT NULL,
      is_dense INTEGER NOT NULL CHECK (is_dense IN (0, 1)),
      PRIMARY KEY (session_path, document_id)
    ) STRICT, WITHOUT ROWID;

    CREATE TABLE invocations (
      invocation_id INTEGER PRIMARY KEY,
      session_path TEXT NOT NULL REFERENCES physical_sessions(session_path) ON DELETE CASCADE,
      kind TEXT NOT NULL CHECK (kind IN ('tool_call', 'bash_execution')),
      tool_name TEXT NOT NULL,
      tool_call_id TEXT,
      session_id TEXT NOT NULL,
      entry_id TEXT NOT NULL,
      source_line_start INTEGER NOT NULL CHECK (source_line_start >= 1),
      source_line_end INTEGER NOT NULL CHECK (source_line_end >= source_line_start),
      source_block_index INTEGER CHECK (source_block_index IS NULL OR source_block_index >= 0),
      timestamp TEXT NOT NULL,
      session_origin TEXT NOT NULL,
      project_identity TEXT,
      project_identity_source TEXT,
      is_error INTEGER CHECK (is_error IS NULL OR is_error IN (0, 1)),
      searchable_text TEXT NOT NULL CHECK (length(searchable_text) <= 4096),
      CHECK ((project_identity IS NULL) = (project_identity_source IS NULL))
    ) STRICT;

    CREATE INDEX invocations_project_identity_index ON invocations(project_identity);

    CREATE VIRTUAL TABLE invocations_fts USING fts5(
      tool_name,
      searchable_text,
      content = 'invocations',
      content_rowid = 'invocation_id',
      tokenize = 'unicode61'
    );

    CREATE TRIGGER invocations_after_insert AFTER INSERT ON invocations BEGIN
      INSERT INTO invocations_fts(rowid, tool_name, searchable_text)
      VALUES (new.invocation_id, new.tool_name, new.searchable_text);
    END;

    CREATE TRIGGER invocations_after_delete AFTER DELETE ON invocations BEGIN
      INSERT INTO invocations_fts(invocations_fts, rowid, tool_name, searchable_text)
      VALUES ('delete', old.invocation_id, old.tool_name, old.searchable_text);
    END;

    CREATE TRIGGER invocations_after_update AFTER UPDATE ON invocations BEGIN
      INSERT INTO invocations_fts(invocations_fts, rowid, tool_name, searchable_text)
      VALUES ('delete', old.invocation_id, old.tool_name, old.searchable_text);
      INSERT INTO invocations_fts(rowid, tool_name, searchable_text)
      VALUES (new.invocation_id, new.tool_name, new.searchable_text);
    END;

    CREATE VIRTUAL TABLE invocations_fts_vocabulary USING fts5vocab(invocations_fts, instance);

    CREATE TABLE dense_projects (
      project_key INTEGER PRIMARY KEY,
      project_identity TEXT UNIQUE,
      identity_key TEXT NOT NULL UNIQUE,
      CHECK (identity_key = coalesce(project_identity, ''))
    ) STRICT;

    CREATE TABLE dense_documents (
      rowid INTEGER PRIMARY KEY,
      document_id TEXT NOT NULL UNIQUE,
      session_path TEXT NOT NULL REFERENCES physical_sessions(session_path) ON DELETE CASCADE,
      project_key INTEGER NOT NULL REFERENCES dense_projects(project_key),
      metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json))
    ) STRICT;

    CREATE INDEX dense_documents_session_path_index ON dense_documents(session_path);
    CREATE INDEX dense_documents_project_key_index ON dense_documents(project_key);

    CREATE VIRTUAL TABLE dense_global_vectors USING vec0(
      embedding FLOAT[1024] DISTANCE_METRIC=cosine
    );

    CREATE VIRTUAL TABLE dense_project_vectors USING vec0(
      embedding FLOAT[1024] DISTANCE_METRIC=cosine,
      project_bucket INTEGER PARTITION KEY,
      project_key INTEGER
    );

    PRAGMA user_version = 2;
  `);
}

function readSqliteRecallSchemaVersion(database: DatabaseSync): number {
  return readRequiredNumber(database.prepare('PRAGMA user_version').get(), 'user_version');
}

function readSqliteRecallIdentity(
  database: DatabaseSync,
  databasePath: string,
  queryOnly: boolean,
): Readonly<SqliteRecallDatabaseIdentity> {
  let stored: SqliteRow | undefined;
  let runtime: SqliteRow | undefined;
  let journalMode: SqliteRow | undefined;
  try {
    stored = database.prepare('SELECT * FROM recall_database_identity WHERE singleton = 1').get();
    runtime = database
      .prepare('SELECT sqlite_version() AS sqlite_version, vec_version() AS sqlite_vec_version')
      .get();
    journalMode = database.prepare('PRAGMA journal_mode').get();
  } catch (error) {
    throw new Error(
      `SQLite Recall database identity incompatible at ${databasePath}; rebuild with psr index --rebuild`,
      { cause: error },
    );
  }
  const identity: SqliteRecallDatabaseIdentity = {
    schemaVersion: readSqliteRecallSchemaVersion(database),
    storageLayout: readRequiredString(stored, 'storage_layout'),
    embeddingDimensions: readRequiredNumber(stored, 'embedding_dimensions'),
    vectorEncoding: readRequiredString(stored, 'vector_encoding'),
    distanceMetric: readRequiredString(stored, 'distance_metric'),
    projectBucketCount: readRequiredNumber(stored, 'project_bucket_count'),
    sqliteVecVersion: readRequiredString(runtime, 'sqlite_vec_version'),
    sqliteVersion: readRequiredString(runtime, 'sqlite_version'),
    journalMode: readRequiredString(journalMode, 'journal_mode'),
    queryOnly,
  };
  if (identity.journalMode !== 'wal') {
    throw new Error(
      `SQLite Recall database identity incompatible at ${databasePath}: journal mode is ${identity.journalMode}, expected wal; rebuild with psr index --rebuild`,
    );
  }
  const compatible =
    identity.schemaVersion === SQLITE_RECALL_SCHEMA_VERSION &&
    identity.storageLayout === SQLITE_RECALL_STORAGE_LAYOUT &&
    identity.embeddingDimensions === SQLITE_RECALL_EMBEDDING_DIMENSIONS &&
    identity.vectorEncoding === SQLITE_RECALL_VECTOR_ENCODING &&
    identity.distanceMetric === SQLITE_RECALL_DISTANCE_METRIC &&
    identity.projectBucketCount === SQLITE_RECALL_PROJECT_BUCKET_COUNT &&
    readRequiredString(stored, 'sqlite_vec_version') === SQLITE_RECALL_VEC_RUNTIME_VERSION &&
    identity.sqliteVecVersion === SQLITE_RECALL_VEC_RUNTIME_VERSION;
  if (!compatible) {
    throw new Error(
      `SQLite Recall database identity incompatible at ${databasePath}; rebuild with psr index --rebuild`,
    );
  }
  return Object.freeze(identity);
}

function parseDenseProjectAttribution(
  value: ReturnType<
    typeof Value.Parse<typeof DENSE_DOCUMENT_METADATA_SCHEMA>
  >['projectAttribution'],
): ResolvedProjectIdentity | null {
  if (value === null) {
    return null;
  }
  const projectIdentity = parseProjectIdentity(value.projectIdentity);
  switch (value.identitySource) {
    case 'non_git_session_origin':
      return {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
      };
    case 'git_origin':
    case 'git_common_directory':
    case 'configured_project_lineage': {
      if (!isCanonicalRepositoryIdentity(projectIdentity)) {
        throw new Error(
          `SQLite Recall dense metadata invalid: ${value.identitySource} requires a repository identity`,
        );
      }
      const identitySource =
        value.identitySource === 'git_origin'
          ? RecallProjectIdentitySource.GIT_ORIGIN
          : value.identitySource === 'git_common_directory'
            ? RecallProjectIdentitySource.GIT_COMMON_DIRECTORY
            : RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE;
      return { projectIdentity, identitySource };
    }
    default:
      throw new Error('SQLite Recall dense metadata project identity source invalid');
  }
}

function parseDenseDocumentMetadata(
  metadataJson: string,
  documentId: string,
): SessionConversationChunk {
  const decoded: unknown = JSON.parse(metadataJson);
  if (!Value.Check(DENSE_DOCUMENT_METADATA_SCHEMA, decoded)) {
    throw new Error(`SQLite Recall dense metadata invalid for document ${documentId}`);
  }
  const parsed = decoded;
  if (parsed.id !== documentId) {
    throw new Error(
      `SQLite Recall dense metadata invalid for document ${documentId}: embedded identity differs`,
    );
  }
  return {
    ...parsed,
    projectAttribution: parseDenseProjectAttribution(parsed.projectAttribution),
  };
}

function serializeDenseDocumentMetadata(document: SessionConversationChunk): string {
  const metadataJson = JSON.stringify(document);
  if (metadataJson === undefined) {
    throw new Error(`SQLite Recall dense metadata invalid for document ${document.id}`);
  }
  parseDenseDocumentMetadata(metadataJson, document.id);
  return metadataJson;
}

function createFtsPhrase(query: string): string {
  return `"${query.replaceAll('"', '""')}"`;
}

function parseInvocationProjectAttribution(row: SqliteRow): ResolvedProjectIdentity | null {
  const projectIdentityValue = row.project_identity;
  const identitySourceValue = row.project_identity_source;
  if (projectIdentityValue === null && identitySourceValue === null) {
    return null;
  }
  if (typeof projectIdentityValue !== 'string' || typeof identitySourceValue !== 'string') {
    throw new Error('SQLite Recall Invocation project attribution invalid');
  }
  const projectIdentity = parseProjectIdentity(projectIdentityValue);
  if (identitySourceValue === 'non_git_session_origin') {
    return {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
    };
  }
  if (!isCanonicalRepositoryIdentity(projectIdentity)) {
    throw new Error(
      `SQLite Recall Invocation project attribution invalid: ${identitySourceValue} requires a repository identity`,
    );
  }
  switch (identitySourceValue) {
    case 'git_origin':
      return { projectIdentity, identitySource: RecallProjectIdentitySource.GIT_ORIGIN };
    case 'git_common_directory':
      return {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.GIT_COMMON_DIRECTORY,
      };
    case 'configured_project_lineage':
      return {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE,
      };
    default:
      throw new Error(
        `SQLite Recall Invocation project identity source invalid: ${identitySourceValue}`,
      );
  }
}

function decodeInvocationSearchResult(row: SqliteRow): SqliteRecallInvocationSearchResult {
  const kind = readRequiredString(row, 'kind');
  if (kind !== 'tool_call' && kind !== 'bash_execution') {
    throw new Error(`SQLite Recall Invocation kind invalid: ${kind}`);
  }
  const toolCallId = row.tool_call_id;
  const sourceBlockIndex = row.source_block_index;
  const isError = row.is_error;
  if (toolCallId !== null && typeof toolCallId !== 'string') {
    throw new Error('SQLite Recall Invocation tool call identity invalid');
  }
  if (sourceBlockIndex !== null && typeof sourceBlockIndex !== 'number') {
    throw new Error('SQLite Recall Invocation source block index invalid');
  }
  if (isError !== null && isError !== 0 && isError !== 1) {
    throw new Error('SQLite Recall Invocation error status invalid');
  }
  return {
    kind,
    toolName: readRequiredString(row, 'tool_name'),
    toolCallId,
    sessionPath: readRequiredString(row, 'session_path'),
    sessionId: readRequiredString(row, 'session_id'),
    entryId: readRequiredString(row, 'entry_id'),
    sourceLineStart: readRequiredInteger(row, 'source_line_start'),
    sourceLineEnd: readRequiredInteger(row, 'source_line_end'),
    sourceBlockIndex,
    timestamp: readRequiredString(row, 'timestamp'),
    sessionOrigin: readRequiredString(row, 'session_origin'),
    projectAttribution: parseInvocationProjectAttribution(row),
    isError: isError === null ? null : isError === 1,
    searchableText: readRequiredString(row, 'searchable_text'),
    rank: readRequiredNumber(row, 'rank'),
  };
}

function createInvocationPersistenceValues(
  invocation: InvocationRecord,
): Array<null | number | string> {
  return [
    invocation.sessionPath,
    invocation.kind,
    invocation.toolName,
    invocation.toolCallId,
    invocation.sessionId,
    invocation.entryId,
    invocation.sourceLineStart,
    invocation.sourceLineEnd,
    invocation.sourceBlockIndex,
    invocation.timestamp,
    invocation.sessionOrigin,
    invocation.projectAttribution?.projectIdentity ?? null,
    invocation.projectAttribution?.identitySource ?? null,
    invocation.isError === null ? null : invocation.isError ? 1 : 0,
    invocation.searchableText,
  ];
}

function assertDenseDocument(document: SessionConversationChunk): void {
  const isConversation =
    document.documentKind === 'conversation' &&
    document.summaryKind === null &&
    document.evidenceKind === 'conversation' &&
    (document.role === 'user' || document.role === 'assistant' || document.role === 'custom');
  const isTurnContext =
    document.documentKind === 'turn_context' &&
    document.summaryKind === null &&
    document.evidenceKind === 'turn_context' &&
    document.role === 'turn';
  const isCompactionSummary =
    document.documentKind === 'summary' &&
    document.summaryKind === 'compaction' &&
    document.evidenceKind === 'compaction_summary' &&
    document.role === 'summary';
  const isBranchSummary =
    document.documentKind === 'summary' &&
    document.summaryKind === 'branch' &&
    document.evidenceKind === 'branch_summary' &&
    document.role === 'summary';
  if (
    document.isDenseSearchable !== true ||
    document.evidencePart !== 'content' ||
    (!isConversation && !isTurnContext && !isCompactionSummary && !isBranchSummary) ||
    document.toolCallId !== null ||
    document.toolName !== null ||
    document.toolCallEntryId !== null ||
    document.toolResultEntryId !== null ||
    document.toolError !== null
  ) {
    throw new Error(
      `SQLite Recall dense document invalid for ${document.id}: only conversation, summary, branch-summary, and turn-context documents are allowed`,
    );
  }
}

function encodeDenseEmbedding(embedding: readonly number[], subject: string): Uint8Array {
  if (embedding.length !== SQLITE_RECALL_EMBEDDING_DIMENSIONS) {
    throw new Error(
      `SQLite Recall dense embedding invalid for ${subject}: expected ${SQLITE_RECALL_EMBEDDING_DIMENSIONS} dimensions, received ${embedding.length}`,
    );
  }
  const vector = new Float32Array(SQLITE_RECALL_EMBEDDING_DIMENSIONS);
  let hasNonZeroComponent = false;
  for (const [index, value] of embedding.entries()) {
    const fp32Value = Math.fround(value);
    if (!Number.isFinite(value) || !Number.isFinite(fp32Value)) {
      throw new Error(
        `SQLite Recall dense embedding invalid for ${subject}: component ${index} must be finite FP32`,
      );
    }
    vector[index] = fp32Value;
    hasNonZeroComponent ||= fp32Value !== 0;
  }
  if (!hasNonZeroComponent) {
    throw new Error(
      `SQLite Recall dense embedding invalid for ${subject}: zero vectors are not allowed`,
    );
  }
  return new Uint8Array(vector.buffer);
}

function decodeDenseEmbedding(value: SQLOutputValue, documentId: string): number[] {
  if (
    !(value instanceof Uint8Array) ||
    value.byteLength !== SQLITE_RECALL_EMBEDDING_DIMENSIONS * Float32Array.BYTES_PER_ELEMENT
  ) {
    throw new Error(`SQLite Recall dense vector invalid for document ${documentId}`);
  }
  const copiedBytes = Uint8Array.from(value);
  return Array.from(new Float32Array(copiedBytes.buffer), Number);
}

function prepareDenseDocuments(
  replacement: SqliteRecallPhysicalSessionReplacement,
): PreparedDenseDocument[] {
  if (
    !replacement.sessionPath ||
    !Number.isSafeInteger(replacement.size) ||
    replacement.size < 0 ||
    !Number.isFinite(replacement.mtimeMs) ||
    replacement.mtimeMs < 0
  ) {
    throw new Error(
      `SQLite Recall physical session replacement invalid for ${replacement.sessionPath || '<empty>'}`,
    );
  }
  const documentIds = new Set(replacement.documentIds);
  if (
    documentIds.size !== replacement.documentIds.length ||
    replacement.documentIds.some((documentId) => !documentId)
  ) {
    throw new Error(
      `SQLite Recall physical session document membership invalid for ${replacement.sessionPath}`,
    );
  }
  const denseDocumentIds = new Set<string>();
  const prepared: PreparedDenseDocument[] = [];
  for (const document of replacement.denseDocuments) {
    if (
      denseDocumentIds.has(document.id) ||
      !documentIds.has(document.id) ||
      document.sessionPath !== replacement.sessionPath
    ) {
      throw new Error(
        `SQLite Recall dense document ownership invalid for ${document.id}: expected physical session ${replacement.sessionPath}`,
      );
    }
    const embedding = replacement.denseEmbeddings.get(document.id);
    if (!embedding) {
      throw new Error(`SQLite Recall dense embedding missing for document ${document.id}`);
    }
    assertDenseDocument(document);
    denseDocumentIds.add(document.id);
    prepared.push({
      document,
      metadataJson: serializeDenseDocumentMetadata(document),
      projectIdentity: document.projectAttribution?.projectIdentity ?? null,
      vectorBlob: encodeDenseEmbedding(embedding, `document ${document.id}`),
    });
  }
  if (
    replacement.denseEmbeddings.size !== denseDocumentIds.size ||
    [...replacement.denseEmbeddings.keys()].some((documentId) => !denseDocumentIds.has(documentId))
  ) {
    throw new Error(
      `SQLite Recall dense embeddings do not match current documents for ${replacement.sessionPath}`,
    );
  }
  for (const invocation of replacement.invocations) {
    if (invocation.sessionPath !== replacement.sessionPath) {
      throw new Error(
        `SQLite Recall Invocation ownership invalid for ${invocation.entryId}: expected physical session ${replacement.sessionPath}`,
      );
    }
  }
  return prepared;
}

/** Opens or creates the schema-v2 WAL database, loading only pinned sqlite-vec before disabling extensions. */
export function openSqliteRecallDatabase(
  databasePath: string,
  options: OpenSqliteRecallDatabaseOptions = {},
): SqliteRecallDatabase {
  const readOnly = options.readOnly ?? false;
  const databaseExists = existsSync(databasePath);
  if (readOnly && !databaseExists) {
    throw new Error(
      `SQLite Recall database missing at ${databasePath}; rebuild with psr index --rebuild`,
    );
  }
  const legacyStates =
    !readOnly && !databaseExists && options.legacyStatePath
      ? readLegacyPhysicalSessionStates(options.legacyStatePath)
      : [];
  if (!readOnly) {
    mkdirSync(dirname(databasePath), { recursive: true });
  }

  const database = new DatabaseSync(databasePath, {
    allowExtension: true,
    readOnly,
    timeout: SQLITE_RECALL_BUSY_TIMEOUT_MILLISECONDS,
  });
  try {
    loadPinnedSqliteVec(database);
    configureSqliteRecallConnection(database, readOnly);
    const schemaVersion = readSqliteRecallSchemaVersion(database);
    if (schemaVersion === 0 && !readOnly) {
      runSqliteRecallTransaction(database, () => {
        createSqliteRecallSchema(database);
        importLegacyPhysicalSessionStates(database, legacyStates);
      });
    } else if (schemaVersion !== SQLITE_RECALL_SCHEMA_VERSION) {
      throw new Error(
        `SQLite Recall database schema incompatible at ${databasePath}: found version ${schemaVersion}, expected ${SQLITE_RECALL_SCHEMA_VERSION}; rebuild with psr index --rebuild`,
      );
    }
    const identity = readSqliteRecallIdentity(database, databasePath, readOnly);

    const readSession = database.prepare(
      'SELECT size, mtime_ms FROM physical_sessions WHERE session_path = ?',
    );
    const listSessionDocuments = database.prepare(
      'SELECT document_id, is_dense FROM session_documents WHERE session_path = ? ORDER BY document_id',
    );
    const listSessions = database.prepare(
      'SELECT session_path FROM physical_sessions ORDER BY session_path',
    );
    const readInvocationBackfill = database.prepare(
      'SELECT invocations_indexed FROM physical_sessions WHERE session_path = ?',
    );
    const readDenseDocument = database.prepare(
      'SELECT metadata_json FROM dense_documents WHERE document_id = ?',
    );
    const readDenseVector = database.prepare(`
      SELECT vector.embedding
      FROM dense_documents AS document
      JOIN dense_global_vectors AS vector ON vector.rowid = document.rowid
      WHERE document.document_id = ?
    `);
    const searchGlobalDenseCandidates = database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM dense_global_vectors
        WHERE embedding MATCH ? AND k = ?
      )
      SELECT document.document_id, document.metadata_json, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `);
    const searchProjectDenseCandidates = database.prepare(`
      WITH nearest AS (
        SELECT rowid, distance
        FROM dense_project_vectors
        WHERE embedding MATCH ? AND k = ?
          AND project_bucket = ? AND project_key = ?
      )
      SELECT document.document_id, document.metadata_json, nearest.distance
      FROM nearest
      JOIN dense_documents AS document ON document.rowid = nearest.rowid
      ORDER BY nearest.distance, document.document_id
    `);
    const searchInvocationCandidates = database.prepare(`
      SELECT invocation.*, bm25(invocations_fts) AS rank
      FROM invocations_fts
      JOIN invocations AS invocation ON invocation.invocation_id = invocations_fts.rowid
      WHERE invocations_fts MATCH ?
      ORDER BY rank, invocation.invocation_id
      LIMIT ?
    `);
    const searchProjectInvocationCandidates = database.prepare(`
      SELECT invocation.*, bm25(invocations_fts) AS rank
      FROM invocations_fts
      JOIN invocations AS invocation ON invocation.invocation_id = invocations_fts.rowid
      WHERE invocations_fts MATCH ? AND invocation.project_identity = ?
      ORDER BY rank, invocation.invocation_id
      LIMIT ?
    `);
    const readProjectionCounts = database.prepare(`
      SELECT
        (SELECT count(*) FROM physical_sessions) AS physical_sessions,
        (SELECT count(*) FROM session_documents) AS session_documents,
        (SELECT count(*) FROM invocations) AS invocations,
        (SELECT count(*) FROM dense_documents) AS dense_documents,
        (SELECT count(*) FROM dense_global_vectors) AS dense_global_vectors,
        (SELECT count(*) FROM dense_project_vectors) AS dense_project_vectors,
        (SELECT count(*) FROM dense_projects) AS dense_projects
    `);
    const readInvocationFtsParity = database.prepare(`
      SELECT
        (SELECT count(DISTINCT doc) FROM invocations_fts_vocabulary)
          AS invocation_fts_documents,
        (SELECT count(*)
          FROM invocations AS invocation
          WHERE NOT EXISTS (
            SELECT 1 FROM invocations_fts_vocabulary AS vocabulary
            WHERE vocabulary.doc = invocation.invocation_id
          )) AS invocations_missing_fts,
        (SELECT count(*)
          FROM (SELECT DISTINCT doc FROM invocations_fts_vocabulary) AS vocabulary
          WHERE NOT EXISTS (
            SELECT 1 FROM invocations AS invocation
            WHERE invocation.invocation_id = vocabulary.doc
          )) AS fts_documents_missing_invocation
    `);
    const readVectorParity = database.prepare(`
      SELECT
        (SELECT count(*) FROM dense_documents) AS dense_documents,
        (SELECT count(*) FROM dense_global_vectors) AS global_vectors,
        (SELECT count(*) FROM dense_project_vectors) AS project_vectors,
        (SELECT count(*)
          FROM dense_documents AS document
          LEFT JOIN dense_global_vectors AS vector ON vector.rowid = document.rowid
          WHERE vector.rowid IS NULL) AS dense_documents_missing_global_vector,
        (SELECT count(*)
          FROM dense_global_vectors AS vector
          LEFT JOIN dense_documents AS document ON document.rowid = vector.rowid
          WHERE document.rowid IS NULL) AS global_vectors_missing_dense_document,
        (SELECT count(*)
          FROM dense_documents AS document
          LEFT JOIN dense_project_vectors AS vector ON vector.rowid = document.rowid
          WHERE vector.rowid IS NULL) AS dense_documents_missing_project_vector,
        (SELECT count(*)
          FROM dense_project_vectors AS vector
          LEFT JOIN dense_documents AS document ON document.rowid = vector.rowid
          WHERE document.rowid IS NULL) AS project_vectors_missing_dense_document,
        (SELECT count(*)
          FROM dense_documents AS document
          JOIN dense_project_vectors AS vector ON vector.rowid = document.rowid
          WHERE vector.project_key != document.project_key
            OR vector.project_bucket != document.project_key % ${SQLITE_RECALL_PROJECT_BUCKET_COUNT})
          AS project_metadata_mismatches,
        (SELECT count(*)
          FROM dense_global_vectors AS global_vector
          JOIN dense_project_vectors AS project_vector ON project_vector.rowid = global_vector.rowid
          WHERE global_vector.embedding != project_vector.embedding)
          AS vector_value_mismatches
    `);
    const readSqliteIntegrity = database.prepare('PRAGMA integrity_check');
    const readForeignKeyViolations = database.prepare('PRAGMA foreign_key_check');
    const verifyInvocationFtsIntegrity = database.prepare(`
      INSERT INTO invocations_fts(invocations_fts, rank) VALUES ('integrity-check', 1)
    `);
    const upsertSession = database.prepare(`
      INSERT INTO physical_sessions(session_path, size, mtime_ms, invocations_indexed)
      VALUES (?, ?, ?, 1)
      ON CONFLICT(session_path) DO UPDATE SET
        size = excluded.size,
        mtime_ms = excluded.mtime_ms,
        invocations_indexed = 1
    `);
    const readSessionDenseRows = database.prepare(
      'SELECT rowid, document_id FROM dense_documents WHERE session_path = ?',
    );
    const readDenseOwner = database.prepare(
      'SELECT rowid, session_path FROM dense_documents WHERE document_id = ?',
    );
    const deleteGlobalVector = database.prepare('DELETE FROM dense_global_vectors WHERE rowid = ?');
    const deleteProjectVector = database.prepare(
      'DELETE FROM dense_project_vectors WHERE rowid = ?',
    );
    const deleteDenseDocument = database.prepare('DELETE FROM dense_documents WHERE rowid = ?');
    const updateDenseDocument = database.prepare(`
      UPDATE dense_documents
      SET session_path = ?, project_key = ?, metadata_json = ?
      WHERE rowid = ?
    `);
    const insertDenseDocument = database.prepare(`
      INSERT INTO dense_documents(document_id, session_path, project_key, metadata_json)
      VALUES (?, ?, ?, ?)
    `);
    const insertGlobalVector = database.prepare(
      'INSERT INTO dense_global_vectors(rowid, embedding) VALUES (?, ?)',
    );
    const insertProjectVector = database.prepare(`
      INSERT INTO dense_project_vectors(rowid, embedding, project_bucket, project_key)
      VALUES (?, ?, ?, ?)
    `);
    const readProject = database.prepare(
      'SELECT project_key FROM dense_projects WHERE identity_key = ?',
    );
    const insertProject = database.prepare(
      'INSERT INTO dense_projects(project_identity, identity_key) VALUES (?, ?)',
    );
    const deleteSessionDocuments = database.prepare(
      'DELETE FROM session_documents WHERE session_path = ?',
    );
    const insertSessionDocument = database.prepare(
      'INSERT INTO session_documents(session_path, document_id, is_dense) VALUES (?, ?, ?)',
    );
    const deleteInvocations = database.prepare('DELETE FROM invocations WHERE session_path = ?');
    const insertInvocation = database.prepare(`
      INSERT INTO invocations(
        session_path, kind, tool_name, tool_call_id, session_id, entry_id,
        source_line_start, source_line_end, source_block_index, timestamp,
        session_origin, project_identity, project_identity_source, is_error, searchable_text
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    const deleteSession = database.prepare('DELETE FROM physical_sessions WHERE session_path = ?');
    const deleteOrphanProjects = database.prepare(`
      DELETE FROM dense_projects
      WHERE NOT EXISTS (
        SELECT 1 FROM dense_documents AS document
        WHERE document.project_key = dense_projects.project_key
      )
    `);

    const readPhysicalSessionState = (
      sessionPath: string,
    ): SqliteRecallPhysicalSessionState | null =>
      runSqliteRecallRead(database, () => {
        const row = readSession.get(sessionPath);
        if (!row) {
          return null;
        }
        const documentIds: string[] = [];
        const denseDocumentIds: string[] = [];
        for (const document of listSessionDocuments.all(sessionPath)) {
          const documentId = readRequiredString(document, 'document_id');
          const isDense = readRequiredInteger(document, 'is_dense');
          if (isDense !== 0 && isDense !== 1) {
            throw new Error(
              `SQLite Recall document membership invalid for physical session ${sessionPath}`,
            );
          }
          documentIds.push(documentId);
          if (isDense === 1) {
            denseDocumentIds.push(documentId);
          }
        }
        return {
          size: readRequiredInteger(row, 'size'),
          mtimeMs: readRequiredNumber(row, 'mtime_ms'),
          documentIds,
          denseDocumentIds,
        };
      });

    const resolveProjectKey = (projectIdentity: string | null): number => {
      const identityKey = projectIdentity ?? '';
      const existing = readProject.get(identityKey);
      if (existing) {
        return readRequiredInteger(existing, 'project_key');
      }
      const result = insertProject.run(projectIdentity, identityKey);
      return convertSqliteInteger(result.lastInsertRowid, 'project key');
    };

    const assertWritable = (operation: string): void => {
      if (readOnly) {
        throw new Error(`SQLite Recall database is read-only: ${operation} is unavailable`);
      }
    };

    return {
      identity,
      readPhysicalSessionState,
      listPhysicalSessionPaths() {
        return listSessions.all().map((row) => readRequiredString(row, 'session_path'));
      },
      requiresInvocationBackfill(sessionPath) {
        return readInvocationBackfill.get(sessionPath)?.invocations_indexed === 0;
      },
      fetchDenseDocuments(ids) {
        return runSqliteRecallRead(database, () => {
          const documents = new Map<string, SessionConversationChunk>();
          for (const id of ids) {
            const row = readDenseDocument.get(id);
            if (row) {
              documents.set(
                id,
                parseDenseDocumentMetadata(readRequiredString(row, 'metadata_json'), id),
              );
            }
          }
          return documents;
        });
      },
      fetchDenseVectors(ids) {
        return runSqliteRecallRead(database, () => {
          const vectors = new Map<string, number[]>();
          for (const id of ids) {
            const row = readDenseVector.get(id);
            if (row) {
              vectors.set(id, decodeDenseEmbedding(row.embedding ?? null, id));
            }
          }
          return vectors;
        });
      },
      searchDenseCandidates(embedding, limit, projectIdentity) {
        if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
          throw new Error(
            'SQLite Recall dense candidate limit invalid: expected an integer from 1 to 200',
          );
        }
        const queryVector = encodeDenseEmbedding(embedding, 'query');
        return runSqliteRecallRead(database, () => {
          let rows: SqliteRow[];
          if (projectIdentity === undefined) {
            rows = searchGlobalDenseCandidates.all(queryVector, BigInt(limit));
          } else {
            const projectRow = readProject.get(projectIdentity);
            if (!projectRow) {
              return [];
            }
            const projectKey = readRequiredInteger(projectRow, 'project_key');
            rows = searchProjectDenseCandidates.all(
              queryVector,
              BigInt(limit),
              BigInt(projectKey % SQLITE_RECALL_PROJECT_BUCKET_COUNT),
              BigInt(projectKey),
            );
          }
          return rows.map((row) => {
            const documentId = readRequiredString(row, 'document_id');
            return {
              ...parseDenseDocumentMetadata(readRequiredString(row, 'metadata_json'), documentId),
              cosineDistance: readRequiredNumber(row, 'distance'),
            };
          });
        });
      },
      searchInvocations(query, limit, projectIdentity) {
        if (!query.trim() || !Number.isInteger(limit) || limit < 1) {
          return [];
        }
        const phrase = createFtsPhrase(query.trim());
        const rows =
          projectIdentity === undefined
            ? searchInvocationCandidates.all(phrase, limit)
            : searchProjectInvocationCandidates.all(phrase, projectIdentity, limit);
        return rows.map(decodeInvocationSearchResult);
      },
      readCounts() {
        const row = readProjectionCounts.get();
        return {
          physicalSessions: readRequiredInteger(row, 'physical_sessions'),
          sessionDocuments: readRequiredInteger(row, 'session_documents'),
          invocations: readRequiredInteger(row, 'invocations'),
          denseDocuments: readRequiredInteger(row, 'dense_documents'),
          denseGlobalVectors: readRequiredInteger(row, 'dense_global_vectors'),
          denseProjectVectors: readRequiredInteger(row, 'dense_project_vectors'),
          denseProjects: readRequiredInteger(row, 'dense_projects'),
        };
      },
      checkIntegrity() {
        return runSqliteRecallRead(database, () => {
          if (!readOnly) {
            verifyInvocationFtsIntegrity.run();
          }
          const sqliteIntegrity = readSqliteIntegrity
            .all()
            .map((row) => readRequiredString(row, 'integrity_check'));
          const foreignKeyViolations = readForeignKeyViolations.all().length;
          const ftsParity = readInvocationFtsParity.get();
          const vectorRow = readVectorParity.get();
          const vectorParity: SqliteRecallVectorParityDiagnostics = {
            denseDocuments: readRequiredInteger(vectorRow, 'dense_documents'),
            globalVectors: readRequiredInteger(vectorRow, 'global_vectors'),
            projectVectors: readRequiredInteger(vectorRow, 'project_vectors'),
            denseDocumentsMissingGlobalVector: readRequiredInteger(
              vectorRow,
              'dense_documents_missing_global_vector',
            ),
            globalVectorsMissingDenseDocument: readRequiredInteger(
              vectorRow,
              'global_vectors_missing_dense_document',
            ),
            denseDocumentsMissingProjectVector: readRequiredInteger(
              vectorRow,
              'dense_documents_missing_project_vector',
            ),
            projectVectorsMissingDenseDocument: readRequiredInteger(
              vectorRow,
              'project_vectors_missing_dense_document',
            ),
            projectMetadataMismatches: readRequiredInteger(
              vectorRow,
              'project_metadata_mismatches',
            ),
            vectorValueMismatches: readRequiredInteger(vectorRow, 'vector_value_mismatches'),
            healthy: false,
          };
          vectorParity.healthy =
            vectorParity.denseDocuments === vectorParity.globalVectors &&
            vectorParity.denseDocuments === vectorParity.projectVectors &&
            vectorParity.denseDocumentsMissingGlobalVector === 0 &&
            vectorParity.globalVectorsMissingDenseDocument === 0 &&
            vectorParity.denseDocumentsMissingProjectVector === 0 &&
            vectorParity.projectVectorsMissingDenseDocument === 0 &&
            vectorParity.projectMetadataMismatches === 0 &&
            vectorParity.vectorValueMismatches === 0;
          const invocationFtsDocuments = readRequiredInteger(ftsParity, 'invocation_fts_documents');
          const invocationsMissingFts = readRequiredInteger(ftsParity, 'invocations_missing_fts');
          const ftsDocumentsMissingInvocation = readRequiredInteger(
            ftsParity,
            'fts_documents_missing_invocation',
          );
          return {
            sqliteIntegrity,
            foreignKeyViolations,
            invocationFtsIntegrityChecked: !readOnly,
            invocationFtsDocuments,
            invocationsMissingFts,
            ftsDocumentsMissingInvocation,
            vectorParity,
            healthy:
              sqliteIntegrity.length === 1 &&
              sqliteIntegrity[0] === 'ok' &&
              foreignKeyViolations === 0 &&
              invocationsMissingFts === 0 &&
              ftsDocumentsMissingInvocation === 0 &&
              vectorParity.healthy,
          };
        });
      },
      replacePhysicalSession(replacement) {
        assertWritable('replacePhysicalSession');
        const denseDocuments = prepareDenseDocuments(replacement);
        const currentDenseIds = new Set(denseDocuments.map(({ document }) => document.id));
        runSqliteRecallTransaction(database, () => {
          upsertSession.run(replacement.sessionPath, replacement.size, replacement.mtimeMs);
          const previousRowids = new Map<string, number>();
          for (const row of readSessionDenseRows.all(replacement.sessionPath)) {
            const rowid = readRequiredInteger(row, 'rowid');
            const documentId = readRequiredString(row, 'document_id');
            previousRowids.set(documentId, rowid);
            deleteGlobalVector.run(BigInt(rowid));
            deleteProjectVector.run(BigInt(rowid));
            if (!currentDenseIds.has(documentId)) {
              deleteDenseDocument.run(rowid);
            }
          }

          for (const prepared of denseDocuments) {
            const projectKey = resolveProjectKey(prepared.projectIdentity);
            const previousRowid = previousRowids.get(prepared.document.id);
            let rowid: number;
            if (previousRowid !== undefined) {
              updateDenseDocument.run(
                replacement.sessionPath,
                projectKey,
                prepared.metadataJson,
                previousRowid,
              );
              rowid = previousRowid;
            } else {
              const owner = readDenseOwner.get(prepared.document.id);
              if (owner) {
                throw new Error(
                  `SQLite Recall dense document ownership conflict for ${prepared.document.id}: already owned by ${readRequiredString(owner, 'session_path')}`,
                );
              }
              const result = insertDenseDocument.run(
                prepared.document.id,
                replacement.sessionPath,
                projectKey,
                prepared.metadataJson,
              );
              rowid = convertSqliteInteger(result.lastInsertRowid, 'dense document rowid');
            }
            insertGlobalVector.run(BigInt(rowid), prepared.vectorBlob);
            insertProjectVector.run(
              BigInt(rowid),
              prepared.vectorBlob,
              BigInt(projectKey % SQLITE_RECALL_PROJECT_BUCKET_COUNT),
              BigInt(projectKey),
            );
          }

          deleteSessionDocuments.run(replacement.sessionPath);
          for (const documentId of replacement.documentIds) {
            insertSessionDocument.run(
              replacement.sessionPath,
              documentId,
              currentDenseIds.has(documentId) ? 1 : 0,
            );
          }

          deleteInvocations.run(replacement.sessionPath);
          for (const invocation of replacement.invocations) {
            insertInvocation.run(...createInvocationPersistenceValues(invocation));
          }
          deleteOrphanProjects.run();
        });
      },
      deletePhysicalSession(sessionPath) {
        assertWritable('deletePhysicalSession');
        let deleted = false;
        runSqliteRecallTransaction(database, () => {
          for (const row of readSessionDenseRows.all(sessionPath)) {
            const rowid = readRequiredInteger(row, 'rowid');
            deleteGlobalVector.run(BigInt(rowid));
            deleteProjectVector.run(BigInt(rowid));
          }
          deleted =
            convertSqliteInteger(deleteSession.run(sessionPath).changes, 'delete count') > 0;
          deleteOrphanProjects.run();
        });
        return deleted;
      },
      close() {
        database.close();
      },
    };
  } catch (error) {
    database.close();
    throw error;
  }
}

import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { DatabaseSync, type StatementSync } from 'node:sqlite';

import { RecallProjectIdentitySource } from './enums.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import type { InvocationRecord } from './createSessionInvocationRecords.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  parseProjectIdentity,
  parseRepositoryIdentity,
  type ProjectIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';

const RECALL_CATALOG_SCHEMA_VERSION = 1;

interface LegacyRecallSessionState extends RecallCatalogSessionState {
  sessionPath: string;
}

/** Options for one catalog open and optional first-open legacy state migration. */
export interface OpenRecallCatalogOptions {
  legacyStatePath?: string;
  readOnly?: boolean;
}

/** Incremental state retained for one physical session file. */
export interface RecallCatalogSessionState {
  size: number;
  mtimeMs: number;
  documentIds: string[];
  denseDocumentIds: string[];
}

/** Complete replacement state for one physical session file and its Invocation records. */
export interface RecallCatalogSessionReplacement extends RecallCatalogSessionState {
  sessionPath: string;
  invocations: readonly InvocationRecord[];
}

/** One compact Invocation search match with its SQLite FTS rank. */
export interface RecallCatalogInvocationSearchResult extends InvocationRecord {
  rank: number;
}

/** Incremental physical-session state and compact Invocation search operations. */
export interface RecallCatalog {
  readPhysicalSessionState(sessionPath: string): RecallCatalogSessionState | null;
  listPhysicalSessionPaths(): string[];
  requiresInvocationBackfill(sessionPath: string): boolean;
  replacePhysicalSession(replacement: RecallCatalogSessionReplacement): void;
  deletePhysicalSession(sessionPath: string): boolean;
  searchInvocations(
    query: string,
    limit: number,
    projectIdentity?: ProjectIdentity,
  ): RecallCatalogInvocationSearchResult[];
  countInvocations(): number;
  close(): void;
}

function readLegacyRecallSessionStates(legacyStatePath: string): LegacyRecallSessionState[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(legacyStatePath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return [];
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall index state invalid at ${legacyStatePath}: ${message}`, {
      cause: error,
    });
  }
  if (
    !isUnknownRecord(parsed) ||
    parsed.version !== 3 ||
    parsed.importPolicyVersion !== SESSION_IMPORT_POLICY_VERSION ||
    !isUnknownRecord(parsed.sessions)
  ) {
    throw new Error(`Recall index state invalid at ${legacyStatePath}: incompatible state schema`);
  }
  return Object.entries(parsed.sessions).map(([sessionPath, value]) => {
    if (
      !isUnknownRecord(value) ||
      typeof value.size !== 'number' ||
      !Number.isFinite(value.size) ||
      value.size < 0 ||
      typeof value.mtimeMs !== 'number' ||
      !Number.isFinite(value.mtimeMs) ||
      value.mtimeMs < 0 ||
      !Array.isArray(value.chunks)
    ) {
      throw new Error(
        `Recall index state invalid at ${legacyStatePath}: invalid session state for ${sessionPath}`,
      );
    }
    const documentIds: string[] = [];
    for (const chunk of value.chunks) {
      if (!isUnknownRecord(chunk) || typeof chunk.id !== 'string' || chunk.id.length === 0) {
        throw new Error(
          `Recall index state invalid at ${legacyStatePath}: invalid document identity for ${sessionPath}`,
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

function createRecallCatalogSchema(database: DatabaseSync): void {
  database.exec(`
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

    CREATE INDEX invocations_project_identity_index
      ON invocations(project_identity);

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

  `);
  database.exec('PRAGMA user_version = 1');
}

function importLegacyRecallSessionStates(
  database: DatabaseSync,
  states: readonly LegacyRecallSessionState[],
): void {
  const insertSession = database.prepare(
    'INSERT INTO physical_sessions(session_path, size, mtime_ms, invocations_indexed) VALUES (?, ?, ?, 0)',
  );
  const insertDocument = database.prepare(
    'INSERT INTO session_documents(session_path, document_id, is_dense) VALUES (?, ?, 0)',
  );
  for (const state of states) {
    insertSession.run(state.sessionPath, state.size, state.mtimeMs);
    for (const documentId of state.documentIds) {
      insertDocument.run(state.sessionPath, documentId);
    }
  }
}

function readRecallCatalogSchemaVersion(database: DatabaseSync): number {
  const row = database.prepare('PRAGMA user_version').get();
  const version = row?.user_version;
  if (typeof version !== 'number') {
    throw new Error('Recall catalog schema version could not be read');
  }
  return version;
}

function runRecallCatalogTransaction(database: DatabaseSync, operation: () => void): void {
  database.exec('BEGIN IMMEDIATE');
  try {
    operation();
    database.exec('COMMIT');
  } catch (error) {
    database.exec('ROLLBACK');
    throw error;
  }
}

function insertInvocationRecord(statement: StatementSync, invocation: InvocationRecord): void {
  statement.run(...createInvocationPersistenceValues(invocation));
}

function createFtsPhrase(query: string): string {
  return `"${query.replaceAll('"', '""')}"`;
}

function readRequiredString(
  row: Record<string, null | number | bigint | string | Uint8Array>,
  column: string,
): string {
  const value = row[column];
  if (typeof value !== 'string') {
    throw new Error(`Recall catalog Invocation search column invalid: ${column}`);
  }
  return value;
}

function readRequiredNumber(
  row: Record<string, null | number | bigint | string | Uint8Array>,
  column: string,
): number {
  const value = row[column];
  if (typeof value !== 'number') {
    throw new Error(`Recall catalog Invocation search column invalid: ${column}`);
  }
  return value;
}

function readInvocationProjectAttribution(
  row: Record<string, null | number | bigint | string | Uint8Array>,
): ResolvedProjectIdentity | null {
  const identity = row.project_identity;
  const source = row.project_identity_source;
  if (identity === null && source === null) {
    return null;
  }
  if (typeof identity !== 'string' || typeof source !== 'string') {
    throw new Error('Recall catalog Invocation project attribution invalid');
  }
  switch (source) {
    case 'git_origin':
      return {
        projectIdentity: parseRepositoryIdentity(identity),
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      };
    case 'git_common_directory':
      return {
        projectIdentity: parseRepositoryIdentity(identity),
        identitySource: RecallProjectIdentitySource.GIT_COMMON_DIRECTORY,
      };
    case 'configured_project_lineage':
      return {
        projectIdentity: parseRepositoryIdentity(identity),
        identitySource: RecallProjectIdentitySource.CONFIGURED_PROJECT_LINEAGE,
      };
    case 'non_git_session_origin':
      return {
        projectIdentity: parseProjectIdentity(identity),
        identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
      };
    default:
      throw new Error(`Recall catalog Invocation project identity source invalid: ${source}`);
  }
}

function decodeInvocationSearchResult(
  row: Record<string, null | number | bigint | string | Uint8Array>,
): RecallCatalogInvocationSearchResult {
  const kind = readRequiredString(row, 'kind');
  if (kind !== 'tool_call' && kind !== 'bash_execution') {
    throw new Error(`Recall catalog Invocation kind invalid: ${kind}`);
  }
  const toolCallId = row.tool_call_id;
  const sourceBlockIndex = row.source_block_index;
  const isError = row.is_error;
  if (toolCallId !== null && typeof toolCallId !== 'string') {
    throw new Error('Recall catalog Invocation tool call identity invalid');
  }
  if (sourceBlockIndex !== null && typeof sourceBlockIndex !== 'number') {
    throw new Error('Recall catalog Invocation source block index invalid');
  }
  if (isError !== null && isError !== 0 && isError !== 1) {
    throw new Error('Recall catalog Invocation error status invalid');
  }
  return {
    kind,
    toolName: readRequiredString(row, 'tool_name'),
    toolCallId,
    sessionPath: readRequiredString(row, 'session_path'),
    sessionId: readRequiredString(row, 'session_id'),
    entryId: readRequiredString(row, 'entry_id'),
    sourceLineStart: readRequiredNumber(row, 'source_line_start'),
    sourceLineEnd: readRequiredNumber(row, 'source_line_end'),
    sourceBlockIndex,
    timestamp: readRequiredString(row, 'timestamp'),
    sessionOrigin: readRequiredString(row, 'session_origin'),
    projectAttribution: readInvocationProjectAttribution(row),
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

/** Opens or creates the WAL-mode SQLite catalog used by incremental recall maintenance. */
export function openRecallCatalog(
  catalogPath: string,
  options: OpenRecallCatalogOptions = {},
): RecallCatalog {
  if (options.readOnly && !existsSync(catalogPath)) {
    throw new Error(`Recall catalog missing at ${catalogPath}; rebuild with psr index --rebuild`);
  }
  const legacyStates =
    !options.readOnly && !existsSync(catalogPath) && options.legacyStatePath
      ? readLegacyRecallSessionStates(options.legacyStatePath)
      : [];
  if (!options.readOnly) {
    mkdirSync(dirname(catalogPath), { recursive: true });
  }
  const database = new DatabaseSync(catalogPath, {
    timeout: 5_000,
    readOnly: options.readOnly ?? false,
  });
  database.exec(
    options.readOnly
      ? 'PRAGMA foreign_keys = ON;'
      : 'PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;',
  );
  const schemaVersion = readRecallCatalogSchemaVersion(database);
  if (schemaVersion === 0 && !options.readOnly) {
    runRecallCatalogTransaction(database, () => {
      createRecallCatalogSchema(database);
      importLegacyRecallSessionStates(database, legacyStates);
    });
  } else if (schemaVersion !== RECALL_CATALOG_SCHEMA_VERSION) {
    database.close();
    throw new Error(
      `Recall catalog schema incompatible at ${catalogPath}: found version ${schemaVersion}, expected ${RECALL_CATALOG_SCHEMA_VERSION}; rebuild with psr index --rebuild`,
    );
  }

  const readSession = database.prepare(
    'SELECT size, mtime_ms FROM physical_sessions WHERE session_path = ?',
  );
  const listDocuments = database.prepare(
    'SELECT document_id, is_dense FROM session_documents WHERE session_path = ? ORDER BY document_id',
  );
  const listSessions = database.prepare(
    'SELECT session_path FROM physical_sessions ORDER BY session_path',
  );
  const readInvocationBackfill = database.prepare(
    'SELECT invocations_indexed FROM physical_sessions WHERE session_path = ?',
  );
  const upsertSession = database.prepare(`
    INSERT INTO physical_sessions(session_path, size, mtime_ms, invocations_indexed)
    VALUES (?, ?, ?, 1)
    ON CONFLICT(session_path) DO UPDATE SET
      size = excluded.size,
      mtime_ms = excluded.mtime_ms,
      invocations_indexed = 1
  `);
  const deleteSessionDocuments = database.prepare(
    'DELETE FROM session_documents WHERE session_path = ?',
  );
  const insertSessionDocument = database.prepare(
    'INSERT INTO session_documents(session_path, document_id, is_dense) VALUES (?, ?, ?)',
  );
  const deleteInvocations = database.prepare('DELETE FROM invocations WHERE session_path = ?');
  const listInvocations = database.prepare(`
    SELECT invocation.*, 0 AS rank
    FROM invocations AS invocation
    WHERE invocation.session_path = ?
    ORDER BY invocation.invocation_id
  `);
  const insertInvocation = database.prepare(`
    INSERT INTO invocations(
      session_path, kind, tool_name, tool_call_id, session_id, entry_id,
      source_line_start, source_line_end, source_block_index, timestamp,
      session_origin, project_identity, project_identity_source, is_error, searchable_text
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const deleteSession = database.prepare('DELETE FROM physical_sessions WHERE session_path = ?');
  const countInvocationRows = database.prepare('SELECT count(*) AS count FROM invocations');

  const readPhysicalSessionState = (sessionPath: string): RecallCatalogSessionState | null => {
    const row = readSession.get(sessionPath);
    if (!row) {
      return null;
    }
    const size = row.size;
    const mtimeMs = row.mtime_ms;
    if (typeof size !== 'number' || typeof mtimeMs !== 'number') {
      throw new Error(`Recall catalog session state invalid for ${sessionPath}`);
    }
    const documentRows = listDocuments.all(sessionPath);
    const documentIds: string[] = [];
    const denseDocumentIds: string[] = [];
    for (const document of documentRows) {
      if (
        typeof document.document_id !== 'string' ||
        (document.is_dense !== 0 && document.is_dense !== 1)
      ) {
        throw new Error(`Recall catalog document identity invalid for ${sessionPath}`);
      }
      documentIds.push(document.document_id);
      if (document.is_dense === 1) {
        denseDocumentIds.push(document.document_id);
      }
    }
    return { size, mtimeMs, documentIds, denseDocumentIds };
  };

  return {
    readPhysicalSessionState,

    listPhysicalSessionPaths() {
      return listSessions.all().map((row) => {
        if (typeof row.session_path !== 'string') {
          throw new Error('Recall catalog physical session path invalid');
        }
        return row.session_path;
      });
    },

    requiresInvocationBackfill(sessionPath) {
      const row = readInvocationBackfill.get(sessionPath);
      return row?.invocations_indexed === 0;
    },

    replacePhysicalSession(replacement) {
      const denseDocumentIds = new Set(replacement.denseDocumentIds);
      if (
        replacement.denseDocumentIds.some(
          (documentId) => !replacement.documentIds.includes(documentId),
        )
      ) {
        throw new Error(
          `Recall catalog dense document identity missing from session ${replacement.sessionPath}`,
        );
      }
      const previous = readPhysicalSessionState(replacement.sessionPath);
      const previousInvocations = previous
        ? listInvocations.all(replacement.sessionPath).map(decodeInvocationSearchResult)
        : [];
      const sortedDocumentIds = [...replacement.documentIds].sort();
      const sortedDenseDocumentIds = [...replacement.denseDocumentIds].sort();
      const childRowsUnchanged =
        previous !== null &&
        JSON.stringify(previous.documentIds) === JSON.stringify(sortedDocumentIds) &&
        JSON.stringify(previous.denseDocumentIds) === JSON.stringify(sortedDenseDocumentIds) &&
        JSON.stringify(previousInvocations.map(createInvocationPersistenceValues)) ===
          JSON.stringify(replacement.invocations.map(createInvocationPersistenceValues));
      runRecallCatalogTransaction(database, () => {
        upsertSession.run(replacement.sessionPath, replacement.size, replacement.mtimeMs);
        if (childRowsUnchanged) {
          return;
        }
        deleteSessionDocuments.run(replacement.sessionPath);
        for (const documentId of replacement.documentIds) {
          insertSessionDocument.run(
            replacement.sessionPath,
            documentId,
            denseDocumentIds.has(documentId) ? 1 : 0,
          );
        }
        deleteInvocations.run(replacement.sessionPath);
        for (const invocation of replacement.invocations) {
          insertInvocationRecord(insertInvocation, invocation);
        }
      });
    },

    deletePhysicalSession(sessionPath) {
      return deleteSession.run(sessionPath).changes > 0;
    },

    searchInvocations(query, limit, projectIdentity) {
      if (!query.trim() || !Number.isInteger(limit) || limit < 1) {
        return [];
      }
      const projectClause = projectIdentity ? 'AND invocation.project_identity = ?' : '';
      const statement = database.prepare(`
        SELECT invocation.*, bm25(invocations_fts) AS rank
        FROM invocations_fts
        JOIN invocations AS invocation ON invocation.invocation_id = invocations_fts.rowid
        WHERE invocations_fts MATCH ? ${projectClause}
        ORDER BY rank, invocation.invocation_id
        LIMIT ?
      `);
      const parameters = projectIdentity
        ? [createFtsPhrase(query.trim()), projectIdentity, limit]
        : [createFtsPhrase(query.trim()), limit];
      return statement.all(...parameters).map(decodeInvocationSearchResult);
    },

    countInvocations() {
      const count = countInvocationRows.get()?.count;
      if (typeof count !== 'number') {
        throw new Error('Recall catalog Invocation count invalid');
      }
      return count;
    },

    close() {
      database.close();
    },
  };
}

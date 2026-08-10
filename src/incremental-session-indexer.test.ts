import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, stat, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';

import {
  indexChangedConversationSessions,
  type DenseRecallIndexStore,
} from './incremental-session-indexer.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { RecallProjectIdentitySource } from './enums.js';
import { parseProjectIdentity, type ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
} from './sqlite-recall-database.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type {
  ConversationTextTokenizer,
  SessionConversationChunk,
} from './session-conversation-index.js';
import type { DenseRecallDocument } from './dense-recall-conversation-store.js';

class MemoryConversationStore implements DenseRecallIndexStore {
  readonly chunks = new Map<string, DenseRecallDocument>();

  upsertDocuments(chunks: DenseRecallDocument[]): void {
    for (const chunk of chunks) {
      this.chunks.set(chunk.id, structuredClone(chunk));
    }
  }

  deleteDocuments(ids: string[]): void {
    for (const id of ids) {
      this.chunks.delete(id);
    }
  }

  fetchDocuments(ids: string[]) {
    return new Map(
      ids.flatMap((id) => {
        const chunk = this.chunks.get(id);
        return chunk ? [[id, structuredClone(chunk)]] : [];
      }),
    );
  }

  fetchVectors(ids: string[]) {
    return new Map(
      ids.flatMap((id) => {
        const chunk = this.chunks.get(id);
        return chunk?.isDenseSearchable ? [[id, [...chunk.embedding]]] : [];
      }),
    );
  }
}

type IndexingMaintenanceWorksetEvent = Extract<
  RecallIndexProgressEvent,
  { kind: 'indexing-maintenance-workset' }
>;

function isIndexingMaintenanceWorksetEvent(
  event: RecallIndexProgressEvent,
): event is IndexingMaintenanceWorksetEvent {
  return event.kind === 'indexing-maintenance-workset';
}

function sessionLines(entries: object[]): string {
  return `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`;
}

function readCatalogSessionPaths(databasePath: string): string[] {
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  try {
    return database.listPhysicalSessionPaths();
  } finally {
    database.close();
  }
}

function readCatalogSessionState(databasePath: string, sessionPath: string) {
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  try {
    return database.readPhysicalSessionState(sessionPath);
  } finally {
    database.close();
  }
}

function readSessionDenseDocuments(databasePath: string, sessionPath: string) {
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  try {
    const state = database.readPhysicalSessionState(sessionPath);
    return database.fetchDenseDocuments(state?.denseDocumentIds ?? []);
  } finally {
    database.close();
  }
}

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createTestEmbedding(firstValue = 1): number[] {
  const embedding = new Array<number>(SQLITE_RECALL_EMBEDDING_DIMENSIONS).fill(0);
  embedding[0] = firstValue;
  return embedding;
}

function createRecordingEmbeddingProvider(batches: string[][]): RecallEmbeddingProvider {
  return {
    async embedQuery() {
      const embedding = new Array<number>(SQLITE_RECALL_EMBEDDING_DIMENSIONS).fill(0);
      embedding[0] = 1;
      return embedding;
    },
    async embedDocuments(documents) {
      batches.push([...documents]);
      return documents.map(() => createTestEmbedding());
    },
  };
}

async function writeSimplePhysicalSessionFile(
  sessionPath: string,
  sessionId: string,
  content: string,
): Promise<void> {
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: sessionId, timestamp: '2026-01-01', cwd: '/p' },
      {
        type: 'message',
        id: `user-${sessionId}`,
        parentId: null,
        timestamp: '2026-01-01',
        message: { role: 'user', content },
      },
    ]),
  );
}

function createIndexerOptions(options: {
  sessionsDirectory: string;
  catalogPath: string;
  store: DenseRecallIndexStore;
  embeddingProvider: RecallEmbeddingProvider;
}) {
  return {
    ...options,
    ignoredPhysicalSessionPaths: new Set<string>(),
    tokenizer,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
}

void test('one changed physical session atomically replaces every SQLite Recall projection', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-unified-replacement-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: 'one', timestamp: '2026-01-01', cwd: '/p' },
      {
        type: 'message',
        id: 'user-one',
        parentId: null,
        timestamp: '2026-01-01',
        message: { role: 'user', content: 'atomic projection evidence' },
      },
      {
        type: 'message',
        id: 'assistant-one',
        parentId: 'user-one',
        timestamp: '2026-01-01',
        message: {
          role: 'assistant',
          content: [
            { type: 'toolCall', id: 'call-one', name: 'read', arguments: { path: '/tmp/a' } },
          ],
        },
      },
    ]),
  );

  openSqliteRecallDatabase(databasePath).close();
  const auditDatabase = new DatabaseSync(databasePath);
  auditDatabase.exec(`
    CREATE TABLE replacement_audit (operation TEXT NOT NULL);
    CREATE TRIGGER audit_physical_session_insert
    AFTER INSERT ON physical_sessions
    BEGIN
      INSERT INTO replacement_audit (operation) VALUES ('insert');
    END;
    CREATE TRIGGER audit_physical_session_update
    AFTER UPDATE ON physical_sessions
    BEGIN
      INSERT INTO replacement_audit (operation) VALUES ('update');
    END;
  `);
  auditDatabase.close();

  await indexChangedConversationSessions({
    sessionsDirectory,
    databasePath,
    ignoredPhysicalSessionPaths: new Set(),
    tokenizer,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
    embeddingProvider: {
      async embedQuery() {
        return createTestEmbedding(0);
      },
      async embedDocuments(documents) {
        return documents.map(() => createTestEmbedding());
      },
    },
  });

  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  t.after(() => database.close());
  assert.deepEqual(database.readCounts(), {
    physicalSessions: 1,
    sessionDocuments: 1,
    invocations: 1,
    denseDocuments: 1,
    denseGlobalVectors: 1,
    denseProjectVectors: 1,
    denseProjects: 1,
  });
  assert.equal(database.checkIntegrity().healthy, true);
  assert.equal(database.searchInvocations('/tmp/a', 5).length, 1);
  const auditReader = new DatabaseSync(databasePath, { readOnly: true });
  const replacementCount = auditReader
    .prepare('SELECT count(*) AS count FROM replacement_audit')
    .get()?.count;
  auditReader.close();
  assert.equal(replacementCount, 1);
});

void test('manual index maintenance announces planning and indexing before their work begins', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-phase-order-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'phase order evidence');
  const operationLog: string[] = [];
  const embeddingProvider = createRecordingEmbeddingProvider([]);

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath,
      store: new MemoryConversationStore(),
      embeddingProvider: {
        ...embeddingProvider,
        async embedDocuments(documents, signal) {
          operationLog.push('embed documents');
          return embeddingProvider.embedDocuments(documents, signal);
        },
      },
    }),
    onProgress(event) {
      operationLog.push(`progress: ${event.kind}`);
    },
  });

  assert.deepEqual(operationLog.slice(0, 5), [
    'progress: discovering-physical-session-files',
    'progress: planning-maintenance-workset',
    'progress: maintenance-workset-planned',
    'progress: indexing-changed-physical-session-files',
    'embed documents',
  ]);
});

void test('manual index maintenance plans exact new, changed, unchanged, and missing files', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-workset-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  const changedPath = join(sessionsDirectory, 'changed.jsonl');
  const missingPath = join(sessionsDirectory, 'missing.jsonl');
  const unchangedPath = join(sessionsDirectory, 'unchanged.jsonl');
  await writeSimplePhysicalSessionFile(changedPath, 'changed', 'original changed evidence');
  await writeSimplePhysicalSessionFile(missingPath, 'missing', 'soon missing evidence');
  await writeSimplePhysicalSessionFile(unchangedPath, 'unchanged', 'stable evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);

  await writeSimplePhysicalSessionFile(
    changedPath,
    'changed',
    'updated and longer changed evidence',
  );
  await rm(missingPath);
  await writeSimplePhysicalSessionFile(join(sessionsDirectory, 'new.jsonl'), 'new', 'new evidence');
  const events: RecallIndexProgressEvent[] = [];

  const summary = await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      events.push(event);
    },
  });

  assert.equal(summary.scannedSessions, 3);
  assert.deepEqual(
    events.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 1,
      changedFiles: 1,
      missingFiles: 1,
      ignoredRemovals: 0,
      rebuild: false,
    },
  );

  const unchangedEvents: RecallIndexProgressEvent[] = [];
  await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      unchangedEvents.push(event);
    },
  });
  assert.deepEqual(
    unchangedEvents.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: false,
    },
  );
});

void test('manual index maintenance skips ignored new files before strict parsing', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-ignore-new-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const ignoredPath = join(sessionsDirectory, 'ignored-malformed.jsonl');
  const eligiblePath = join(sessionsDirectory, 'eligible.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(ignoredPath, '{"type":"session"}\n');
  await writeSimplePhysicalSessionFile(eligiblePath, 'eligible', 'eligible exact path evidence');
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const events: RecallIndexProgressEvent[] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
  });

  const ignored = await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set([ignoredPath]),
    onProgress(event) {
      events.push(event);
    },
  });

  assert.equal(ignored.scannedSessions, 2);
  assert.equal(ignored.indexedSessions, 1);
  assert.equal(ignored.failedSessions.length, 0);
  assert.deepEqual(embeddedBatches, [['eligible exact path evidence']]);
  assert.ok([...store.chunks.values()].every((chunk) => chunk.sessionPath === eligiblePath));
  assert.deepEqual(readCatalogSessionPaths(catalogPath), [eligiblePath]);
  assert.deepEqual(
    events.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 2,
      newFiles: 1,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 0,
      rebuild: false,
    },
  );

  const unignored = await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set(),
  });
  assert.equal(unignored.indexedSessions, 0);
  assert.equal(unignored.failedSessions.length, 1);
  assert.equal(unignored.failedSessions[0]?.sessionPath, ignoredPath);
  assert.match(unignored.failedSessions[0]?.error ?? '', /session/iu);
});

void test('manual index maintenance removes ignored indexed files and reindexes after unignore', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-ignore-indexed-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'indexed then ignored evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set(),
  });
  assert.ok(readSessionDenseDocuments(catalogPath, sessionPath).size > 0);
  const events: RecallIndexProgressEvent[] = [];

  const removed = await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set([sessionPath]),
    onProgress(event) {
      events.push(event);
    },
  });

  assert.equal(removed.removedSessions, 1);
  assert.ok(removed.deletedChunks > 0);
  assert.equal(readSessionDenseDocuments(catalogPath, sessionPath).size, 0);
  assert.deepEqual(readCatalogSessionPaths(catalogPath), []);
  assert.deepEqual(
    events.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 1,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 1,
      rebuild: false,
    },
  );
  const progress = events.filter(isIndexingMaintenanceWorksetEvent);
  assert.equal(progress.at(-1)?.completedFiles, 1);
  assert.equal(progress.at(-1)?.totalFiles, 1);

  const reindexed = await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set(),
  });
  assert.equal(reindexed.indexedSessions, 1);
  assert.ok(readSessionDenseDocuments(catalogPath, sessionPath).size > 0);
  assert.deepEqual(readCatalogSessionPaths(catalogPath), [sessionPath]);
});

void test('manual index maintenance classifies an ignored missing indexed path only once', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-ignore-missing-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'ignored missing evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set(),
  });
  await rm(sessionPath);
  const events: RecallIndexProgressEvent[] = [];

  const result = await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set([sessionPath]),
    onProgress(event) {
      events.push(event);
    },
  });

  assert.equal(result.removedSessions, 1);
  assert.deepEqual(
    events.find((event) => event.kind === 'maintenance-workset-planned'),
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 0,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      ignoredRemovals: 1,
      rebuild: false,
    },
  );
});

void test('manual index maintenance resolves a relative sessions directory to absolute catalog paths', async (t) => {
  const root = await mkdtemp(join(process.cwd(), 'recall-indexer-relative-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const relativeSessionsDirectory = relative(process.cwd(), sessionsDirectory);
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'absolute state identity evidence');

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory: relativeSessionsDirectory,
      catalogPath,
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    ignoredPhysicalSessionPaths: new Set(),
  });

  assert.deepEqual(readCatalogSessionPaths(catalogPath), [sessionPath]);
});

void test('manual index maintenance reuses an active vector only for the same canonical checksum', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-active-vector-reuse-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sourceCatalogPath = join(root, 'source-catalog.sqlite');
  const targetCatalogPath = join(root, 'target-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'safe vector reuse evidence');
  const activeStore = new MemoryConversationStore();
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      catalogPath: sourceCatalogPath,
      store: activeStore,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  const activeDatabase = openSqliteRecallDatabase(sourceCatalogPath, { readOnly: true });
  t.after(() => activeDatabase.close());

  const summary = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath: targetCatalogPath,
      store: new MemoryConversationStore(),
      embeddingProvider: {
        embedQuery: async () => [],
        embedDocuments: async () => {
          throw new Error('Matching active vector should avoid embedding');
        },
      },
    }),
    vectorReuseReader: {
      fetchDocuments(ids) {
        const documents = activeDatabase.fetchDenseDocuments(ids);
        assert.equal(documents.size, ids.length);
        return documents;
      },
      fetchVectors(ids) {
        const vectors = activeDatabase.fetchDenseVectors(ids);
        assert.equal(vectors.size, ids.length);
        return vectors;
      },
    },
  });

  assert.equal(summary.newlyEmbeddedChunks, 0);
  assert.equal(summary.reusedVectors, 1);
  assert.deepEqual(
    readSessionDenseDocuments(targetCatalogPath, sessionPath),
    activeDatabase.fetchDenseDocuments(
      activeDatabase.readPhysicalSessionState(sessionPath)?.denseDocumentIds ?? [],
    ),
  );
});

void test('reused SQLite vectors refresh current branch, sibling, project, and source metadata', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-current-metadata-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sourceDatabasePath = join(root, 'source.sqlite');
  const targetDatabasePath = join(root, 'target.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(
    sessionPath,
    'one',
    'metadata reuse evidence with current sibling geometry',
  );
  const currentProjectIdentity = parseProjectIdentity('non-git-session-origin:/current');
  const commonOptions = {
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath: sourceDatabasePath,
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    chunkPolicy: { maxTokens: 3, overlapTokens: 0 },
    resolveProjectIdentity: async (): Promise<ResolvedProjectIdentity> => ({
      projectIdentity: currentProjectIdentity,
      identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
    }),
  };
  await indexChangedConversationSessions(commonOptions);
  const sourceDatabase = openSqliteRecallDatabase(sourceDatabasePath, { readOnly: true });
  t.after(() => sourceDatabase.close());
  const sourceDocumentIds =
    sourceDatabase.readPhysicalSessionState(sessionPath)?.denseDocumentIds ?? [];
  const documentId = sourceDocumentIds[0];
  assert.ok(documentId);
  const sourceDocuments = sourceDatabase.fetchDenseDocuments(sourceDocumentIds);
  const sourceVectors = sourceDatabase.fetchDenseVectors(sourceDocumentIds);
  const currentSourceDocument = sourceDocuments.get(documentId);
  const reusableVector = sourceVectors.get(documentId);
  assert.ok(currentSourceDocument);
  assert.ok(reusableVector);
  const staleProjectIdentity = parseProjectIdentity('non-git-session-origin:/stale');
  const staleReusableDocument: SessionConversationChunk = {
    ...structuredClone(currentSourceDocument),
    projectAttribution: {
      projectIdentity: staleProjectIdentity,
      identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
    },
    currentLeafId: null,
    siblingIds: ['stale-sibling'],
    previousSiblingId: 'stale-sibling',
    sourceLineStart: 99,
    sourceLineEnd: 99,
  };

  const updated = await indexChangedConversationSessions({
    ...commonOptions,
    databasePath: targetDatabasePath,
    embeddingProvider: {
      embedQuery: async () => [],
      embedDocuments: async () => {
        throw new Error('Checksum-compatible legacy vector should avoid embedding');
      },
    },
    vectorReuseReader: {
      fetchDocuments(ids) {
        const documents = new Map<string, SessionConversationChunk>();
        for (const id of ids) {
          const document = id === documentId ? staleReusableDocument : sourceDocuments.get(id);
          if (document) {
            documents.set(id, document);
          }
        }
        return documents;
      },
      fetchVectors(ids) {
        const vectors = new Map<string, number[]>();
        for (const id of ids) {
          const vector = sourceVectors.get(id);
          if (vector) {
            vectors.set(id, vector);
          }
        }
        return vectors;
      },
    },
  });

  const targetDatabase = openSqliteRecallDatabase(targetDatabasePath, { readOnly: true });
  t.after(() => targetDatabase.close());
  const currentDocument = targetDatabase.fetchDenseDocuments([documentId]).get(documentId);
  assert.ok(currentDocument);
  assert.ok(updated.reusedVectors >= 1);
  assert.deepEqual(targetDatabase.fetchDenseVectors([documentId]).get(documentId), reusableVector);
  assert.equal(currentDocument.projectAttribution?.projectIdentity, currentProjectIdentity);
  assert.deepEqual(currentDocument.currentLeafId, currentSourceDocument.currentLeafId);
  assert.deepEqual(currentDocument.siblingIds, currentSourceDocument.siblingIds);
  assert.equal(currentDocument.sourceLineStart, currentSourceDocument.sourceLineStart);
  assert.equal(currentDocument.sourceLineEnd, currentSourceDocument.sourceLineEnd);
});

void test('manual index maintenance rejects active-vector reuse after canonical content changes', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-active-vector-checksum-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'alpha');
  const activeStore = new MemoryConversationStore();
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      catalogPath: join(root, 'source-catalog.sqlite'),
      store: activeStore,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'bravo');
  const embeddedBatches: string[][] = [];

  const summary = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath: join(root, 'target-catalog.sqlite'),
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
    vectorReuseStore: activeStore,
  });

  assert.equal(summary.newlyEmbeddedChunks, 1);
  assert.equal(summary.reusedVectors, 0);
  assert.deepEqual(embeddedBatches, [['bravo']]);
});

void test('manual incremental indexing adds, reuses, changes, and deletes SQLite rows', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  const entries: object[] = [
    {
      type: 'session',
      version: 3,
      id: 'session-1',
      timestamp: '2026-07-24T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'user-1',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content: 'Remember the durable queue' },
    },
  ];
  await writeFile(sessionPath, sessionLines(entries));
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
  });

  const first = await indexChangedConversationSessions(options);
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.newlyEmbeddedChunks, 1);
  assert.equal(first.reusedVectors, 0);
  assert.equal(readSessionDenseDocuments(catalogPath, sessionPath).size, 1);

  const unchanged = await indexChangedConversationSessions(options);
  assert.equal(unchanged.indexedSessions, 0);
  assert.equal(unchanged.embeddingRequestCount, 0);
  assert.equal(embeddedBatches.length, 1);

  entries.push({
    type: 'message',
    id: 'assistant-1',
    parentId: 'user-1',
    timestamp: '2026-07-24T10:02:00Z',
    message: { role: 'assistant', content: 'Use one manually maintained zvec collection.' },
  });
  await writeFile(sessionPath, sessionLines(entries));
  const changed = await indexChangedConversationSessions(options);
  assert.equal(changed.indexedSessions, 1);
  assert.ok(changed.reusedVectors >= 1);
  assert.ok(changed.newlyEmbeddedChunks >= 1);
  assert.equal(embeddedBatches.length, 2);

  await rm(sessionPath);
  const deleted = await indexChangedConversationSessions(options);
  assert.equal(deleted.removedSessions, 1);
  assert.equal(store.chunks.size, 0);
});

void test('manual index maintenance reports cumulative progress and continues after a damaged file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const damagedPath = join(sessionsDirectory, 'a-damaged.jsonl');
  const healthyPath = join(sessionsDirectory, 'z-healthy.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(damagedPath, 'damaged', 'initial damaged evidence');
  await writeSimplePhysicalSessionFile(healthyPath, 'healthy', 'initial healthy evidence');
  const store = new MemoryConversationStore();
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);

  await writeFile(damagedPath, '{"type":"session"}\n');
  await writeSimplePhysicalSessionFile(
    healthyPath,
    'healthy',
    'updated healthy evidence with more words',
  );
  const events: RecallIndexProgressEvent[] = [];
  const result = await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      events.push(event);
    },
  });

  const warningIndex = events.findIndex((event) => event.kind === 'physical-session-file-failed');
  const laterHealthyProgressIndex = events.findIndex(
    (event) =>
      event.kind === 'indexing-maintenance-workset' &&
      event.sessionPath === healthyPath &&
      event.completedFiles === 2,
  );
  assert.ok(warningIndex >= 0);
  assert.ok(laterHealthyProgressIndex > warningIndex);
  assert.deepEqual(events[warningIndex], {
    kind: 'physical-session-file-failed',
    sessionPath: damagedPath,
  });
  assert.equal(result.failedSessions.length, 1);
  assert.match(result.failedSessions[0]?.error ?? '', /session/iu);
  assert.equal(readSessionDenseDocuments(catalogPath, damagedPath).size, 0);
  assert.ok(readSessionDenseDocuments(catalogPath, healthyPath).size > 0);

  const progressEvents = events.filter(isIndexingMaintenanceWorksetEvent);
  for (let index = 1; index < progressEvents.length; index += 1) {
    const previous = progressEvents[index - 1];
    const current = progressEvents[index];
    assert.ok(previous);
    assert.ok(current);
    assert.ok(current.completedFiles >= previous.completedFiles);
    assert.ok(current.indexedSessions >= previous.indexedSessions);
    assert.ok(current.newlyEmbeddedDocuments >= previous.newlyEmbeddedDocuments);
    assert.ok(current.reusedVectors >= previous.reusedVectors);
    assert.ok(current.deletedDocuments >= previous.deletedDocuments);
    assert.ok(current.failedSessions >= previous.failedSessions);
  }
  const finalProgress = progressEvents.at(-1);
  assert.ok(finalProgress);
  assert.equal(finalProgress.completedFiles, finalProgress.totalFiles);
  assert.equal(finalProgress.indexedSessions, result.indexedSessions);
  assert.equal(finalProgress.newlyEmbeddedDocuments, result.newlyEmbeddedChunks);
  assert.equal(finalProgress.reusedVectors, result.reusedVectors);
  assert.equal(finalProgress.deletedDocuments, result.deletedChunks);
  assert.equal(finalProgress.failedSessions, result.failedSessions.length);
});

void test('manual index maintenance reports multiple batches within one large physical session file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-batches-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const sessionPath = join(sessionsDirectory, 'large.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  const messages = Array.from({ length: 140 }, (_, index) => ({
    type: 'message',
    id: `message-${index}`,
    parentId: index === 0 ? null : `message-${index - 1}`,
    timestamp: '2026-01-01',
    message: {
      role: index % 2 === 0 ? 'user' : 'assistant',
      content: `dense evidence message ${index}`,
    },
  }));
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: 'large', timestamp: '2026-01-01', cwd: '/p' },
      ...messages,
    ]),
  );
  const events: RecallIndexProgressEvent[] = [];

  const result = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath,
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    onProgress(event) {
      events.push(event);
    },
  });

  const inFileProgress = events
    .filter(isIndexingMaintenanceWorksetEvent)
    .filter((event) => event.sessionPath === sessionPath && event.completedFiles === 0);
  assert.ok(inFileProgress.length >= 2);
  const firstBatchProgress = inFileProgress[0];
  const lastBatchProgress = inFileProgress.at(-1);
  assert.ok(firstBatchProgress);
  assert.ok(lastBatchProgress);
  assert.ok(lastBatchProgress.newlyEmbeddedDocuments > firstBatchProgress.newlyEmbeddedDocuments);
  assert.equal(result.embeddingRequestCount, inFileProgress.length);
  assert.ok(
    events.some(
      (event) =>
        event.kind === 'indexing-maintenance-workset' &&
        event.sessionPath === sessionPath &&
        event.completedFiles === 1,
    ),
  );
});

void test('manual index maintenance skips an unchanged physical session without parsing it', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-no-parse-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'unchanged.jsonl');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'unchanged', 'metadata skip evidence');
  const stableTimestamp = new Date('2026-08-09T12:00:00.000Z');
  await utimes(sessionPath, stableTimestamp, stableTimestamp);
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store: new MemoryConversationStore(),
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const originalStats = await stat(sessionPath);
  const originalBytes = await readFile(sessionPath);
  await writeFile(sessionPath, `{${' '.repeat(originalBytes.length - 1)}`);
  await utimes(sessionPath, originalStats.atime, originalStats.mtime);
  const disguisedStats = await stat(sessionPath);
  assert.equal(disguisedStats.size, originalStats.size);
  assert.equal(disguisedStats.mtimeMs, originalStats.mtimeMs);

  const skipped = await indexChangedConversationSessions(options);

  assert.equal(skipped.indexedSessions, 0);
  assert.equal(skipped.failedSessions.length, 0);
});

void test('legacy state migrates once without rewriting the corpus-wide JSON file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-migration-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'legacy.jsonl');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const legacyStatePath = join(root, 'index-state.json');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'legacy', 'legacy migration evidence');
  const sourceStats = await stat(sessionPath);
  const legacyState = `${JSON.stringify({
    version: 3,
    importPolicyVersion: SESSION_IMPORT_POLICY_VERSION,
    sessions: {
      [sessionPath]: {
        size: sourceStats.size,
        mtimeMs: sourceStats.mtimeMs,
        chunks: [],
      },
    },
  })}\n`;
  await writeFile(legacyStatePath, legacyState);
  const legacyStateInode = (await stat(legacyStatePath)).ino;
  const options = {
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath,
      store: new MemoryConversationStore(),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    legacyStatePath,
  };

  const migrated = await indexChangedConversationSessions(options);
  const unchanged = await indexChangedConversationSessions(options);

  assert.equal(migrated.indexedSessions, 1);
  assert.equal(unchanged.indexedSessions, 0);
  assert.equal(await readFile(legacyStatePath, 'utf8'), legacyState);
  assert.equal((await stat(legacyStatePath)).ino, legacyStateInode);
  const database = openSqliteRecallDatabase(catalogPath, { readOnly: true });
  assert.equal(database.requiresInvocationBackfill(sessionPath), false);
  database.close();
});

void test('a late SQLite replacement failure leaves the previous physical session intact', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-replacement-failure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'previous committed evidence');
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath: databasePath,
    store: new MemoryConversationStore(),
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const previousState = readCatalogSessionState(databasePath, sessionPath);
  const previousDocuments = readSessionDenseDocuments(databasePath, sessionPath);

  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'replacement evidence is different');
  await assert.rejects(
    indexChangedConversationSessions({
      ...options,
      embeddingProvider: {
        embedQuery: async () => [],
        embedDocuments: async (documents) => documents.map(() => [1, 0, 0]),
      },
    }),
    /dense embedding invalid/u,
  );

  assert.deepEqual(readCatalogSessionState(databasePath, sessionPath), previousState);
  assert.deepEqual(readSessionDenseDocuments(databasePath, sessionPath), previousDocuments);
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  assert.equal(database.checkIntegrity().healthy, true);
  database.close();
});

void test('completed earlier physical sessions survive a later SQLite replacement failure', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-later-failure-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const firstPath = join(sessionsDirectory, 'a-first.jsonl');
  const secondPath = join(sessionsDirectory, 'b-second.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(firstPath, 'first', 'first committed evidence');
  await writeSimplePhysicalSessionFile(secondPath, 'second', 'second rejected evidence');
  let embeddingRequest = 0;

  await assert.rejects(
    indexChangedConversationSessions({
      ...createIndexerOptions({
        sessionsDirectory,
        catalogPath: databasePath,
        store: new MemoryConversationStore(),
        embeddingProvider: {
          embedQuery: async () => [],
          async embedDocuments(documents) {
            embeddingRequest += 1;
            return documents.map(() =>
              embeddingRequest === 1 ? createTestEmbedding() : [1, 0, 0],
            );
          },
        },
      }),
    }),
    /dense embedding invalid/u,
  );

  assert.deepEqual(readCatalogSessionPaths(databasePath), [firstPath]);
  assert.ok(readSessionDenseDocuments(databasePath, firstPath).size > 0);
  assert.equal(readSessionDenseDocuments(databasePath, secondPath).size, 0);
});

void test('interrupted index maintenance commits completed sessions and resumes remaining work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-interrupted-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const firstPath = join(sessionsDirectory, 'a-first.jsonl');
  const secondPath = join(sessionsDirectory, 'b-second.jsonl');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(firstPath, 'first', 'first interruption evidence');
  await writeSimplePhysicalSessionFile(secondPath, 'second', 'second interruption evidence');
  const embeddedBatches: string[][] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store: new MemoryConversationStore(),
    embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
  });
  const abortController = new AbortController();

  await assert.rejects(
    indexChangedConversationSessions({
      ...options,
      signal: abortController.signal,
      onProgress(event) {
        if (event.kind === 'indexing-maintenance-workset' && event.completedFiles === 1) {
          abortController.abort(new Error('stop between physical sessions'));
        }
      },
    }),
    /Recall conversation indexing cancelled/u,
  );
  assert.deepEqual(readCatalogSessionPaths(catalogPath), [firstPath]);
  assert.equal(embeddedBatches.length, 1);

  const resumed = await indexChangedConversationSessions(options);
  assert.equal(resumed.indexedSessions, 1);
  assert.deepEqual(readCatalogSessionPaths(catalogPath), [firstPath, secondPath]);
  assert.equal(embeddedBatches.length, 2);
});

void test('updating one physical session leaves unrelated catalog state unchanged', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-unrelated-state-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const changedPath = join(sessionsDirectory, 'changed.jsonl');
  const unrelatedPath = join(sessionsDirectory, 'unrelated.jsonl');
  const catalogPath = join(root, 'recall-catalog.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(changedPath, 'changed', 'original changed state');
  await writeSimplePhysicalSessionFile(unrelatedPath, 'unrelated', 'stable unrelated state');
  const options = createIndexerOptions({
    sessionsDirectory,
    catalogPath,
    store: new MemoryConversationStore(),
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const unrelatedBefore = readCatalogSessionState(catalogPath, unrelatedPath);

  await writeSimplePhysicalSessionFile(
    changedPath,
    'changed',
    'updated and longer changed state evidence',
  );
  const updated = await indexChangedConversationSessions(options);

  assert.equal(updated.indexedSessions, 1);
  assert.deepEqual(readCatalogSessionState(catalogPath, unrelatedPath), unrelatedBefore);
});

void test('compact indexing keeps tool results and bash output away from tokenizer, embeddings, and dense storage', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-tool-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'tool.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    sessionLines([
      { type: 'session', version: 3, id: 'session-1', timestamp: '2026-01-01', cwd: '/p' },
      {
        type: 'message',
        id: 'assistant-1',
        parentId: null,
        timestamp: '2026-01-01',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'toolCall',
              id: 'call-1',
              name: 'read',
              arguments: {
                path: '/tmp/a',
                content: 'OMITTED_ARGUMENT_PAYLOAD_MUST_NOT_ENTER_SQLITE',
              },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-1',
        parentId: 'assistant-1',
        timestamp: '2026-01-01',
        message: {
          role: 'toolResult',
          toolCallId: 'call-1',
          toolName: 'read',
          content: [{ type: 'text', text: 'TOOL_RESULT_MUST_NOT_ENTER_SQLITE' }],
          isError: false,
        },
      },
      {
        type: 'message',
        id: 'bash-1',
        parentId: 'tool-1',
        timestamp: '2026-01-01',
        message: {
          role: 'bashExecution',
          command: 'rg compact-catalog src',
          output: 'BASH_OUTPUT_MUST_NOT_ENTER_SQLITE',
          exitCode: 0,
          cancelled: false,
        },
      },
    ]),
  );
  const store = new MemoryConversationStore();
  const embeddedBatches: string[][] = [];
  const catalogPath = join(root, 'recall-catalog.sqlite');

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      catalogPath,
      store,
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
    tokenizer: {
      encodeConversationText() {
        throw new Error('Tool-only sessions must not reach the conversation tokenizer');
      },
    },
  });

  assert.equal(embeddedBatches.length, 0);
  assert.equal(readSessionDenseDocuments(catalogPath, sessionPath).size, 0);
  const database = openSqliteRecallDatabase(catalogPath, { readOnly: true });
  assert.equal(database.searchInvocations('/tmp/a', 5).length, 1);
  assert.equal(database.searchInvocations('compact catalog', 5).length, 1);
  database.close();
  const catalogBytes = await readFile(catalogPath);
  assert.ok(!catalogBytes.includes('TOOL_RESULT_MUST_NOT_ENTER_SQLITE'));
  assert.ok(!catalogBytes.includes('BASH_OUTPUT_MUST_NOT_ENTER_SQLITE'));
  assert.ok(!catalogBytes.includes('OMITTED_ARGUMENT_PAYLOAD_MUST_NOT_ENTER_SQLITE'));
});

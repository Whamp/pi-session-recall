import assert from 'node:assert/strict';
import {
  appendFile,
  mkdtemp,
  mkdir,
  readFile,
  rm,
  stat,
  utimes,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';

import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { RecallProjectIdentitySource } from './enums.js';
import { parseProjectIdentity, type ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
} from './sqlite-recall-database.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';

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

function readRecallDatabaseSessionPaths(databasePath: string): string[] {
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  try {
    return database.listPhysicalSessionPaths();
  } finally {
    database.close();
  }
}

function readRecallDatabaseSessionState(databasePath: string, sessionPath: string) {
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
  databasePath: string;
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
    denseVectors: 1,
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
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'phase order evidence');
  const operationLog: string[] = [];
  const embeddingProvider = createRecordingEmbeddingProvider([]);

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath,
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
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  const changedPath = join(sessionsDirectory, 'changed.jsonl');
  const missingPath = join(sessionsDirectory, 'missing.jsonl');
  const unchangedPath = join(sessionsDirectory, 'unchanged.jsonl');
  await writeSimplePhysicalSessionFile(changedPath, 'changed', 'original changed evidence');
  await writeSimplePhysicalSessionFile(missingPath, 'missing', 'soon missing evidence');
  await writeSimplePhysicalSessionFile(unchangedPath, 'unchanged', 'stable evidence');
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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
  const databasePath = join(root, 'recall.sqlite');
  const ignoredPath = join(sessionsDirectory, 'ignored-malformed.jsonl');
  const eligiblePath = join(sessionsDirectory, 'eligible.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(ignoredPath, '{"type":"session"}\n');
  await writeSimplePhysicalSessionFile(eligiblePath, 'eligible', 'eligible exact path evidence');
  const embeddedBatches: string[][] = [];
  const events: RecallIndexProgressEvent[] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [eligiblePath]);
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
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'indexed then ignored evidence');
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions({
    ...options,
    ignoredPhysicalSessionPaths: new Set(),
  });
  assert.ok(readSessionDenseDocuments(databasePath, sessionPath).size > 0);
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
  assert.equal(readSessionDenseDocuments(databasePath, sessionPath).size, 0);
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), []);
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
  assert.ok(readSessionDenseDocuments(databasePath, sessionPath).size > 0);
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [sessionPath]);
});

void test('manual index maintenance classifies an ignored missing indexed path only once', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-ignore-missing-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'ignored missing evidence');
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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

void test('manual index maintenance resolves a relative sessions directory to absolute Recall database paths', async (t) => {
  const root = await mkdtemp(join(process.cwd(), 'recall-indexer-relative-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const relativeSessionsDirectory = relative(process.cwd(), sessionsDirectory);
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'absolute state identity evidence');

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory: relativeSessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    ignoredPhysicalSessionPaths: new Set(),
  });

  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [sessionPath]);
});

void test('manual index maintenance reuses an active vector only for the same canonical checksum', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-active-vector-reuse-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sourceDatabasePath = join(root, 'source-recall.sqlite');
  const targetDatabasePath = join(root, 'target-recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'safe vector reuse evidence');
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath: sourceDatabasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  const activeDatabase = openSqliteRecallDatabase(sourceDatabasePath, { readOnly: true });
  t.after(() => activeDatabase.close());

  const summary = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath: targetDatabasePath,
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
    readSessionDenseDocuments(targetDatabasePath, sessionPath),
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
      databasePath: sourceDatabasePath,
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
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath: join(root, 'source-database.sqlite'),
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  await writeSimplePhysicalSessionFile(sessionPath, 'one', 'bravo');
  const embeddedBatches: string[][] = [];

  const summary = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath: join(root, 'target-recall.sqlite'),
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
    vectorReuseReader: (() => {
      const sourceDatabase = openSqliteRecallDatabase(join(root, 'source-database.sqlite'), {
        readOnly: true,
      });
      t.after(() => sourceDatabase.close());
      return {
        fetchDocuments: (ids: string[]) => sourceDatabase.fetchDenseDocuments(ids),
        fetchVectors: (ids: string[]) => sourceDatabase.fetchDenseVectors(ids),
      };
    })(),
  });

  assert.equal(summary.newlyEmbeddedChunks, 1);
  assert.equal(summary.reusedVectors, 0);
  assert.deepEqual(embeddedBatches, [['bravo']]);
});

void test('manual incremental indexing adds, reuses, changes, and deletes SQLite rows', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
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
  const embeddedBatches: string[][] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
    embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
  });

  const first = await indexChangedConversationSessions(options);
  assert.equal(first.indexedSessions, 1);
  assert.equal(first.newlyEmbeddedChunks, 1);
  assert.equal(first.reusedVectors, 0);
  assert.equal(readSessionDenseDocuments(databasePath, sessionPath).size, 1);

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
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), []);
});

void test('selected physical-session maintenance does not process or remove unselected sessions', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-selected-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const selectedSessionPath = join(sessionsDirectory, 'selected.jsonl');
  const unselectedSessionPath = join(sessionsDirectory, 'unselected.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(
    selectedSessionPath,
    'selected',
    'initial selected evidence',
  );
  await writeSimplePhysicalSessionFile(
    unselectedSessionPath,
    'unselected',
    'initial unselected evidence',
  );
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const unselectedStateBefore = readRecallDatabaseSessionState(databasePath, unselectedSessionPath);

  await writeSimplePhysicalSessionFile(
    selectedSessionPath,
    'selected',
    'updated selected evidence',
  );
  await rm(unselectedSessionPath);
  const summary = await indexChangedConversationSessions({
    ...options,
    selectedPhysicalSessionPaths: [selectedSessionPath],
  });

  assert.equal(summary.scannedSessions, 1);
  assert.equal(summary.indexedSessions, 1);
  assert.equal(summary.removedSessions, 0);
  assert.deepEqual(
    readRecallDatabaseSessionState(databasePath, unselectedSessionPath),
    unselectedStateBefore,
  );
});

void test('manual index maintenance reports cumulative progress and continues after a damaged file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const damagedPath = join(sessionsDirectory, 'a-damaged.jsonl');
  const healthyPath = join(sessionsDirectory, 'z-healthy.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(damagedPath, 'damaged', 'initial damaged evidence');
  await writeSimplePhysicalSessionFile(healthyPath, 'healthy', 'initial healthy evidence');
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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
  assert.equal(readSessionDenseDocuments(databasePath, damagedPath).size, 0);
  assert.ok(readSessionDenseDocuments(databasePath, healthyPath).size > 0);

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

void test('manual index maintenance profiles each changed physical session file by phase', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-profile-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'profiled.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'profiled', 'initial profile evidence');
  let monotonicTime = 0;
  const options = {
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    monotonicNow() {
      monotonicTime += 5;
      return monotonicTime;
    },
  };
  await indexChangedConversationSessions(options);
  const previousSourceBytes = (await stat(sessionPath)).size;

  await writeSimplePhysicalSessionFile(
    sessionPath,
    'profiled',
    'updated profile evidence with appended words',
  );
  const events: RecallIndexProgressEvent[] = [];
  const result = await indexChangedConversationSessions({
    ...options,
    onProgress(event) {
      events.push(event);
    },
  });

  const profileEvents = events.filter((event) => event.kind === 'physical-session-file-profiled');
  assert.equal(profileEvents.length, 1);
  const profile = profileEvents[0];
  assert.ok(profile?.kind === 'physical-session-file-profiled');
  assert.equal(profile.sessionPath, sessionPath);
  assert.equal(profile.change, 'changed');
  assert.equal(profile.indexedSourceBytesBefore, previousSourceBytes);
  assert.equal(profile.sourceBytesAtPlanning, (await stat(sessionPath)).size);
  assert.equal(profile.newlyEmbeddedDocuments, result.newlyEmbeddedChunks);
  assert.equal(profile.reusedVectors, result.reusedVectors);
  assert.ok(profile.denseDocuments > 0);
  assert.ok(profile.totalElapsedMilliseconds > 0);
  assert.deepEqual(Object.keys(profile.phaseElapsedMilliseconds).sort(), [
    'documentConstructionTokenization',
    'embedding',
    'graphValidation',
    'readParse',
    'sqliteReplacement',
    'vectorLookup',
  ]);
  for (const elapsedMilliseconds of Object.values(profile.phaseElapsedMilliseconds)) {
    assert.ok(elapsedMilliseconds > 0);
  }
  assert.deepEqual(Object.keys(profile.documentPhaseElapsedMilliseconds).sort(), [
    'conversationChunkTokenization',
    'metadataInvocationProjectAttribution',
    'pendingAtomicSummaryDocuments',
    'turnContextConstructionBudgetSplitting',
  ]);
  for (const elapsedMilliseconds of Object.values(profile.documentPhaseElapsedMilliseconds)) {
    assert.ok(elapsedMilliseconds > 0);
  }
  assert.doesNotMatch(JSON.stringify(profile), /updated profile evidence/iu);
});

void test('one physical session embeds identical chunk content once without removing occurrences or overlap', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-identical-embedding-inputs-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  const sessionPath = join(sessionsDirectory, 'repeated-content.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  const repeatedContent = Array.from({ length: 600 }, (_, index) => `token-${index}`).join(' ');
  await writeFile(
    sessionPath,
    sessionLines([
      {
        type: 'session',
        version: 3,
        id: 'repeated-content',
        timestamp: '2026-01-01',
        cwd: '/p',
      },
      {
        type: 'custom_message',
        id: 'first-copy',
        parentId: null,
        timestamp: '2026-01-01',
        role: 'custom',
        content: repeatedContent,
        display: true,
      },
      {
        type: 'custom_message',
        id: 'second-copy',
        parentId: 'first-copy',
        timestamp: '2026-01-01',
        role: 'custom',
        content: repeatedContent,
        display: true,
      },
    ]),
  );
  const fullImport = await readSessionConversationImport(sessionPath, {
    tokenizer,
    maxTokens: 512,
    overlapTokens: 64,
  });
  assert.equal(fullImport.chunks.length, 4);
  assert.equal(new Set(fullImport.chunks.map((chunk) => chunk.content)).size, 2);
  assert.equal(fullImport.chunks.filter((chunk) => chunk.overlapTokenCount > 0).length, 2);
  const embeddedBatches: string[][] = [];

  const summary = await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
  );

  const persistedDocuments = readSessionDenseDocuments(databasePath, sessionPath);
  assert.equal(persistedDocuments.size, 4);
  assert.deepEqual(
    new Set(Array.from(persistedDocuments.values()).map((document) => document.id)),
    new Set(fullImport.chunks.map((chunk) => chunk.id)),
  );
  assert.equal(
    Array.from(persistedDocuments.values()).filter((document) => document.overlapTokenCount > 0)
      .length,
    2,
  );
  assert.equal(embeddedBatches.flat().length, 2);
  assert.equal(new Set(embeddedBatches.flat()).size, 2);
  assert.equal(summary.newlyEmbeddedChunks, 2);
  assert.equal(summary.reusedVectors, 2);
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  t.after(() => database.close());
  const persistedVectors = database.fetchDenseVectors(Array.from(persistedDocuments.keys()));
  for (const documents of Map.groupBy(
    persistedDocuments.values(),
    (document) => document.content,
  ).values()) {
    assert.equal(documents.length, 2);
    assert.deepEqual(
      persistedVectors.get(documents[0]?.id ?? ''),
      persistedVectors.get(documents[1]?.id ?? ''),
    );
  }
});

void test('identical embedding input reuse spans batches but remains scoped to one physical session', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-bounded-embedding-inputs-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  const repeatedContent = 'repeated compaction-like evidence';
  for (const sessionId of ['first-session', 'second-session']) {
    const repeatedMessages = Array.from({ length: 130 }, (_, index) => ({
      type: 'custom_message',
      id: `${sessionId}-copy-${index}`,
      parentId: index === 0 ? null : `${sessionId}-copy-${index - 1}`,
      timestamp: '2026-01-01',
      role: 'custom',
      content: repeatedContent,
      display: true,
    }));
    await writeFile(
      join(sessionsDirectory, `${sessionId}.jsonl`),
      sessionLines([
        { type: 'session', version: 3, id: sessionId, timestamp: '2026-01-01', cwd: '/p' },
        ...repeatedMessages,
      ]),
    );
  }
  const embeddedBatches: string[][] = [];

  const summary = await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
  );

  assert.deepEqual(embeddedBatches, [[repeatedContent], [repeatedContent]]);
  assert.equal(summary.indexedSessions, 2);
  assert.equal(summary.newlyEmbeddedChunks, 2);
  assert.equal(summary.reusedVectors, 258);
  for (const sessionId of ['first-session', 'second-session']) {
    assert.equal(
      readSessionDenseDocuments(databasePath, join(sessionsDirectory, `${sessionId}.jsonl`)).size,
      130,
    );
  }
});

void test('manual index maintenance reports multiple batches within one large physical session file', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-batches-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const databasePath = join(root, 'recall.sqlite');
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
      databasePath,
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
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'unchanged', 'metadata skip evidence');
  const stableTimestamp = new Date('2026-08-09T12:00:00.000Z');
  await utimes(sessionPath, stableTimestamp, stableTimestamp);
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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

  const database = openSqliteRecallDatabase(databasePath);
  t.after(() => database.close());
  let completeStateReads = 0;
  const planningDatabase = {
    ...database,
    readPhysicalSessionState(path: string) {
      completeStateReads += 1;
      return database.readPhysicalSessionState(path);
    },
  };
  const skipped = await indexChangedConversationSessions({
    ...options,
    database: planningDatabase,
  });

  assert.equal(skipped.indexedSessions, 0);
  assert.equal(skipped.failedSessions.length, 0);
  assert.equal(completeStateReads, 0);
});

void test('metadata-only dirty detection skips parsing, tokenization, embedding, and replacement', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-content-identical-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'content-identical.jsonl');
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(
    sessionPath,
    'content-identical',
    'content-identical source evidence',
  );
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  const previousState = readRecallDatabaseSessionState(databasePath, sessionPath);
  assert.ok(previousState);
  assert.match(previousState.sourceSha256, /^[a-f0-9]{64}$/u);
  const changedTimestamp = new Date(previousState.mtimeMs + 10_000);
  await utimes(sessionPath, changedTimestamp, changedTimestamp);
  const progressEvents: RecallIndexProgressEvent[] = [];

  const result = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: {
        async embedQuery() {
          throw new Error('Content-identical fast path must not embed a query');
        },
        async embedDocuments() {
          throw new Error('Content-identical fast path must not embed documents');
        },
      },
    }),
    tokenizer: {
      encodeConversationText() {
        throw new Error('Content-identical fast path must not tokenize');
      },
    },
    onProgress(event) {
      progressEvents.push(event);
    },
  });

  assert.equal(result.indexedSessions, 1);
  assert.equal(result.newlyEmbeddedChunks, 0);
  assert.equal(result.reusedVectors, 0);
  const currentState = readRecallDatabaseSessionState(databasePath, sessionPath);
  assert.ok(currentState);
  assert.equal(currentState.sourceSha256, previousState.sourceSha256);
  assert.equal(currentState.size, previousState.size);
  assert.ok(currentState.mtimeMs > previousState.mtimeMs);
  const profile = progressEvents.find((event) => event.kind === 'physical-session-file-profiled');
  assert.ok(profile?.kind === 'physical-session-file-profiled');
  assert.equal(profile.phaseElapsedMilliseconds.graphValidation, 0);
  assert.equal(profile.phaseElapsedMilliseconds.documentConstructionTokenization, 0);
  assert.deepEqual(Object.values(profile.documentPhaseElapsedMilliseconds), [0, 0, 0, 0]);
});

void test('changed session reuses old token geometry and equals a clean full import', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-projection-cache-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'projection-cache.jsonl');
  const databasePath = join(root, 'recall.sqlite');
  const oldContent = 'historical token geometry must be reused';
  const newContent = 'new custom projection input';
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'projection-cache', oldContent);
  await indexChangedConversationSessions(
    createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
  );
  await appendFile(
    sessionPath,
    `${JSON.stringify({
      type: 'custom_message',
      id: 'custom-appended',
      parentId: 'user-projection-cache',
      timestamp: '2026-01-01',
      role: 'custom',
      content: newContent,
      display: true,
    })}\n`,
  );
  const encodedTexts: string[] = [];
  const cachingTokenizer: ConversationTextTokenizer = {
    encodeConversationText(text) {
      encodedTexts.push(text);
      if (text.includes(oldContent)) {
        throw new Error('Projection cache retokenized unchanged historical content');
      }
      return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
    },
  };

  const result = await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider([]),
    }),
    tokenizer: cachingTokenizer,
  });

  assert.equal(result.failedSessions.length, 0);
  assert.ok(encodedTexts.some((text) => text.includes(newContent)));
  const optimizedDocuments = readSessionDenseDocuments(databasePath, sessionPath);
  const fullImport = await readSessionConversationImport(sessionPath, {
    tokenizer,
    maxTokens: 512,
    overlapTokens: 64,
  });
  const expectedDocuments = new Map(
    fullImport.chunks
      .filter((chunk) => chunk.isDenseSearchable)
      .map((chunk) => [chunk.id, chunk] as const),
  );
  assert.deepEqual(optimizedDocuments, expectedDocuments);
  const historicalDocument = Array.from(optimizedDocuments.values()).find((document) =>
    document.content.includes(oldContent),
  );
  assert.deepEqual(
    historicalDocument?.childEntryIds.map((entryId) => entryId.value),
    ['custom-appended'],
  );
});

void test('mismatched projection cache state falls back to current tokenization', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-projection-cache-miss-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'projection-cache-miss.jsonl');
  const databasePath = join(root, 'recall.sqlite');
  const oldContent = 'cache mismatch must retokenize this history';
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(sessionPath, 'projection-cache-miss', oldContent);
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const rawDatabase = new DatabaseSync(databasePath);
  rawDatabase
    .prepare('UPDATE conversation_projection_input_documents SET input_checksum = ?')
    .run('f'.repeat(64));
  rawDatabase.close();
  await appendFile(
    sessionPath,
    `${JSON.stringify({
      type: 'custom_message',
      id: 'cache-miss-appended',
      parentId: 'user-projection-cache-miss',
      timestamp: '2026-01-01',
      content: 'cache miss appended evidence',
      display: true,
    })}\n`,
  );
  const encodedTexts: string[] = [];

  const result = await indexChangedConversationSessions({
    ...options,
    tokenizer: {
      encodeConversationText(text) {
        encodedTexts.push(text);
        return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
      },
    },
  });

  assert.equal(result.failedSessions.length, 0);
  assert.ok(encodedTexts.some((text) => text.includes(oldContent)));
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  assert.equal(database.checkIntegrity().healthy, true);
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
    databasePath: databasePath,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const previousState = readRecallDatabaseSessionState(databasePath, sessionPath);
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

  assert.deepEqual(readRecallDatabaseSessionState(databasePath, sessionPath), previousState);
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
        databasePath: databasePath,
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

  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [firstPath]);
  assert.ok(readSessionDenseDocuments(databasePath, firstPath).size > 0);
  assert.equal(readSessionDenseDocuments(databasePath, secondPath).size, 0);
});

void test('interrupted index maintenance commits completed sessions and resumes remaining work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-interrupted-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const firstPath = join(sessionsDirectory, 'a-first.jsonl');
  const secondPath = join(sessionsDirectory, 'b-second.jsonl');
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(firstPath, 'first', 'first interruption evidence');
  await writeSimplePhysicalSessionFile(secondPath, 'second', 'second interruption evidence');
  const embeddedBatches: string[][] = [];
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
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
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [firstPath]);
  assert.equal(embeddedBatches.length, 1);

  const resumed = await indexChangedConversationSessions(options);
  assert.equal(resumed.indexedSessions, 1);
  assert.deepEqual(readRecallDatabaseSessionPaths(databasePath), [firstPath, secondPath]);
  assert.equal(embeddedBatches.length, 2);
});

void test('updating one physical session leaves unrelated Recall database state unchanged', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-indexer-unrelated-state-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const changedPath = join(sessionsDirectory, 'changed.jsonl');
  const unrelatedPath = join(sessionsDirectory, 'unrelated.jsonl');
  const databasePath = join(root, 'recall.sqlite');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeSimplePhysicalSessionFile(changedPath, 'changed', 'original changed state');
  await writeSimplePhysicalSessionFile(unrelatedPath, 'unrelated', 'stable unrelated state');
  const options = createIndexerOptions({
    sessionsDirectory,
    databasePath,
    embeddingProvider: createRecordingEmbeddingProvider([]),
  });
  await indexChangedConversationSessions(options);
  const unrelatedBefore = readRecallDatabaseSessionState(databasePath, unrelatedPath);

  await writeSimplePhysicalSessionFile(
    changedPath,
    'changed',
    'updated and longer changed state evidence',
  );
  const updated = await indexChangedConversationSessions(options);

  assert.equal(updated.indexedSessions, 1);
  assert.deepEqual(readRecallDatabaseSessionState(databasePath, unrelatedPath), unrelatedBefore);
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
  const embeddedBatches: string[][] = [];
  const databasePath = join(root, 'recall.sqlite');

  await indexChangedConversationSessions({
    ...createIndexerOptions({
      sessionsDirectory,
      databasePath,
      embeddingProvider: createRecordingEmbeddingProvider(embeddedBatches),
    }),
    tokenizer: {
      encodeConversationText() {
        throw new Error('Tool-only sessions must not reach the conversation tokenizer');
      },
    },
  });

  assert.equal(embeddedBatches.length, 0);
  assert.equal(readSessionDenseDocuments(databasePath, sessionPath).size, 0);
  const database = openSqliteRecallDatabase(databasePath, { readOnly: true });
  assert.equal(database.searchInvocations('/tmp/a', 5).length, 1);
  assert.equal(database.searchInvocations('compact catalog', 5).length, 1);
  database.close();
  const catalogBytes = await readFile(databasePath);
  assert.ok(!catalogBytes.includes('TOOL_RESULT_MUST_NOT_ENTER_SQLITE'));
  assert.ok(!catalogBytes.includes('BASH_OUTPUT_MUST_NOT_ENTER_SQLITE'));
  assert.ok(!catalogBytes.includes('OMITTED_ARGUMENT_PAYLOAD_MUST_NOT_ENTER_SQLITE'));
});

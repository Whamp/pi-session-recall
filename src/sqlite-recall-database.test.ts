import assert from 'node:assert/strict';
import { spawn, type ChildProcess } from 'node:child_process';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import test from 'node:test';
import { DatabaseSync } from 'node:sqlite';

import { RecallProjectIdentitySource } from './enums.js';
import type { InvocationRecord } from './createSessionInvocationRecords.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
} from './sqlite-recall-database.js';

const TEST_SOURCE_SHA256 = 'a'.repeat(64);

function createUnitEmbedding(componentIndex: number): number[] {
  const embedding = Array.from({ length: SQLITE_RECALL_EMBEDDING_DIMENSIONS }, () => 0);
  embedding[componentIndex] = 1;
  return embedding;
}

function isSqliteWriterTransactionActive(databasePath: string): boolean {
  const probe = new DatabaseSync(databasePath, { timeout: 0 });
  try {
    probe.exec('BEGIN IMMEDIATE');
    probe.exec('ROLLBACK');
    return false;
  } catch (error) {
    if (
      readNodeErrorCode(error) === 'SQLITE_BUSY' ||
      (isUnknownRecord(error) && error.errcode === 5)
    ) {
      return true;
    }
    throw error;
  } finally {
    probe.close();
  }
}

async function waitForSqliteWriterTransaction(databasePath: string): Promise<void> {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    if (isSqliteWriterTransactionActive(databasePath)) {
      return;
    }
    await sleep(2);
  }
  throw new Error('SQLite Recall test writer did not open its replacement transaction');
}

function waitForChildProcess(child: ChildProcess): Promise<{
  code: number | null;
  signal: NodeJS.Signals | null;
}> {
  return new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({ code, signal }));
  });
}

function createInvocationRecord(
  sessionPath: string,
  overrides: Partial<InvocationRecord> = {},
): InvocationRecord {
  return {
    kind: 'tool_call',
    toolName: 'read',
    toolCallId: 'call-1',
    sessionPath,
    sessionId: 'session-1',
    entryId: 'assistant-1',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockIndex: 0,
    timestamp: '2026-08-10T12:00:00Z',
    sessionOrigin: '/project',
    projectAttribution: null,
    isError: false,
    searchableText: 'tool="read"\npath="/project/src/sqlite-recall-database.ts"',
    ...overrides,
  };
}

void test('SQLite Recall database exposes its production schema and runtime identity', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-identity-'));
  const databasePath = join(directory, 'recall.sqlite');
  t.after(() => rm(directory, { recursive: true, force: true }));

  const writer = openSqliteRecallDatabase(databasePath);
  const { sqliteVersion, ...writerIdentity } = writer.identity;
  assert.match(sqliteVersion, /^3\.\d+\.\d+$/u);
  assert.deepEqual(writerIdentity, {
    schemaVersion: 4,
    storageLayout: 'unified-sqlite-vec',
    embeddingDimensions: 1_024,
    vectorEncoding: 'float32',
    distanceMetric: 'cosine',
    projectBucketCount: 16,
    sqliteVecVersion: 'v0.1.9',
    journalMode: 'wal',
    queryOnly: false,
  });
  writer.close();

  const schemaProbe = new DatabaseSync(databasePath, { readOnly: true });
  const vectorTables = schemaProbe
    .prepare(
      "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%USING vec0%' ORDER BY name",
    )
    .all()
    .map((row) => row.name);
  schemaProbe.close();
  assert.deepEqual(vectorTables, ['dense_vectors']);

  const reader = openSqliteRecallDatabase(databasePath, { readOnly: true });
  t.after(() => reader.close());
  assert.deepEqual(reader.identity, { ...writer.identity, queryOnly: true });
  assert.equal(reader.checkIntegrity().invocationFtsIntegrityChecked, false);
  assert.equal(reader.checkIntegrity().healthy, true);
  assert.throws(
    () => reader.deletePhysicalSession('/sessions/read-only.jsonl'),
    /SQLite Recall database is read-only: deletePhysicalSession is unavailable/u,
  );
});

void test('SQLite Recall database rejects incompatible schema and runtime identity with a rebuild action', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-incompatible-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const incompatibleVersionPath = join(directory, 'incompatible-version.sqlite');
  const incompatibleVersion = new DatabaseSync(incompatibleVersionPath);
  incompatibleVersion.exec('PRAGMA user_version = 99');
  incompatibleVersion.close();
  assert.throws(
    () => openSqliteRecallDatabase(incompatibleVersionPath),
    /SQLite Recall database schema incompatible.*found version 99.*psr index --rebuild/u,
  );

  const missingIdentityPath = join(directory, 'missing-identity.sqlite');
  const missingIdentity = new DatabaseSync(missingIdentityPath);
  missingIdentity.exec('PRAGMA user_version = 4');
  missingIdentity.close();
  assert.throws(
    () => openSqliteRecallDatabase(missingIdentityPath),
    /SQLite Recall database identity incompatible.*psr index --rebuild/u,
  );

  const nonWalPath = join(directory, 'non-wal.sqlite');
  openSqliteRecallDatabase(nonWalPath).close();
  const nonWal = new DatabaseSync(nonWalPath);
  nonWal.exec('PRAGMA journal_mode = DELETE');
  nonWal.close();
  assert.throws(
    () => openSqliteRecallDatabase(nonWalPath, { readOnly: true }),
    /SQLite Recall database identity incompatible.*journal mode.*psr index --rebuild/u,
  );
});

void test('SQLite Recall database losslessly round-trips dense metadata and vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-round-trip-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const sessionPath = '/sessions/round-trip.jsonl';
  const projectIdentity = parseRepositoryIdentity('git-origin:github.com/Whamp/round-trip');
  const document = createTestSessionConversationChunk({
    id: 'dense-round-trip',
    schemaVersion: 8,
    documentKind: 'summary',
    summaryKind: 'branch',
    evidenceKind: 'branch_summary',
    sessionPath,
    parentSessionPath: '/sessions/parent.jsonl',
    cwd: '/worktrees/round-trip',
    projectPath: '/projects/round-trip',
    projectAttribution: {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    parentEntryId: { value: 'entry-parent' },
    childEntryIds: [{ value: 'entry-child-a' }, { value: 'entry-child-b' }],
    contributingEntryIds: [{ value: 'entry-source-a' }, { value: 'entry-source-b' }],
    branchPathLeafIds: [{ value: 'leaf-a' }, { value: 'leaf-b' }],
    compactedByEntryIds: [{ value: 'compaction-a' }],
    compactionFirstKeptEntryId: { value: 'entry-kept' },
    branchSummaryFromEntryId: { value: 'entry-summary-source' },
    role: 'summary',
    sourceLineStart: 14,
    sourceLineEnd: 19,
    sourceBlockStart: null,
    sourceBlockEnd: null,
    siblingIds: ['dense-before', 'dense-after'],
    previousSiblingId: 'dense-before',
    nextSiblingId: 'dense-after',
    content: 'Lossless branch summary metadata.',
  });
  const embedding = createUnitEmbedding(17);

  database.replacePhysicalSession({
    sessionPath,
    size: 12_345,
    mtimeMs: 67_890.5,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [
      {
        projectionInputId: 'b'.repeat(40),
        inputChecksum: 'c'.repeat(64),
        documents: [document],
      },
    ],
    documentIds: ['compatibility-only', document.id],
    denseDocuments: [document],
    denseEmbeddings: new Map([[document.id, embedding]]),
    invocations: [],
  });

  assert.deepEqual(database.readPhysicalSessionState(sessionPath), {
    size: 12_345,
    mtimeMs: 67_890.5,
    sourceSha256: TEST_SOURCE_SHA256,
    invocationCount: 0,
    documentIds: ['compatibility-only', document.id],
    denseDocumentIds: [document.id],
  });
  assert.deepEqual(
    database.readConversationProjectionInputs(sessionPath),
    new Map([
      [
        'b'.repeat(40),
        {
          projectionInputId: 'b'.repeat(40),
          inputChecksum: 'c'.repeat(64),
          documents: [document],
        },
      ],
    ]),
  );
  assert.deepEqual(database.listPhysicalSessionPaths(), [sessionPath]);
  assert.equal(database.requiresInvocationBackfill(sessionPath), false);
  assert.deepEqual(database.fetchDenseDocuments([document.id]), new Map([[document.id, document]]));
  assert.deepEqual(database.fetchDenseVectors([document.id]), new Map([[document.id, embedding]]));
});

void test('SQLite Recall database routes global and pre-k exact project dense search', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-routing-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const sessionPath = '/sessions/routing.jsonl';
  const targetProject = parseRepositoryIdentity('git-origin:github.com/Whamp/project-target');
  const collisionProject = parseRepositoryIdentity(
    'git-origin:github.com/Whamp/project-bucket-collision',
  );
  const target = createTestSessionConversationChunk({
    id: 'target-project-result',
    sessionPath,
    projectAttribution: {
      projectIdentity: targetProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
  });
  const otherDocuments = Array.from({ length: 16 }, (_, index) => {
    const projectIdentity =
      index === 15
        ? collisionProject
        : parseRepositoryIdentity(`git-origin:github.com/Whamp/project-${index}`);
    return createTestSessionConversationChunk({
      id: index === 15 ? 'closer-same-bucket-result' : `other-project-${index}`,
      sessionPath,
      projectAttribution: {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      },
    });
  });
  const targetEmbedding = createUnitEmbedding(0);
  targetEmbedding[0] = 0.8;
  targetEmbedding[1] = 0.6;
  const denseEmbeddings = new Map<string, readonly number[]>([[target.id, targetEmbedding]]);
  for (const [index, document] of otherDocuments.entries()) {
    denseEmbeddings.set(document.id, createUnitEmbedding(index === 15 ? 0 : 2));
  }

  database.replacePhysicalSession({
    sessionPath,
    size: 1,
    mtimeMs: 2,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [target.id, ...otherDocuments.map(({ id }) => id)],
    denseDocuments: [target, ...otherDocuments],
    denseEmbeddings,
    invocations: [],
  });

  assert.equal(
    database.searchDenseCandidates(createUnitEmbedding(0), 1)[0]?.id,
    otherDocuments[15]?.id,
  );
  const projectResults = database.searchDenseCandidates(createUnitEmbedding(0), 1, targetProject);
  assert.equal(projectResults[0]?.id, target.id);
  assert.ok(Math.abs((projectResults[0]?.cosineDistance ?? 1) - 0.2) < 0.000_001);
  assert.deepEqual(
    database.searchDenseCandidates(
      createUnitEmbedding(0),
      1,
      parseRepositoryIdentity('git-origin:github.com/Whamp/project-missing'),
    ),
    [],
  );
});

void test('SQLite Recall database rejects zero vectors and tool rows', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-dense-rejection-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const sessionPath = '/sessions/rejected.jsonl';
  const conversation = createTestSessionConversationChunk({ id: 'zero-vector', sessionPath });

  assert.throws(
    () =>
      database.replacePhysicalSession({
        sessionPath,
        size: 1,
        mtimeMs: 2,
        sourceSha256: TEST_SOURCE_SHA256,
        conversationProjectionInputs: [],
        documentIds: [conversation.id],
        denseDocuments: [conversation],
        denseEmbeddings: new Map([
          [conversation.id, Array.from({ length: SQLITE_RECALL_EMBEDDING_DIMENSIONS }, () => 0)],
        ]),
        invocations: [],
      }),
    /zero vectors are not allowed/u,
  );

  const toolDocument = createTestSessionConversationChunk({
    id: 'tool-row',
    sessionPath,
    documentKind: 'tool',
    evidenceKind: 'tool_call',
    evidencePart: 'name',
    role: 'tool',
    toolCallId: 'call-1',
    toolName: 'read',
  });
  assert.throws(
    () =>
      database.replacePhysicalSession({
        sessionPath,
        size: 1,
        mtimeMs: 2,
        sourceSha256: TEST_SOURCE_SHA256,
        conversationProjectionInputs: [],
        documentIds: [toolDocument.id],
        denseDocuments: [toolDocument],
        denseEmbeddings: new Map([[toolDocument.id, createUnitEmbedding(0)]]),
        invocations: [],
      }),
    /only conversation, summary, branch-summary, and turn-context documents are allowed/u,
  );
  const disguisedToolResult = createTestSessionConversationChunk({
    id: 'disguised-tool-result',
    sessionPath,
    evidencePart: 'result',
  });
  assert.throws(
    () =>
      database.replacePhysicalSession({
        sessionPath,
        size: 1,
        mtimeMs: 2,
        sourceSha256: TEST_SOURCE_SHA256,
        conversationProjectionInputs: [],
        documentIds: [disguisedToolResult.id],
        denseDocuments: [disguisedToolResult],
        denseEmbeddings: new Map([[disguisedToolResult.id, createUnitEmbedding(0)]]),
        invocations: [],
      }),
    /only conversation, summary, branch-summary, and turn-context documents are allowed/u,
  );
  assert.equal(database.readPhysicalSessionState(sessionPath), null);
});

void test('SQLite Recall database searches compact Invocations and reports projection counts', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-invocations-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const alphaProject = parseRepositoryIdentity('git-origin:github.com/Whamp/invocation-alpha');
  const betaProject = parseRepositoryIdentity('git-origin:github.com/Whamp/invocation-beta');

  for (const [index, projectIdentity] of [alphaProject, betaProject].entries()) {
    const sessionPath = `/sessions/invocation-${index}.jsonl`;
    const document = createTestSessionConversationChunk({
      id: `invocation-dense-${index}`,
      sessionPath,
      projectAttribution: {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      },
    });
    database.replacePhysicalSession({
      sessionPath,
      size: 100 + index,
      mtimeMs: 200 + index,
      sourceSha256: TEST_SOURCE_SHA256,
      conversationProjectionInputs: [],
      documentIds: [document.id],
      denseDocuments: [document],
      denseEmbeddings: new Map([[document.id, createUnitEmbedding(index)]]),
      invocations: [
        createInvocationRecord(sessionPath, {
          toolCallId: `call-${index}`,
          entryId: `assistant-${index}`,
          sourceLineStart: 10 + index,
          sourceLineEnd: 10 + index,
          projectAttribution: {
            projectIdentity,
            identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
          },
          searchableText: `tool="read"\npath="/project/src/compact-database-${index}.ts"`,
        }),
      ],
    });
  }

  assert.deepEqual(
    database.searchInvocations('compact database', 10).map((match) => ({
      sessionPath: match.sessionPath,
      entryId: match.entryId,
      sourceLineStart: match.sourceLineStart,
    })),
    [
      { sessionPath: '/sessions/invocation-0.jsonl', entryId: 'assistant-0', sourceLineStart: 10 },
      { sessionPath: '/sessions/invocation-1.jsonl', entryId: 'assistant-1', sourceLineStart: 11 },
    ],
  );
  const projectMatches = database.searchInvocations('compact database', 10, betaProject);
  assert.equal(projectMatches.length, 1);
  assert.equal(projectMatches[0]?.sessionPath, '/sessions/invocation-1.jsonl');
  assert.equal(projectMatches[0]?.projectAttribution?.projectIdentity, betaProject);
  assert.deepEqual(database.readCounts(), {
    physicalSessions: 2,
    sessionDocuments: 2,
    invocations: 2,
    denseDocuments: 2,
    denseVectors: 2,
    denseProjects: 2,
  });
});

void test('one physical-session replacement commits every Recall projection together', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-complete-replacement-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const sessionPath = '/sessions/complete.jsonl';
  const projectIdentity = parseRepositoryIdentity('git-origin:github.com/Whamp/complete');
  const document = createTestSessionConversationChunk({
    id: 'complete-dense-document',
    sessionPath,
    projectAttribution: {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    content: 'Atomic database projection evidence',
  });
  const embedding = createUnitEmbedding(23);
  const invocation = createInvocationRecord(sessionPath, {
    projectAttribution: {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    searchableText: 'tool="read"\npath="/project/atomic-projection.ts"',
  });

  database.replacePhysicalSession({
    sessionPath,
    size: 55,
    mtimeMs: 66,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [document.id],
    denseDocuments: [document],
    denseEmbeddings: new Map([[document.id, embedding]]),
    invocations: [invocation],
  });

  assert.deepEqual(database.readPhysicalSessionState(sessionPath), {
    size: 55,
    mtimeMs: 66,
    sourceSha256: TEST_SOURCE_SHA256,
    invocationCount: 1,
    documentIds: [document.id],
    denseDocumentIds: [document.id],
  });
  assert.deepEqual(database.fetchDenseDocuments([document.id]), new Map([[document.id, document]]));
  assert.deepEqual(database.fetchDenseVectors([document.id]), new Map([[document.id, embedding]]));
  assert.equal(database.searchDenseCandidates(embedding, 1, projectIdentity)[0]?.id, document.id);
  assert.equal(
    database.searchInvocations('atomic projection', 1, projectIdentity)[0]?.entryId,
    invocation.entryId,
  );
  assert.deepEqual(database.checkIntegrity(), {
    sqliteIntegrity: ['ok'],
    foreignKeyViolations: 0,
    invocationFtsIntegrityChecked: true,
    invocationFtsDocuments: 1,
    invocationsMissingFts: 0,
    ftsDocumentsMissingInvocation: 0,
    projectionInputsMissingDenseDocument: 0,
    vectorParity: {
      denseDocuments: 1,
      vectors: 1,
      denseDocumentsMissingVector: 0,
      vectorsMissingDenseDocument: 0,
      projectMetadataMismatches: 0,
      healthy: true,
    },
    healthy: true,
  });
});

void test('read-only integrity detects missing Invocation FTS document rows', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-fts-integrity-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const databasePath = join(directory, 'recall.sqlite');
  const writer = openSqliteRecallDatabase(databasePath);
  const sessionPath = '/sessions/fts-integrity.jsonl';
  writer.replacePhysicalSession({
    sessionPath,
    size: 10,
    mtimeMs: 20,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [],
    denseDocuments: [],
    denseEmbeddings: new Map(),
    invocations: [createInvocationRecord(sessionPath)],
  });
  writer.close();

  const corruption = new DatabaseSync(databasePath);
  const invocation = corruption
    .prepare('SELECT invocation_id, tool_name, searchable_text FROM invocations')
    .get();
  const invocationId = invocation?.invocation_id;
  const toolName = invocation?.tool_name;
  const searchableText = invocation?.searchable_text;
  if (invocationId === undefined || toolName === undefined || searchableText === undefined) {
    throw new Error('SQLite Recall FTS integrity fixture Invocation missing');
  }
  corruption
    .prepare(`
      INSERT INTO invocations_fts(invocations_fts, rowid, tool_name, searchable_text)
      VALUES ('delete', ?, ?, ?)
    `)
    .run(invocationId, toolName, searchableText);
  corruption.close();

  const reader = openSqliteRecallDatabase(databasePath, { readOnly: true });
  t.after(() => reader.close());
  const diagnostics = reader.checkIntegrity();

  assert.equal(diagnostics.invocationFtsDocuments, 0);
  assert.equal(diagnostics.invocationsMissingFts, 1);
  assert.equal(diagnostics.ftsDocumentsMissingInvocation, 0);
  assert.equal(diagnostics.healthy, false);
});

void test('replacement refreshes same-ID metadata, vectors, project scope, and Invocation FTS', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-refresh-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const sessionPath = '/sessions/refresh.jsonl';
  const oldProject = parseRepositoryIdentity('git-origin:github.com/Whamp/refresh-old');
  const currentProject = parseRepositoryIdentity('git-origin:github.com/Whamp/refresh-current');
  const oldDocument = createTestSessionConversationChunk({
    id: 'stable-document-id',
    sessionPath,
    projectAttribution: {
      projectIdentity: oldProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    siblingIds: ['stale-sibling'],
    nextSiblingId: 'stale-sibling',
    content: 'stale metadata',
  });
  database.replacePhysicalSession({
    sessionPath,
    size: 90,
    mtimeMs: 100,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [oldDocument.id],
    denseDocuments: [oldDocument],
    denseEmbeddings: new Map([[oldDocument.id, createUnitEmbedding(60)]]),
    invocations: [
      createInvocationRecord(sessionPath, {
        searchableText: 'tool="read"\npath="/project/stale-invocation.ts"',
      }),
    ],
  });
  const currentDocument = {
    ...oldDocument,
    checksum: 'current-checksum',
    projectAttribution: {
      projectIdentity: currentProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    siblingIds: ['current-sibling'],
    previousSiblingId: 'current-sibling',
    nextSiblingId: null,
    content: 'current metadata',
  };
  const currentEmbedding = createUnitEmbedding(61);

  database.replacePhysicalSession({
    sessionPath,
    size: 91,
    mtimeMs: 101,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [currentDocument.id],
    denseDocuments: [currentDocument],
    denseEmbeddings: new Map([[currentDocument.id, currentEmbedding]]),
    invocations: [
      createInvocationRecord(sessionPath, {
        searchableText: 'tool="read"\npath="/project/current-invocation.ts"',
      }),
    ],
  });

  assert.deepEqual(database.readPhysicalSessionState(sessionPath), {
    size: 91,
    mtimeMs: 101,
    sourceSha256: TEST_SOURCE_SHA256,
    invocationCount: 1,
    documentIds: [currentDocument.id],
    denseDocumentIds: [currentDocument.id],
  });
  assert.deepEqual(
    database.fetchDenseDocuments([currentDocument.id]),
    new Map([[currentDocument.id, currentDocument]]),
  );
  assert.deepEqual(
    database.fetchDenseVectors([currentDocument.id]),
    new Map([[currentDocument.id, currentEmbedding]]),
  );
  assert.deepEqual(database.searchDenseCandidates(currentEmbedding, 1, oldProject), []);
  assert.equal(
    database.searchDenseCandidates(currentEmbedding, 1, currentProject)[0]?.id,
    currentDocument.id,
  );
  assert.equal(database.searchInvocations('stale invocation', 10).length, 0);
  assert.equal(database.searchInvocations('current invocation', 10).length, 1);
  assert.equal(database.readCounts().denseProjects, 1);
  assert.equal(database.checkIntegrity().healthy, true);
});

void test('a late replacement constraint failure rolls back all projections and preserves unrelated sessions', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-rollback-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const replacedSessionPath = '/sessions/replaced.jsonl';
  const unrelatedSessionPath = '/sessions/unrelated.jsonl';
  const initialProject = parseRepositoryIdentity('git-origin:github.com/Whamp/rollback-initial');
  const failedProject = parseRepositoryIdentity('git-origin:github.com/Whamp/rollback-failed');
  const unrelatedProject = parseRepositoryIdentity(
    'git-origin:github.com/Whamp/rollback-unrelated',
  );
  const initialDocument = createTestSessionConversationChunk({
    id: 'rollback-before',
    sessionPath: replacedSessionPath,
    projectAttribution: {
      projectIdentity: initialProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    content: 'prior dense projection',
  });
  const unrelatedDocument = createTestSessionConversationChunk({
    id: 'unrelated-before',
    sessionPath: unrelatedSessionPath,
    projectAttribution: {
      projectIdentity: unrelatedProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    content: 'unrelated dense projection',
  });
  const initialEmbedding = createUnitEmbedding(31);
  const unrelatedEmbedding = createUnitEmbedding(32);
  database.replacePhysicalSession({
    sessionPath: replacedSessionPath,
    size: 10,
    mtimeMs: 20,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [initialDocument.id],
    denseDocuments: [initialDocument],
    denseEmbeddings: new Map([[initialDocument.id, initialEmbedding]]),
    invocations: [
      createInvocationRecord(replacedSessionPath, {
        searchableText: 'tool="read"\npath="/project/rollback-prior.ts"',
      }),
    ],
  });
  database.replacePhysicalSession({
    sessionPath: unrelatedSessionPath,
    size: 30,
    mtimeMs: 40,
    sourceSha256: TEST_SOURCE_SHA256,
    conversationProjectionInputs: [],
    documentIds: [unrelatedDocument.id],
    denseDocuments: [unrelatedDocument],
    denseEmbeddings: new Map([[unrelatedDocument.id, unrelatedEmbedding]]),
    invocations: [
      createInvocationRecord(unrelatedSessionPath, {
        toolCallId: 'call-unrelated',
        entryId: 'entry-unrelated',
        searchableText: 'tool="read"\npath="/project/unrelated-stable.ts"',
      }),
    ],
  });
  const unrelatedBefore = {
    state: database.readPhysicalSessionState(unrelatedSessionPath),
    documents: [...database.fetchDenseDocuments([unrelatedDocument.id])],
    vectors: [...database.fetchDenseVectors([unrelatedDocument.id])],
    invocations: database.searchInvocations('unrelated stable', 10),
  };
  const countsBefore = database.readCounts();
  const failedDocument = createTestSessionConversationChunk({
    id: 'rollback-after',
    sessionPath: replacedSessionPath,
    projectAttribution: {
      projectIdentity: failedProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    content: 'this projection must roll back',
  });

  assert.throws(() =>
    database.replacePhysicalSession({
      sessionPath: replacedSessionPath,
      size: 11,
      mtimeMs: 21,
      sourceSha256: TEST_SOURCE_SHA256,
      conversationProjectionInputs: [],
      documentIds: [failedDocument.id],
      denseDocuments: [failedDocument],
      denseEmbeddings: new Map([[failedDocument.id, createUnitEmbedding(33)]]),
      invocations: [
        createInvocationRecord(replacedSessionPath, {
          sourceLineStart: 0,
          sourceLineEnd: 0,
          searchableText: 'tool="read"\npath="/project/rollback-failed.ts"',
        }),
      ],
    }),
  );

  assert.deepEqual(database.readPhysicalSessionState(replacedSessionPath), {
    size: 10,
    mtimeMs: 20,
    sourceSha256: TEST_SOURCE_SHA256,
    invocationCount: 1,
    documentIds: [initialDocument.id],
    denseDocumentIds: [initialDocument.id],
  });
  assert.deepEqual(
    database.fetchDenseDocuments([initialDocument.id]),
    new Map([[initialDocument.id, initialDocument]]),
  );
  assert.deepEqual(
    database.fetchDenseVectors([initialDocument.id]),
    new Map([[initialDocument.id, initialEmbedding]]),
  );
  assert.equal(database.fetchDenseDocuments([failedDocument.id]).size, 0);
  assert.equal(database.searchInvocations('rollback prior', 10).length, 1);
  assert.equal(database.searchInvocations('rollback failed', 10).length, 0);
  assert.deepEqual(
    {
      state: database.readPhysicalSessionState(unrelatedSessionPath),
      documents: [...database.fetchDenseDocuments([unrelatedDocument.id])],
      vectors: [...database.fetchDenseVectors([unrelatedDocument.id])],
      invocations: database.searchInvocations('unrelated stable', 10),
    },
    unrelatedBefore,
  );
  assert.deepEqual(database.readCounts(), countsBefore);
  assert.equal(database.checkIntegrity().healthy, true);
});

void test('physical-session deletion atomically removes only its complete projection', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-delete-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const database = openSqliteRecallDatabase(join(directory, 'recall.sqlite'));
  t.after(() => database.close());
  const deletedSessionPath = '/sessions/delete-me.jsonl';
  const retainedSessionPath = '/sessions/retain-me.jsonl';
  const deletedProject = parseRepositoryIdentity('git-origin:github.com/Whamp/delete-me');
  const retainedProject = parseRepositoryIdentity('git-origin:github.com/Whamp/retain-me');
  const deletedDocument = createTestSessionConversationChunk({
    id: 'deleted-document',
    sessionPath: deletedSessionPath,
    projectAttribution: {
      projectIdentity: deletedProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
  });
  const retainedDocument = createTestSessionConversationChunk({
    id: 'retained-document',
    sessionPath: retainedSessionPath,
    projectAttribution: {
      projectIdentity: retainedProject,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
  });
  const deletedEmbedding = createUnitEmbedding(40);
  const retainedEmbedding = createUnitEmbedding(41);
  for (const [sessionPath, document, embedding, searchTerm] of [
    [deletedSessionPath, deletedDocument, deletedEmbedding, 'delete projection'],
    [retainedSessionPath, retainedDocument, retainedEmbedding, 'retain projection'],
  ] as const) {
    database.replacePhysicalSession({
      sessionPath,
      size: 1,
      mtimeMs: 2,
      sourceSha256: TEST_SOURCE_SHA256,
      conversationProjectionInputs: [],
      documentIds: [document.id],
      denseDocuments: [document],
      denseEmbeddings: new Map([[document.id, embedding]]),
      invocations: [
        createInvocationRecord(sessionPath, {
          toolCallId: `call-${document.id}`,
          entryId: `entry-${document.id}`,
          searchableText: `tool="read"\npath="/project/${searchTerm.replace(' ', '-')}.ts"`,
        }),
      ],
    });
  }
  const retainedBefore = {
    state: database.readPhysicalSessionState(retainedSessionPath),
    document: database.fetchDenseDocuments([retainedDocument.id]),
    vector: database.fetchDenseVectors([retainedDocument.id]),
    invocation: database.searchInvocations('retain projection', 10),
  };

  assert.equal(database.deletePhysicalSession(deletedSessionPath), true);
  assert.equal(database.deletePhysicalSession(deletedSessionPath), false);
  assert.equal(database.readPhysicalSessionState(deletedSessionPath), null);
  assert.equal(database.fetchDenseDocuments([deletedDocument.id]).size, 0);
  assert.equal(database.fetchDenseVectors([deletedDocument.id]).size, 0);
  assert.equal(database.searchInvocations('delete projection', 10).length, 0);
  assert.deepEqual(
    {
      state: database.readPhysicalSessionState(retainedSessionPath),
      document: database.fetchDenseDocuments([retainedDocument.id]),
      vector: database.fetchDenseVectors([retainedDocument.id]),
      invocation: database.searchInvocations('retain projection', 10),
    },
    retainedBefore,
  );
  assert.deepEqual(database.readCounts(), {
    physicalSessions: 1,
    sessionDocuments: 1,
    invocations: 1,
    denseDocuments: 1,
    denseVectors: 1,
    denseProjects: 1,
  });
  assert.equal(database.checkIntegrity().healthy, true);
});

void test(
  'combined normal search reads dense, Invocation, neighbors, and counts from one committed snapshot',
  { timeout: 30_000 },
  async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-search-snapshot-'));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const databasePath = join(directory, 'recall.sqlite');
    const payloadPath = join(directory, 'replacement.json');
    const sessionPath = '/sessions/search-snapshot.jsonl';
    const priorDocument = createTestSessionConversationChunk({
      id: 'search-snapshot-prior',
      sessionPath,
      content: 'prior coherent snapshot',
    });
    const embedding = createUnitEmbedding(71);
    const writer = openSqliteRecallDatabase(databasePath);
    writer.replacePhysicalSession({
      sessionPath,
      size: 1,
      mtimeMs: 1,
      sourceSha256: TEST_SOURCE_SHA256,
      conversationProjectionInputs: [],
      documentIds: [priorDocument.id],
      denseDocuments: [priorDocument],
      denseEmbeddings: new Map([[priorDocument.id, embedding]]),
      invocations: [
        createInvocationRecord(sessionPath, {
          searchableText: 'tool="read"\npath="/project/prior-coherent.ts"',
        }),
      ],
    });
    writer.close();

    const replacementDocuments = ['a', 'b'].map((suffix, index) =>
      createTestSessionConversationChunk({
        id: `search-snapshot-replacement-${suffix}`,
        sessionPath,
        content: `replacement committed snapshot ${index}`,
      }),
    );
    await writeFile(
      payloadPath,
      JSON.stringify({
        sessionPath,
        size: 2,
        mtimeMs: 2,
        sourceSha256: TEST_SOURCE_SHA256,
        conversationProjectionInputs: [],
        documentIds: replacementDocuments.map(({ id }) => id),
        denseDocuments: replacementDocuments,
        denseEmbeddings: replacementDocuments.map((document) => [document.id, embedding]),
        invocations: replacementDocuments.map((_, index) =>
          createInvocationRecord(sessionPath, {
            entryId: `replacement-invocation-${index}`,
            searchableText: `tool="read"\npath="/project/replacement-coherent-${index}.ts"`,
          }),
        ),
      }),
    );

    const reader = openSqliteRecallDatabase(databasePath, { readOnly: true });
    t.after(() => reader.close());
    const snapshot = await reader.searchRecallSnapshot({
      embedding,
      query: 'coherent',
      denseLimit: 8,
      invocationLimit: 8,
      async onSnapshotEstablished() {
        const childScript = `
          import { readFileSync } from 'node:fs';
          const { openSqliteRecallDatabase } = await import(process.argv[1]);
          const replacement = JSON.parse(readFileSync(process.argv[3], 'utf8'));
          replacement.denseEmbeddings = new Map(replacement.denseEmbeddings);
          const database = openSqliteRecallDatabase(process.argv[2]);
          database.replacePhysicalSession(replacement);
          database.close();
        `;
        const child = spawn(
          process.execPath,
          [
            '--import',
            'tsx',
            '--input-type=module',
            '--eval',
            childScript,
            new URL('./sqlite-recall-database.ts', import.meta.url).href,
            databasePath,
            payloadPath,
          ],
          { stdio: 'ignore' },
        );
        assert.deepEqual(await waitForChildProcess(child), { code: 0, signal: null });
      },
    });

    assert.deepEqual(
      snapshot.denseCandidates.map(({ id }) => id),
      [priorDocument.id],
    );
    assert.deepEqual(
      snapshot.invocationCandidates.map(({ entryId }) => entryId),
      ['assistant-1'],
    );
    assert.deepEqual(snapshot.denseDocuments, new Map([[priorDocument.id, priorDocument]]));
    assert.equal(snapshot.counts.denseDocuments, 1);
    assert.equal(snapshot.counts.invocations, 1);

    const committed = openSqliteRecallDatabase(databasePath, { readOnly: true });
    t.after(() => committed.close());
    assert.equal(committed.readCounts().denseDocuments, 2);
    assert.equal(committed.readCounts().invocations, 2);
  },
);

void test(
  'a concurrent reader sees the prior complete snapshot and SIGKILL restores it',
  { timeout: 30_000 },
  async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'sqlite-recall-sigkill-'));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const databasePath = join(directory, 'recall.sqlite');
    const payloadPath = join(directory, 'replacement.json');
    const sessionPath = '/sessions/sigkill.jsonl';
    const projectIdentity = parseRepositoryIdentity('git-origin:github.com/Whamp/sigkill');
    const priorDocument = createTestSessionConversationChunk({
      id: 'sigkill-prior',
      sessionPath,
      projectAttribution: {
        projectIdentity,
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      },
      content: 'prior committed snapshot',
    });
    const priorEmbedding = createUnitEmbedding(51);
    const initial = openSqliteRecallDatabase(databasePath);
    initial.replacePhysicalSession({
      sessionPath,
      size: 70,
      mtimeMs: 80,
      sourceSha256: TEST_SOURCE_SHA256,
      conversationProjectionInputs: [],
      documentIds: [priorDocument.id],
      denseDocuments: [priorDocument],
      denseEmbeddings: new Map([[priorDocument.id, priorEmbedding]]),
      invocations: [
        createInvocationRecord(sessionPath, {
          searchableText: 'tool="read"\npath="/project/sigkill-prior.ts"',
        }),
      ],
    });
    initial.close();

    const replacementDocuments = Array.from({ length: 2_000 }, (_, index) =>
      createTestSessionConversationChunk({
        id: `sigkill-replacement-${index.toString().padStart(4, '0')}`,
        sessionPath,
        projectAttribution: {
          projectIdentity,
          identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
        },
        content: `uncommitted replacement ${index}`,
      }),
    );
    const replacementEmbeddings = replacementDocuments.map((document, index) => [
      document.id,
      createUnitEmbedding(index % SQLITE_RECALL_EMBEDDING_DIMENSIONS),
    ]);
    await writeFile(
      payloadPath,
      JSON.stringify({
        sessionPath,
        size: 71,
        mtimeMs: 81,
        sourceSha256: TEST_SOURCE_SHA256,
        conversationProjectionInputs: [],
        documentIds: replacementDocuments.map(({ id }) => id),
        denseDocuments: replacementDocuments,
        denseEmbeddings: replacementEmbeddings,
        invocations: [
          createInvocationRecord(sessionPath, {
            searchableText: 'tool="read"\npath="/project/sigkill-uncommitted.ts"',
          }),
        ],
      }),
    );

    const reader = openSqliteRecallDatabase(databasePath, { readOnly: true });
    t.after(() => reader.close());
    const childScript = `
      import { readFileSync } from 'node:fs';
      const { openSqliteRecallDatabase } = await import(process.argv[1]);
      const replacement = JSON.parse(readFileSync(process.argv[3], 'utf8'));
      replacement.denseEmbeddings = new Map(replacement.denseEmbeddings);
      const database = openSqliteRecallDatabase(process.argv[2]);
      database.replacePhysicalSession(replacement);
      database.close();
    `;
    const child = spawn(
      process.execPath,
      [
        '--import',
        'tsx',
        '--input-type=module',
        '--eval',
        childScript,
        new URL('./sqlite-recall-database.ts', import.meta.url).href,
        databasePath,
        payloadPath,
      ],
      { stdio: 'ignore' },
    );
    const childCompletion = waitForChildProcess(child);
    await waitForSqliteWriterTransaction(databasePath);

    assert.deepEqual(reader.readPhysicalSessionState(sessionPath), {
      size: 70,
      mtimeMs: 80,
      sourceSha256: TEST_SOURCE_SHA256,
      invocationCount: 1,
      documentIds: [priorDocument.id],
      denseDocumentIds: [priorDocument.id],
    });
    assert.deepEqual(
      reader.fetchDenseDocuments([priorDocument.id]),
      new Map([[priorDocument.id, priorDocument]]),
    );
    assert.deepEqual(
      reader.fetchDenseVectors([priorDocument.id]),
      new Map([[priorDocument.id, priorEmbedding]]),
    );
    assert.equal(reader.searchInvocations('sigkill prior', 10).length, 1);
    assert.deepEqual(reader.readCounts(), {
      physicalSessions: 1,
      sessionDocuments: 1,
      invocations: 1,
      denseDocuments: 1,
      denseVectors: 1,
      denseProjects: 1,
    });
    assert.equal(isSqliteWriterTransactionActive(databasePath), true);

    assert.equal(child.kill('SIGKILL'), true);
    const termination = await childCompletion;
    assert.equal(termination.signal, 'SIGKILL');

    const recovered = openSqliteRecallDatabase(databasePath);
    t.after(() => recovered.close());
    assert.deepEqual(recovered.readPhysicalSessionState(sessionPath), {
      size: 70,
      mtimeMs: 80,
      sourceSha256: TEST_SOURCE_SHA256,
      invocationCount: 1,
      documentIds: [priorDocument.id],
      denseDocumentIds: [priorDocument.id],
    });
    assert.deepEqual(
      recovered.fetchDenseDocuments([priorDocument.id]),
      new Map([[priorDocument.id, priorDocument]]),
    );
    assert.deepEqual(
      recovered.fetchDenseVectors([priorDocument.id]),
      new Map([[priorDocument.id, priorEmbedding]]),
    );
    assert.equal(recovered.searchInvocations('sigkill prior', 10).length, 1);
    assert.equal(recovered.searchInvocations('sigkill uncommitted', 10).length, 0);
    assert.equal(recovered.checkIntegrity().healthy, true);
  },
);

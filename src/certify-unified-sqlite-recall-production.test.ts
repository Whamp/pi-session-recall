import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, symlinkSync } from 'node:fs';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  assertCertificationScratchRoot,
  certifyDisposableUnifiedSqliteClone,
  DENSE_CERTIFICATION_QUERIES,
  evaluateUnifiedSqliteCertificationGates,
  formatUnifiedSqliteCertificationProgress,
  MAXIMUM_CERTIFIED_STORAGE_BYTES,
  readUnifiedSqliteCertificationArguments,
  resolveCertifiedCandidateDirectory,
  sanitizeUnifiedSqliteCertificationReport,
  snapshotUnifiedSqliteActivePointer,
  SOURCE_CERTIFICATION_PROBES,
} from './certify-unified-sqlite-recall-production.js';
import { RecallProjectIdentitySource } from './enums.js';
import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { parseProjectIdentity } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';
import {
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
  type SqliteRecallDatabase,
} from './sqlite-recall-database.js';

function unitEmbedding(): number[] {
  const embedding = Array.from({ length: SQLITE_RECALL_EMBEDDING_DIMENSIONS }, () => 0);
  embedding[0] = 1;
  return embedding;
}

function hashFile(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

void test('certification progress renders bounded stage counters', () => {
  assert.equal(
    formatUnifiedSqliteCertificationProgress({
      stage: 'clone-churn',
      completed: 40,
      total: 100,
    }),
    'Certification: clone churn (40/100)',
  );
  assert.equal(
    formatUnifiedSqliteCertificationProgress({ stage: 'candidate-integrity' }),
    'Certification: candidate integrity',
  );
});

void test('certification gate calculations enforce all production thresholds', () => {
  const passing = evaluateUnifiedSqliteCertificationGates({
    storageBytes: MAXIMUM_CERTIFIED_STORAGE_BYTES,
    projectP95Milliseconds: 99.9,
    globalP95Milliseconds: 699.9,
    invocationP95Milliseconds: 4.9,
    denseProbePasses: DENSE_CERTIFICATION_QUERIES.map(() => true),
    invocationProbePasses: [true, true, true, true, true, true],
    sourceProbePasses: SOURCE_CERTIFICATION_PROBES.map(() => true),
    integrityHealthy: true,
    linuxX64LoadPassed: true,
    candidateInactive: true,
    clonePassed: true,
  });
  assert.ok(Object.values(passing).every((value) => value === true));

  const failing = evaluateUnifiedSqliteCertificationGates({
    storageBytes: MAXIMUM_CERTIFIED_STORAGE_BYTES + 1,
    projectP95Milliseconds: 100,
    globalP95Milliseconds: 700,
    invocationP95Milliseconds: 5,
    denseProbePasses: [true],
    invocationProbePasses: [true],
    sourceProbePasses: [true],
    integrityHealthy: false,
    linuxX64LoadPassed: false,
    candidateInactive: false,
    clonePassed: null,
  });
  assert.deepEqual(failing, {
    storage: false,
    projectLatency: false,
    globalLatency: false,
    invocationLatency: false,
    invocationProbes: false,
    denseProbes: false,
    sourceProvenance: false,
    integrity: false,
    linuxX64Load: false,
    candidateInactive: false,
    clone: null,
  });
});

void test('certification arguments require exact output and complete clone flags', () => {
  const base = [
    '--data-root',
    '/safe/data',
    '--candidate-target',
    'generations/generation-fixture',
    '--project-identity',
    'git-origin:github.com/Whamp/pi-session-recall',
  ];
  assert.equal(readUnifiedSqliteCertificationArguments(base).outputPath, null);
  assert.throws(
    () => readUnifiedSqliteCertificationArguments([...base, '--scratch-root', '/safe/scratch']),
    /requires --scratch-root, --representative-session, and --block-device together/u,
  );
  assert.throws(
    () => readUnifiedSqliteCertificationArguments([...base, '--output', '/tmp/result.json']),
    /output must be docs\/research\/unified-sqlite-production-recall-certification\.json/u,
  );
  assert.throws(
    () =>
      readUnifiedSqliteCertificationArguments([
        ...base,
        '--scratch-root',
        '/safe/scratch',
        '--representative-session',
        '/safe/session.jsonl',
        '--block-device',
        '../../unexpected',
      ]),
    /block device name is invalid/u,
  );
});

void test('candidate and scratch path guards reject traversal and overlap', () => {
  assert.equal(
    resolveCertifiedCandidateDirectory('/data/recall', 'generations/generation-safe'),
    '/data/recall/generations/generation-safe',
  );
  for (const target of ['generation-safe', 'generations/../active', '/data/other']) {
    assert.throws(
      () => resolveCertifiedCandidateDirectory('/data/recall', target),
      /not an exact staged generation/u,
    );
  }
  assert.throws(
    () =>
      assertCertificationScratchRoot(
        '/data/recall/scratch',
        '/data/recall',
        '/data/recall/generations/generation-safe',
      ),
    /must be disposable and disjoint/u,
  );
});

const certificationTokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

const resolveCertificationProjectIdentity = async () => ({
  projectIdentity: parseProjectIdentity('non-git-session-origin:/fixture'),
  identitySource: RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN as const,
});

const certificationEmbeddingProvider: RecallEmbeddingProvider = {
  embedQuery(query) {
    assert.notEqual(query.trim(), '');
    return Promise.resolve(unitEmbedding());
  },
  embedDocuments(documents) {
    assert.ok(documents.length > 0);
    return Promise.resolve(documents.map(() => unitEmbedding()));
  },
};

async function writeCertificationSession(
  sessionPath: string,
  sessionId: string,
  content: string,
): Promise<void> {
  await writeFile(
    sessionPath,
    `${[
      { type: 'session', version: 3, id: sessionId, timestamp: '2026-08-10', cwd: '/fixture' },
      {
        type: 'message',
        id: `user-${sessionId}`,
        parentId: null,
        timestamp: '2026-08-10',
        message: { role: 'user', content },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n')}\n`,
  );
}

void test('clone certification runs one real changed-session index and rejects dishonest fixture evidence', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'unified-certification-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataRoot = join(root, 'data');
  const candidateDirectory = join(dataRoot, 'generations', 'generation-fixture');
  const candidateDatabasePath = join(candidateDirectory, 'recall.sqlite');
  const sessionsDirectory = join(root, 'sessions');
  const scratchRoot = join(root, 'scratch');
  const representativeSessionPath = join(sessionsDirectory, 'representative.jsonl');
  const unrelatedSessionPath = join(sessionsDirectory, 'unrelated.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  mkdirSync(candidateDirectory, { recursive: true });
  mkdirSync(scratchRoot, { recursive: true });
  await writeCertificationSession(
    representativeSessionPath,
    'representative',
    'representative certification evidence',
  );
  await writeCertificationSession(unrelatedSessionPath, 'unrelated', 'unrelated stable evidence');

  await indexChangedConversationSessions({
    sessionsDirectory,
    databasePath: candidateDatabasePath,
    embeddingProvider: certificationEmbeddingProvider,
    tokenizer: certificationTokenizer,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
    ignoredPhysicalSessionPaths: new Set(),
    resolveProjectIdentity: resolveCertificationProjectIdentity,
  });
  symlinkSync('generations/generation-active', join(dataRoot, 'active'));
  const pointerBefore = snapshotUnifiedSqliteActivePointer(dataRoot);
  const candidateHashBefore = hashFile(candidateDatabasePath);

  const runRealChangedSessionIndex = (
    database: SqliteRecallDatabase,
    onProgress: (event: RecallIndexProgressEvent) => void,
  ) =>
    indexChangedConversationSessions({
      sessionsDirectory,
      database,
      embeddingProvider: certificationEmbeddingProvider,
      tokenizer: certificationTokenizer,
      chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
      ignoredPhysicalSessionPaths: new Set(),
      resolveProjectIdentity: resolveCertificationProjectIdentity,
      onProgress,
    });
  const certify = (
    overrides: Partial<Parameters<typeof certifyDisposableUnifiedSqliteClone>[0]> = {},
  ) => {
    const deviceWriteSamples = [0, 1 * 1024 ** 2, 1 * 1024 ** 2, 101 * 1024 ** 2];
    return certifyDisposableUnifiedSqliteClone({
      candidateDirectory,
      candidateDatabasePath,
      scratchRoot,
      dataRoot,
      representativeSessionPath,
      blockDevice: 'fixture-device',
      readDeviceWrittenBytes: () => deviceWriteSamples.shift() ?? null,
      flushFilesystemWrites() {},
      runChangedSessionIndex: runRealChangedSessionIndex,
      ...overrides,
    });
  };

  const cloneProgress: string[] = [];
  const result = await certify({
    onProgress(event) {
      cloneProgress.push(formatUnifiedSqliteCertificationProgress(event));
    },
  });
  assert.equal(result.passed, true, JSON.stringify(result));
  assert.equal(cloneProgress.includes('Certification: clone copy'), true);
  assert.equal(cloneProgress.includes('Certification: clone churn (100/100)'), true);
  assert.equal(cloneProgress.at(-1), 'Certification: clone complete');
  assert.deepEqual(result.changedSessionIndexedPhysicalSessionPaths, [representativeSessionPath]);
  assert.deepEqual(result.changedSessionIndexSummary, {
    scannedSessions: 2,
    indexedSessions: 1,
    removedSessions: 0,
    reusedVectors: 1,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
    deletedChunks: 0,
    failedSessions: [],
  });
  assert.equal(result.changedSessionIndexReusedExpectedVectors, true);
  assert.equal(result.unrelatedPhysicalSessionUnchanged, true);
  assert.equal(result.databaseCountsUnchanged, true);
  assert.equal(result.concurrentReaderSawCommittedState, true);
  assert.equal(result.explicitRollbackRestoredState, true);
  assert.equal(result.forcedTerminationRestoredState, true);
  assert.equal(result.directDatabaseChurnCycles, 100);
  assert.equal(result.allocatedGrowthBytes, 0);
  assert.equal(result.freePageGrowth, 0);
  assert.equal(hashFile(candidateDatabasePath), candidateHashBefore);
  assert.deepEqual(snapshotUnifiedSqliteActivePointer(dataRoot), pointerBefore);

  await assert.rejects(
    certifyDisposableUnifiedSqliteClone({
      candidateDirectory,
      candidateDatabasePath,
      scratchRoot,
      dataRoot,
      representativeSessionPath,
      blockDevice: 'fixture-device',
      readDeviceWrittenBytes: () => 0,
    }),
    /real changed-session index callback is required/u,
  );
  const wrongCount = await certify({
    async runChangedSessionIndex(database, onProgress) {
      const summary = await runRealChangedSessionIndex(database, onProgress);
      return { ...summary, indexedSessions: 2 };
    },
  });
  assert.equal(wrongCount.passed, false);

  const changedUnrelated = await certify({
    async runChangedSessionIndex(database, onProgress) {
      const summary = await runRealChangedSessionIndex(database, onProgress);
      const unrelated = database.readPhysicalSessionReplacement(unrelatedSessionPath);
      assert.ok(unrelated);
      database.replacePhysicalSession({ ...unrelated, size: unrelated.size + 1 });
      return summary;
    },
  });
  assert.equal(changedUnrelated.passed, false);
  assert.equal(changedUnrelated.unrelatedPhysicalSessionUnchanged, false);

  const unavailableWrites = await certify({ readDeviceWrittenBytes: () => null });
  assert.equal(unavailableWrites.passed, false);
  assert.equal(unavailableWrites.deviceWriteMeasurement, 'unavailable');
});

void test('durable report sanitization removes machine roots and raw evidence fields', () => {
  const sanitized = sanitizeUnifiedSqliteCertificationReport(
    {
      candidate: '/private/data/generations/generation-a',
      result: {
        sessionPath: '/home/example/.pi/agent/sessions/a.jsonl',
        text: 'raw command output',
        searchableText: 'secret locator payload',
        embedding: [1, 2, 3],
      },
    },
    { '/private/data': '$DATA_ROOT', '/home/example': '$HOME' },
  );
  const encoded = JSON.stringify(sanitized);
  assert.match(encoded, /\$DATA_ROOT\/generations\/generation-a/u);
  assert.match(encoded, /\$HOME\/\.pi\/agent\/sessions\/a\.jsonl/u);
  assert.doesNotMatch(encoded, /raw command output|secret locator payload|\[1,2,3\]/u);
});

void test('command reports safe usage before reading any default production path', () => {
  const scriptPath = new URL('./certify-unified-sqlite-recall-production.ts', import.meta.url)
    .pathname;
  const result = spawnSync(process.execPath, ['--import', 'tsx', scriptPath], {
    encoding: 'utf8',
  });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unified SQLite recall certification usage/u);
  assert.doesNotMatch(result.stderr, /\.pi\/agent\/recall|\.pi\/agent\/sessions/u);

  const help = spawnSync(process.execPath, ['--import', 'tsx', scriptPath, '--help'], {
    encoding: 'utf8',
  });
  assert.equal(help.status, 0, help.stderr);
  assert.match(help.stdout, /Unified SQLite recall certification usage/u);
  assert.equal(help.stderr, '');
});

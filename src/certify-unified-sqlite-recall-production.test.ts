import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, symlinkSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  ZVecCollectionSchema,
  ZVecCreateAndOpen,
  ZVecDataType,
  ZVecIndexType,
  ZVecMetricType,
} from '@zvec/zvec';

import {
  assertCertificationScratchRoot,
  certifyDisposableUnifiedSqliteClone,
  DENSE_CERTIFICATION_QUERIES,
  evaluateUnifiedSqliteCertificationGates,
  MAXIMUM_CERTIFIED_STORAGE_BYTES,
  readUnifiedSqliteCertificationArguments,
  resolveCertifiedCandidateDirectory,
  sanitizeUnifiedSqliteCertificationReport,
  snapshotUnifiedSqliteActivePointer,
  SOURCE_CERTIFICATION_PROBES,
} from './certify-unified-sqlite-recall-production.js';
import type { InvocationRecord } from './createSessionInvocationRecords.js';
import { RecallProjectIdentitySource } from './enums.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
} from './sqlite-recall-database.js';
import { readZvecConversationDenseIndexType } from './zvec-conversation-store.js';

function unitEmbedding(): number[] {
  const embedding = Array.from({ length: SQLITE_RECALL_EMBEDDING_DIMENSIONS }, () => 0);
  embedding[0] = 1;
  return embedding;
}

function hashFile(path: string): string {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

void test('certification gate calculations enforce all production thresholds', () => {
  const passing = evaluateUnifiedSqliteCertificationGates({
    storageBytes: MAXIMUM_CERTIFIED_STORAGE_BYTES,
    projectP95Milliseconds: 99.9,
    globalP95Milliseconds: 499.9,
    invocationP95Milliseconds: 4.9,
    denseTopResultsMatch: DENSE_CERTIFICATION_QUERIES.map(() => true),
    globalTopEightOverlaps: DENSE_CERTIFICATION_QUERIES.map(() => 7),
    invocationProbePasses: [true, true, true, true, true, true],
    sourceProbePasses: SOURCE_CERTIFICATION_PROBES.map(() => true),
    integrityHealthy: true,
    linuxX64LoadPassed: true,
    macOsPackagesAvailable: true,
    candidateInactive: true,
    clonePassed: true,
  });
  assert.ok(Object.values(passing).every((value) => value === true));

  const failing = evaluateUnifiedSqliteCertificationGates({
    storageBytes: MAXIMUM_CERTIFIED_STORAGE_BYTES + 1,
    projectP95Milliseconds: 100,
    globalP95Milliseconds: 500,
    invocationP95Milliseconds: 5,
    denseTopResultsMatch: [true],
    globalTopEightOverlaps: [6],
    invocationProbePasses: [true],
    sourceProbePasses: [true],
    integrityHealthy: false,
    linuxX64LoadPassed: false,
    macOsPackagesAvailable: false,
    candidateInactive: false,
    clonePassed: null,
  });
  assert.deepEqual(failing, {
    storage: false,
    projectLatency: false,
    globalLatency: false,
    invocationLatency: false,
    invocationProbes: false,
    denseTopResults: false,
    globalOverlap: false,
    sourceProvenance: false,
    integrity: false,
    linuxX64Load: false,
    macOsPackageAvailability: false,
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
    '--control-zvec',
    '/safe/control',
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

void test('tiny candidate and flat-Zvec control preserve active pointer while clone probes mutate only scratch', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'unified-certification-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataRoot = join(root, 'data');
  const candidateDirectory = join(dataRoot, 'generations', 'generation-fixture');
  const candidateDatabasePath = join(candidateDirectory, 'recall.sqlite');
  const scratchRoot = join(root, 'scratch');
  const controlPath = join(root, 'flat-control');
  mkdirSync(candidateDirectory, { recursive: true });
  mkdirSync(scratchRoot, { recursive: true });

  const control = ZVecCreateAndOpen(
    controlPath,
    new ZVecCollectionSchema({
      name: 'flat_control',
      vectors: {
        name: 'embedding',
        dataType: ZVecDataType.VECTOR_FP32,
        dimension: SQLITE_RECALL_EMBEDDING_DIMENSIONS,
        indexParams: { indexType: ZVecIndexType.FLAT, metricType: ZVecMetricType.IP },
      },
      fields: [],
    }),
  );
  control.closeSync();
  assert.equal(readZvecConversationDenseIndexType(controlPath), ZVecIndexType.FLAT);

  const sessionPath = '/fixture/sessions/representative.jsonl';
  const projectIdentity = parseRepositoryIdentity('git-origin:github.com/Whamp/pi-session-recall');
  const document = createTestSessionConversationChunk({
    id: 'fixture-document',
    sessionPath,
    projectAttribution: {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
  });
  const invocation: InvocationRecord = {
    kind: 'tool_call',
    toolName: 'brain_query',
    toolCallId: 'call-fixture',
    sessionPath,
    sessionId: 'session-fixture',
    entryId: 'entry-fixture',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockIndex: 0,
    timestamp: '2026-08-10T12:00:00Z',
    sessionOrigin: '/fixture',
    projectAttribution: {
      projectIdentity,
      identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
    },
    isError: false,
    searchableText: 'tool="brain_query" query="fixture"',
  };
  const database = openSqliteRecallDatabase(candidateDatabasePath);
  database.replacePhysicalSession({
    sessionPath,
    size: 123,
    mtimeMs: 456,
    documentIds: [document.id],
    denseDocuments: [document],
    denseEmbeddings: new Map([[document.id, unitEmbedding()]]),
    invocations: [invocation],
  });
  database.close();

  symlinkSync('generations/generation-active', join(dataRoot, 'active'));
  const pointerBefore = snapshotUnifiedSqliteActivePointer(dataRoot);
  assert.equal(pointerBefore.exists, true);
  assert.equal(pointerBefore.target, 'generations/generation-active');
  const candidateHashBefore = hashFile(candidateDatabasePath);
  const deviceWriteSamples = [0, 1 * 1024 ** 2, 1 * 1024 ** 2, 101 * 1024 ** 2];
  const result = certifyDisposableUnifiedSqliteClone({
    candidateDirectory,
    candidateDatabasePath,
    scratchRoot,
    dataRoot,
    representativeSessionPath: sessionPath,
    blockDevice: 'fixture-device',
    minimumFreeBytes: 0,
    readDeviceWrittenBytes: () => deviceWriteSamples.shift() ?? null,
  });
  assert.equal(result.passed, true, JSON.stringify(result));
  assert.equal(result.concurrentReaderSawCommittedState, true);
  assert.equal(result.explicitRollbackRestoredState, true);
  assert.equal(result.forcedTerminationRestoredState, true);
  assert.equal(result.churnCycles, 100);
  assert.equal(result.allocatedGrowthBytes, 0);
  assert.equal(result.freePageGrowth, 0);
  assert.equal(hashFile(candidateDatabasePath), candidateHashBefore);
  assert.deepEqual(snapshotUnifiedSqliteActivePointer(dataRoot), pointerBefore);
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

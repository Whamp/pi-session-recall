import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallMetadataSweepStatus } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  scanRecallSessionMetadata,
  type RecallMetadataSweepContinuation,
  type RecallMetadataSweepContinuationStore,
  type RecallSessionMetadataFilesystem,
} from './scan-recall-session-metadata.js';

interface FakeMetadataFilesystem extends RecallSessionMetadataFilesystem {
  calls: string[];
}

function createFlatMetadataFilesystem(entryCount: number): FakeMetadataFilesystem {
  const names = Array.from(
    { length: entryCount },
    (_, index) => `session-${String(index).padStart(5, '0')}.jsonl`,
  );
  const calls: string[] = [];
  return {
    calls,
    async readDirectory(path) {
      calls.push(`readdir:${path}`);
      return names;
    },
    async statPath(path) {
      calls.push(`stat:${path}`);
      return {
        isDirectory: false,
        isFile: true,
        sizeBytes: 10,
        modifiedAtEpochMilliseconds: 20,
        sourceDevice: '10',
        sourceInode: path,
      };
    },
  };
}

function createMemoryContinuationStore(): RecallMetadataSweepContinuationStore & {
  continuation: RecallMetadataSweepContinuation | null;
} {
  return {
    continuation: null,
    async readContinuation() {
      return this.continuation;
    },
    async writeContinuation(continuation) {
      this.continuation = continuation;
    },
    async clearContinuation() {
      this.continuation = null;
    },
  };
}

void test('metadata sweep stops at 10,000 files and resumes its persisted continuation', async () => {
  const filesystem = createFlatMetadataFilesystem(10_001);
  const continuationStore = createMemoryContinuationStore();

  const first = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    monotonicNowMilliseconds: () => 0,
  });

  assert.equal(first.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  assert.equal(first.deletionConfirmationSuppressed, true);
  assert.equal(first.scannedFileCount, 10_000);
  assert.equal(filesystem.calls.filter((call) => call.startsWith('stat:')).length, 10_000);
  assert.notEqual(continuationStore.continuation, null);

  filesystem.calls.length = 0;
  const second = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    monotonicNowMilliseconds: () => 0,
  });

  assert.equal(second.status, RecallMetadataSweepStatus.COMPLETE);
  assert.equal(second.scannedFileCount, 1);
  assert.equal(filesystem.calls.filter((call) => call.startsWith('stat:')).length, 1);
  assert.equal(continuationStore.continuation, null);
  assert.equal(
    filesystem.calls.every((call) => /^(readdir|stat):/u.test(call)),
    true,
  );
});

void test('metadata sweep resumes newly inserted names before the lexical cursor', async () => {
  const continuationStore = createMemoryContinuationStore();
  let names = ['b.jsonl', 'c.jsonl'];
  const filesystem: RecallSessionMetadataFilesystem = {
    async readDirectory() {
      return names;
    },
    async statPath(path) {
      return {
        isDirectory: false,
        isFile: true,
        sizeBytes: 10,
        modifiedAtEpochMilliseconds: 20,
        sourceDevice: '10',
        sourceInode: path,
      };
    },
  };
  const first = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 1,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(first.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  names = ['a.jsonl', 'b.jsonl', 'c.jsonl'];

  const second = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 10,
    monotonicNowMilliseconds: () => 0,
  });

  assert.equal(second.status, RecallMetadataSweepStatus.COMPLETE);
  assert.deepEqual(
    second.observedSessionMetadata.map(({ relativePath }) => relativePath).toSorted(),
    ['a.jsonl', 'c.jsonl'],
  );
});

void test('metadata sweep final rescan observes insertions in a completed parent directory', async () => {
  const continuationStore = createMemoryContinuationStore();
  let rootNames = ['child'];
  const filesystem: RecallSessionMetadataFilesystem = {
    async readDirectory(path) {
      return path.endsWith('/child') ? ['first.jsonl', 'second.jsonl'] : rootNames;
    },
    async statPath(path) {
      return {
        isDirectory: path.endsWith('/child'),
        isFile: !path.endsWith('/child'),
        sizeBytes: 10,
        modifiedAtEpochMilliseconds: 20,
        sourceDevice: '10',
        sourceInode: path,
      };
    },
  };
  const first = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 1,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(first.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  rootNames = ['child', 'inserted.jsonl'];

  const second = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 10,
    monotonicNowMilliseconds: () => 0,
  });

  assert.equal(second.status, RecallMetadataSweepStatus.COMPLETE);
  assert.deepEqual(
    second.observedSessionMetadata.map(({ relativePath }) => relativePath).toSorted(),
    ['child/second.jsonl', 'inserted.jsonl'],
  );
});

void test('metadata sweep independently stops when its monotonic 500 ms budget is reached', async () => {
  const filesystem = createFlatMetadataFilesystem(100);
  const continuationStore = createMemoryContinuationStore();
  let clockReading = -100;

  const result = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    monotonicNowMilliseconds() {
      clockReading += 100;
      return clockReading;
    },
  });

  assert.equal(result.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  assert.equal(result.scannedFileCount < 100, true);
  assert.notEqual(continuationStore.continuation, null);
});

void test('metadata sweep persists a strict scalar continuation outside session files', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'scan-recall-session-metadata-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionRootDirectory = join(directory, 'sessions');
  const controlDirectory = join(directory, 'recall', 'markers', 'control');
  await mkdir(sessionRootDirectory, { recursive: true });
  await writeFile(join(sessionRootDirectory, 'a.jsonl'), 'private session body');
  await writeFile(join(sessionRootDirectory, 'b.jsonl'), 'another private session body');

  const first = await scanRecallSessionMetadata({
    sessionRootDirectory,
    controlDirectory,
    maxFiles: 1,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(first.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  const continuationPath = join(controlDirectory, 'metadata-sweep-continuation.json');
  const continuation: unknown = JSON.parse(await readFile(continuationPath, 'utf8'));
  assert.equal(isUnknownRecord(continuation), true);
  if (!isUnknownRecord(continuation)) {
    throw new Error('Expected strict scalar metadata continuation');
  }
  assert.deepEqual(Object.keys(continuation).toSorted(), [
    'currentRelativeDirectory',
    'directoryCheckpoints',
    'observedKnownSourceIdentities',
    'observedPhysicalSessionIds',
    'observedSessionFileCount',
    'pendingRelativeDirectories',
    'rescanStarted',
    'sweepId',
    'version',
  ]);
  assert.equal(JSON.stringify(continuation).includes('private session body'), false);

  const second = await scanRecallSessionMetadata({
    sessionRootDirectory,
    controlDirectory,
    maxFiles: 10,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(second.status, RecallMetadataSweepStatus.COMPLETE);
  await assert.rejects(() => readFile(continuationPath, 'utf8'), { code: 'ENOENT' });
});

void test('metadata sweep classifies unavailable roots and suspicious broad loss without deletion', async () => {
  const unavailableFilesystem: RecallSessionMetadataFilesystem = {
    async readDirectory() {
      const error = new Error('missing root');
      Object.assign(error, { code: 'ENOENT' });
      throw error;
    },
    async statPath() {
      throw new Error('stat should not run');
    },
  };
  const unavailable = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/missing',
    controlDirectory: '/isolated/control',
    filesystem: unavailableFilesystem,
    continuationStore: createMemoryContinuationStore(),
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(unavailable.status, RecallMetadataSweepStatus.ROOT_UNAVAILABLE);
  assert.equal(unavailable.rootHealthy, false);
  assert.equal(unavailable.deletionConfirmationSuppressed, true);

  const mountFailureFilesystem: RecallSessionMetadataFilesystem = {
    async readDirectory() {
      const error = new Error('stale mount');
      Object.assign(error, { code: 'ESTALE' });
      throw error;
    },
    async statPath() {
      throw new Error('stat should not run');
    },
  };
  const mountFailure = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/stale-mount',
    controlDirectory: '/isolated/control',
    filesystem: mountFailureFilesystem,
    continuationStore: createMemoryContinuationStore(),
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(mountFailure.status, RecallMetadataSweepStatus.ROOT_UNAVAILABLE);
  assert.equal(mountFailure.deletionConfirmationSuppressed, true);

  const permissionDeniedFilesystem: RecallSessionMetadataFilesystem = {
    async readDirectory() {
      const error = new Error('permission denied');
      Object.assign(error, { code: 'EACCES' });
      throw error;
    },
    async statPath() {
      throw new Error('stat should not run');
    },
  };
  const permissionDenied = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/denied',
    controlDirectory: '/isolated/control',
    filesystem: permissionDeniedFilesystem,
    continuationStore: createMemoryContinuationStore(),
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(permissionDenied.status, RecallMetadataSweepStatus.PERMISSION_DENIED);
  assert.equal(permissionDenied.rootHealthy, false);
  assert.equal(permissionDenied.deletionConfirmationSuppressed, true);

  const filesystem = createFlatMetadataFilesystem(1);
  const suspicious = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore: createMemoryContinuationStore(),
    knownSources: [
      { physicalSessionId: 'physical-1', relativePath: 'session-00000.jsonl' },
      { physicalSessionId: 'physical-2', relativePath: 'missing-1.jsonl' },
      { physicalSessionId: 'physical-3', relativePath: 'missing-2.jsonl' },
    ],
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 1,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(suspicious.status, RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS);
  assert.equal(suspicious.rootHealthy, true);
  assert.equal(suspicious.deletionConfirmationSuppressed, true);
  assert.deepEqual(suspicious.missingPhysicalSessionIds, ['physical-2', 'physical-3']);
});

void test('metadata sweep keeps one sweep ID across continuation and reports stable source identity', async () => {
  const filesystem = createFlatMetadataFilesystem(2);
  const continuationStore = createMemoryContinuationStore();
  let nextSweep = 0;
  const createSweepId = () => `sweep-${++nextSweep}`;

  const first = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 1,
    monotonicNowMilliseconds: () => 0,
    createSweepId,
    knownSources: [{ physicalSessionId: 'physical-1', relativePath: 'session-00000.jsonl' }],
  });
  const second = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore,
    maxFiles: 10,
    monotonicNowMilliseconds: () => 0,
    createSweepId,
    knownSources: [{ physicalSessionId: 'physical-1', relativePath: 'session-00000.jsonl' }],
  });

  assert.equal(first.sweepId, 'sweep-1');
  assert.equal(second.sweepId, 'sweep-1');
  assert.equal(nextSweep, 1);
  assert.deepEqual(second.observedKnownSourceIdentities, [
    {
      physicalSessionId: 'physical-1',
      sourceDevice: '10',
      sourceInode: '/isolated/sessions/session-00000.jsonl',
    },
  ]);
  assert.deepEqual(second.observedSessionMetadata[0], {
    physicalSessionId: null,
    relativePath: 'session-00001.jsonl',
    sizeBytes: 10,
    modifiedAtEpochMilliseconds: 20,
    sourceDevice: '10',
    sourceInode: '/isolated/sessions/session-00001.jsonl',
  });
});

void test('metadata sweep halts above either configured mass-loss limit and permits the boundary', async () => {
  const filesystem = createFlatMetadataFilesystem(8);
  const knownSources = Array.from({ length: 10 }, (_, index) => ({
    physicalSessionId: `physical-${index}`,
    relativePath: `session-${String(index).padStart(5, '0')}.jsonl`,
  }));
  const atBoundary = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore: createMemoryContinuationStore(),
    knownSources,
    confirmedDeletionMaxMissingSourceCount: 2,
    confirmedDeletionMaxMissingSourceRatio: 0.2,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(atBoundary.status, RecallMetadataSweepStatus.COMPLETE);

  const aboveRatio = await scanRecallSessionMetadata({
    sessionRootDirectory: '/isolated/sessions',
    controlDirectory: '/isolated/control',
    filesystem,
    continuationStore: createMemoryContinuationStore(),
    knownSources,
    confirmedDeletionMaxMissingSourceCount: 10,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(aboveRatio.status, RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS);
});

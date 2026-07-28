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
    'afterEntryName',
    'currentRelativeDirectory',
    'observedPhysicalSessionIds',
    'observedSessionFileCount',
    'pendingRelativeDirectories',
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
    suspiciousMassLossMinimumMissingSources: 2,
    monotonicNowMilliseconds: () => 0,
  });
  assert.equal(suspicious.status, RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS);
  assert.equal(suspicious.rootHealthy, true);
  assert.equal(suspicious.deletionConfirmationSuppressed, true);
  assert.deepEqual(suspicious.missingPhysicalSessionIds, ['physical-2', 'physical-3']);
});

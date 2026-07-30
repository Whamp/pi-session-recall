import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import test from 'node:test';

import { assertRecallTestDataRoot } from './assert-recall-test-data-root.js';
import { createRecallDiagnosticHostIdentity } from './create-recall-diagnostic-host-identity.js';
import {
  RECALL_METADATA_SWEEP_MAX_ELAPSED_MILLISECONDS,
  scanRecallSessionMetadata,
  type RecallMetadataSweepContinuation,
  type RecallMetadataSweepContinuationStore,
  type RecallSessionMetadataFilesystem,
} from './scan-recall-session-metadata.js';

const DIAGNOSTIC_SAMPLE_COUNT = 25;
const DIAGNOSTIC_FILE_COUNT = 10_000;

interface RecallMetadataSweepDiagnosticReport {
  hostIdentity: ReturnType<typeof createRecallDiagnosticHostIdentity>;
  sampleMilliseconds: number[];
  sampleCount: number;
  fileCount: number;
  p95Milliseconds: number;
  maximumMilliseconds: number;
  acceptanceBoundMilliseconds: number;
  accepted: boolean;
}

function percentile95(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(Math.ceil(sorted.length * 0.95) - 1, 0)] ?? 0;
}

function createGeneratedMetadataFilesystem(): RecallSessionMetadataFilesystem {
  const names = Array.from(
    { length: DIAGNOSTIC_FILE_COUNT },
    (_, index) => `session-${String(index).padStart(5, '0')}.jsonl`,
  );
  return {
    async readDirectory() {
      return names;
    },
    async statPath() {
      return {
        isDirectory: false,
        isFile: true,
        sizeBytes: 1_024,
        modifiedAtEpochMilliseconds: 1_753_315_200_000,
        sourceDevice: '10',
        sourceInode: '20',
      };
    },
  };
}

function createDiagnosticContinuationStore(): RecallMetadataSweepContinuationStore {
  let continuation: RecallMetadataSweepContinuation | null = null;
  return {
    async readContinuation() {
      return continuation;
    },
    async writeContinuation(nextContinuation) {
      continuation = nextContinuation;
    },
    async clearContinuation() {
      continuation = null;
    },
  };
}

async function measureRecallMetadataSweep(): Promise<RecallMetadataSweepDiagnosticReport> {
  const directory = await mkdtemp(join(tmpdir(), 'recall-metadata-diagnostic-'));
  try {
    await assertRecallTestDataRoot({ testDataRoot: directory, repositoryRoot: process.cwd() });
    const filesystem = createGeneratedMetadataFilesystem();
    const samples: number[] = [];
    for (let sample = 0; sample < DIAGNOSTIC_SAMPLE_COUNT; sample += 1) {
      const startedAt = performance.now();
      await scanRecallSessionMetadata({
        sessionRootDirectory: join(directory, 'session-metadata'),
        controlDirectory: join(directory, 'control'),
        filesystem,
        continuationStore: createDiagnosticContinuationStore(),
        monotonicNowMilliseconds: performance.now.bind(performance),
      });
      samples.push(performance.now() - startedAt);
    }
    const p95Milliseconds = percentile95(samples);
    return {
      hostIdentity: createRecallDiagnosticHostIdentity(),
      sampleMilliseconds: samples,
      sampleCount: DIAGNOSTIC_SAMPLE_COUNT,
      fileCount: DIAGNOSTIC_FILE_COUNT,
      p95Milliseconds,
      maximumMilliseconds: Math.max(...samples),
      acceptanceBoundMilliseconds: RECALL_METADATA_SWEEP_MAX_ELAPSED_MILLISECONDS,
      accepted: p95Milliseconds <= RECALL_METADATA_SWEEP_MAX_ELAPSED_MILLISECONDS,
    };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

void test(
  'target host generated metadata sweep stays within 500 ms p95 at 10,000 files',
  { skip: process.env.PI_RECALL_RUN_METADATA_DIAGNOSTIC !== '1' },
  async () => {
    const report = await measureRecallMetadataSweep();
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    const evidenceOutputDirectory = process.env.PI_RECALL_EVIDENCE_OUTPUT_DIRECTORY;
    if (evidenceOutputDirectory !== undefined) {
      await writeFile(
        join(evidenceOutputDirectory, 'metadata-sweep.json'),
        `${JSON.stringify(report, null, 2)}\n`,
      );
    }
    assert.equal(report.accepted, true, JSON.stringify(report));
  },
);

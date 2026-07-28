import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { performance } from 'node:perf_hooks';
import test from 'node:test';

import { RecallWorkMarkerTrigger } from './enums.js';
import { publishRecallWorkMarker } from './publish-recall-work-marker.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  createRecallWorkMarkerId,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
} from './recall-work-marker.js';

const DIAGNOSTIC_SAMPLE_COUNT = 50;
const MAX_COMBINED_P95_MILLISECONDS = 25;

interface RecallMarkerPublicationDiagnosticReport {
  sampleCount: number;
  publicationP95Milliseconds: number;
  detachedSpawnP95Milliseconds: number;
  combinedP95Milliseconds: number;
  maximumCombinedMilliseconds: number;
  acceptanceBoundMilliseconds: number;
  accepted: boolean;
}

function percentile95(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(Math.ceil(sorted.length * 0.95) - 1, 0)] ?? 0;
}

async function measureRecallMarkerPublication(): Promise<RecallMarkerPublicationDiagnosticReport> {
  const directory = await mkdtemp(join(tmpdir(), 'recall-marker-diagnostic-'));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'recall', 'markers', 'pending');
  const physicalSessionPath = join(sessionsDirectory, 'diagnostic-session.jsonl');
  const publicationMilliseconds: number[] = [];
  const detachedSpawnMilliseconds: number[] = [];
  const combinedMilliseconds: number[] = [];
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');
  try {
    for (let sample = 1; sample <= DIAGNOSTIC_SAMPLE_COUNT; sample += 1) {
      const identity = {
        version: RECALL_WORK_MARKER_VERSION,
        physicalSessionId: 'diagnostic-physical-session',
        physicalSessionPath,
        runtimeInstanceId: 'diagnostic-runtime-instance',
        runtimeSequence: sample,
        createdAtEpochMilliseconds: sample,
        trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
      } as const;
      const marker: RecallWorkMarker = {
        ...identity,
        markerId: createRecallWorkMarkerId(identity),
      };
      let spawnMilliseconds = 0;
      const combinedStartedAt = performance.now();
      await publishRecallWorkMarker(marker, {
        markerSpoolDirectory,
        trustedSessionRoots: [sessionsDirectory],
        workerSignal: {
          signalDetachedWorker() {
            const spawnStartedAt = performance.now();
            const child = spawn(process.execPath, ['-e', 'process.exit(0)'], {
              detached: true,
              stdio: 'ignore',
            });
            child.once('error', (error) => {
              process.emitWarning(
                `Recall marker diagnostic detached child failed [${readNodeErrorCode(error) ?? 'UNKNOWN'}]`,
              );
            });
            child.unref();
            spawnMilliseconds = performance.now() - spawnStartedAt;
          },
        },
      });
      const combined = performance.now() - combinedStartedAt;
      detachedSpawnMilliseconds.push(spawnMilliseconds);
      publicationMilliseconds.push(Math.max(combined - spawnMilliseconds, 0));
      combinedMilliseconds.push(combined);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
  const combinedP95Milliseconds = percentile95(combinedMilliseconds);
  return {
    sampleCount: DIAGNOSTIC_SAMPLE_COUNT,
    publicationP95Milliseconds: percentile95(publicationMilliseconds),
    detachedSpawnP95Milliseconds: percentile95(detachedSpawnMilliseconds),
    combinedP95Milliseconds,
    maximumCombinedMilliseconds: Math.max(...combinedMilliseconds),
    acceptanceBoundMilliseconds: MAX_COMBINED_P95_MILLISECONDS,
    accepted: combinedP95Milliseconds <= MAX_COMBINED_P95_MILLISECONDS,
  };
}

void test(
  'target host recall marker publication and detached spawn stay within 25 ms p95',
  { skip: process.env.PI_RECALL_RUN_MARKER_DIAGNOSTIC !== '1' },
  async () => {
    const report = await measureRecallMarkerPublication();
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    assert.equal(report.accepted, true, JSON.stringify(report));
  },
);

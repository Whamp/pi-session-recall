import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import test from 'node:test';

import {
  RecallBackgroundIndexProcessState,
  RecallFixedSnapshotBuildOperationPhase,
} from './enums.js';
import type { RecallBackgroundIndexGenerationStatus } from './recall-background-index-build.js';
import { shouldTerminateStalledRecallBackgroundIndexWorker } from './recall-background-index-worker.js';

void test('background index worker refuses to run without a request path', async () => {
  const child = spawn(
    process.execPath,
    ['--import', 'tsx', new URL('./recall-background-index-worker.ts', import.meta.url).pathname],
    {
      stdio: 'ignore',
    },
  );
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });

  assert.equal(exitCode, 1);
});

const OBSERVED_AT = Date.parse('2026-08-01T01:00:00.000Z');
const FAILURE_THRESHOLD_MILLISECONDS = 30 * 60 * 1_000;

function createWatchdogStatus(
  operationStartedAt: string,
  heartbeatAt: string,
): RecallBackgroundIndexGenerationStatus {
  return {
    version: 1,
    buildId: 'build-watchdog',
    generationId: 'generation-watchdog',
    embeddingProfileId: 'embedding-profile-watchdog',
    processId: 123,
    processState: RecallBackgroundIndexProcessState.RUNNING,
    startedAt: '2026-08-01T00:00:00.000Z',
    updatedAt: heartbeatAt,
    heartbeatAt,
    completedAt: null,
    progress: null,
    activeOperation: {
      phase: RecallFixedSnapshotBuildOperationPhase.DENSE_STORE_CLOSE,
      startedAt: operationStartedAt,
      sessionsRootRelativePath: 'stuck-source.jsonl',
    },
    latestCompletedOperation: null,
    stallDiagnostic: null,
    latestCheckpoint: null,
    latestActionableError: null,
  };
}

void test('background rebuild watchdog requires both a stale operation and stale heartbeat', () => {
  assert.equal(
    shouldTerminateStalledRecallBackgroundIndexWorker(
      createWatchdogStatus('2026-08-01T00:20:00.000Z', '2026-08-01T00:20:00.000Z'),
      OBSERVED_AT,
      FAILURE_THRESHOLD_MILLISECONDS,
    ),
    true,
  );
  assert.equal(
    shouldTerminateStalledRecallBackgroundIndexWorker(
      createWatchdogStatus('2026-08-01T00:20:00.000Z', '2026-08-01T00:59:55.000Z'),
      OBSERVED_AT,
      FAILURE_THRESHOLD_MILLISECONDS,
    ),
    false,
  );
});

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

import {
  RecallGenerationCutoverState,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
} from './recall-generation-state.js';
import {
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import {
  createRecallWorkMarkerId,
  encodeRecallWorkMarker,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
} from './recall-work-marker.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { runRecallIncrementalWorker } from './run-recall-incremental-worker.js';

interface WorkerFixture {
  controlDirectory: string;
  markerQuarantineDirectory: string;
  markerSpoolDirectory: string;
  sessionsDirectory: string;
  workerOwnershipLockPath: string;
  marker: RecallWorkMarker;
  publishMarker(): Promise<void>;
}

async function createWorkerFixture(
  t: test.TestContext,
  trigger:
    | RecallWorkMarkerTrigger.ACTIVITY
    | RecallWorkMarkerTrigger.ARRIVAL = RecallWorkMarkerTrigger.ACTIVITY,
): Promise<WorkerFixture> {
  const directory = await mkdtemp(join(tmpdir(), 'run-recall-incremental-worker-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'recall', 'markers', 'pending');
  const markerQuarantineDirectory = join(directory, 'recall', 'markers', 'quarantine');
  const controlDirectory = join(directory, 'recall', 'markers', 'control');
  const workerOwnershipLockPath = join(directory, 'recall', 'incremental-worker.lock');
  const physicalSessionPath = join(sessionsDirectory, 'session.jsonl');
  await mkdir(markerSpoolDirectory, { recursive: true });
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');
  const identity = {
    version: RECALL_WORK_MARKER_VERSION,
    physicalSessionId: 'physical-session-1',
    physicalSessionPath,
    runtimeInstanceId: 'runtime-1',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1_753_315_200_000,
    trigger: { kind: trigger },
  } as const;
  const marker = { ...identity, markerId: createRecallWorkMarkerId(identity) };
  return {
    controlDirectory,
    markerQuarantineDirectory,
    markerSpoolDirectory,
    sessionsDirectory,
    workerOwnershipLockPath,
    marker,
    async publishMarker() {
      await writeFile(
        join(markerSpoolDirectory, `${marker.markerId}.json`),
        await encodeRecallWorkMarker(marker, { trustedSessionRoots: [sessionsDirectory] }),
      );
    },
  };
}

void test('incremental worker launch uses nonblocking kernel flock without PID or lease inference', async () => {
  const source = await readFile(
    new URL('./publish-recall-work-marker.ts', import.meta.url),
    'utf8',
  );
  assert.match(source, /spawn\(\s*'\/usr\/bin\/flock'/u);
  assert.match(source, /'--nonblock'/u);
  assert.doesNotMatch(source, /pid.?file|process.?alive|stale.?time|kill\([^)]*,\s*0\)/iu);
});

void test('incremental worker exits empty before heavy imports and a later signal drains visible work', async (t) => {
  const fixture = await createWorkerFixture(t);
  let heavyImportCount = 0;
  const options = {
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
  };

  const beforePublication = await runRecallIncrementalWorker(options);
  assert.equal(beforePublication.workPlan.workItems.length, 0);
  assert.equal(beforePublication.heavyDependenciesLoaded, false);
  assert.equal(heavyImportCount, 0);

  await fixture.publishMarker();
  const afterLaterSignal = await runRecallIncrementalWorker(options);
  assert.equal(afterLaterSignal.workPlan.workItems.length, 1);
  assert.equal(afterLaterSignal.heavyDependenciesLoaded, true);
  assert.equal(heavyImportCount, 1);
  assert.equal(
    await readFile(
      join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`),
      'utf8',
    ).then((source) => source.length > 0),
    true,
  );
});

function createWorkerPhysicalProjection(fixture: WorkerFixture): PhysicalSessionProjection {
  const generationId = 'generation-1';
  const physicalSessionId = fixture.marker.physicalSessionId;
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId,
    physicalSessionId,
    sourcePath: fixture.marker.physicalSessionPath,
    sourceDevice: '10',
    sourceInode: '20',
    appendCursorBytes: 3,
    appendCursorLines: 1,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: null,
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

void test('building generation freezes incremental commits while retaining published markers', async (t) => {
  const fixture = await createWorkerFixture(t);
  await fixture.publishMarker();
  const generationRegistryPath = join(fixture.controlDirectory, 'generation-registry.json');
  const activePointer = createRecallActiveGenerationPointer('generation-1');
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: 'generation-1',
    buildingGenerationId: 'generation-building',
    rollbackGenerationId: null,
    activePointerChecksum: activePointer.checksum,
    generations: [
      {
        generationId: 'generation-1',
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 2,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
      },
      {
        generationId: 'generation-building',
        state: RecallGenerationCutoverState.BUILDING,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 2,
        indexManifestFingerprint: 'b'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 3,
        stateChangedAtEpochMilliseconds: 3,
        rebuildStartMarkerId: fixture.marker.markerId,
      },
    ],
  });
  let heavyImportCount = 0;

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    generationRegistryPath,
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
  });

  assert.equal(result.commitsFrozen, true);
  assert.equal(result.heavyDependenciesLoaded, false);
  assert.equal(result.workPlan.workItems.length, 1);
  assert.equal(heavyImportCount, 0);
  assert.equal(
    await readFile(
      join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`),
      'utf8',
    ).then((source) => source.length > 0),
    true,
  );
});

void test('empty ordinary worker pass completes replay-pending generation backlog', async (t) => {
  const fixture = await createWorkerFixture(t);
  const generationRegistryPath = join(fixture.controlDirectory, 'generation-registry.json');
  const activeGenerationPointerPath = join(fixture.controlDirectory, 'active-generation.json');
  const backlogSummaryPath = join(fixture.controlDirectory, 'backlog-summary.json');
  const lockPath = join(fixture.controlDirectory, 'operation.lock');
  const pointer = createRecallActiveGenerationPointer('generation-1');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, pointer);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: 'generation-1',
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: 'generation-1',
        state: RecallGenerationCutoverState.REPLAY_PENDING,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 2,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    generationRegistryPath,
    generationReplayCompletion: {
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath,
      lockPath,
    },
    trustedSessionRoots: [fixture.sessionsDirectory],
  });

  assert.equal(result.generationReplayCompleted, true);
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.generations[0]?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
});

void test('arrival metadata sweep lazily loads active projections and invokes deletion reconciliation', async (t) => {
  const fixture = await createWorkerFixture(t, RecallWorkMarkerTrigger.ARRIVAL);
  await fixture.publishMarker();
  const physicalProjection = createWorkerPhysicalProjection(fixture);
  let inventoryLoadCount = 0;
  let reconciliationCount = 0;

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadKnownSourceInventory() {
      inventoryLoadCount += 1;
      return {
        knownSources: [
          {
            physicalSessionId: physicalProjection.physicalSessionId,
            relativePath: 'session.jsonl',
          },
        ],
        physicalProjections: [physicalProjection],
      };
    },
    async loadHeavyDependencies() {},
    async reconcileDeletion(metadataSweep, physicalProjections) {
      reconciliationCount += 1;
      assert.equal(metadataSweep.status, 'complete');
      assert.deepEqual(physicalProjections, [physicalProjection]);
    },
  });

  assert.equal(result.heavyDependenciesLoaded, true);
  assert.equal(inventoryLoadCount, 1);
  assert.equal(reconciliationCount, 1);
});

async function waitForProbeLines(probePath: string, expectedLineCount: number): Promise<void> {
  const deadline = performance.now() + 3_000;
  while (performance.now() < deadline) {
    try {
      const lines = (await readFile(probePath, 'utf8')).trim().split('\n').filter(Boolean);
      if (lines.length >= expectedLineCount) {
        return;
      }
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ENOENT') {
        throw error;
      }
    }
    await new Promise((resolve) => {
      setTimeout(resolve, 10);
    });
  }
  throw new Error('Incremental worker import probe timed out');
}

function spawnLockedWorker(lockPath: string, harnessPath: string) {
  const startedAt = performance.now();
  const child = spawn(
    '/usr/bin/flock',
    ['--nonblock', lockPath, process.execPath, '--import', 'tsx', harnessPath],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );
  let standardError = '';
  child.stderr.on('data', (chunk: Buffer) => {
    standardError += chunk.toString('utf8');
  });
  return {
    child,
    completed: new Promise<{
      code: number | null;
      elapsedMilliseconds: number;
      standardError: string;
    }>((resolve) => {
      child.once('exit', (code) => {
        resolve({
          code,
          elapsedMilliseconds: performance.now() - startedAt,
          standardError,
        });
      });
    }),
  };
}

async function writeWorkerHarness(
  harnessPath: string,
  fixture: WorkerFixture,
  probePath: string,
  delayMilliseconds: number,
  crashAfterImport: boolean,
): Promise<void> {
  const moduleUrl = pathToFileURL(
    fileURLToPath(new URL('./run-recall-incremental-worker.ts', import.meta.url)),
  ).href;
  await writeFile(
    harnessPath,
    `import { appendFile } from 'node:fs/promises';\n` +
      `import { runRecallIncrementalWorker } from ${JSON.stringify(moduleUrl)};\n` +
      `await runRecallIncrementalWorker({\n` +
      `  markerSpoolDirectory: ${JSON.stringify(fixture.markerSpoolDirectory)},\n` +
      `  markerQuarantineDirectory: ${JSON.stringify(fixture.markerQuarantineDirectory)},\n` +
      `  controlDirectory: ${JSON.stringify(fixture.controlDirectory)},\n` +
      `  targetGenerationId: 'generation-1',\n` +
      `  trustedSessionRoots: [${JSON.stringify(fixture.sessionsDirectory)}],\n` +
      `  async loadHeavyDependencies() {\n` +
      `    await appendFile(${JSON.stringify(probePath)}, String(process.pid) + '\\n');\n` +
      (crashAfterImport
        ? `    process.kill(process.pid, 'SIGKILL');\n`
        : `    await new Promise((resolve) => setTimeout(resolve, ${delayMilliseconds}));\n`) +
      `  },\n` +
      `});\n`,
  );
}

void test('kernel flock admits one worker, rejects losers promptly, and releases after winner crash', async (t) => {
  const fixture = await createWorkerFixture(t);
  await fixture.publishMarker();
  const harnessPath = join(fixture.controlDirectory, 'worker-harness.mts');
  const probePath = join(fixture.controlDirectory, 'heavy-imports.txt');
  await mkdir(fixture.controlDirectory, { recursive: true });
  await writeWorkerHarness(harnessPath, fixture, probePath, 500, false);

  const winner = spawnLockedWorker(fixture.workerOwnershipLockPath, harnessPath);
  const earlyWinnerResult = await Promise.race([
    waitForProbeLines(probePath, 1).then(() => null),
    winner.completed,
  ]);
  if (earlyWinnerResult !== null) {
    throw new Error(
      `Incremental worker exited before import probe: ${earlyWinnerResult.standardError}`,
    );
  }
  const loser = spawnLockedWorker(fixture.workerOwnershipLockPath, harnessPath);
  const loserResult = await loser.completed;
  assert.notEqual(loserResult.code, 0);
  assert.equal(loserResult.elapsedMilliseconds < 250, true);
  assert.equal((await readFile(probePath, 'utf8')).trim().split('\n').length, 1);
  assert.equal((await winner.completed).code, 0);

  await writeWorkerHarness(harnessPath, fixture, probePath, 0, true);
  const crashingWinner = spawnLockedWorker(fixture.workerOwnershipLockPath, harnessPath);
  await waitForProbeLines(probePath, 2);
  assert.notEqual((await crashingWinner.completed).code, 0);

  await writeWorkerHarness(harnessPath, fixture, probePath, 0, false);
  const recoveredWinner = spawnLockedWorker(fixture.workerOwnershipLockPath, harnessPath);
  assert.equal((await recoveredWinner.completed).code, 0);
  assert.equal((await readFile(probePath, 'utf8')).trim().split('\n').length, 3);
});

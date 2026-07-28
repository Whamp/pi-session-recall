import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

import {
  RecallBacklogFailureCategory,
  RecallDiagnosticErrorCategory,
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallEligibilityThreshold,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallMetadataSweepStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  createRecallActiveGenerationPointer,
  decodeRecallBacklogSummary,
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
  type RecallWorkMarkerIdentity,
  type RecallWorkMarkerTriggerPayload,
} from './recall-work-marker.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  runRecallIncrementalWorker,
  runRecallIncrementalWorkerDiagnosticBoundary,
  writeRecallIncrementalWorkerBacklog,
  type RunRecallIncrementalWorkerOptions,
} from './run-recall-incremental-worker.js';
import type { RecallIncrementalDiagnosticCompletion } from './recall-operation-diagnostics.js';
import type { CommittedIncrementalRecallWorkPlan } from './transfer-incremental-recall-work-plan.js';

async function writeFailureVisibilityRegistry(
  generationRegistryPath: string,
  generationId = 'generation-detached',
): Promise<void> {
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'd'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 10,
        stateChangedAtEpochMilliseconds: 20,
        rebuildStartMarkerId: null,
      },
    ],
  });
}

async function waitForDetachedWorkerCompletion(completionPath: string): Promise<void> {
  const deadline = performance.now() + 5_000;
  while (performance.now() < deadline) {
    try {
      await access(completionPath);
      return;
    } catch (error) {
      if (readNodeErrorCode(error) !== 'ENOENT') {
        throw error;
      }
    }
    await new Promise((resolve) => {
      setTimeout(resolve, 10);
    });
  }
  throw new Error('Detached incremental worker failure visibility probe timed out');
}

interface DetachedFailureVisibilityResult {
  backlogSummaryPath: string;
  diagnosticLogPath: string;
  warningPath: string;
}

type DetachedFailureVisibilityScenario =
  | 'both_sinks_fail'
  | 'deletion_halt'
  | 'diagnostics_fallback'
  | 'successful_clear'
  | 'top_level_failure';

const PRIVATE_FAILURE_SENTINEL =
  'private conversation; private query; private tool output; private embedding; /private/source/path';

async function runDetachedFailureVisibilityProbe(
  t: test.TestContext,
  scenario: DetachedFailureVisibilityScenario,
): Promise<DetachedFailureVisibilityResult> {
  const directory = await mkdtemp(join(tmpdir(), `recall-detached-${scenario}-`));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const backlogSummaryPath = join(directory, 'backlog-summary.json');
  const diagnosticLogPath = join(directory, 'incremental-diagnostics.jsonl');
  const warningPath = join(directory, 'warnings.txt');
  const completionPath = join(directory, 'complete');
  const harnessPath = join(directory, 'detached-worker-harness.mts');
  const generationId = 'generation-detached';
  await writeFailureVisibilityRegistry(generationRegistryPath, generationId);

  if (scenario === 'top_level_failure') {
    await writeRecallIncrementalWorkerBacklog(
      {
        backlogSummaryPath,
        generationRegistryPath,
        targetGenerationId: generationId,
        nowEpochMilliseconds: () => 100,
      },
      null,
      {
        workPlan: {
          workItems: [
            {
              marker: {
                physicalSessionId: 'physical-detached',
                createdAtEpochMilliseconds: 50,
              },
            },
          ],
        },
      },
    );
    await writeFile(generationRegistryPath, 'invalid registry recovery state');
  }
  if (scenario === 'diagnostics_fallback' || scenario === 'both_sinks_fail') {
    await mkdir(diagnosticLogPath);
  }
  if (scenario === 'both_sinks_fail') {
    await mkdir(backlogSummaryPath);
  }

  const workerModuleUrl = pathToFileURL(
    fileURLToPath(new URL('./run-recall-incremental-worker.ts', import.meta.url)),
  ).href;
  const diagnosticsModuleUrl = pathToFileURL(
    fileURLToPath(new URL('./recall-operation-diagnostics.ts', import.meta.url)),
  ).href;
  const enumsModuleUrl = pathToFileURL(fileURLToPath(new URL('./enums.ts', import.meta.url))).href;
  const scenarioBody = {
    both_sinks_fail: `for (let index = 0; index < 2; index += 1) { try { await runRecallIncrementalWorkerDiagnosticBoundary({ operationDiagnostics: diagnostics, persistFailure: () => writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED), async run() { throw new Error('private query ' + index); } }); } catch {} }\n`,
    deletion_halt: `await reportRecallIncrementalWorkerDeletionHalt({ operationDiagnostics: diagnostics, generationId: ${JSON.stringify(generationId)}, generationState: RecallGenerationCutoverState.ACTIVE, haltResult: { halted: true, consideredPhysicalSessionCount: 1, sourceMissingRecordedCount: 0, sourceMissingClearedCount: 0, confirmedSourceDeletionCount: 0, removedEvidenceOccurrenceCount: 0, removedLogicalProjectionCount: 0, removedPhysicalProjectionCount: 0, acknowledgedCheckpointCount: 0, haltCategoryCounts: { [RecallConfirmedDeletionHaltCategory.ACTIVE_GENERATION_CHANGED]: 1 } }, workPlan, nowEpochMilliseconds: () => 100, persistFailure: () => writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.CONFIRMED_DELETION_HALTED, { workPlan }) });\n`,
    diagnostics_fallback: `try { await runRecallIncrementalWorkerDiagnosticBoundary({ operationDiagnostics: diagnostics, persistFailure: () => writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED), async run() { throw new Error('private tool output must not persist'); } }); } catch {}\n`,
    successful_clear: `await writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED, { workPlan });\nawait writeRecallIncrementalWorkerBacklog(paths, null, { workPlan });\n`,
    top_level_failure: `try { await runRecallIncrementalWorkerDiagnosticBoundary({ operationDiagnostics: diagnostics, persistFailure: () => writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED), async run() { throw new Error(${JSON.stringify(PRIVATE_FAILURE_SENTINEL)}); } }); } catch {}\n`,
  }[scenario];

  await writeFile(
    harnessPath,
    `import { appendFileSync } from 'node:fs';\n` +
      `import { writeFile } from 'node:fs/promises';\n` +
      `import { RecallBacklogFailureCategory, RecallConfirmedDeletionHaltCategory, RecallDiagnosticsMode, RecallGenerationCutoverState } from ${JSON.stringify(enumsModuleUrl)};\n` +
      `import { createRecallOperationDiagnostics } from ${JSON.stringify(diagnosticsModuleUrl)};\n` +
      `import { reportRecallIncrementalWorkerDeletionHalt, runRecallIncrementalWorkerDiagnosticBoundary, writeRecallIncrementalWorkerBacklog, writeRecallIncrementalWorkerFailureBacklog } from ${JSON.stringify(workerModuleUrl)};\n` +
      `const paths = { backlogSummaryPath: ${JSON.stringify(backlogSummaryPath)}, generationRegistryPath: ${JSON.stringify(generationRegistryPath)}, targetGenerationId: ${JSON.stringify(generationId)}, nowEpochMilliseconds: () => 100 };\n` +
      `const notifyWarning = (message) => appendFileSync(${JSON.stringify(warningPath)}, message + '\\n');\n` +
      `const diagnostics = createRecallOperationDiagnostics({ mode: RecallDiagnosticsMode.ALL, activeLogPath: ${JSON.stringify(diagnosticLogPath)}, retainedLogPath: ${JSON.stringify(join(directory, 'incremental-diagnostics.previous.jsonl'))}, notifyWarning, onPersistenceFailure: () => writeRecallIncrementalWorkerFailureBacklog(paths, RecallBacklogFailureCategory.DIAGNOSTICS_PERSISTENCE_FAILED) });\n` +
      `const workPlan = { workItems: [{ marker: { physicalSessionId: 'physical-detached', createdAtEpochMilliseconds: 50 } }] };\n` +
      scenarioBody +
      `await writeFile(${JSON.stringify(completionPath)}, 'complete');\n`,
  );
  const child = spawn(process.execPath, ['--import', 'tsx', harnessPath], {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();
  await waitForDetachedWorkerCompletion(completionPath);
  return { backlogSummaryPath, diagnosticLogPath, warningPath };
}

void test('detached ignored-stdio deletion halt persists its scalar safeguard state', async (t) => {
  const result = await runDetachedFailureVisibilityProbe(t, 'deletion_halt');
  const diagnosticSource = await readFile(result.diagnosticLogPath, 'utf8');
  const backlogSource = await readFile(result.backlogSummaryPath, 'utf8');
  assert.match(diagnosticSource, /"operationKind":"deletion_reconciliation"/u);
  assert.match(diagnosticSource, /"deletionSafeguardCategory":"active_generation_changed"/u);
  assert.match(diagnosticSource, /"backlogPendingEligibleSessionCount":1/u);
  assert.match(diagnosticSource, /"backlogOldestEligibleMarkerAgeMilliseconds":50/u);
  assert.equal(
    decodeRecallBacklogSummary(backlogSource).lastFailureCategory,
    RecallBacklogFailureCategory.CONFIRMED_DELETION_HALTED,
  );
  await assert.rejects(() => readFile(result.warningPath), { code: 'ENOENT' });
});

void test('detached diagnostics sink failure falls back to the atomic scalar backlog', async (t) => {
  const result = await runDetachedFailureVisibilityProbe(t, 'diagnostics_fallback');
  const backlogSource = await readFile(result.backlogSummaryPath, 'utf8');
  assert.equal(
    decodeRecallBacklogSummary(backlogSource).lastFailureCategory,
    RecallBacklogFailureCategory.DIAGNOSTICS_PERSISTENCE_FAILED,
  );
  assert.doesNotMatch(backlogSource, /private tool output/u);
  await assert.rejects(() => readFile(result.warningPath), { code: 'ENOENT' });
});

void test('detached failures warn once only when both durable scalar sinks fail', async (t) => {
  const result = await runDetachedFailureVisibilityProbe(t, 'both_sinks_fail');
  const warnings = (await readFile(result.warningPath, 'utf8')).trim().split('\n');
  assert.equal(warnings.length, 1);
  assert.equal(
    warnings[0],
    'Recall diagnostics disabled after local log and fallback persistence failed.',
  );
  assert.doesNotMatch(warnings[0] ?? '', /private query/u);
});

void test('detached successful backlog update clears failure and refreshes scalar count and age', async (t) => {
  const result = await runDetachedFailureVisibilityProbe(t, 'successful_clear');
  const summary = decodeRecallBacklogSummary(await readFile(result.backlogSummaryPath, 'utf8'));
  assert.equal(summary.lastFailureCategory, null);
  assert.equal(summary.pendingEligibleSessionCount, 1);
  assert.equal(summary.oldestEligibleMarkerAgeMilliseconds, 50);
  assert.equal(summary.observedAtEpochMilliseconds, 100);
});

void test('detached ignored-stdio top-level failure persists scalar diagnostics and backlog', async (t) => {
  const result = await runDetachedFailureVisibilityProbe(t, 'top_level_failure');
  const diagnosticSource = await readFile(result.diagnosticLogPath, 'utf8');
  const backlogSource = await readFile(result.backlogSummaryPath, 'utf8');
  assert.match(diagnosticSource, /"operationKind":"incremental_worker"/u);
  assert.match(diagnosticSource, /"status":"failed"/u);
  const backlogSummary = decodeRecallBacklogSummary(backlogSource);
  assert.equal(
    backlogSummary.lastFailureCategory,
    RecallBacklogFailureCategory.INCREMENTAL_WORKER_FAILED,
  );
  assert.equal(backlogSummary.pendingEligibleSessionCount, 1);
  assert.equal(backlogSummary.oldestEligibleMarkerAgeMilliseconds, 50);
  assert.doesNotMatch(
    `${diagnosticSource}${backlogSource}`,
    new RegExp(PRIVATE_FAILURE_SENTINEL, 'u'),
  );
  await assert.rejects(() => readFile(result.warningPath), { code: 'ENOENT' });
});

void test('worker diagnostic boundary records and flushes an early executable failure', async () => {
  const diagnostics: RecallIncrementalDiagnosticCompletion[] = [];
  let flushed = false;
  let monotonicMilliseconds = 0;

  await assert.rejects(
    () =>
      runRecallIncrementalWorkerDiagnosticBoundary({
        operationDiagnostics: {
          recordDurableIncrementalFailure(completion) {
            diagnostics.push(completion);
          },
          async flush() {
            flushed = true;
          },
        },
        async persistFailure() {},
        monotonicMilliseconds() {
          monotonicMilliseconds += 5;
          return monotonicMilliseconds;
        },
        async run() {
          throw new Error('early recovery failed');
        },
      }),
    /early recovery failed/u,
  );

  assert.equal(flushed, true);
  assert.equal(diagnostics.length, 1);
  assert.equal(diagnostics[0]?.operationKind, RecallDiagnosticOperationKind.INCREMENTAL_WORKER);
  assert.equal(diagnostics[0]?.status, RecallDiagnosticStatus.FAILED);
  assert.equal(diagnostics[0]?.errorCategory, RecallDiagnosticErrorCategory.OPERATION_FAILED);
  assert.equal(diagnostics[0]?.metrics.elapsedMilliseconds, 5);
});

interface WorkerFixture {
  controlDirectory: string;
  markerQuarantineDirectory: string;
  markerSpoolDirectory: string;
  sessionsDirectory: string;
  physicalSessionPath: string;
  workerOwnershipLockPath: string;
  marker: RecallWorkMarker;
  publishMarker(marker?: RecallWorkMarker): Promise<void>;
}

async function createWorkerFixture(
  t: test.TestContext,
  trigger: RecallWorkMarkerTriggerPayload = { kind: RecallWorkMarkerTrigger.ACTIVITY },
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
  const createdAtEpochMilliseconds = 1_753_315_200_000;
  await utimes(
    physicalSessionPath,
    createdAtEpochMilliseconds / 1_000,
    createdAtEpochMilliseconds / 1_000,
  );
  const identity: RecallWorkMarkerIdentity = {
    version: RECALL_WORK_MARKER_VERSION,
    physicalSessionId: 'physical-session-1',
    physicalSessionPath,
    runtimeInstanceId: 'runtime-1',
    runtimeSequence: 1,
    createdAtEpochMilliseconds,
    trigger,
  };
  const marker: RecallWorkMarker = {
    ...identity,
    markerId: createRecallWorkMarkerId(identity),
  };
  return {
    controlDirectory,
    markerQuarantineDirectory,
    markerSpoolDirectory,
    sessionsDirectory,
    physicalSessionPath,
    workerOwnershipLockPath,
    marker,
    async publishMarker(markerToPublish = marker) {
      await writeFile(
        join(markerSpoolDirectory, `${markerToPublish.markerId}.json`),
        await encodeRecallWorkMarker(markerToPublish, {
          trustedSessionRoots: [sessionsDirectory],
        }),
      );
    },
  };
}

function committedTransferOutcome(
  committedDocumentCount: number,
): CommittedIncrementalRecallWorkPlan {
  return {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount,
  };
}

void test('incremental worker launch uses kernel flock without PID or lease inference', async () => {
  const publicationSource = await readFile(
    new URL('./publish-recall-work-marker.ts', import.meta.url),
    'utf8',
  );
  const signalSource = await readFile(
    new URL('./create-recall-detached-worker-signal.ts', import.meta.url),
    'utf8',
  );
  const workerSource = await readFile(
    new URL('./run-recall-incremental-worker.ts', import.meta.url),
    'utf8',
  );
  const scheduleSource = await readFile(
    new URL('./recall-incremental-worker-schedule.ts', import.meta.url),
    'utf8',
  );
  assert.match(publicationSource, /createRecallDetachedWorkerSignal/u);
  assert.match(signalSource, /spawn\(\s*'\/bin\/sh'/u);
  assert.match(signalSource, /\/usr\/bin\/flock --nonblock/u);
  assert.match(scheduleSource, /sleep .*exec \/usr\/bin\/flock/u);
  assert.doesNotMatch(workerSource, /from '\.\/octen-conversation-tokenizer\.js'/u);
  assert.doesNotMatch(
    workerSource,
    /import \{\s*transferIncrementalRecallWorkPlan[^;]*from '\.\/transfer-incremental-recall-work-plan\.js'/u,
  );
  assert.doesNotMatch(
    `${publicationSource}\n${signalSource}\n${scheduleSource}`,
    /pid.?file|process.?alive|stale.?time|kill\([^)]*,\s*0\)/iu,
  );
});

void test('incremental worker exits empty before heavy imports and a later signal drains visible work', async (t) => {
  const fixture = await createWorkerFixture(t);
  let heavyImportCount = 0;
  let transferCount = 0;
  const options = {
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
    async transferWorkPlan(workPlan: { workItems: readonly unknown[] }) {
      transferCount += 1;
      assert.equal(workPlan.workItems.length, 1);
      return committedTransferOutcome(0);
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
  assert.equal(transferCount, 1);
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

const explicitEligibilityScenarios: Array<{
  name: string;
  trigger: RecallWorkMarkerTriggerPayload;
}> = [
  {
    name: 'compaction',
    trigger: {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'logical-session-1',
      compactionEntryId: 'compaction-1',
    },
  },
  {
    name: 'branch exit',
    trigger: {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical-session-1',
      oldLeafEntryId: 'old-leaf',
      newLeafEntryId: 'new-leaf',
    },
  },
  { name: 'departure', trigger: { kind: RecallWorkMarkerTrigger.DEPARTURE } },
];

for (const scenario of explicitEligibilityScenarios) {
  void test(`incremental worker persists ${scenario.name} until 60 seconds without growth`, async (t) => {
    const fixture = await createWorkerFixture(t, scenario.trigger);
    await fixture.publishMarker();
    let heavyImportCount = 0;
    let transferCount = 0;
    const baseEpochMilliseconds = fixture.marker.createdAtEpochMilliseconds;
    const options = {
      markerSpoolDirectory: fixture.markerSpoolDirectory,
      markerQuarantineDirectory: fixture.markerQuarantineDirectory,
      controlDirectory: fixture.controlDirectory,
      targetGenerationId: 'generation-1',
      trustedSessionRoots: [fixture.sessionsDirectory],
      async loadHeavyDependencies() {
        heavyImportCount += 1;
      },
      async transferWorkPlan() {
        transferCount += 1;
        return committedTransferOutcome(1);
      },
    };

    const deferred = await runRecallIncrementalWorker({
      ...options,
      nowEpochMilliseconds: () => baseEpochMilliseconds + 59_999,
    });
    assert.deepEqual(deferred.transferOutcomes, [
      {
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: RecallEligibilityThreshold.EXPLICIT_EXIT_QUIET,
        readyAtEpochMilliseconds: baseEpochMilliseconds + 60_000,
      },
    ]);
    assert.equal(deferred.heavyDependenciesLoaded, false);
    assert.equal(heavyImportCount, 0);
    assert.equal(transferCount, 0);
    await access(join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`));

    const committed = await runRecallIncrementalWorker({
      ...options,
      nowEpochMilliseconds: () => baseEpochMilliseconds + 60_000,
    });
    assert.deepEqual(committed.transferOutcomes, [
      {
        kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
        committedDocumentCount: 1,
      },
    ]);
    assert.equal(committed.heavyDependenciesLoaded, true);
    assert.equal(heavyImportCount, 1);
    assert.equal(transferCount, 1);
  });
}

void test('incremental worker persists crash-only activity across restarts until 30 minutes', async (t) => {
  const fixture = await createWorkerFixture(t);
  await fixture.publishMarker();
  let heavyImportCount = 0;
  let transferCount = 0;
  const baseEpochMilliseconds = fixture.marker.createdAtEpochMilliseconds;
  const options = {
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
    async transferWorkPlan() {
      transferCount += 1;
      return committedTransferOutcome(1);
    },
  };

  for (let restart = 0; restart < 2; restart += 1) {
    const deferred = await runRecallIncrementalWorker({
      ...options,
      nowEpochMilliseconds: () => baseEpochMilliseconds + 30 * 60_000 - 1,
    });
    assert.deepEqual(deferred.transferOutcomes, [
      {
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
        readyAtEpochMilliseconds: baseEpochMilliseconds + 30 * 60_000,
      },
    ]);
  }
  assert.equal(heavyImportCount, 0);
  assert.equal(transferCount, 0);
  await access(join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`));

  const committed = await runRecallIncrementalWorker({
    ...options,
    nowEpochMilliseconds: () => baseEpochMilliseconds + 30 * 60_000,
  });
  assert.equal(committed.transferOutcomes[0]?.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  assert.equal(heavyImportCount, 1);
  assert.equal(transferCount, 1);
});

void test('incremental worker resets persisted eligibility when the source grows', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.DEPARTURE });
  await fixture.publishMarker();
  const baseEpochMilliseconds = fixture.marker.createdAtEpochMilliseconds;
  await utimes(
    fixture.physicalSessionPath,
    (baseEpochMilliseconds + 50_000) / 1_000,
    (baseEpochMilliseconds + 50_000) / 1_000,
  );
  let heavyImportCount = 0;
  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => baseEpochMilliseconds + 60_000,
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
    async transferWorkPlan() {
      throw new Error('Recall transfer must not run before the reset quiet deadline');
    },
  });

  assert.deepEqual(result.transferOutcomes, [
    {
      kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
      threshold: RecallEligibilityThreshold.EXPLICIT_EXIT_QUIET,
      readyAtEpochMilliseconds: baseEpochMilliseconds + 110_000,
    },
  ]);
  assert.equal(heavyImportCount, 0);
});

void test('arrival does not revoke an already established explicit-exit deadline', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.DEPARTURE });
  await fixture.publishMarker();
  const arrivalIdentity: RecallWorkMarkerIdentity = {
    version: RECALL_WORK_MARKER_VERSION,
    physicalSessionId: fixture.marker.physicalSessionId,
    physicalSessionPath: fixture.marker.physicalSessionPath,
    runtimeInstanceId: fixture.marker.runtimeInstanceId,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: fixture.marker.createdAtEpochMilliseconds + 50_000,
    trigger: { kind: RecallWorkMarkerTrigger.ARRIVAL },
  };
  const arrivalMarker: RecallWorkMarker = {
    ...arrivalIdentity,
    markerId: createRecallWorkMarkerId(arrivalIdentity),
  };
  await fixture.publishMarker(arrivalMarker);
  let transferCount = 0;

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => fixture.marker.createdAtEpochMilliseconds + 60_000,
    async loadHeavyDependencies() {},
    async transferWorkPlan() {
      transferCount += 1;
      return committedTransferOutcome(1);
    },
  });

  assert.equal(transferCount, 1);
  assert.equal(result.transferOutcomes[0]?.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
});

void test('incremental worker retains a prepared transfer deferred to five minutes without preparing twice', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.DEPARTURE });
  await fixture.publishMarker();
  const readyAtEpochMilliseconds = fixture.marker.createdAtEpochMilliseconds + 5 * 60_000;
  let heavyImportCount = 0;
  let transferCount = 0;
  const workerOptions: RunRecallIncrementalWorkerOptions = {
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => fixture.marker.createdAtEpochMilliseconds + 60_000,
    async loadHeavyDependencies() {
      heavyImportCount += 1;
    },
    async transferWorkPlan() {
      transferCount += 1;
      return {
        kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
        threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
        readyAtEpochMilliseconds,
      };
    },
  };

  const result = await runRecallIncrementalWorker(workerOptions);
  const restarted = await runRecallIncrementalWorker({
    ...workerOptions,
    persistedLargeTransferDeferrals: result.largeTransferDeferrals,
  });

  assert.deepEqual(result.transferOutcomes, [
    {
      kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
      threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
      readyAtEpochMilliseconds,
    },
  ]);
  assert.deepEqual(restarted.transferOutcomes, result.transferOutcomes);
  assert.equal(heavyImportCount, 1);
  assert.equal(transferCount, 1);
  assert.equal(result.nextWakeAtEpochMilliseconds, readyAtEpochMilliseconds);
  await access(join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`));
});

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
        sessionProjectionSchemaVersion: 3,
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
        sessionProjectionSchemaVersion: 3,
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

  const backlogSummaryPath = join(fixture.controlDirectory, 'backlog-summary.json');
  await writeRecallIncrementalWorkerBacklog(
    {
      backlogSummaryPath,
      generationRegistryPath,
      targetGenerationId: 'generation-1',
      nowEpochMilliseconds: () => fixture.marker.createdAtEpochMilliseconds + 31 * 60_000,
    },
    null,
    result,
  );
  let backlog = decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  assert.equal(backlog.pendingEligibleSessionCount, 1);
  assert.equal(backlog.oldestEligibleMarkerAgeMilliseconds, 31 * 60_000);
  await writeRecallIncrementalWorkerBacklog(
    {
      backlogSummaryPath,
      generationRegistryPath,
      targetGenerationId: 'generation-1',
      nowEpochMilliseconds: () => fixture.marker.createdAtEpochMilliseconds + 31 * 60_000,
    },
    RecallBacklogFailureCategory.WRITE_FAILED,
  );
  backlog = decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8'));
  assert.equal(backlog.lastFailureCategory, 'write_failed');
});

void test('ordinary worker exposes quarantine failure until replay can complete', async (t) => {
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
        sessionProjectionSchemaVersion: 3,
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

  await writeFile(join(fixture.markerSpoolDirectory, 'corrupt-marker.json'), 'not-json\n');
  const workerOptions = {
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
  };
  const quarantined = await runRecallIncrementalWorker(workerOptions);

  assert.equal(quarantined.generationReplayCompleted, false);
  assert.equal(
    quarantined.replayBlockingFailureCategory,
    RecallBacklogFailureCategory.MARKER_DECODE_FAILED,
  );
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.generations[0]?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );

  await rm(fixture.markerQuarantineDirectory, { recursive: true, force: true });
  const completed = await runRecallIncrementalWorker(workerOptions);
  assert.equal(completed.generationReplayCompleted, true);
  assert.equal(completed.replayBlockingFailureCategory, null);
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.generations[0]?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
});

void test('missing marker-backed source reaches deletion reconciliation before transfer', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.ARRIVAL });
  await fixture.publishMarker();
  const physicalProjection = createWorkerPhysicalProjection(fixture);
  await rm(fixture.physicalSessionPath);
  let reconciliationCount = 0;

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    async loadKnownSourceInventory() {
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
    async loadHeavyDependencies() {
      throw new Error('Recall missing-source reconciliation must not load heavy dependencies');
    },
    async reconcileDeletion(metadataSweep, physicalProjections, missingSourceWorkPlans) {
      reconciliationCount += 1;
      assert.deepEqual(metadataSweep.missingPhysicalSessionIds, [fixture.marker.physicalSessionId]);
      assert.deepEqual(physicalProjections, [physicalProjection]);
      assert.deepEqual(missingSourceWorkPlans?.[0]?.sourceMarkerIds, [fixture.marker.markerId]);
    },
  });

  assert.equal(reconciliationCount, 1);
  assert.equal(result.heavyDependenciesLoaded, false);
});

void test('metadata sweep continuation schedules another worker pass without unrelated activity', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.ARRIVAL });
  await fixture.publishMarker();
  await rm(fixture.physicalSessionPath);
  const nowEpochMilliseconds = fixture.marker.createdAtEpochMilliseconds + 1;

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    controlDirectory: fixture.controlDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => nowEpochMilliseconds,
    async scanSessionMetadata() {
      return {
        sweepId: 'bounded-sweep',
        status: RecallMetadataSweepStatus.CONTINUATION_REQUIRED,
        rootHealthy: true,
        deletionConfirmationSuppressed: true,
        scannedFileCount: 10_000,
        observedSessionFileCount: 0,
        observedSessionMetadata: [],
        observedKnownSourceIdentities: [],
        missingPhysicalSessionIds: [],
        continuationPersisted: true,
        elapsedMilliseconds: 500,
      };
    },
  });

  assert.equal(result.metadataSweep?.status, RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
  assert.equal(result.nextWakeAtEpochMilliseconds, nowEpochMilliseconds);
});

void test('arrival metadata sweep lazily loads active projections and invokes deletion reconciliation', async (t) => {
  const fixture = await createWorkerFixture(t, { kind: RecallWorkMarkerTrigger.ARRIVAL });
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

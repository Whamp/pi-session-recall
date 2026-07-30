import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import {
  access,
  appendFile,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  utimes,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import { promisify } from 'node:util';

import {
  RecallBackgroundIndexProcessState,
  RecallGenerationCutoverState,
  RecallWorkMarkerTrigger,
} from './enums.js';
import { createRecallBackgroundIndexWorkerFixtureService } from './createRecallBackgroundIndexWorkerFixtureService.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { publishRecallWorkMarker } from './publish-recall-work-marker.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { tryAcquireRecallRebuildOwnershipLock } from './recall-rebuild-ownership-lock.js';
import {
  createRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const CLI_PATH = new URL('./pi-session-recall-cli.ts', import.meta.url).pathname;
const CLI_TEST_PATH = new URL('./pi-session-recall-cli.test-utils.ts', import.meta.url).pathname;

async function runRecallOperatorCommand(
  executablePath: string,
  argumentsList: readonly string[],
  dataDirectory: string,
  sessionsDirectory: string,
  environmentOverrides: Readonly<Record<string, string>> = {},
): Promise<{ stdout: string; stderr: string }> {
  return EXEC_FILE_ASYNC(process.execPath, ['--import', 'tsx', executablePath, ...argumentsList], {
    env: {
      ...process.env,
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
      ...environmentOverrides,
    },
  });
}

async function waitForRecallOperatorProcessState(
  dataDirectory: string,
  sessionsDirectory: string,
  expectedState: RecallBackgroundIndexProcessState,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const result = await runRecallOperatorCommand(
      CLI_PATH,
      ['status'],
      dataDirectory,
      sessionsDirectory,
    );
    const output: unknown = JSON.parse(result.stdout);
    if (
      isUnknownRecord(output) &&
      isUnknownRecord(output.process) &&
      output.process.processState === expectedState
    ) {
      return output;
    }
    await sleep(20);
  }
  assert.fail(`Timed out waiting for recall operator process state ${expectedState}`);
}

void test('standalone setup presents stored-width defaults and evidence status', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-setup-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  const packageJson: unknown = JSON.parse(
    await readFile(new URL('../package.json', import.meta.url), 'utf8'),
  );
  assert.ok(isUnknownRecord(packageJson));
  assert.ok(isUnknownRecord(packageJson.bin));
  assert.equal(packageJson.bin['pi-session-recall'], './src/pi-session-recall-cli.ts');

  const result = await EXEC_FILE_ASYNC(CLI_PATH, ['setup'], {
    env: {
      ...process.env,
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
    },
  });
  const output: unknown = JSON.parse(result.stdout);
  assert.ok(isUnknownRecord(output));
  assert.equal(output.command, 'setup');
  assert.deepEqual(output.profiles, [
    {
      profile: 'octen-embedding-4b',
      nativeDimensions: 2560,
      defaultStoredDimensions: 1024,
      allowedStoredDimensions: { minimum: 1, maximum: 2560 },
      evidenceStatus: 'vendor-supported-prefix',
      evidenceSources: ['https://docs.octen.ai/api-reference/embedding'],
    },
    {
      profile: 'embeddinggemma-300m',
      nativeDimensions: 768,
      defaultStoredDimensions: 768,
      allowedStoredDimensions: [768, 512, 256, 128],
      evidenceStatus: 'verified-mrl',
      evidenceSources: ['https://huggingface.co/google/embeddinggemma-300m'],
    },
  ]);
});

void test('standalone setup routes guided status and stored-width selection arguments', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-guided-setup-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const environment = {
    ...process.env,
    PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
    PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
  };

  const statusResult = await EXEC_FILE_ASYNC(
    process.execPath,
    ['--import', 'tsx', CLI_PATH, 'setup', 'status'],
    { env: environment },
  );
  const status: unknown = JSON.parse(statusResult.stdout);
  assert.ok(isUnknownRecord(status));
  assert.equal(status.action, 'status');
  assert.deepEqual(status.configuration, { state: 'unconfigured', embedding: null });

  await assert.rejects(
    () =>
      EXEC_FILE_ASYNC(
        process.execPath,
        [
          '--import',
          'tsx',
          CLI_PATH,
          'setup',
          'select-embeddinggemma',
          '--stored-dimensions',
          '512',
        ],
        { env: environment },
      ),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /requires explicit --approve-download/u);
      return true;
    },
  );
});

void test('standalone status reports fresh recall as not ready without opening generation stores', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  const result = await EXEC_FILE_ASYNC(process.execPath, ['--import', 'tsx', CLI_PATH, 'status'], {
    env: {
      ...process.env,
      PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
      PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
    },
  });
  const output: unknown = JSON.parse(result.stdout);
  assert.deepEqual(output, {
    command: 'status',
    readiness: 'not-ready',
    activeGeneration: null,
    stagingGeneration: null,
    process: null,
    sourceProgress: null,
    latestDurablePhysicalProjection: null,
    replay: { state: 'none', snapshotFileName: null },
    recovery: { required: false, generationIds: [] },
    backlog: null,
    latestActionableError: null,
  });
});

void test('standalone status reports active replay, recovery, and bounded backlog diagnostics', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-populated-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');
  const generationId = 'generation_active';
  const generationDirectory = join(dataDirectory, 'generations', generationId);
  await mkdir(generationDirectory, { recursive: true });
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallActiveGenerationPointer(join(dataDirectory, 'active-generation.json'), pointer);
  await writeRecallGenerationRegistry(join(dataDirectory, 'generation-registry.json'), {
    version: 1,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.REPLAY_PENDING,
        embeddingProfileId: 'embedding-profile-active',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 100,
        stateChangedAtEpochMilliseconds: 200,
        rebuildStartMarkerId: 'marker_1',
        rebuildMarkerWatermark: ['marker_1'],
        replaySnapshotFileName: 'generation-replay-snapshot-activation.json',
        validatedAtEpochMilliseconds: 150,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const backlog = {
    version: 1 as const,
    pendingEligibleSessionCount: 3,
    oldestEligibleMarkerAgeMilliseconds: 500,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    generationState: RecallGenerationCutoverState.REPLAY_PENDING,
    activeGenerationAgeMilliseconds: 100,
    rebuildAgeMilliseconds: null,
    lastFailureCategory: null,
    observedAtEpochMilliseconds: 300,
  };
  await writeRecallBacklogSummary(join(dataDirectory, 'backlog-summary.json'), backlog);
  await writeFile(join(generationDirectory, 'write-recovery.json'), '{}\n', 'utf8');

  const result = await runRecallOperatorCommand(
    CLI_PATH,
    ['status'],
    dataDirectory,
    sessionsDirectory,
  );
  const output: unknown = JSON.parse(result.stdout);
  assert.ok(isUnknownRecord(output));
  assert.equal(output.readiness, 'ready');
  assert.ok(isUnknownRecord(output.activeGeneration));
  assert.equal(output.activeGeneration.generationId, generationId);
  assert.equal(output.activeGeneration.embeddingProfileId, 'embedding-profile-active');
  assert.deepEqual(output.replay, {
    state: 'pending',
    snapshotFileName: 'generation-replay-snapshot-activation.json',
  });
  assert.deepEqual(output.recovery, { required: true, generationIds: [generationId] });
  assert.deepEqual(output.backlog, backlog);
  assert.match(String(output.latestActionableError), /recovery required/u);
});

void test('standalone rebuild stops, resumes the same snapshot, and discards inactive work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-rebuild-'));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');
  const releasePath = join(dataDirectory, 'fixture-embedding-release');
  t.after(async () => {
    await writeFile(releasePath, 'release\n', 'utf8').catch(() => undefined);
    await sleep(100);
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(sessionsDirectory, { recursive: true });
  const sessionPath = join(sessionsDirectory, 'operator-session.jsonl');
  await writeFile(
    sessionPath,
    `${JSON.stringify({
      type: 'session',
      version: 3,
      id: 'operator-session',
      timestamp: '2026-08-10T10:00:00.000Z',
      cwd: '/operator-project',
    })}\n${JSON.stringify({
      type: 'message',
      id: 'operator-entry',
      parentId: null,
      timestamp: '2026-08-10T10:00:01.000Z',
      message: { role: 'assistant', content: 'detached target generation evidence' },
    })}\n`,
    'utf8',
  );
  await mkdir(dataDirectory, { recursive: true });
  await writeFile(join(dataDirectory, 'fixture-pause-embedding'), 'pause\n', 'utf8');

  const startedResult = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['rebuild'],
    dataDirectory,
    sessionsDirectory,
  );
  const started: unknown = JSON.parse(startedResult.stdout);
  assert.ok(isUnknownRecord(started));
  assert.equal(started.command, 'rebuild');
  assert.ok(isUnknownRecord(started.process));
  assert.equal(started.process.processState, RecallBackgroundIndexProcessState.STARTING);
  assert.equal(typeof started.process.generationId, 'string');
  const generationId = String(started.process.generationId);

  for (let attempt = 0; attempt < 500; attempt += 1) {
    try {
      await access(join(dataDirectory, 'fixture-embedding-started'));
      break;
    } catch {
      await sleep(20);
    }
  }
  await access(join(dataDirectory, 'fixture-embedding-started'));

  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['rebuild'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /already running|already exists/u);
      return true;
    },
  );
  await assert.rejects(
    () => runRecallOperatorCommand(CLI_PATH, ['discard'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /owned by a live rebuild|stop it before discard/u);
      return true;
    },
  );

  await runRecallOperatorCommand(CLI_PATH, ['stop'], dataDirectory, sessionsDirectory);
  await waitForRecallOperatorProcessState(
    dataDirectory,
    sessionsDirectory,
    RecallBackgroundIndexProcessState.STOPPED,
  );
  await rm(sessionPath);
  await writeFile(releasePath, 'release\n', 'utf8');

  const resumedResult = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['resume'],
    dataDirectory,
    sessionsDirectory,
  );
  const resumed: unknown = JSON.parse(resumedResult.stdout);
  assert.ok(isUnknownRecord(resumed));
  assert.ok(isUnknownRecord(resumed.process));
  assert.equal(resumed.process.generationId, generationId);
  const succeeded = await waitForRecallOperatorProcessState(
    dataDirectory,
    sessionsDirectory,
    RecallBackgroundIndexProcessState.SUCCEEDED,
  );
  assert.ok(isUnknownRecord(succeeded.process));
  assert.equal(typeof succeeded.process.embeddingProfileId, 'string');
  assert.match(String(succeeded.process.embeddingProfileId), /^embedding-profile-/u);
  assert.deepEqual(succeeded.sourceProgress, {
    scannedSessions: 1,
    totalSessions: 1,
    sessionPath: 'operator-session.jsonl',
  });
  assert.ok(isUnknownRecord(succeeded.latestDurablePhysicalProjection));
  assert.equal(succeeded.latestDurablePhysicalProjection.checkpointedSessions, 1);
  assert.equal(succeeded.latestDurablePhysicalProjection.totalSessions, 1);
  assert.equal(succeeded.latestDurablePhysicalProjection.sessionPath, 'operator-session.jsonl');
  assert.equal(typeof succeeded.latestDurablePhysicalProjection.physicalSourceIdentity, 'string');
  assert.ok(isUnknownRecord(succeeded.stagingGeneration));
  assert.equal(succeeded.stagingGeneration.generationId, generationId);
  assert.equal(succeeded.stagingGeneration.status, 'ready');
  assert.equal(succeeded.activeGeneration, null);
  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['resume'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /requires resumable state, received ready/u);
      return true;
    },
  );

  const discardedResult = await runRecallOperatorCommand(
    CLI_PATH,
    ['discard'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(discardedResult.stdout), {
    command: 'discard',
    discarded: true,
  });
  await assert.rejects(() => access(join(dataDirectory, 'generations', generationId)), {
    code: 'ENOENT',
  });
});

void test('standalone status detects a crashed detached owner and permits abandoned discard', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-crash-'));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');
  const releasePath = join(dataDirectory, 'fixture-embedding-release');
  t.after(async () => {
    await writeFile(releasePath, 'release\n', 'utf8').catch(() => undefined);
    await sleep(100);
    await rm(root, { recursive: true, force: true });
  });
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    join(sessionsDirectory, 'crash-session.jsonl'),
    `${JSON.stringify({
      type: 'session',
      version: 3,
      id: 'crash-session',
      timestamp: '2026-08-11T10:00:00.000Z',
      cwd: '/operator-project',
    })}\n${JSON.stringify({
      type: 'message',
      id: 'crash-entry',
      parentId: null,
      timestamp: '2026-08-11T10:00:01.000Z',
      message: { role: 'assistant', content: 'crash probe evidence' },
    })}\n`,
    'utf8',
  );
  await mkdir(dataDirectory, { recursive: true });
  await writeFile(join(dataDirectory, 'fixture-pause-embedding'), 'pause\n', 'utf8');

  const launched = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['rebuild'],
    dataDirectory,
    sessionsDirectory,
  );
  const launchOutput: unknown = JSON.parse(launched.stdout);
  assert.ok(isUnknownRecord(launchOutput));
  assert.ok(isUnknownRecord(launchOutput.process));
  assert.equal(typeof launchOutput.process.processId, 'number');
  const processId = Number(launchOutput.process.processId);
  for (let attempt = 0; attempt < 500; attempt += 1) {
    try {
      await access(join(dataDirectory, 'fixture-embedding-started'));
      break;
    } catch {
      await sleep(20);
    }
  }
  await access(join(dataDirectory, 'fixture-embedding-started'));
  process.kill(processId, 'SIGKILL');

  const crashed = await waitForRecallOperatorProcessState(
    dataDirectory,
    sessionsDirectory,
    RecallBackgroundIndexProcessState.CRASHED,
  );
  assert.ok(isUnknownRecord(crashed.process));
  assert.match(
    String(crashed.process.latestActionableError),
    /exited without a completion record/u,
  );
  assert.ok(isUnknownRecord(crashed.stagingGeneration));
  assert.equal(crashed.stagingGeneration.status, 'resumable');

  const discarded = await runRecallOperatorCommand(
    CLI_PATH,
    ['discard'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(discarded.stdout), { command: 'discard', discarded: true });
});

void test('standalone catch-up reports no active work and refuses a concurrent worker owner', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-catch-up-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');

  const noWork = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['catch-up'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(noWork.stdout), {
    command: 'catch-up',
    activeGenerationId: null,
    processedPhysicalSourceCount: 0,
    remainingEligibleSessionCount: 0,
    replayCompleted: null,
  });

  const generationId = 'generation_active';
  await mkdir(join(dataDirectory, 'generations', generationId), { recursive: true });
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallActiveGenerationPointer(join(dataDirectory, 'active-generation.json'), pointer);
  await writeRecallGenerationRegistry(join(dataDirectory, 'generation-registry.json'), {
    version: 1,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: 'embedding-profile-active',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 100,
        stateChangedAtEpochMilliseconds: 200,
        rebuildStartMarkerId: 'marker_1',
        rebuildMarkerWatermark: ['marker_1'],
        validatedAtEpochMilliseconds: 150,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const owner = await tryAcquireRecallRebuildOwnershipLock(
    join(dataDirectory, 'incremental-worker.lock'),
  );
  assert.ok(owner);
  try {
    await assert.rejects(
      () => runRecallOperatorCommand(CLI_TEST_PATH, ['catch-up'], dataDirectory, sessionsDirectory),
      (error: unknown) => {
        assert.ok(isUnknownRecord(error));
        assert.equal(error.code, 1);
        assert.match(String(error.stderr), /incremental worker is already running/u);
        return true;
      },
    );
  } finally {
    await owner.release();
  }
});

void test('standalone catch-up drains one marker through the configured target worker', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-catch-up-target-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');
  const sessionPath = join(sessionsDirectory, 'catch-up-session.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    `${JSON.stringify({
      type: 'session',
      version: 3,
      id: 'catch-up-session',
      timestamp: '2026-08-12T10:00:00.000Z',
      cwd: '/operator-project',
    })}\n${JSON.stringify({
      type: 'message',
      id: 'catch-up-entry',
      parentId: null,
      timestamp: '2026-08-12T10:00:01.000Z',
      message: { role: 'assistant', content: 'catch up target evidence' },
    })}\n`,
    'utf8',
  );
  await utimes(sessionPath, 1, 1);
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const generationId = 'generation_catch_up';
  const opened = await service.createEmptyRecallGeneration({ generationId });
  const rawManifest: unknown = JSON.parse(await readFile(opened.manifestPath, 'utf8'));
  assert.ok(isUnknownRecord(rawManifest));
  assert.ok(isUnknownRecord(rawManifest.embeddingProfile));
  const embeddingProfileId = rawManifest.embeddingProfile.profileId;
  assert.ok(typeof embeddingProfileId === 'string');
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: 1,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: opened.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 100,
        stateChangedAtEpochMilliseconds: 200,
        rebuildStartMarkerId: 'marker_build',
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: opened.validatedAtEpochMilliseconds,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const markerIdentity = {
    version: 1 as const,
    physicalSessionId: 'catch-up-session',
    physicalSessionPath: sessionPath,
    runtimeInstanceId: 'runtime-catch-up',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: Date.now() - 120_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: 'catch-up-session',
      leafEntryId: 'catch-up-entry',
    },
  } as const;
  const marker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  await publishRecallWorkMarker(marker, {
    markerSpoolDirectory: config.markerSpoolDirectory,
    trustedSessionRoots: [sessionsDirectory],
    workerSignal: { signalDetachedWorker() {} },
  });

  const caughtUp = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['catch-up'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(caughtUp.stdout), {
    command: 'catch-up',
    activeGenerationId: generationId,
    processedPhysicalSourceCount: 1,
    remainingEligibleSessionCount: 0,
    replayCompleted: null,
  });
  const caughtUpStatus = await runRecallOperatorCommand(
    CLI_PATH,
    ['status'],
    dataDirectory,
    sessionsDirectory,
  );
  const caughtUpStatusOutput: unknown = JSON.parse(caughtUpStatus.stdout);
  assert.ok(isUnknownRecord(caughtUpStatusOutput));
  assert.ok(isUnknownRecord(caughtUpStatusOutput.backlog));
  assert.equal(caughtUpStatusOutput.backlog.pendingEligibleSessionCount, 0);
  const lexical = await service.searchRecallGenerationLexical(
    generationId,
    'catch up target evidence',
    5,
  );
  assert.equal(lexical[0]?.content, 'catch up target evidence');

  await appendFile(
    sessionPath,
    `${JSON.stringify({
      type: 'message',
      id: 'recovery-entry',
      parentId: 'catch-up-entry',
      timestamp: '2026-08-12T10:00:02.000Z',
      message: { role: 'assistant', content: 'recovered target evidence' },
    })}\n`,
    'utf8',
  );
  await utimes(sessionPath, 1, 1);
  const recoveryMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: Date.now() - 120_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: 'catch-up-session',
      leafEntryId: 'recovery-entry',
    },
  } as const;
  const recoveryMarker: RecallWorkMarker = {
    ...recoveryMarkerIdentity,
    markerId: createRecallWorkMarkerId(recoveryMarkerIdentity),
  };
  await publishRecallWorkMarker(recoveryMarker, {
    markerSpoolDirectory: config.markerSpoolDirectory,
    trustedSessionRoots: [sessionsDirectory],
    workerSignal: { signalDetachedWorker() {} },
  });
  await writeFile(
    join(dataDirectory, 'fixture-interrupt-incremental-after-lexical-source-write'),
    'interrupt\n',
    'utf8',
  );

  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['catch-up'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /fixture incremental interruption/u);
      return true;
    },
  );
  const recoveryRecordPath = join(
    dataDirectory,
    'generations',
    generationId,
    'write-recovery.json',
  );
  const validRecoveryRecord = await readFile(recoveryRecordPath, 'utf8');
  const recoveryStatus = await runRecallOperatorCommand(
    CLI_PATH,
    ['status'],
    dataDirectory,
    sessionsDirectory,
  );
  const recoveryStatusOutput: unknown = JSON.parse(recoveryStatus.stdout);
  assert.ok(isUnknownRecord(recoveryStatusOutput));
  assert.deepEqual(recoveryStatusOutput.recovery, {
    required: true,
    generationIds: [generationId],
  });
  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['catch-up'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /recovery required; run pi-session-recall recover/u);
      return true;
    },
  );

  await writeFile(recoveryRecordPath, '{}\n', 'utf8');
  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['recover'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /incremental recovery record does not match/u);
      return true;
    },
  );
  await writeFile(recoveryRecordPath, validRecoveryRecord, 'utf8');
  const recovered = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['recover'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(recovered.stdout), {
    command: 'recover',
    recovered: true,
    recoveryKind: 'incremental-write',
  });
  const recoveredLexical = await service.searchRecallGenerationLexical(
    generationId,
    'recovered target evidence',
    5,
  );
  assert.equal(recoveredLexical[0]?.content, 'recovered target evidence');

  const deletionMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 3,
    createdAtEpochMilliseconds: Date.now() - 120_000,
  } as const;
  const deletionMarker: RecallWorkMarker = {
    ...deletionMarkerIdentity,
    markerId: createRecallWorkMarkerId(deletionMarkerIdentity),
  };
  await publishRecallWorkMarker(deletionMarker, {
    markerSpoolDirectory: config.markerSpoolDirectory,
    trustedSessionRoots: [sessionsDirectory],
    workerSignal: { signalDetachedWorker() {} },
  });
  await rm(sessionPath);
  const deletionEnvironment = {
    PI_RECALL_CONFIRMED_DELETION_MAX_MISSING_SOURCE_RATIO: '1',
  };
  await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['catch-up'],
    dataDirectory,
    sessionsDirectory,
    deletionEnvironment,
  );
  await sleep(100);
  await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['catch-up'],
    dataDirectory,
    sessionsDirectory,
    deletionEnvironment,
  );
  assert.deepEqual(
    await service.searchRecallGenerationLexical(generationId, 'recovered target evidence', 5),
    [],
  );
});

void test('standalone rollback recovers interruption and cleanup deletes only collectible generations', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-rollback-cleanup-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');
  await mkdir(sessionsDirectory, { recursive: true });
  const config = await loadRecallConversationConfig({
    environment: {
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
    },
  });
  const service = createRecallBackgroundIndexWorkerFixtureService(config);
  const firstGenerationId = 'generation_cli_rollback_a';
  const secondGenerationId = 'generation_cli_rollback_b';
  const thirdGenerationId = 'generation_cli_rollback_c';
  await service.createEmptyRecallGeneration({ generationId: firstGenerationId });
  await service.activateValidatedRecallGeneration(firstGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await service.createEmptyRecallGeneration({ generationId: secondGenerationId });
  await service.activateValidatedRecallGeneration(secondGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);

  await writeFile(join(dataDirectory, 'fixture-interrupt-rollback-after_registry'), 'interrupt\n');
  await assert.rejects(
    () => runRecallOperatorCommand(CLI_TEST_PATH, ['rollback'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.match(String(error.stderr), /fixture rollback interruption/u);
      return true;
    },
  );
  const interruptedStatus = await runRecallOperatorCommand(
    CLI_PATH,
    ['status'],
    dataDirectory,
    sessionsDirectory,
  );
  const interruptedStatusOutput: unknown = JSON.parse(interruptedStatus.stdout);
  assert.ok(isUnknownRecord(interruptedStatusOutput));
  assert.deepEqual(interruptedStatusOutput.recovery, {
    required: true,
    generationIds: [firstGenerationId],
  });
  const recovered = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['recover'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(recovered.stdout), {
    command: 'recover',
    recovered: true,
    recoveryKind: 'generation-cutover',
  });
  assert.equal(await service.completeRecallGenerationReplay(), true);

  const switchedBack = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['rollback'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(switchedBack.stdout), {
    command: 'rollback',
    activeGenerationId: secondGenerationId,
    rollbackGenerationId: firstGenerationId,
    restoredMarkerCount: 0,
  });
  assert.equal(await service.completeRecallGenerationReplay(), true);
  const protectedCleanup = await runRecallOperatorCommand(
    CLI_PATH,
    ['cleanup'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(protectedCleanup.stdout), {
    command: 'cleanup',
    deletedGenerationIds: [],
  });
  await access(join(config.generationRootDirectory, firstGenerationId));

  await service.createEmptyRecallGeneration({ generationId: thirdGenerationId });
  await service.activateValidatedRecallGeneration(thirdGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  const registry = await readRecallGenerationRegistry(config.generationRegistryPath);
  assert.ok(registry);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    ...registry,
    generations: registry.generations.map((entry) =>
      entry.generationId === firstGenerationId
        ? { ...entry, retireAfterEpochMilliseconds: 0 }
        : entry,
    ),
  });
  const cleaned = await runRecallOperatorCommand(
    CLI_PATH,
    ['cleanup'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(cleaned.stdout), {
    command: 'cleanup',
    deletedGenerationIds: [firstGenerationId],
  });
  await assert.rejects(() => access(join(config.generationRootDirectory, firstGenerationId)), {
    code: 'ENOENT',
  });
});

void test('standalone maintenance reports no-op recovery and cleanup and unavailable rollback', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-operator-maintenance-empty-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'data');
  const sessionsDirectory = join(root, 'sessions');

  const recovered = await runRecallOperatorCommand(
    CLI_TEST_PATH,
    ['recover'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(recovered.stdout), {
    command: 'recover',
    recovered: false,
    recoveryKind: 'none',
  });

  const generationId = 'generation_active';
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallActiveGenerationPointer(join(dataDirectory, 'active-generation.json'), pointer);
  await writeRecallGenerationRegistry(join(dataDirectory, 'generation-registry.json'), {
    version: 1,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: 'embedding-profile-active',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 100,
        stateChangedAtEpochMilliseconds: 200,
        rebuildStartMarkerId: 'marker_1',
        rebuildMarkerWatermark: ['marker_1'],
        validatedAtEpochMilliseconds: 150,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });

  const cleaned = await runRecallOperatorCommand(
    CLI_PATH,
    ['cleanup'],
    dataDirectory,
    sessionsDirectory,
  );
  assert.deepEqual(JSON.parse(cleaned.stdout), {
    command: 'cleanup',
    deletedGenerationIds: [],
  });

  await assert.rejects(
    () => runRecallOperatorCommand(CLI_PATH, ['rollback'], dataDirectory, sessionsDirectory),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.equal(error.code, 1);
      assert.match(String(error.stderr), /no retained target generation/u);
      return true;
    },
  );
});

void test('standalone CLI rejects unknown commands with a machine-detectable exit status', async () => {
  await assert.rejects(
    () => EXEC_FILE_ASYNC(process.execPath, ['--import', 'tsx', CLI_PATH, 'unknown-command']),
    (error: unknown) => {
      assert.ok(isUnknownRecord(error));
      assert.equal(error.code, 1);
      assert.match(String(error.stderr), /Pi session recall command invalid/u);
      return true;
    },
  );
});

import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';
import { promisify } from 'node:util';

import { RecallBackgroundIndexProcessState, RecallGenerationCutoverState } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  createRecallActiveGenerationPointer,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const CLI_PATH = new URL('./pi-session-recall-cli.ts', import.meta.url).pathname;
const CLI_TEST_PATH = new URL('./pi-session-recall-cli.test-utils.ts', import.meta.url).pathname;

async function runRecallOperatorCommand(
  executablePath: string,
  argumentsList: readonly string[],
  dataDirectory: string,
  sessionsDirectory: string,
): Promise<{ stdout: string; stderr: string }> {
  return EXEC_FILE_ASYNC(process.execPath, ['--import', 'tsx', executablePath, ...argumentsList], {
    env: {
      ...process.env,
      PI_RECALL_DATA_DIRECTORY: dataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: sessionsDirectory,
      PI_RECALL_EMBEDDING_DIMENSIONS: '3',
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

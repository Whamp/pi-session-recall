import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  inspectRecallWriteWindow,
  recallWriteWindowStatePaths,
} from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
} from './recall-generation-state.js';
import { recoverRecallGenerationCutover } from './recover-recall-generation-cutover.js';

void test('consistent active pointer requires no generation cutover recovery', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recover-recall-generation-cutover-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const pointer = createRecallActiveGenerationPointer('generation_active');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, pointer);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: pointer.activeGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: pointer.activeGenerationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });

  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
    }),
    false,
  );
});

void test('consistent cutover no-op does not clear stale write recovery state', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recover-recall-cutover-stale-window-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const lockPath = join(directory, 'operation.lock');
  const pointer = createRecallActiveGenerationPointer('generation_active');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, pointer);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: pointer.activeGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: pointer.activeGenerationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const statePaths = recallWriteWindowStatePaths(lockPath);
  await writeFile(
    statePaths.recoveryRequiredPath,
    `${JSON.stringify({ version: 1, state: 'recovery_required' })}\n`,
  );

  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath,
    }),
    false,
  );
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: true,
    recoveryRequired: true,
  });
});

void test('registry-first rollback cutover recovery publishes the retained target pointer', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recover-recall-rollback-cutover-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const oldPointer = createRecallActiveGenerationPointer('generation_current');
  const targetPointer = createRecallActiveGenerationPointer('generation_target');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, oldPointer);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: targetPointer.activeGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: oldPointer.activeGenerationId,
    activePointerChecksum: targetPointer.checksum,
    generations: [
      {
        generationId: targetPointer.activeGenerationId,
        state: RecallGenerationCutoverState.REPLAY_PENDING,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
      {
        generationId: oldPointer.activeGenerationId,
        state: RecallGenerationCutoverState.ROLLBACK,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'b'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: 50_000,
      },
    ],
  });

  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
    }),
    true,
  );
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    targetPointer.activeGenerationId,
  );
});

void test('registry-first legacy adoption recovery creates a missing active pointer', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recover-recall-legacy-cutover-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const targetPointer = createRecallActiveGenerationPointer('legacy-aaaaaaaaaaaaaaaaaaaaaaaa');
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: targetPointer.activeGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: targetPointer.checksum,
    generations: [
      {
        generationId: targetPointer.activeGenerationId,
        state: RecallGenerationCutoverState.LEGACY_READ_ONLY,
        indexManifestVersion: 5,
        markerSchemaVersion: null,
        sessionProjectionSchemaVersion: null,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });

  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
    }),
    true,
  );
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    targetPointer.activeGenerationId,
  );
});

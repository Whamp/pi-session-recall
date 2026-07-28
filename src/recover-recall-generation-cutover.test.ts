import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { completeRecallGenerationReplay } from './complete-recall-generation-replay.js';
import {
  assertRecallWriteWindowAvailableForRead,
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
      generationRootDirectory: join(directory, 'generations'),
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
      embeddingDimensions: 3,
    }),
    false,
  );
});

void test('consistent replay-pending cutover recovers empty stores before empty-spool replay', async (t) => {
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
  const statePaths = recallWriteWindowStatePaths(lockPath);
  await Promise.all([
    writeFile(
      statePaths.currentWindowPath,
      `${JSON.stringify({ version: 1, state: 'current_window' })}\n`,
    ),
    writeFile(
      statePaths.recoveryRequiredPath,
      `${JSON.stringify({ version: 1, state: 'recovery_required' })}\n`,
    ),
  ]);
  const recoveredStores: string[] = [];
  await mkdir(join(directory, 'generations', pointer.activeGenerationId), { recursive: true });
  const backlogSummaryPath = join(directory, 'backlog-summary.json');
  const markerSpoolDirectory = join(directory, 'markers', 'pending');

  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath,
      lockPath,
      generationRootDirectory: join(directory, 'generations'),
      embeddingDimensions: 3,
      openWriteEvidenceStore(databasePath, embeddingDimensions) {
        assert.equal(
          databasePath,
          join(directory, 'generations', pointer.activeGenerationId, 'zvec'),
        );
        assert.equal(embeddingDimensions, 3);
        recoveredStores.push('evidence-open');
        return { close: () => recoveredStores.push('evidence-close') };
      },
      openWriteProjectionStore(databasePath, generationId) {
        assert.equal(
          databasePath,
          join(directory, 'generations', pointer.activeGenerationId, 'session-projections'),
        );
        assert.equal(generationId, pointer.activeGenerationId);
        recoveredStores.push('projection-open');
        return { close: () => recoveredStores.push('projection-close') };
      },
    }),
    true,
  );
  assert.deepEqual(recoveredStores, [
    'evidence-open',
    'projection-open',
    'projection-close',
    'evidence-close',
  ]);
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });
  assert.equal(
    await completeRecallGenerationReplay({
      activeGenerationPointerPath,
      generationRegistryPath,
      backlogSummaryPath,
      markerSpoolDirectory,
      markerQuarantineDirectory: join(directory, 'markers', 'quarantine'),
      lockPath,
    }),
    true,
  );
  await assertRecallWriteWindowAvailableForRead(lockPath);
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
      generationRootDirectory: join(directory, 'generations'),
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
      embeddingDimensions: 3,
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
      generationRootDirectory: join(directory, 'generations'),
      backlogSummaryPath: join(directory, 'backlog-summary.json'),
      lockPath: join(directory, 'operation.lock'),
      embeddingDimensions: 3,
    }),
    true,
  );
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    targetPointer.activeGenerationId,
  );
});

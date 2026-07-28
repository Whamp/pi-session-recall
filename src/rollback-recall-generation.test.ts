import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { recallWriteWindowStatePaths } from './coordinate-recall-write-window.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
} from './recall-generation-state.js';
import { recoverRecallGenerationCutover } from './recover-recall-generation-cutover.js';
import { rollbackRecallGeneration } from './rollback-recall-generation.js';

void test('explicit rollback restores retained markers and wakes replay after partial failures', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'rollback-recall-generation-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const markerSpoolDirectory = join(directory, 'markers', 'pending');
  const retainedMarkerDirectory = join(directory, 'markers', 'rollback-retained');
  for (const generationId of ['generation_old', 'generation_new']) {
    await mkdir(join(generationRootDirectory, generationId, 'zvec'), { recursive: true });
  }
  await mkdir(retainedMarkerDirectory, { recursive: true });
  await writeFile(join(retainedMarkerDirectory, 'marker_branch_only.json'), 'branch transition\n');
  const activePointer = createRecallActiveGenerationPointer('generation_new');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, activePointer);
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: 'generation_new',
    buildingGenerationId: null,
    rollbackGenerationId: 'generation_old',
    activePointerChecksum: activePointer.checksum,
    generations: [
      {
        generationId: 'generation_old',
        state: RecallGenerationCutoverState.ROLLBACK,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: 50_000,
      },
      {
        generationId: 'generation_new',
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'b'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 3,
        stateChangedAtEpochMilliseconds: 4,
        rebuildStartMarkerId: 'marker_branch_only',
        rebuildMarkerWatermark: ['marker_branch_only'],
        validatedAtEpochMilliseconds: 4,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });

  const lockPath = join(directory, 'operation.lock');
  const backlogSummaryPath = join(directory, 'backlog-summary.json');
  let workerSignalCount = 0;
  const workerSignal = {
    signalDetachedWorker() {
      workerSignalCount += 1;
      if (workerSignalCount === 1) {
        assert.equal(readFileSyncState().currentWindow, false);
      }
    },
  };
  const result = await rollbackRecallGeneration({
    activeGenerationPointerPath,
    generationRegistryPath,
    generationRootDirectory,
    backlogSummaryPath,
    markerSpoolDirectory,
    retainedMarkerDirectory,
    lockPath,
    workerSignal,
    rollbackRetentionMilliseconds: 1_000,
    nowEpochMilliseconds: () => 10_000,
  });

  function readFileSyncState(): { currentWindow: boolean; recoveryRequired: boolean } {
    const statePaths = recallWriteWindowStatePaths(lockPath);
    const currentWindow = existsSync(statePaths.currentWindowPath);
    const recoveryRequired = existsSync(statePaths.recoveryRequiredPath);
    return { currentWindow, recoveryRequired };
  }

  assert.equal(workerSignalCount, 1);
  assert.deepEqual(result, {
    activeGenerationId: 'generation_old',
    rollbackGenerationId: 'generation_new',
    restoredMarkerCount: 1,
  });
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    'generation_old',
  );
  assert.equal(
    await readFile(join(markerSpoolDirectory, 'marker_branch_only.json'), 'utf8'),
    'branch transition\n',
  );
  await access(join(retainedMarkerDirectory, 'marker_branch_only.json'));
  const registry = await readRecallGenerationRegistry(generationRegistryPath);
  assert.equal(registry?.activeGenerationId, 'generation_old');
  assert.equal(registry?.rollbackGenerationId, 'generation_new');
  assert.equal(
    registry?.generations.find(({ generationId }) => generationId === 'generation_old')?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );

  await rm(backlogSummaryPath, { force: true });
  await mkdir(backlogSummaryPath);
  await assert.rejects(
    () =>
      rollbackRecallGeneration({
        activeGenerationPointerPath,
        generationRegistryPath,
        generationRootDirectory,
        backlogSummaryPath,
        markerSpoolDirectory,
        retainedMarkerDirectory,
        lockPath,
        workerSignal,
        rollbackRetentionMilliseconds: 1_000,
        nowEpochMilliseconds: () => 10_500,
      }),
    /EISDIR|directory/iu,
  );
  assert.equal(workerSignalCount, 2);
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    'generation_new',
  );
  await rm(backlogSummaryPath, { recursive: true, force: true });
  assert.equal(
    await recoverRecallGenerationCutover({
      activeGenerationPointerPath,
      generationRegistryPath,
      generationRootDirectory,
      backlogSummaryPath,
      lockPath,
      embeddingDimensions: 3,
      openWriteEvidenceStore() {
        return { close() {} };
      },
      openWriteProjectionStore() {
        return { close() {} };
      },
    }),
    true,
  );

  await mkdir(join(retainedMarkerDirectory, 'marker_invalid.json'));
  await assert.rejects(
    () =>
      rollbackRecallGeneration({
        activeGenerationPointerPath,
        generationRegistryPath,
        generationRootDirectory,
        backlogSummaryPath,
        markerSpoolDirectory,
        retainedMarkerDirectory,
        lockPath,
        workerSignal,
        rollbackRetentionMilliseconds: 1_000,
        nowEpochMilliseconds: () => 10_750,
      }),
    /EISDIR|directory/iu,
  );
  assert.equal(workerSignalCount, 3);
  assert.equal(
    (await readRecallActiveGenerationPointer(activeGenerationPointerPath))?.activeGenerationId,
    'generation_new',
  );
});

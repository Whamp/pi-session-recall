import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
} from './recall-generation-state.js';
import { rollbackRecallGeneration } from './rollback-recall-generation.js';

void test('explicit rollback atomically restores the retained generation and archived markers', async (t) => {
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
        sessionProjectionSchemaVersion: 2,
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
        sessionProjectionSchemaVersion: 2,
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

  const result = await rollbackRecallGeneration({
    activeGenerationPointerPath,
    generationRegistryPath,
    generationRootDirectory,
    backlogSummaryPath: join(directory, 'backlog-summary.json'),
    markerSpoolDirectory,
    retainedMarkerDirectory,
    lockPath: join(directory, 'operation.lock'),
    rollbackRetentionMilliseconds: 1_000,
    nowEpochMilliseconds: () => 10_000,
  });

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
});

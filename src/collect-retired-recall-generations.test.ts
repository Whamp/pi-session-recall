import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { collectRetiredRecallGenerations } from './collect-retired-recall-generations.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallGenerationRegistry,
  type RecallGenerationRegistryEntry,
} from './recall-generation-state.js';

function createGenerationEntry(
  generationId: string,
  state: RecallGenerationCutoverState,
  retireAfterEpochMilliseconds: number | null,
): RecallGenerationRegistryEntry {
  return {
    generationId,
    state,
    indexManifestVersion: 6,
    markerSchemaVersion: 1,
    sessionProjectionSchemaVersion: 2,
    indexManifestFingerprint: createRecallActiveGenerationPointer(generationId).checksum,
    rebuildStartedAtEpochMilliseconds: 1,
    stateChangedAtEpochMilliseconds: 2,
    rebuildStartMarkerId: null,
    rebuildMarkerWatermark: [],
    validatedAtEpochMilliseconds: 2,
    retireAfterEpochMilliseconds,
  };
}

void test('generation collection waits for replay and retention then deletes only inactive validated rollback', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'collect-retired-recall-generations-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const lockPath = join(directory, 'operation.lock');
  const retainedMarkerDirectory = join(directory, 'markers', 'rollback-retained');
  for (const generationId of ['generation_active', 'generation_rollback', 'generation_failed']) {
    await mkdir(join(generationRootDirectory, generationId), { recursive: true });
  }
  await mkdir(retainedMarkerDirectory, { recursive: true });
  const activePointer = createRecallActiveGenerationPointer('generation_active');
  await writeRecallActiveGenerationPointer(activeGenerationPointerPath, activePointer);
  const replayPendingRegistry: RecallGenerationRegistry = {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: 'generation_active',
    buildingGenerationId: null,
    rollbackGenerationId: 'generation_rollback',
    activePointerChecksum: activePointer.checksum,
    generations: [
      createGenerationEntry('generation_active', RecallGenerationCutoverState.REPLAY_PENDING, null),
      createGenerationEntry('generation_rollback', RecallGenerationCutoverState.ROLLBACK, 5_000),
      {
        ...createGenerationEntry('generation_failed', RecallGenerationCutoverState.FAILED, 5_000),
        validatedAtEpochMilliseconds: null,
      },
    ],
  };
  await writeRecallGenerationRegistry(generationRegistryPath, replayPendingRegistry);

  assert.deepEqual(
    await collectRetiredRecallGenerations({
      activeGenerationPointerPath,
      generationRegistryPath,
      generationRootDirectory,
      lockPath,
      nowEpochMilliseconds: () => 10_000,
    }),
    { deletedGenerationIds: [] },
  );
  await access(join(generationRootDirectory, 'generation_rollback'));

  await writeRecallGenerationRegistry(generationRegistryPath, {
    ...replayPendingRegistry,
    generations: replayPendingRegistry.generations.map((entry) =>
      entry.generationId === 'generation_active'
        ? { ...entry, state: RecallGenerationCutoverState.ACTIVE }
        : entry,
    ),
  });
  assert.deepEqual(
    await collectRetiredRecallGenerations({
      activeGenerationPointerPath,
      generationRegistryPath,
      generationRootDirectory,
      lockPath,
      nowEpochMilliseconds: () => 4_999,
    }),
    { deletedGenerationIds: [] },
  );
  assert.deepEqual(
    await collectRetiredRecallGenerations({
      activeGenerationPointerPath,
      generationRegistryPath,
      generationRootDirectory,
      lockPath,
      retainedMarkerDirectory,
      nowEpochMilliseconds: () => 5_000,
    }),
    { deletedGenerationIds: ['generation_rollback'] },
  );
  await assert.rejects(() => access(join(generationRootDirectory, 'generation_rollback')));
  await access(join(generationRootDirectory, 'generation_active'));
  await access(join(generationRootDirectory, 'generation_failed'));
  await assert.rejects(() => access(retainedMarkerDirectory));
  const collectedRegistry = await readRecallGenerationRegistry(generationRegistryPath);
  assert.equal(collectedRegistry?.rollbackGenerationId, null);
  assert.deepEqual(
    collectedRegistry?.generations.map(({ generationId }) => generationId).toSorted(),
    ['generation_active', 'generation_failed'],
  );
});

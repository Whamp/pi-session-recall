import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  decodeRecallBacklogSummary,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
  type RecallGenerationRegistry,
} from './recall-generation-state.js';
import { completeRecallGenerationReplayTransition } from './recall-generation-transitions.js';
import { RECALL_SESSION_PROJECTION_SCHEMA_VERSION } from './recall-session-projection.js';

function createReplayRegistry(generationId: string): RecallGenerationRegistry {
  const pointer = createRecallActiveGenerationPointer(generationId);
  return {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.REPLAY_PENDING,
        embeddingProfileId: 'embedding-profile-v1',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 10_000,
        stateChangedAtEpochMilliseconds: 11_000,
        rebuildStartMarkerId: 'marker_start',
        rebuildMarkerWatermark: ['marker_replay'],
        validatedAtEpochMilliseconds: 12_000,
        retireAfterEpochMilliseconds: null,
      },
    ],
  };
}

void test('replay completion transition publishes active registry then active backlog', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-replay-transition-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  const backlogPath = join(root, 'backlog-summary.json');
  const generationId = 'generation_replay';
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer(generationId),
  );
  await writeRecallGenerationRegistry(registryPath, createReplayRegistry(generationId));
  let proofCount = 0;

  const completed = await completeRecallGenerationReplayTransition({
    activeGenerationPointerPath: pointerPath,
    generationRegistryPath: registryPath,
    backlogSummaryPath: backlogPath,
    nowEpochMilliseconds: () => 20_000,
    async proveReplayWorkComplete() {
      proofCount += 1;
      return true;
    },
  });

  assert.equal(completed, true);
  assert.equal(proofCount, 1);
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations[0]?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
  const backlog = decodeRecallBacklogSummary(await readFile(backlogPath, 'utf8'));
  assert.equal(backlog.activeGenerationId, generationId);
  assert.equal(backlog.generationState, RecallGenerationCutoverState.ACTIVE);
  assert.equal(backlog.observedAtEpochMilliseconds, 20_000);
});

void test('active replay completion repairs backlog without rechecking marker work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-active-replay-transition-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  const backlogPath = join(root, 'backlog-summary.json');
  const generationId = 'generation_active';
  const registry = createReplayRegistry(generationId);
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer(generationId),
  );
  await writeRecallGenerationRegistry(registryPath, {
    ...registry,
    generations: registry.generations.map((entry) => ({
      ...entry,
      state: RecallGenerationCutoverState.ACTIVE,
    })),
  });

  const completed = await completeRecallGenerationReplayTransition({
    activeGenerationPointerPath: pointerPath,
    generationRegistryPath: registryPath,
    backlogSummaryPath: backlogPath,
    nowEpochMilliseconds: () => 30_000,
    async proveReplayWorkComplete() {
      throw new Error('active transition must not inspect marker work');
    },
  });

  assert.equal(completed, true);
  assert.equal(
    decodeRecallBacklogSummary(await readFile(backlogPath, 'utf8')).observedAtEpochMilliseconds,
    30_000,
  );
});

void test('replay completion transition rejects pointer and registry disagreement', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-replay-transition-mismatch-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_pointer'),
  );
  await writeRecallGenerationRegistry(registryPath, createReplayRegistry('generation_registry'));

  await assert.rejects(
    () =>
      completeRecallGenerationReplayTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: join(root, 'backlog-summary.json'),
        async proveReplayWorkComplete() {
          return true;
        },
      }),
    /replay completion found pointer and registry disagreement/u,
  );
});

import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm } from 'node:fs/promises';
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
import {
  activateReadyRecallGenerationTransition,
  completeRecallGenerationReplayTransition,
  completeStagingRecallGenerationDiscardTransition,
  prepareStagingRecallGenerationDiscardTransition,
  rollbackRecallGenerationTransition,
} from './recall-generation-transitions.js';
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

function createStagingDiscardRegistry(
  stagingState: RecallGenerationCutoverState,
): RecallGenerationRegistry {
  const activeRegistry = createReplayRegistry('generation_active');
  return {
    ...activeRegistry,
    buildingGenerationId:
      stagingState === RecallGenerationCutoverState.BUILDING ||
      stagingState === RecallGenerationCutoverState.READY
        ? 'generation_staging'
        : null,
    rollbackGenerationId:
      stagingState === RecallGenerationCutoverState.ROLLBACK ? 'generation_staging' : null,
    generations: [
      ...activeRegistry.generations.map((entry) => ({
        ...entry,
        state: RecallGenerationCutoverState.ACTIVE,
      })),
      {
        generationId: 'generation_staging',
        state: stagingState,
        embeddingProfileId: 'embedding-profile-v2',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: '0'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 13_000,
        stateChangedAtEpochMilliseconds: 14_000,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds:
          stagingState === RecallGenerationCutoverState.ROLLBACK ? 14_000 : null,
        retireAfterEpochMilliseconds:
          stagingState === RecallGenerationCutoverState.ROLLBACK ? 30_000 : null,
      },
    ],
  };
}

void test('activation preserves READY registry state when pointer publication fails', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-activation-pointer-fault-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  const frozenRegistry = createStagingDiscardRegistry(RecallGenerationCutoverState.BUILDING);
  const readyRegistry = createStagingDiscardRegistry(RecallGenerationCutoverState.READY);
  const readyEntry = readyRegistry.generations.find(
    ({ generationId }) => generationId === 'generation_staging',
  );
  assert.ok(readyEntry);
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_active'),
  );
  await writeRecallGenerationRegistry(registryPath, frozenRegistry);
  let recoveryRequired = false;

  await assert.rejects(
    () =>
      activateReadyRecallGenerationTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        expectedActivePointer: createRecallActiveGenerationPointer('generation_active'),
        expectedFrozenRegistry: frozenRegistry,
        readyRegistry,
        readyEntry,
        activatedAtEpochMilliseconds: 20_000,
        async beforePointerSwap() {
          await rm(pointerPath);
          await mkdir(pointerPath);
        },
        throwIfCancelled() {},
        retainRecoveryRequired() {
          recoveryRequired = true;
        },
      }),
    /EISDIR|directory/iu,
  );

  const interruptedRegistry = await readRecallGenerationRegistry(registryPath);
  assert.equal(interruptedRegistry?.buildingGenerationId, 'generation_staging');
  assert.equal(
    interruptedRegistry?.generations.find(
      ({ generationId }) => generationId === 'generation_staging',
    )?.state,
    RecallGenerationCutoverState.READY,
  );
  assert.equal(recoveryRequired, true);
});

void test('staging discard rejects a retained rollback generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-staging-discard-source-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_active'),
  );
  await writeRecallGenerationRegistry(
    registryPath,
    createStagingDiscardRegistry(RecallGenerationCutoverState.ROLLBACK),
  );

  await assert.rejects(
    () =>
      prepareStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: join(root, 'backlog-summary.json'),
        discardedGenerationId: 'generation_staging',
        discardedAtEpochMilliseconds: 20_000,
      }),
    /staging discard requires failed or abandoned building state/u,
  );
});

void test('staging discard completion rejects an entry not prepared for deletion', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-staging-discard-completion-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_active'),
  );
  await writeRecallGenerationRegistry(
    registryPath,
    createStagingDiscardRegistry(RecallGenerationCutoverState.FAILED),
  );

  await assert.rejects(
    () =>
      completeStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        discardedGenerationId: 'generation_staging',
      }),
    /staging discard completion requires a retired entry/u,
  );
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations.some(
      ({ generationId }) => generationId === 'generation_staging',
    ),
    true,
  );
});

void test('staging discard retires failed and abandoned build states before removal', async (t) => {
  for (const stagingState of [
    RecallGenerationCutoverState.FAILED,
    RecallGenerationCutoverState.BUILDING,
    RecallGenerationCutoverState.READY,
  ]) {
    await t.test(stagingState, async (stateTest) => {
      const root = await mkdtemp(join(tmpdir(), `recall-staging-discard-${stagingState}-`));
      stateTest.after(() => rm(root, { recursive: true, force: true }));
      const pointerPath = join(root, 'active-generation.json');
      const registryPath = join(root, 'generation-registry.json');
      await writeRecallActiveGenerationPointer(
        pointerPath,
        createRecallActiveGenerationPointer('generation_active'),
      );
      await writeRecallGenerationRegistry(registryPath, createStagingDiscardRegistry(stagingState));

      await prepareStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: join(root, 'backlog-summary.json'),
        discardedGenerationId: 'generation_staging',
        discardedAtEpochMilliseconds: 20_000,
      });

      const preparedRegistry = await readRecallGenerationRegistry(registryPath);
      const preparedEntry = preparedRegistry?.generations.find(
        ({ generationId }) => generationId === 'generation_staging',
      );
      assert.equal(preparedRegistry?.buildingGenerationId, null);
      assert.equal(preparedEntry?.state, RecallGenerationCutoverState.RETIRED);
      assert.equal(preparedEntry?.validatedAtEpochMilliseconds, 20_000);
      assert.equal(preparedEntry?.retireAfterEpochMilliseconds, 20_000);
      await completeStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        discardedGenerationId: 'generation_staging',
      });
      assert.equal(
        (await readRecallGenerationRegistry(registryPath))?.generations.some(
          ({ generationId }) => generationId === 'generation_staging',
        ),
        false,
      );
    });
  }
});

void test('staging discard remains retired when backlog publication fails', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-staging-discard-backlog-fault-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  const backlogPath = join(root, 'backlog-summary.json');
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_active'),
  );
  await writeRecallGenerationRegistry(
    registryPath,
    createStagingDiscardRegistry(RecallGenerationCutoverState.FAILED),
  );
  await mkdir(backlogPath);

  await assert.rejects(
    () =>
      prepareStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: backlogPath,
        discardedGenerationId: 'generation_staging',
        discardedAtEpochMilliseconds: 20_000,
      }),
    /EISDIR|directory/iu,
  );
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_staging',
    )?.state,
    RecallGenerationCutoverState.RETIRED,
  );
});

void test('staging discard rejects pointer and registry disagreement', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-staging-discard-pointer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  await writeRecallActiveGenerationPointer(
    pointerPath,
    createRecallActiveGenerationPointer('generation_other'),
  );
  await writeRecallGenerationRegistry(
    registryPath,
    createStagingDiscardRegistry(RecallGenerationCutoverState.FAILED),
  );

  await assert.rejects(
    () =>
      prepareStagingRecallGenerationDiscardTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: join(root, 'backlog-summary.json'),
        discardedGenerationId: 'generation_staging',
        discardedAtEpochMilliseconds: 20_000,
      }),
    /staging discard found pointer and registry disagreement/u,
  );
});

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

void test('replay completion leaves active registry recoverable when backlog publication fails', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-replay-backlog-fault-'));
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
  await mkdir(backlogPath);

  await assert.rejects(
    () =>
      completeRecallGenerationReplayTransition({
        activeGenerationPointerPath: pointerPath,
        generationRegistryPath: registryPath,
        backlogSummaryPath: backlogPath,
        async proveReplayWorkComplete() {
          return true;
        },
      }),
    /EISDIR|directory/iu,
  );
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations[0]?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
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

void test('rollback transition validates target then publishes registry pointer and backlog', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-rollback-transition-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const pointerPath = join(root, 'active-generation.json');
  const registryPath = join(root, 'generation-registry.json');
  const backlogPath = join(root, 'backlog-summary.json');
  const activePointer = createRecallActiveGenerationPointer('generation_new');
  await writeRecallActiveGenerationPointer(pointerPath, activePointer);
  await writeRecallGenerationRegistry(registryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: 'generation_new',
    buildingGenerationId: null,
    rollbackGenerationId: 'generation_old',
    activePointerChecksum: activePointer.checksum,
    generations: [
      {
        generationId: 'generation_old',
        state: RecallGenerationCutoverState.ROLLBACK,
        embeddingProfileId: 'embedding-profile-v1',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1_000,
        stateChangedAtEpochMilliseconds: 2_000,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: 2_000,
        retireAfterEpochMilliseconds: 50_000,
      },
      {
        generationId: 'generation_new',
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: 'embedding-profile-v1',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'b'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 3_000,
        stateChangedAtEpochMilliseconds: 4_000,
        rebuildStartMarkerId: 'marker_start',
        rebuildMarkerWatermark: ['marker_replay'],
        validatedAtEpochMilliseconds: 4_000,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const events: string[] = [];

  const transition = await rollbackRecallGenerationTransition({
    activeGenerationPointerPath: pointerPath,
    generationRegistryPath: registryPath,
    backlogSummaryPath: backlogPath,
    rollbackRetentionMilliseconds: 1_000,
    nowEpochMilliseconds: () => 10_000,
    async validateRollbackGeneration(generationId) {
      events.push(`validate:${generationId}`);
    },
    async prepareRollbackReplay() {
      events.push('restore-markers');
      return {
        restoredMarkerCount: 2,
        replayMarkerIds: ['marker_replay'],
        replaySnapshotFileName: 'generation-replay-snapshot-transition.json',
      };
    },
    retainRecoveryRequired() {
      events.push('retain-recovery');
    },
  });

  assert.deepEqual(events, ['validate:generation_old', 'restore-markers']);
  assert.deepEqual(transition, {
    result: {
      activeGenerationId: 'generation_old',
      rollbackGenerationId: 'generation_new',
      restoredMarkerCount: 2,
    },
    replayRequired: true,
  });
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_old',
    )?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
  assert.equal(
    (await readRecallGenerationRegistry(registryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_new',
    )?.retireAfterEpochMilliseconds,
    11_000,
  );
  assert.equal(
    decodeRecallBacklogSummary(await readFile(backlogPath, 'utf8')).pendingEligibleSessionCount,
    2,
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

import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
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
  decodeRecallBacklogSummary,
  readRecallActiveGenerationPointer,
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
} from './recall-generation-state.js';
import { recoverRecallGenerationCutover } from './recover-recall-generation-cutover.js';
import {
  rebuildRecallGeneration,
  RecallGenerationCutoverStage,
  type RecallGenerationBuildPaths,
} from './rebuild-recall-generation.js';

interface RebuildFixture {
  directory: string;
  generationRootDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  backlogSummaryPath: string;
  markerSpoolDirectory: string;
  lockPath: string;
  oldGenerationId: string;
  oldGenerationDirectory: string;
  workerSignal: { signalDetachedWorker(): void };
}

async function createRebuildFixture(t: test.TestContext): Promise<RebuildFixture> {
  const directory = await mkdtemp(join(tmpdir(), 'rebuild-recall-generation-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const oldGenerationId = 'generation_old';
  const oldGenerationDirectory = join(generationRootDirectory, oldGenerationId);
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const markerSpoolDirectory = join(directory, 'markers', 'pending');
  await mkdir(join(oldGenerationDirectory, 'zvec'), { recursive: true });
  await mkdir(markerSpoolDirectory, { recursive: true });
  await writeFile(join(oldGenerationDirectory, 'index-manifest.json'), '{"manifestVersion":6}\n');
  await writeRecallActiveGenerationPointer(
    activeGenerationPointerPath,
    createRecallActiveGenerationPointer(oldGenerationId),
  );
  return {
    directory,
    generationRootDirectory,
    activeGenerationPointerPath,
    generationRegistryPath: join(directory, 'generation-registry.json'),
    backlogSummaryPath: join(directory, 'backlog-summary.json'),
    markerSpoolDirectory,
    lockPath: join(directory, 'operation.lock'),
    oldGenerationId,
    oldGenerationDirectory,
    workerSignal: { signalDetachedWorker() {} },
  };
}

async function createBuiltGeneration(paths: RecallGenerationBuildPaths): Promise<void> {
  await mkdir(paths.databasePath, { recursive: true });
  await mkdir(paths.projectionDatabasePath, { recursive: true });
  await writeFile(paths.statePath, '{"version":2,"importPolicyVersion":3,"sessions":{}}\n');
  await writeFile(paths.manifestPath, '{"manifestVersion":6}\n');
}

void test('side-by-side rebuild keeps old search selected through build and optimize then cuts over briefly', async (t) => {
  const fixture = await createRebuildFixture(t);
  await writeFile(join(fixture.markerSpoolDirectory, 'marker_before.json'), '{}\n');
  const selectedDuringMaintenance: string[] = [];
  const cutoverStages: RecallGenerationCutoverStage[] = [];
  let optimizeCalls = 0;
  let closeCalls = 0;

  const rebuilt = await rebuildRecallGeneration({
    ...fixture,
    generationId: 'generation_new',
    rollbackRetentionMilliseconds: 1_000,
    nowEpochMilliseconds: () => 10_000,
    async captureBuildSnapshot() {
      assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
        currentWindow: true,
        recoveryRequired: true,
      });
      assert.equal(await readRecallGenerationRegistry(fixture.generationRegistryPath), null);
      return 'approved-projection-snapshot';
    },
    async buildGeneration(paths, approvedProjectionSnapshot) {
      assert.equal(approvedProjectionSnapshot, 'approved-projection-snapshot');
      assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
        currentWindow: false,
        recoveryRequired: false,
      });
      assert.equal(
        decodeRecallBacklogSummary(await readFile(fixture.backlogSummaryPath, 'utf8'))
          .generationState,
        RecallGenerationCutoverState.BUILDING,
      );
      for (let index = 0; index < 20; index += 1) {
        selectedDuringMaintenance.push(
          (
            await readRecallActiveGenerationSelection(
              fixture.activeGenerationPointerPath,
              fixture.generationRootDirectory,
            )
          ).activeGenerationId,
        );
      }
      await writeFile(join(fixture.markerSpoolDirectory, 'marker_during.json'), '{}\n');
      await createBuiltGeneration(paths);
      return {
        result: { totalChunks: 7 },
        async optimize() {
          optimizeCalls += 1;
          selectedDuringMaintenance.push(
            (
              await readRecallActiveGenerationSelection(
                fixture.activeGenerationPointerPath,
                fixture.generationRootDirectory,
              )
            ).activeGenerationId,
          );
        },
        async close() {
          closeCalls += 1;
        },
      };
    },
    async validateGeneration() {
      return { indexManifestFingerprint: 'a'.repeat(64) };
    },
    async onCutoverStage(stage) {
      cutoverStages.push(stage);
      assert.equal((await inspectRecallWriteWindow(fixture.lockPath)).currentWindow, true);
      const expectedGenerationId =
        stage === RecallGenerationCutoverStage.BEFORE_POINTER_SWAP
          ? fixture.oldGenerationId
          : 'generation_new';
      assert.equal(
        (await readRecallActiveGenerationPointer(fixture.activeGenerationPointerPath))
          ?.activeGenerationId,
        expectedGenerationId,
      );
    },
  });

  assert.deepEqual(new Set(selectedDuringMaintenance), new Set([fixture.oldGenerationId]));
  assert.equal(optimizeCalls, 1);
  assert.equal(closeCalls, 1);
  assert.deepEqual(cutoverStages, [
    RecallGenerationCutoverStage.BEFORE_POINTER_SWAP,
    RecallGenerationCutoverStage.AFTER_POINTER_SWAP,
  ]);
  assert.deepEqual(rebuilt, {
    result: { totalChunks: 7 },
    previousGenerationId: fixture.oldGenerationId,
    activeGenerationId: 'generation_new',
    replayMarkerWatermark: ['marker_before', 'marker_during'],
  });
  assert.equal(
    (
      await readRecallActiveGenerationSelection(
        fixture.activeGenerationPointerPath,
        fixture.generationRootDirectory,
      )
    ).activeGenerationId,
    'generation_new',
  );
  const registry = await readRecallGenerationRegistry(fixture.generationRegistryPath);
  assert.equal(registry?.activeGenerationId, 'generation_new');
  assert.equal(registry?.buildingGenerationId, null);
  assert.equal(registry?.rollbackGenerationId, fixture.oldGenerationId);
  assert.equal(
    registry?.generations.find(({ generationId }) => generationId === 'generation_new')?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
  assert.equal(
    registry?.generations.find(({ generationId }) => generationId === fixture.oldGenerationId)
      ?.state,
    RecallGenerationCutoverState.ROLLBACK,
  );
  assert.equal(
    decodeRecallBacklogSummary(await readFile(fixture.backlogSummaryPath, 'utf8')).generationState,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
  assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });
});

void test('successful empty-spool cutover wakes the ordinary worker after lock release', async (t) => {
  const fixture = await createRebuildFixture(t);
  const statePaths = recallWriteWindowStatePaths(fixture.lockPath);
  let workerSignalCalls = 0;

  await rebuildRecallGeneration({
    ...fixture,
    generationId: 'generation_idle_cutover',
    workerSignal: {
      signalDetachedWorker() {
        workerSignalCalls += 1;
        assert.equal(existsSync(statePaths.currentWindowPath), false);
        assert.equal(existsSync(statePaths.recoveryRequiredPath), false);
        assert.equal(
          decodeRecallBacklogSummary(readFileSync(fixture.backlogSummaryPath, 'utf8'))
            .generationState,
          RecallGenerationCutoverState.REPLAY_PENDING,
        );
      },
    },
    async buildGeneration(paths) {
      await createBuiltGeneration(paths);
      return { result: null, async close() {} };
    },
    async validateGeneration() {
      return { indexManifestFingerprint: 'c'.repeat(64) };
    },
  });

  assert.equal(workerSignalCalls, 1);
  assert.equal(
    (await readRecallGenerationRegistry(fixture.generationRegistryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_idle_cutover',
    )?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
});

void test('build failure leaves the old generation active and wakes its accumulated marker work', async (t) => {
  const fixture = await createRebuildFixture(t);
  const statePaths = recallWriteWindowStatePaths(fixture.lockPath);
  let workerSignalCalls = 0;
  await writeFile(join(fixture.markerSpoolDirectory, 'marker_during_failure.json'), '{}\n');
  await assert.rejects(
    () =>
      rebuildRecallGeneration({
        ...fixture,
        generationId: 'generation_failed',
        workerSignal: {
          signalDetachedWorker() {
            workerSignalCalls += 1;
            assert.equal(existsSync(statePaths.currentWindowPath), false);
            assert.equal(existsSync(statePaths.recoveryRequiredPath), false);
          },
        },
        async buildGeneration() {
          throw new Error('replacement build failed');
        },
        async validateGeneration() {
          assert.fail('failed build must not validate');
        },
      }),
    /replacement build failed/,
  );

  assert.equal(workerSignalCalls, 1);
  assert.equal(
    (await readRecallActiveGenerationPointer(fixture.activeGenerationPointerPath))
      ?.activeGenerationId,
    fixture.oldGenerationId,
  );
  const registry = await readRecallGenerationRegistry(fixture.generationRegistryPath);
  assert.equal(registry?.buildingGenerationId, null);
  assert.equal(
    registry?.generations.find(({ generationId }) => generationId === 'generation_failed')?.state,
    RecallGenerationCutoverState.FAILED,
  );
});

void test('optimization cancellation closes the replacement and never swaps the pointer', async (t) => {
  const fixture = await createRebuildFixture(t);
  const abortController = new AbortController();
  let closeCalls = 0;
  await assert.rejects(
    () =>
      rebuildRecallGeneration({
        ...fixture,
        generationId: 'generation_cancelled',
        signal: abortController.signal,
        async buildGeneration(paths) {
          await createBuiltGeneration(paths);
          return {
            result: null,
            async optimize() {
              abortController.abort(new Error('optimization policy cancelled cutover'));
            },
            async close() {
              closeCalls += 1;
            },
          };
        },
        async validateGeneration() {
          assert.fail('cancelled optimization must not validate');
        },
      }),
    /optimization policy cancelled cutover/,
  );
  assert.equal(closeCalls, 1);
  assert.equal(
    (await readRecallActiveGenerationPointer(fixture.activeGenerationPointerPath))
      ?.activeGenerationId,
    fixture.oldGenerationId,
  );
});

for (const faultStage of [
  RecallGenerationCutoverStage.BEFORE_POINTER_SWAP,
  RecallGenerationCutoverStage.AFTER_POINTER_SWAP,
]) {
  void test(`cutover fault at ${faultStage} preserves atomic pointer recovery semantics`, async (t) => {
    const fixture = await createRebuildFixture(t);
    await assert.rejects(
      () =>
        rebuildRecallGeneration({
          ...fixture,
          generationId: `generation_${faultStage}`,
          async buildGeneration(paths) {
            await createBuiltGeneration(paths);
            return { result: null, async close() {} };
          },
          async validateGeneration() {
            return { indexManifestFingerprint: 'b'.repeat(64) };
          },
          async onCutoverStage(stage) {
            if (stage === faultStage) {
              throw new Error(`injected ${faultStage} failure`);
            }
          },
        }),
      new RegExp(`injected ${faultStage} failure`, 'u'),
    );
    const activeGenerationId = (
      await readRecallActiveGenerationPointer(fixture.activeGenerationPointerPath)
    )?.activeGenerationId;
    assert.equal(
      activeGenerationId,
      faultStage === RecallGenerationCutoverStage.BEFORE_POINTER_SWAP
        ? fixture.oldGenerationId
        : `generation_${faultStage}`,
    );
    assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
      currentWindow: faultStage === RecallGenerationCutoverStage.AFTER_POINTER_SWAP,
      recoveryRequired: faultStage === RecallGenerationCutoverStage.AFTER_POINTER_SWAP,
    });
    assert.equal(
      await recoverRecallGenerationCutover({
        activeGenerationPointerPath: fixture.activeGenerationPointerPath,
        generationRegistryPath: fixture.generationRegistryPath,
        generationRootDirectory: fixture.generationRootDirectory,
        backlogSummaryPath: fixture.backlogSummaryPath,
        lockPath: fixture.lockPath,
        embeddingDimensions: 3,
        nowEpochMilliseconds: () => 20_000,
        openWriteEvidenceStore() {
          return { close() {} };
        },
        openWriteProjectionStore() {
          return { close() {} };
        },
      }),
      true,
    );
    const recoveredActiveGenerationId = (
      await readRecallGenerationRegistry(fixture.generationRegistryPath)
    )?.activeGenerationId;
    assert.equal(
      recoveredActiveGenerationId,
      faultStage === RecallGenerationCutoverStage.BEFORE_POINTER_SWAP
        ? fixture.oldGenerationId
        : `generation_${faultStage}`,
    );
    assert.equal(
      (await readRecallActiveGenerationPointer(fixture.activeGenerationPointerPath))
        ?.activeGenerationId,
      recoveredActiveGenerationId,
    );
    assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
      currentWindow: false,
      recoveryRequired: false,
    });
  });
}

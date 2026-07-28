import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { completeRecallGenerationReplay } from './complete-recall-generation-replay.js';
import { RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  decodeRecallBacklogSummary,
  readRecallGenerationRegistry,
  writeRecallActiveGenerationPointer,
} from './recall-generation-state.js';
import { rebuildRecallGeneration } from './rebuild-recall-generation.js';

void test('replay failure retains backlog and successful empty-spool replay activates replacement', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'complete-recall-generation-replay-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const oldGenerationDirectory = join(generationRootDirectory, 'generation_old');
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  const backlogSummaryPath = join(directory, 'backlog-summary.json');
  const markerSpoolDirectory = join(directory, 'markers', 'pending');
  const lockPath = join(directory, 'operation.lock');
  await mkdir(join(oldGenerationDirectory, 'zvec'), { recursive: true });
  await mkdir(markerSpoolDirectory, { recursive: true });
  await writeFile(join(oldGenerationDirectory, 'index-manifest.json'), '{"manifestVersion":6}\n');
  await writeFile(join(markerSpoolDirectory, 'marker_replay.json'), '{}\n');
  await writeRecallActiveGenerationPointer(
    activeGenerationPointerPath,
    createRecallActiveGenerationPointer('generation_old'),
  );
  await rebuildRecallGeneration({
    generationRootDirectory,
    activeGenerationPointerPath,
    generationRegistryPath,
    backlogSummaryPath,
    markerSpoolDirectory,
    lockPath,
    generationId: 'generation_new',
    workerSignal: { signalDetachedWorker() {} },
    async buildGeneration(paths) {
      await mkdir(paths.databasePath, { recursive: true });
      await writeFile(paths.manifestPath, '{"manifestVersion":6}\n');
      return { result: null, async close() {} };
    },
    async validateGeneration() {
      return { indexManifestFingerprint: 'a'.repeat(64) };
    },
  });

  const incomplete = await completeRecallGenerationReplay({
    activeGenerationPointerPath,
    generationRegistryPath,
    backlogSummaryPath,
    markerSpoolDirectory,
    lockPath,
  });
  assert.equal(incomplete, false);
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_new',
    )?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );

  await rm(join(markerSpoolDirectory, 'marker_replay.json'));
  const completed = await completeRecallGenerationReplay({
    activeGenerationPointerPath,
    generationRegistryPath,
    backlogSummaryPath,
    markerSpoolDirectory,
    lockPath,
    nowEpochMilliseconds: () => 20_000,
  });
  assert.equal(completed, true);
  assert.equal(
    (await readRecallGenerationRegistry(generationRegistryPath))?.generations.find(
      ({ generationId }) => generationId === 'generation_new',
    )?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
  assert.equal(
    decodeRecallBacklogSummary(await readFile(backlogSummaryPath, 'utf8')).generationState,
    RecallGenerationCutoverState.ACTIVE,
  );
});

import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  coordinateRecallWriteWindow,
  inspectRecallWriteWindow,
  recallWriteWindowStatePaths,
} from './coordinate-recall-write-window.js';

void test('recall write window leaves no coordination state after a normal close path', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-write-window-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lockPath = join(directory, 'operation.lock');
  let observedDuringWindow = false;

  await coordinateRecallWriteWindow({ lockPath, allowRecovery: false }, async ({ recovering }) => {
    assert.equal(recovering, false);
    const state = await inspectRecallWriteWindow(lockPath);
    observedDuringWindow = state.currentWindow && state.recoveryRequired;
  });

  assert.equal(observedDuringWindow, true);
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });
});

void test('only a recovery-capable writer clears explicit interrupted-window state', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-write-recovery-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lockPath = join(directory, 'operation.lock');
  const paths = recallWriteWindowStatePaths(lockPath);
  const interruptedState = `${JSON.stringify({ version: 1, state: 'recovery_required' })}\n`;
  await writeFile(paths.recoveryRequiredPath, interruptedState);
  await writeFile(
    paths.currentWindowPath,
    `${JSON.stringify({ version: 1, state: 'current_window' })}\n`,
  );

  await assert.rejects(
    () => coordinateRecallWriteWindow({ lockPath, allowRecovery: false }, async () => undefined),
    /Recall write recovery required/u,
  );
  assert.equal(await readFile(paths.recoveryRequiredPath, 'utf8'), interruptedState);

  let recovering = false;
  await coordinateRecallWriteWindow({ lockPath, allowRecovery: true }, async (window) => {
    recovering = window.recovering;
  });
  assert.equal(recovering, true);
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });
});

void test('kernel-backed write window waits cancellably without deleting another writer state', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-write-contention-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lockPath = join(directory, 'operation.lock');
  const firstEntered = Promise.withResolvers<void>();
  const releaseFirst = Promise.withResolvers<void>();
  const first = coordinateRecallWriteWindow({ lockPath, allowRecovery: false }, async () => {
    firstEntered.resolve();
    await releaseFirst.promise;
  });
  await firstEntered.promise;

  await assert.rejects(
    () =>
      coordinateRecallWriteWindow(
        {
          lockPath,
          allowRecovery: false,
          signal: AbortSignal.timeout(20),
        },
        async () => undefined,
      ),
    /Recall conversation operation cancelled/u,
  );
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: true,
    recoveryRequired: true,
  });

  releaseFirst.resolve();
  await first;
});

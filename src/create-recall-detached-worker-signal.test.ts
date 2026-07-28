import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';

async function waitForPath(path: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (Date.now() < deadline) {
    try {
      await access(path);
      return;
    } catch {
      await new Promise((resolve) => {
        setTimeout(resolve, 10);
      });
    }
  }
  throw new Error(`Recall detached worker signal test timed out waiting for ${path}`);
}

void test('detached worker signals coalesce behind an active ownership lock without losing the successor', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-detached-worker-signal-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const workerOwnershipLockPath = join(directory, 'worker.lock');
  const holderReadyPath = join(directory, 'holder-ready');
  const workerProbePath = join(directory, 'worker-probe');
  const harnessPath = join(directory, 'worker-harness.mts');
  await writeFile(
    harnessPath,
    `import { appendFile, writeFile } from 'node:fs/promises';\n` +
      `const [mode, readyPath, probePath] = process.argv.slice(2);\n` +
      `if (mode === 'hold') {\n` +
      `  await writeFile(readyPath, 'ready\\n');\n` +
      `  await new Promise((resolve) => setTimeout(resolve, 250));\n` +
      `} else {\n` +
      `  await appendFile(probePath, 'run-start\\n');\n` +
      `  await new Promise((resolve) => setTimeout(resolve, 250));\n` +
      `  await appendFile(probePath, 'run-end\\n');\n` +
      `}\n`,
  );
  const holder = spawn(
    '/usr/bin/flock',
    [
      workerOwnershipLockPath,
      process.execPath,
      '--import',
      'tsx',
      harnessPath,
      'hold',
      holderReadyPath,
      workerProbePath,
    ],
    { stdio: 'ignore' },
  );
  await waitForPath(holderReadyPath);

  const workerSignal = createRecallDetachedWorkerSignal(workerOwnershipLockPath, {
    workerExecutablePath: harnessPath,
    workerArguments: ['work', holderReadyPath, workerProbePath],
    workingDirectory: process.cwd(),
  });
  workerSignal.signalDetachedWorker();
  workerSignal.signalDetachedWorker();

  await waitForPath(workerProbePath);
  workerSignal.signalDetachedWorker();
  workerSignal.signalDetachedWorker();
  const deadline = Date.now() + 3_000;
  let probe = '';
  while (Date.now() < deadline) {
    probe = await readFile(workerProbePath, 'utf8');
    if (probe.split('run-end').length === 3) {
      break;
    }
    await new Promise((resolve) => {
      setTimeout(resolve, 10);
    });
  }
  await new Promise((resolve) => {
    setTimeout(resolve, 100);
  });
  assert.equal(await readFile(workerProbePath, 'utf8'), 'run-start\nrun-end\nrun-start\nrun-end\n');
  assert.equal(holder.exitCode, 0);
});

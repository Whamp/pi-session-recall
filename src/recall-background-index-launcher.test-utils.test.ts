import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import test from 'node:test';

void test('background launcher test utility requires data and session directories', async () => {
  const child = spawn(
    process.execPath,
    [
      '--import',
      'tsx',
      new URL('./recall-background-index-launcher.test-utils.ts', import.meta.url).pathname,
    ],
    { stdio: 'ignore' },
  );
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });

  assert.notEqual(exitCode, 0);
});

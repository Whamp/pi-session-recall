import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import test from 'node:test';

void test('background index worker refuses to run without a request path', async () => {
  const child = spawn(
    process.execPath,
    ['--import', 'tsx', new URL('./recall-background-index-worker.ts', import.meta.url).pathname],
    {
      stdio: 'ignore',
    },
  );
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });

  assert.equal(exitCode, 1);
});

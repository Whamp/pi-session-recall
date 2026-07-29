import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

void test('first-index CLI reports fresh mixed inference as not ready', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-setup-cli-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const child = spawn(
    process.execPath,
    [
      '--import',
      'tsx',
      new URL('./runRecallFirstIndexSetupCli.ts', import.meta.url).pathname,
      'inference',
      'status',
    ],
    {
      env: {
        ...process.env,
        PI_RECALL_DATA_DIRECTORY: join(root, 'data'),
        PI_RECALL_SESSIONS_DIRECTORY: join(root, 'sessions'),
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  let stdout = '';
  let stderr = '';
  child.stdout.setEncoding('utf8').on('data', (chunk: string) => {
    stdout += chunk;
  });
  child.stderr.setEncoding('utf8').on('data', (chunk: string) => {
    stderr += chunk;
  });
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });

  assert.equal(exitCode, 0, stderr);
  assert.match(stdout, /"ready":false/u);
});

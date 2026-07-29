import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { runRecallInferenceSetupCommand } from './runRecallInferenceSetupCommand.js';

void test('inference setup status reports the missing required embedding', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-command-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const output: string[] = [];

  await runRecallInferenceSetupCommand(['status'], {
    statePath: join(root, 'inference.json'),
    candidates: [],
    writeOutput(value) {
      output.push(value);
    },
  });

  const status: unknown = JSON.parse(output[0] ?? 'null');
  assert.ok(status && typeof status === 'object');
  assert.equal(Reflect.get(status, 'ready'), false);
});

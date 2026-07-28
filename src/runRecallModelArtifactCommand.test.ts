import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createRecallModelArtifactFixtureGguf,
  createRecallModelArtifactFixtureProfile,
} from './recall-model-artifact.test-utils.js';
import { runRecallModelArtifactCommand } from './runRecallModelArtifactCommand.js';

void test('model artifact command inspects without mutating or downloading', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-artifact-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = createRecallModelArtifactFixtureGguf();
  const output: string[] = [];

  await runRecallModelArtifactCommand(['inspect'], {
    commandUsage: 'usage: fixture inspect',
    errorPrefix: 'Recall fixture model',
    profile: createRecallModelArtifactFixtureProfile(artifact),
    cacheDirectory: root,
    writeOutput(value) {
      output.push(value);
    },
  });

  assert.match(output[0] ?? '', /"state":"missing"/u);
});

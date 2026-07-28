import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { readRecallIndexGenerationStatus } from './recall-index-generations.js';

void test('missing generation selectors and legacy manifest report no active or staging generation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-generations-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  assert.deepEqual(
    await readRecallIndexGenerationStatus({
      legacyPaths: {
        databasePath: join(root, 'zvec'),
        statePath: join(root, 'state.json'),
        manifestPath: join(root, 'manifest.json'),
        lockPath: join(root, 'lock'),
      },
      generationsDirectory: join(root, 'generations'),
      activeGenerationPath: join(root, 'active.json'),
      stagingGenerationPath: join(root, 'staging.json'),
    }),
    { active: null, staging: null },
  );
});

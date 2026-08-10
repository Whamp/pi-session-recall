import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  activateRecallDatabaseCandidate,
  createRecallDatabaseCandidate,
  resolveActiveRecallDatabasePaths,
  type RecallDatabaseGenerationConfig,
} from './recall-database-generation.js';

void test('missing candidate database cannot replace the active legacy database', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-missing-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config: RecallDatabaseGenerationConfig = {
    databasePath: join(dataDirectory, 'zvec'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
  };
  const candidate = await createRecallDatabaseCandidate(config);
  await rm(candidate.directoryPath, { recursive: true });

  await assert.rejects(
    activateRecallDatabaseCandidate(config, candidate),
    /Recall candidate database incomplete/u,
  );
  assert.deepEqual(await resolveActiveRecallDatabasePaths(config), {
    databasePath: config.databasePath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    indexMaintenanceStatusPath: config.indexMaintenanceStatusPath,
  });
});

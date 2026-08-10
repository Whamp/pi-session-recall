import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readlink, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  activateRecallDatabaseCandidate,
  activateStagedRecallDatabase,
  createRecallDatabaseCandidate,
  resolveActiveRecallDatabasePaths,
  resumeRecallDatabaseCandidate,
  stageRecallDatabaseCandidate,
  type RecallDatabaseCandidate,
  type RecallDatabaseGenerationConfig,
} from './recall-database-generation.js';

function createGenerationConfig(dataDirectory: string): RecallDatabaseGenerationConfig {
  return {
    databasePath: join(dataDirectory, 'zvec'),
    catalogPath: join(dataDirectory, 'recall-catalog.sqlite'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
  };
}

async function completeRecallDatabaseCandidate(candidate: RecallDatabaseCandidate): Promise<void> {
  await mkdir(candidate.paths.databasePath);
  await Promise.all([
    writeFile(candidate.paths.catalogPath, 'catalog'),
    writeFile(candidate.paths.manifestPath, 'manifest'),
    writeFile(candidate.paths.indexMaintenanceStatusPath, 'status'),
  ]);
}

void test('missing candidate database cannot replace the active legacy database', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-missing-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await rm(candidate.directoryPath, { recursive: true });

  await assert.rejects(
    activateRecallDatabaseCandidate(config, candidate),
    /Recall candidate database incomplete/u,
  );
  assert.deepEqual(await resolveActiveRecallDatabasePaths(config), {
    databasePath: config.databasePath,
    catalogPath: config.catalogPath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    indexMaintenanceStatusPath: config.indexMaintenanceStatusPath,
  });
});

void test('candidate resume requires exactly one interrupted candidate and preserves its files', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-resume-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);

  await assert.rejects(
    resumeRecallDatabaseCandidate(config),
    /Recall candidate resume requires exactly one interrupted candidate; found 0/u,
  );
  const candidate = await createRecallDatabaseCandidate(config);
  await writeFile(candidate.paths.catalogPath, 'partial catalog');

  const resumed = await resumeRecallDatabaseCandidate(config);

  assert.equal(resumed.directoryPath, candidate.directoryPath);
  assert.equal(await readFile(resumed.paths.catalogPath, 'utf8'), 'partial catalog');
});

void test('staged candidate stays inactive until its exact database target is activated', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-staged-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  await mkdir(config.databasePath);
  await writeFile(config.manifestPath, 'legacy manifest');
  await writeFile(config.statePath, 'legacy state');
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);

  const staged = await stageRecallDatabaseCandidate(config, candidate);

  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
  assert.match(staged.databaseTarget, /^generations\/generation-/u);
  assert.deepEqual(await resolveActiveRecallDatabasePaths(config), {
    databasePath: config.databasePath,
    catalogPath: config.catalogPath,
    statePath: config.statePath,
    manifestPath: config.manifestPath,
    indexMaintenanceStatusPath: config.indexMaintenanceStatusPath,
  });

  const activation = await activateStagedRecallDatabase(config, staged.databaseTarget);

  assert.equal(await readlink(join(dataDirectory, 'active')), staged.databaseTarget);
  assert.deepEqual(activation, { previousAvailable: true });
  const previousRecordPath = join(staged.directoryPath, '.previous-database.json');
  assert.deepEqual(JSON.parse(await readFile(previousRecordPath, 'utf8')), {
    version: 1,
    target: '.',
  });

  await assert.rejects(
    activateStagedRecallDatabase(config, staged.databaseTarget),
    /Recall staged database is already active/u,
  );
  assert.deepEqual(JSON.parse(await readFile(previousRecordPath, 'utf8')), {
    version: 1,
    target: '.',
  });
});

void test('staged activation rejects incomplete or unmanaged database targets without changing active', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-invalid-staged-target-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);
  const staged = await stageRecallDatabaseCandidate(config, candidate);
  await rm(join(staged.directoryPath, 'index-maintenance-status.json'));

  await assert.rejects(
    activateStagedRecallDatabase(config, staged.databaseTarget),
    /Recall staged database incomplete/u,
  );
  await assert.rejects(
    activateStagedRecallDatabase(config, '../outside'),
    /Recall database pointer target escapes the managed data directory/u,
  );
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
});

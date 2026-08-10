import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, readlink, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  activateRecallDatabaseCandidate,
  activateStagedRecallDatabase,
  createRecallDatabaseCandidate,
  resolveActiveRecallDatabasePaths,
  restorePreviousRecallDatabase,
  resumeRecallDatabaseCandidate,
  stageRecallDatabaseCandidate,
  type RecallDatabaseCandidate,
  type RecallDatabaseGenerationConfig,
} from './recall-database-generation.js';
import {
  createRecallIndexManifest,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
} from './recall-index-manifest.js';

const OCTEN_IDENTITY: RecallEmbeddingModelIdentity = {
  requestModel: 'octen-embed',
  servedModelId: 'Octen/Octen-Embedding-4B',
  nativeDimensions: 2_560,
  storedDimensions: 1_024,
  transformation: 'vendor-prefix-then-l2-v1',
};

function createGenerationConfig(dataDirectory: string): RecallDatabaseGenerationConfig {
  return {
    sqliteDatabasePath: join(dataDirectory, 'recall.sqlite'),
    legacyV6ZvecDatabasePath: join(dataDirectory, 'zvec'),
    legacyV6StatePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
  };
}

async function writeVersion8Manifest(manifestPath: string): Promise<void> {
  await writeRecallIndexManifest(
    manifestPath,
    createRecallIndexManifest({ embeddingIdentity: OCTEN_IDENTITY }),
  );
}

async function completeRecallDatabaseCandidate(candidate: RecallDatabaseCandidate): Promise<void> {
  await Promise.all([
    writeFile(candidate.paths.sqliteDatabasePath, 'sqlite database'),
    writeVersion8Manifest(candidate.paths.manifestPath),
    writeFile(candidate.paths.indexMaintenanceStatusPath, 'status'),
  ]);
}

async function createLegacyVersion6Database(config: RecallDatabaseGenerationConfig): Promise<void> {
  await mkdir(config.legacyV6ZvecDatabasePath);
  await Promise.all([
    writeFile(config.legacyV6StatePath, 'legacy state'),
    writeFile(config.manifestPath, '{"manifestVersion":6}\n'),
  ]);
}

void test('version 8 candidate stages and activates with one recall.sqlite database file', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-v8-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);

  const staged = await stageRecallDatabaseCandidate(config, candidate);

  assert.equal(await readFile(staged.paths.sqliteDatabasePath, 'utf8'), 'sqlite database');
  assert.deepEqual(Object.keys(staged.paths).sort(), [
    'indexMaintenanceStatusPath',
    'legacyV6StatePath',
    'legacyV6ZvecDatabasePath',
    'manifestPath',
    'sqliteDatabasePath',
  ]);
  await assert.rejects(readFile(staged.paths.legacyV6ZvecDatabasePath), { code: 'ENOENT' });
  assert.deepEqual((await readdir(staged.directoryPath)).sort(), [
    'index-maintenance-status.json',
    'index-manifest.json',
    'recall.sqlite',
  ]);
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });

  assert.deepEqual(await activateStagedRecallDatabase(config, staged.databaseTarget), {
    previousAvailable: false,
  });
  assert.equal(await readlink(join(dataDirectory, 'active')), staged.databaseTarget);
  assert.equal(
    (await resolveActiveRecallDatabasePaths(config)).sqliteDatabasePath,
    staged.paths.sqliteDatabasePath,
  );
});

void test('activation records root version 6 and rollback restores it atomically', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-v6-rollback-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  await createLegacyVersion6Database(config);
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);

  const activation = await activateRecallDatabaseCandidate(config, candidate);
  const activeTarget = await readlink(join(dataDirectory, 'active'));

  assert.deepEqual(activation, { previousAvailable: true });
  await restorePreviousRecallDatabase(config);
  assert.equal(await readlink(join(dataDirectory, 'active')), '.');
  assert.equal(
    (await resolveActiveRecallDatabasePaths(config)).legacyV6ZvecDatabasePath,
    config.legacyV6ZvecDatabasePath,
  );
  assert.notEqual(activeTarget, '.');
});

void test('candidate missing recall.sqlite is rejected with an actionable rebuild message', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-incomplete-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await Promise.all([
    writeVersion8Manifest(candidate.paths.manifestPath),
    writeFile(candidate.paths.indexMaintenanceStatusPath, 'status'),
  ]);

  await assert.rejects(
    stageRecallDatabaseCandidate(config, candidate),
    /Recall candidate database incomplete[\s\S]*recall\.sqlite[\s\S]*psr index --rebuild/u,
  );
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
});

void test('existing staged version 7 layout is rejected and never activates', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-v7-generation-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const stagedDirectory = join(config.databaseGenerationRootPath ?? '', 'generation-draft-v7');
  await mkdir(stagedDirectory, { recursive: true });
  await Promise.all([
    writeFile(join(stagedDirectory, 'recall.sqlite'), 'not a unified database'),
    writeFile(join(stagedDirectory, 'index-manifest.json'), '{"manifestVersion":7}\n'),
    writeFile(join(stagedDirectory, 'index-maintenance-status.json'), 'status'),
  ]);

  await assert.rejects(
    activateStagedRecallDatabase(config, 'generations/generation-draft-v7'),
    /version 7[\s\S]*incompatible[\s\S]*psr index --rebuild/u,
  );
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
});

void test('rollback rejects an incomplete root version 6 target without changing active', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-invalid-rollback-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  await createLegacyVersion6Database(config);
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);
  await activateRecallDatabaseCandidate(config, candidate);
  const activeTarget = await readlink(join(dataDirectory, 'active'));
  await rm(config.legacyV6StatePath);

  await assert.rejects(
    restorePreviousRecallDatabase(config),
    /Previous recall database incomplete[\s\S]*index-state\.json[\s\S]*psr index --rebuild/u,
  );
  assert.equal(await readlink(join(dataDirectory, 'active')), activeTarget);
});

void test('candidate resume preserves the sole interrupted recall.sqlite file', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-resume-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);

  await assert.rejects(
    resumeRecallDatabaseCandidate(config),
    /Recall candidate resume requires exactly one interrupted candidate; found 0/u,
  );
  const candidate = await createRecallDatabaseCandidate(config);
  await writeFile(candidate.paths.sqliteDatabasePath, 'partial database');

  const resumed = await resumeRecallDatabaseCandidate(config);

  assert.equal(resumed.directoryPath, candidate.directoryPath);
  assert.equal(await readFile(resumed.paths.sqliteDatabasePath, 'utf8'), 'partial database');
});

void test('fresh candidate creation removes stale candidates but preserves generations', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-stale-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const stale = await createRecallDatabaseCandidate(config);
  await writeFile(stale.paths.sqliteDatabasePath, 'partial database');
  const generationPath = join(config.databaseGenerationRootPath ?? '', 'generation-certified');
  await mkdir(generationPath);

  const candidate = await createRecallDatabaseCandidate(config);

  assert.equal(candidate.staleCandidatesRemoved, 1);
  await assert.rejects(readFile(stale.paths.sqliteDatabasePath), { code: 'ENOENT' });
  assert.equal((await stat(generationPath)).isDirectory(), true);
  await assert.rejects(readFile(candidate.paths.sqliteDatabasePath), { code: 'ENOENT' });
});

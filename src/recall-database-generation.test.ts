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
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    indexMaintenanceStatusPath: join(dataDirectory, 'index-maintenance-status.json'),
    databaseGenerationRootPath: join(dataDirectory, 'generations'),
  };
}

async function writeCurrentManifest(manifestPath: string): Promise<void> {
  await writeRecallIndexManifest(
    manifestPath,
    createRecallIndexManifest({ embeddingIdentity: OCTEN_IDENTITY }),
  );
}

async function completeRecallDatabaseCandidate(candidate: RecallDatabaseCandidate): Promise<void> {
  await Promise.all([
    writeFile(candidate.paths.sqliteDatabasePath, 'sqlite database'),
    writeCurrentManifest(candidate.paths.manifestPath),
    writeFile(candidate.paths.indexMaintenanceStatusPath, 'status'),
  ]);
}

void test('current-format candidate stages and activates one recall.sqlite database', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-current-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(candidate);

  const staged = await stageRecallDatabaseCandidate(config, candidate);

  assert.equal(await readFile(staged.paths.sqliteDatabasePath, 'utf8'), 'sqlite database');
  assert.deepEqual(Object.keys(staged.paths).sort(), [
    'indexMaintenanceStatusPath',
    'manifestPath',
    'sqliteDatabasePath',
  ]);
  assert.deepEqual((await readdir(staged.directoryPath)).sort(), [
    'index-maintenance-status.json',
    'index-manifest.json',
    'recall.sqlite',
  ]);
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });

  await activateStagedRecallDatabase(config, staged.databaseTarget);
  assert.equal(await readlink(join(dataDirectory, 'active')), staged.databaseTarget);
  assert.equal(
    (await resolveActiveRecallDatabasePaths(config)).sqliteDatabasePath,
    staged.paths.sqliteDatabasePath,
  );
  assert.equal((await readdir(staged.directoryPath)).includes('.previous-database.json'), false);
});

void test('candidate activation replaces the active pointer without compatibility metadata', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-current-activation-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);

  const first = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(first);
  await activateRecallDatabaseCandidate(config, first);
  const firstTarget = await readlink(join(dataDirectory, 'active'));

  const second = await createRecallDatabaseCandidate(config);
  await completeRecallDatabaseCandidate(second);
  await activateRecallDatabaseCandidate(config, second);
  const secondTarget = await readlink(join(dataDirectory, 'active'));

  assert.notEqual(secondTarget, firstTarget);
  assert.equal((await stat(join(dataDirectory, firstTarget))).isDirectory(), true);
  assert.equal(
    (await readdir(join(dataDirectory, secondTarget))).includes('.previous-database.json'),
    false,
  );
});

void test('candidate missing recall.sqlite is rejected with an actionable rebuild message', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-incomplete-candidate-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const candidate = await createRecallDatabaseCandidate(config);
  await Promise.all([
    writeCurrentManifest(candidate.paths.manifestPath),
    writeFile(candidate.paths.indexMaintenanceStatusPath, 'status'),
  ]);

  await assert.rejects(
    stageRecallDatabaseCandidate(config, candidate),
    /Recall candidate database incomplete[\s\S]*recall\.sqlite[\s\S]*psr index --rebuild/u,
  );
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
});

void test('obsolete staged layouts are rejected and never activate', async (t) => {
  const dataDirectory = await mkdtemp(join(tmpdir(), 'recall-obsolete-generation-'));
  t.after(() => rm(dataDirectory, { recursive: true, force: true }));
  const config = createGenerationConfig(dataDirectory);
  const stagedDirectory = join(config.databaseGenerationRootPath ?? '', 'generation-obsolete');
  await mkdir(stagedDirectory, { recursive: true });
  await Promise.all([
    writeFile(join(stagedDirectory, 'recall.sqlite'), 'not a current database'),
    writeFile(join(stagedDirectory, 'index-manifest.json'), '{"manifestVersion":7}\n'),
    writeFile(join(stagedDirectory, 'index-maintenance-status.json'), 'status'),
  ]);

  await assert.rejects(
    activateStagedRecallDatabase(config, 'generations/generation-obsolete'),
    /version 7[\s\S]*incompatible[\s\S]*psr index --rebuild/u,
  );
  await assert.rejects(readlink(join(dataDirectory, 'active')), { code: 'ENOENT' });
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

import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { assertRecallTestDataRoot } from './assert-recall-test-data-root.js';

void test('recall test data root accepts an isolated temporary directory', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-test-root-safe-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const repositoryRoot = join(directory, 'repository');
  const homeDirectory = join(directory, 'home');
  const testDataRoot = join(directory, 'scratch');
  await Promise.all([mkdir(repositoryRoot), mkdir(homeDirectory), mkdir(testDataRoot)]);

  assert.equal(
    await assertRecallTestDataRoot({ testDataRoot, repositoryRoot, homeDirectory }),
    testDataRoot,
  );
});

void test('recall test data root rejects defaults, configured paths, settings, and committed evidence', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-test-root-protected-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const repositoryRoot = join(directory, 'repository');
  const homeDirectory = join(directory, 'home');
  const configuredRecallRoot = join(directory, 'configured-recall');
  await Promise.all([
    mkdir(join(repositoryRoot, 'docs', 'evaluation'), { recursive: true }),
    mkdir(join(repositoryRoot, 'evaluation', 'corpus'), { recursive: true }),
    mkdir(join(homeDirectory, '.pi', 'agent', 'recall'), { recursive: true }),
    mkdir(join(homeDirectory, '.pi', 'agent', 'sessions'), { recursive: true }),
    mkdir(configuredRecallRoot),
  ]);

  const protectedCandidates = [
    join(homeDirectory, '.pi', 'agent', 'recall', 'benchmark'),
    join(homeDirectory, '.pi', 'agent', 'sessions'),
    join(homeDirectory, '.pi', 'agent', 'settings.json'),
    join(homeDirectory, '.pi', 'agent', 'recall.json'),
    join(configuredRecallRoot, 'embedding-cache'),
    join(repositoryRoot, 'docs', 'evaluation', 'scratch'),
    join(repositoryRoot, 'evaluation', 'corpus'),
  ];
  for (const testDataRoot of protectedCandidates) {
    await assert.rejects(
      () =>
        assertRecallTestDataRoot({
          testDataRoot,
          repositoryRoot,
          homeDirectory,
          configuredProtectedPaths: [
            configuredRecallRoot,
            join(configuredRecallRoot, 'operation.lock'),
            join(configuredRecallRoot, 'markers'),
            join(configuredRecallRoot, 'embedding-cache'),
          ],
        }),
      /Recall test data root overlaps protected path/u,
    );
  }
});

void test('recall test data root resolves symlink aliases before checking overlap', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-test-root-symlink-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const repositoryRoot = join(directory, 'repository');
  const homeDirectory = join(directory, 'home');
  const productionRecallRoot = join(homeDirectory, '.pi', 'agent', 'recall');
  await Promise.all([mkdir(repositoryRoot), mkdir(productionRecallRoot, { recursive: true })]);
  const alias = join(directory, 'apparently-safe');
  await symlink(productionRecallRoot, alias);

  await assert.rejects(
    () => assertRecallTestDataRoot({ testDataRoot: alias, repositoryRoot, homeDirectory }),
    /Recall test data root overlaps protected path/u,
  );
});

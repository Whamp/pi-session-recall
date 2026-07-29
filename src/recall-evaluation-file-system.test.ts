import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createPrivateRecallEvaluationConfig,
  isPathInsideRecallEvaluationArea,
  writeAtomicRecallEvaluationFile,
  type RecallEvaluationFileSystem,
} from './recall-evaluation-file-system.js';

void test('recall evaluation paths stay bounded and publishable files replace atomically', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-evaluation-file-system-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evidencePath = join(directory, 'nested', 'evidence.json');

  assert.equal(isPathInsideRecallEvaluationArea(directory, directory), true);
  assert.equal(isPathInsideRecallEvaluationArea(directory, evidencePath), true);
  assert.equal(isPathInsideRecallEvaluationArea(directory, `${directory}-nearby`), false);

  await writeAtomicRecallEvaluationFile(evidencePath, 'first');
  await writeAtomicRecallEvaluationFile(evidencePath, 'second');
  assert.equal(await readFile(evidencePath, 'utf8'), 'second');
  assert.deepEqual(
    (await readdir(join(directory, 'nested'))).filter((name) => name.endsWith('.tmp')),
    [],
  );
});

void test('durable evaluation replacement syncs and closes the temp before rename and syncs the destination directory', async () => {
  const operations: string[] = [];
  const temporaryPathPattern = /evidence\.json\.\d+\.[0-9a-f-]+\.tmp$/u;
  const fileSystem: RecallEvaluationFileSystem = {
    async mkdir() {
      return undefined;
    },
    async open(path, flags) {
      const label = flags === 'wx' ? 'temp' : 'directory';
      if (label === 'temp') {
        assert.match(path, temporaryPathPattern);
        operations.push('create temp');
      } else {
        operations.push('open directory');
      }
      return {
        async writeFile(content) {
          assert.equal(content, 'evidence');
          operations.push('write temp');
        },
        async sync() {
          operations.push(`sync ${label}`);
        },
        async close() {
          operations.push(`close ${label}`);
        },
      };
    },
    async rename(from, to) {
      assert.match(from, temporaryPathPattern);
      assert.equal(to, '/publication/evidence.json');
      operations.push('replace target');
    },
    async rm() {
      operations.push('remove temp');
    },
    async readdir() {
      return [];
    },
  };

  await writeAtomicRecallEvaluationFile('/publication/evidence.json', 'evidence', fileSystem);

  assert.deepEqual(operations, [
    'create temp',
    'write temp',
    'sync temp',
    'close temp',
    'replace target',
    'open directory',
    'sync directory',
    'close directory',
  ]);
});

void test('durable evaluation replacement persists each newly created parent entry', async () => {
  const syncedDirectories: string[] = [];
  const fileSystem: RecallEvaluationFileSystem = {
    async mkdir() {
      return '/publication/new-parent';
    },
    async open(path, flags) {
      const directoryPath = flags === 'r' ? path : null;
      return {
        async writeFile() {},
        async sync() {
          if (directoryPath) {
            syncedDirectories.push(directoryPath);
          }
        },
        async close() {},
      };
    },
    async rename() {},
    async rm() {},
    async readdir() {
      return [];
    },
  };

  await writeAtomicRecallEvaluationFile(
    '/publication/new-parent/nested/evidence.json',
    'evidence',
    fileSystem,
  );

  assert.deepEqual(syncedDirectories, [
    '/publication',
    '/publication/new-parent',
    '/publication/new-parent/nested',
    '/publication/new-parent/nested',
  ]);
});

void test('durable evaluation replacement cannot resolve successfully before final directory sync', async () => {
  const finalSyncFailure = new Error('directory sync failed');
  const fileSystem: RecallEvaluationFileSystem = {
    async mkdir() {
      return undefined;
    },
    async open(path, flags) {
      assert.ok(path);
      return {
        async writeFile() {},
        async sync() {
          if (flags === 'r') {
            throw finalSyncFailure;
          }
        },
        async close() {},
      };
    },
    async rename() {},
    async rm() {},
    async readdir() {
      return [];
    },
  };

  await assert.rejects(
    () => writeAtomicRecallEvaluationFile('/publication/evidence.json', 'evidence', fileSystem),
    (error: unknown) => error === finalSyncFailure,
  );
});

void test('durable evaluation replacement preserves the original failure when cleanup also fails', async () => {
  const originalFailure = new Error('replace failed');
  let tempClosed = false;
  const fileSystem: RecallEvaluationFileSystem = {
    async mkdir() {
      return undefined;
    },
    async open() {
      return {
        async writeFile() {},
        async sync() {},
        async close() {
          tempClosed = true;
        },
      };
    },
    async rename() {
      throw originalFailure;
    },
    async rm() {
      throw new Error('cleanup failed');
    },
    async readdir() {
      return [];
    },
  };

  await assert.rejects(
    () => writeAtomicRecallEvaluationFile('/publication/evidence.json', 'evidence', fileSystem),
    (error: unknown) => error === originalFailure,
  );
  assert.equal(tempClosed, true);
});

void test('private recall evaluation config replaces every production mutable and selector path', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'private-recall-evaluation-config-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const productionDataDirectory = join(directory, 'production-recall');
  const evaluationRootDirectory = join(directory, 'private-evaluation');
  const snapshotDirectory = join(evaluationRootDirectory, 'snapshots');
  const workDirectory = join(evaluationRootDirectory, 'evaluation-work');
  await mkdir(evaluationRootDirectory);
  const baseConfig = await loadRecallConversationConfig({
    homeDirectory: directory,
    environment: {
      PI_RECALL_DATA_DIRECTORY: productionDataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: join(directory, 'production-sessions'),
    },
  });

  const privateConfig = createPrivateRecallEvaluationConfig({
    baseConfig,
    evaluationRootDirectory,
    workDirectory,
    sessionsDirectory: snapshotDirectory,
    immutableInputPaths: [snapshotDirectory],
    candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  });
  const writableAndSelectorPaths = [
    privateConfig.databasePath,
    privateConfig.statePath,
    privateConfig.manifestPath,
    privateConfig.tokenizerCacheDirectory,
    privateConfig.embeddingCacheDirectory,
    privateConfig.lockPath,
    privateConfig.generationsDirectory,
    privateConfig.activeGenerationPath,
    privateConfig.stagingGenerationPath,
    privateConfig.backgroundIndexStatusPath,
    privateConfig.backgroundIndexRequestPath,
    privateConfig.diagnosticLogPath,
    privateConfig.retainedDiagnosticLogPath,
  ];
  for (const path of writableAndSelectorPaths) {
    assert.ok(path);
    assert.equal(isPathInsideRecallEvaluationArea(workDirectory, path), true);
    assert.equal(isPathInsideRecallEvaluationArea(productionDataDirectory, path), false);
  }
  assert.equal(privateConfig.sessionsDirectory, snapshotDirectory);
  assert.notEqual(privateConfig.activeGenerationPath, baseConfig.activeGenerationPath);
  assert.notEqual(privateConfig.stagingGenerationPath, baseConfig.stagingGenerationPath);

  assert.throws(
    () =>
      createPrivateRecallEvaluationConfig({
        baseConfig,
        evaluationRootDirectory,
        workDirectory: join(snapshotDirectory, 'unsafe-work'),
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [snapshotDirectory],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /work area overlaps an immutable input/u,
  );
  assert.throws(
    () =>
      createPrivateRecallEvaluationConfig({
        baseConfig,
        evaluationRootDirectory,
        workDirectory,
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [join(workDirectory, 'plans.json')],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /work area overlaps an immutable input/u,
  );

  assert.throws(
    () =>
      createPrivateRecallEvaluationConfig({
        baseConfig: {
          ...baseConfig,
          activeGenerationPath: join(workDirectory, 'production-active-generation.json'),
        },
        evaluationRootDirectory,
        workDirectory,
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [snapshotDirectory],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /work area overlaps a production path/u,
  );

  await mkdir(workDirectory);
  const symlinkedProductionDataDirectory = join(directory, 'production-recall-alias');
  await symlink(workDirectory, symlinkedProductionDataDirectory, 'dir');
  const symlinkedProductionConfig = await loadRecallConversationConfig({
    homeDirectory: directory,
    environment: {
      PI_RECALL_DATA_DIRECTORY: symlinkedProductionDataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: join(directory, 'production-sessions'),
    },
  });
  assert.throws(
    () =>
      createPrivateRecallEvaluationConfig({
        baseConfig: symlinkedProductionConfig,
        evaluationRootDirectory,
        workDirectory,
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [snapshotDirectory],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /work area overlaps a production path/u,
  );

  const physicalPrivateDirectory = join(directory, 'physical-private-evaluation');
  const externalDirectory = join(directory, 'external');
  await mkdir(physicalPrivateDirectory, { recursive: true });
  await mkdir(externalDirectory);
  await symlink(externalDirectory, join(physicalPrivateDirectory, 'escape'), 'dir');
  assert.throws(
    () =>
      createPrivateRecallEvaluationConfig({
        baseConfig,
        evaluationRootDirectory: physicalPrivateDirectory,
        workDirectory: join(physicalPrivateDirectory, 'escape', 'evaluation-work'),
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [snapshotDirectory],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /symbolic link/u,
  );
});

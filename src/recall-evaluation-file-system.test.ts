import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createPrivateRecallEvaluationConfig,
  isPathInsideRecallEvaluationArea,
  writeAtomicRecallEvaluationFile,
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

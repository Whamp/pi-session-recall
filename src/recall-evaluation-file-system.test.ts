import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
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
  const snapshotDirectory = join(directory, 'private-evaluation', 'snapshots');
  const workDirectory = join(directory, 'private-evaluation', 'evaluation-work');
  const baseConfig = await loadRecallConversationConfig({
    homeDirectory: directory,
    environment: {
      PI_RECALL_DATA_DIRECTORY: productionDataDirectory,
      PI_RECALL_SESSIONS_DIRECTORY: join(directory, 'production-sessions'),
    },
  });

  const privateConfig = createPrivateRecallEvaluationConfig({
    baseConfig,
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
        workDirectory,
        sessionsDirectory: snapshotDirectory,
        immutableInputPaths: [join(workDirectory, 'plans.json')],
        candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
      }),
    /work area overlaps an immutable input/u,
  );
});

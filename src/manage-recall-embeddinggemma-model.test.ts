import assert from 'node:assert/strict';
import { copyFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { runRecallEmbeddingGemmaModelCommand } from './manage-recall-embeddinggemma-model.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import {
  createRecallModelArtifactFixtureGguf as createCommandFixtureGguf,
  createRecallModelArtifactFixtureProfile as createCommandFixtureProfile,
} from './recall-model-artifact-test-fixture.js';

const commandStatusSchema = Type.Object({ state: Type.String() });
const commandDiagnosisSchema = Type.Object({ healthy: Type.Boolean() });
const commandInspectionSchema = Type.Object({
  profile: Type.Object({
    source: Type.Object({ revision: Type.String() }),
    license: Type.Object({ distributionStatus: Type.String() }),
  }),
  status: commandStatusSchema,
});

void test('EmbeddingGemma model command inspects metadata and gates mutations with approve', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = createCommandFixtureGguf();
  const sourcePath = join(root, 'source.gguf');
  await writeFile(sourcePath, artifact);
  let downloadCount = 0;
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact(_sourceUrl, destinationPath) {
      downloadCount += 1;
      await copyFile(sourcePath, destinationPath);
    },
  };
  const profile = createCommandFixtureProfile(artifact);
  const cacheDirectory = join(root, 'models');
  const output: string[] = [];
  const options = {
    cacheDirectory,
    profile,
    transport,
    writeOutput(value: string) {
      output.push(value);
    },
  };

  await runRecallEmbeddingGemmaModelCommand(['inspect'], options);
  const inspection = Value.Parse(commandInspectionSchema, JSON.parse(output.pop() ?? ''));
  assert.equal(inspection.profile.source.revision, profile.source.revision);
  assert.equal(inspection.profile.license.distributionStatus, 'review-required');
  assert.equal(inspection.status.state, 'missing');

  await assert.rejects(
    () => runRecallEmbeddingGemmaModelCommand(['download'], options),
    /Recall model download approval required/u,
  );
  assert.equal(downloadCount, 0);

  await runRecallEmbeddingGemmaModelCommand(['download', '--approve'], options);
  assert.equal(downloadCount, 1);
  assert.equal(Value.Parse(commandStatusSchema, JSON.parse(output.pop() ?? '')).state, 'valid');

  await runRecallEmbeddingGemmaModelCommand(['doctor'], options);
  assert.equal(Value.Parse(commandDiagnosisSchema, JSON.parse(output.pop() ?? '')).healthy, true);

  await assert.rejects(
    () => runRecallEmbeddingGemmaModelCommand(['remove'], options),
    /Recall model removal approval required/u,
  );
  await runRecallEmbeddingGemmaModelCommand(['remove', '--approve'], options);
  assert.equal(Value.Parse(commandStatusSchema, JSON.parse(output.pop() ?? '')).state, 'missing');
});

void test('EmbeddingGemma model command rejects ambiguous actions and arguments', async () => {
  await assert.rejects(
    () => runRecallEmbeddingGemmaModelCommand(['download', '--approve', '--mystery']),
    /Recall EmbeddingGemma model command arguments invalid/u,
  );
  await assert.rejects(
    () => runRecallEmbeddingGemmaModelCommand(['unknown']),
    /usage: npm run model:embeddinggemma/u,
  );
});

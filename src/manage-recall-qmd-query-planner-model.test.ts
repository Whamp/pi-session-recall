import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { runRecallQmdQueryPlannerModelCommand } from './manage-recall-qmd-query-planner-model.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import { createRecallModelArtifactFixtureGguf } from './recall-model-artifact-test-fixture.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

const commandStatusSchema = Type.Object({ state: Type.String() });
const commandInspectionSchema = Type.Object({
  profile: Type.Object({
    profileId: Type.String(),
    source: Type.Object({ revision: Type.String(), byteSize: Type.Number() }),
  }),
  status: commandStatusSchema,
});

void test('QMD query planner model command inspects exact metadata and gates mutations', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-qmd-query-planner-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = createRecallModelArtifactFixtureGguf();
  const sourcePath = join(root, 'source.gguf');
  await writeFile(sourcePath, artifact);
  let downloadCount = 0;
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact(_sourceUrl, destinationPath) {
      downloadCount += 1;
      await copyFile(sourcePath, destinationPath);
    },
  };
  const recommended = createRecommendedQmdQueryPlanningModelProfile();
  const profile = {
    ...recommended,
    source: {
      ...recommended.source,
      byteSize: artifact.length,
      sha256: createHash('sha256').update(artifact).digest('hex'),
      downloadUrl: 'https://models.invalid/pinned/qmd-query-planner-fixture.gguf',
    },
  };
  const output: string[] = [];
  const options = {
    cacheDirectory: join(root, 'models'),
    profile,
    transport,
    writeOutput(value: string) {
      output.push(value);
    },
  };

  await runRecallQmdQueryPlannerModelCommand(['inspect'], options);
  const inspection = Value.Parse(commandInspectionSchema, JSON.parse(output.pop() ?? ''));
  assert.equal(inspection.profile.profileId, 'qmd-query-expansion-1.7b-q4-k-m-v1');
  assert.equal(inspection.profile.source.revision, recommended.source.revision);
  assert.equal(inspection.profile.source.byteSize, artifact.length);
  assert.equal(inspection.status.state, 'missing');
  assert.equal(downloadCount, 0);

  await assert.rejects(
    () => runRecallQmdQueryPlannerModelCommand(['download'], options),
    /Recall model download approval required/u,
  );
  assert.equal(downloadCount, 0);

  await runRecallQmdQueryPlannerModelCommand(['download', '--approve'], options);
  assert.equal(downloadCount, 1);
  assert.equal(Value.Parse(commandStatusSchema, JSON.parse(output.pop() ?? '')).state, 'valid');

  await runRecallQmdQueryPlannerModelCommand(['repair', '--approve'], options);
  assert.equal(downloadCount, 1);
  assert.equal(Value.Parse(commandStatusSchema, JSON.parse(output.pop() ?? '')).state, 'valid');

  await assert.rejects(
    () => runRecallQmdQueryPlannerModelCommand(['remove'], options),
    /Recall model removal approval required/u,
  );
  await runRecallQmdQueryPlannerModelCommand(['remove', '--approve'], options);
  assert.equal(Value.Parse(commandStatusSchema, JSON.parse(output.pop() ?? '')).state, 'missing');
});

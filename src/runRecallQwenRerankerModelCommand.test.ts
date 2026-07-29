import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { copyFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { runRecallQwenRerankerModelCommand } from './runRecallQwenRerankerModelCommand.js';
import type { RecallModelArtifactTransport } from './recall-model-artifact-cache.js';
import { createRecallModelArtifactFixtureGguf } from './recall-model-artifact.test-utils.js';
import { createRecommendedQwenRerankingModelProfile } from './recall-model-profiles.js';

const COMMAND_STATUS_SCHEMA = Type.Object({ state: Type.String() });
const COMMAND_INSPECTION_SCHEMA = Type.Object({
  profile: Type.Object({
    profileId: Type.String(),
    source: Type.Object({ revision: Type.String(), byteSize: Type.Number() }),
  }),
  status: COMMAND_STATUS_SCHEMA,
});

void test('Qwen reranker model command inspects metadata and gates mutations with approve', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-qwen-reranker-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = createRecallModelArtifactFixtureGguf();
  const sourcePath = join(root, 'source.gguf');
  await writeFile(sourcePath, artifact);
  let downloadCount = 0;
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact(sourceUrl, destinationPath) {
      void sourceUrl;
      downloadCount += 1;
      await copyFile(sourcePath, destinationPath);
    },
  };
  const recommended = createRecommendedQwenRerankingModelProfile();
  const profile = {
    ...recommended,
    source: {
      ...recommended.source,
      byteSize: artifact.length,
      sha256: createHash('sha256').update(artifact).digest('hex'),
      downloadUrl: 'https://models.invalid/pinned/qwen-reranker-fixture.gguf',
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

  await runRecallQwenRerankerModelCommand(['inspect'], options);
  const inspection = Value.Parse(COMMAND_INSPECTION_SCHEMA, JSON.parse(output.pop() ?? ''));
  assert.equal(inspection.profile.profileId, 'qwen3-reranker-0.6b-q8-0-v1');
  assert.equal(inspection.profile.source.revision, recommended.source.revision);
  assert.equal(inspection.status.state, 'missing');
  assert.equal(downloadCount, 0);

  await assert.rejects(
    () => runRecallQwenRerankerModelCommand(['download'], options),
    /Recall model download approval required/u,
  );
  assert.equal(downloadCount, 0);

  await runRecallQwenRerankerModelCommand(['download', '--approve'], options);
  assert.equal(downloadCount, 1);
  assert.equal(Value.Parse(COMMAND_STATUS_SCHEMA, JSON.parse(output.pop() ?? '')).state, 'valid');

  await assert.rejects(
    () => runRecallQwenRerankerModelCommand(['remove'], options),
    /Recall model removal approval required/u,
  );
  await runRecallQwenRerankerModelCommand(['remove', '--approve'], options);
  assert.equal(Value.Parse(COMMAND_STATUS_SCHEMA, JSON.parse(output.pop() ?? '')).state, 'missing');
});

void test('Qwen reranker model command rejects ambiguous actions and arguments', async () => {
  await assert.rejects(
    () => runRecallQwenRerankerModelCommand(['download', '--approve', '--mystery']),
    /Recall Qwen reranker model command arguments invalid/u,
  );
  await assert.rejects(
    () => runRecallQwenRerankerModelCommand(['unknown']),
    /usage: npm run model:qwen-reranker/u,
  );
});

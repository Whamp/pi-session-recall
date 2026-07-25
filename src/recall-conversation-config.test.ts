import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('recall config uses local octen embeddings and supports file plus environment overrides', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');
  await writeFile(
    configPath,
    JSON.stringify({ embeddingBatchSize: 4, dataDirectory: join(directory, 'file-data') }),
  );

  const config = await loadRecallConversationConfig({
    homeDirectory: directory,
    configPath,
    environment: {
      PI_RECALL_EMBEDDING_MODEL: 'environment-model',
      PI_RECALL_EMBEDDING_BATCH_SIZE: '12',
    },
  });

  assert.equal(config.embeddingBaseUrl, 'http://192.168.0.67:8090/v1');
  assert.equal(config.embeddingModel, 'environment-model');
  assert.equal(config.embeddingDimensions, 2560);
  assert.equal(config.embeddingBatchSize, 12);
  assert.equal(config.embeddingServedModelId, 'Octen/Octen-Embedding-4B');
  assert.equal(config.embeddingArtifact, 'Octen-Embedding-4B.Q8_0.gguf');
  assert.equal(config.embeddingQuantization, 'Q8_0');
  assert.equal(config.embeddingPooling, 'last');
  assert.equal(config.databasePath, join(directory, 'file-data', 'zvec'));
  assert.equal(config.manifestPath, join(directory, 'file-data', 'index-manifest.json'));
  assert.equal(config.tokenizerCacheDirectory, join(directory, 'file-data', 'tokenizers'));
  assert.equal(config.embeddingCacheDirectory, join(directory, 'file-data', 'embedding-cache'));
  assert.equal(config.sessionsDirectory, join(directory, '.pi', 'agent', 'sessions'));
});

void test('recall config rejects invalid numeric environment settings', async () => {
  await assert.rejects(
    () =>
      loadRecallConversationConfig({
        homeDirectory: '/tmp',
        configPath: '/missing',
        environment: { PI_RECALL_EMBEDDING_DIMENSIONS: 'zero' },
      }),
    /Recall configuration invalid integer/,
  );
});

import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('recall config defaults to one Octen profile and the frozen search policy', async () => {
  const home = await mkdtemp(join(tmpdir(), 'recall-config-'));
  try {
    const config = await loadRecallConversationConfig({ homeDirectory: home, environment: {} });

    assert.equal(config.embeddingBaseUrl, 'http://192.168.0.67:8090/v1');
    assert.equal(config.embeddingModel, 'octen-embed');
    assert.equal(config.embeddingServedModelId, 'Octen/Octen-Embedding-4B');
    assert.equal(config.embeddingNativeDimensions, 2_560);
    assert.equal(config.embeddingStoredDimensions, 1_024);
    assert.deepEqual(config.chunkPolicy, { maxTokens: 512, overlapTokens: 64 });
    assert.deepEqual(config.searchCandidateLimits, { dense: 8, invocation: 8 });
    assert.equal(config.databasePath, join(home, '.pi', 'agent', 'recall', 'zvec'));
    assert.equal(config.catalogPath, join(home, '.pi', 'agent', 'recall', 'recall-catalog.sqlite'));
    assert.equal(config.statePath, join(home, '.pi', 'agent', 'recall', 'index-state.json'));
    assert.equal(config.manifestPath, join(home, '.pi', 'agent', 'recall', 'index-manifest.json'));
    assert.equal(
      config.databaseGenerationRootPath,
      join(home, '.pi', 'agent', 'recall', 'generations'),
    );
    assert.equal(
      config.indexMaintenanceStatusPath,
      join(home, '.pi', 'agent', 'recall', 'index-maintenance-status.json'),
    );
    assert.equal(
      config.physicalSessionIgnoreStatePath,
      join(home, '.pi', 'agent', 'recall', 'physical-session-ignore.json'),
    );
    assert.equal('embeddingCacheDirectory' in config, false);
    assert.equal('diagnosticLogPath' in config, false);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});

void test('recall config accepts direct Octen HTTP and fixed-width profile overrides', async () => {
  const home = await mkdtemp(join(tmpdir(), 'recall-config-env-'));
  try {
    const config = await loadRecallConversationConfig({
      homeDirectory: home,
      environment: {
        PI_RECALL_EMBEDDING_BASE_URL: 'http://127.0.0.1:8090/v1',
        PI_RECALL_EMBEDDING_MODEL: 'octen-test',
        PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS: '2560',
        PI_RECALL_EMBEDDING_STORED_DIMENSIONS: '1024',
        PI_RECALL_EMBEDDING_BATCH_SIZE: '4',
      },
    });

    assert.equal(config.embeddingBaseUrl, 'http://127.0.0.1:8090/v1');
    assert.equal(config.embeddingModel, 'octen-test');
    assert.equal(config.embeddingStoredDimensions, 1_024);
    assert.equal(config.embeddingBatchSize, 4);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
});

void test('recall config rejects widths that cannot back the fixed compact dense store', async () => {
  await assert.rejects(
    loadRecallConversationConfig({
      homeDirectory: '/tmp/recall-config-wrong-width',
      environment: {
        PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS: '2560',
        PI_RECALL_EMBEDDING_STORED_DIMENSIONS: '768',
      },
    }),
    /stored dimensions 768 do not match the compact dense store width 1024/,
  );

  await assert.rejects(
    loadRecallConversationConfig({
      homeDirectory: '/tmp/recall-config-invalid',
      environment: {
        PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS: '512',
        PI_RECALL_EMBEDDING_STORED_DIMENSIONS: '1024',
      },
    }),
    /stored dimensions 1024 exceed native dimensions 512/,
  );
});

void test('recall config rejects removed model-management and diagnostics settings', async () => {
  const root = await mkdtemp(join(tmpdir(), 'recall-config-removed-'));
  const path = join(root, 'recall.json');
  try {
    await writeFile(path, '{"rerankerModel":"qwen","diagnostics":"all"}\n', 'utf8');

    await assert.rejects(
      loadRecallConversationConfig({ configPath: path, homeDirectory: root, environment: {} }),
      /Recall configuration invalid/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

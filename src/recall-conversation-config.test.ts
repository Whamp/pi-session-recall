import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallDiagnosticsMode } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';

void test('recall config uses local octen embeddings and supports file plus environment overrides', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');
  await writeFile(
    configPath,
    JSON.stringify({
      embeddingBatchSize: 4,
      rerankerModel: 'file-reranker-model',
      denseCandidateLimit: 7,
      dataDirectory: join(directory, 'file-data'),
    }),
  );

  const config = await loadRecallConversationConfig({
    homeDirectory: directory,
    configPath,
    environment: {
      PI_RECALL_EMBEDDING_MODEL: 'environment-model',
      PI_RECALL_EMBEDDING_BATCH_SIZE: '12',
      PI_RECALL_RERANKER_BASE_URL: 'http://reranker.test/v1',
      PI_RECALL_LEXICAL_CANDIDATE_LIMIT: '9',
    },
  });

  assert.equal(config.embeddingBaseUrl, 'http://192.168.0.67:8090/v1');
  assert.equal(config.embeddingModel, 'environment-model');
  assert.equal(config.embeddingDimensions, 2560);
  assert.equal(config.embeddingBatchSize, 12);
  assert.equal(config.rerankerBaseUrl, 'http://reranker.test/v1');
  assert.equal(config.rerankerModel, 'file-reranker-model');
  assert.deepEqual(config.searchCandidateLimits, { dense: 7, lexical: 9, identifier: 40 });
  assert.equal(config.embeddingServedModelId, 'Octen/Octen-Embedding-4B');
  assert.equal(config.embeddingArtifact, 'Octen-Embedding-4B.Q8_0.gguf');
  assert.equal(config.embeddingQuantization, 'Q8_0');
  assert.equal(config.embeddingPooling, 'last');
  assert.equal(config.databasePath, join(directory, 'file-data', 'zvec'));
  assert.equal(config.manifestPath, join(directory, 'file-data', 'index-manifest.json'));
  assert.equal(config.tokenizerCacheDirectory, join(directory, 'file-data', 'tokenizers'));
  assert.equal(config.embeddingCacheDirectory, join(directory, 'file-data', 'embedding-cache'));
  assert.equal(config.diagnosticsMode, RecallDiagnosticsMode.SLOW);
  assert.equal(config.diagnosticLogPath, join(directory, 'file-data', 'diagnostics.jsonl'));
  assert.equal(
    config.retainedDiagnosticLogPath,
    join(directory, 'file-data', 'diagnostics.previous.jsonl'),
  );
  assert.equal(config.sessionsDirectory, join(directory, '.pi', 'agent', 'sessions'));
});

void test('recall config accepts canonical repository identities mapped to absolute historical roots', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-lineage-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');
  const prototypeRoot = join(directory, 'historical-prototype');
  const localPrototypeRoot = join(directory, 'historical-local-prototype');
  const localRepositoryIdentity = `git-common-directory:${join(directory, 'successor', '.git')}`;
  await writeFile(
    configPath,
    JSON.stringify({
      projectLineages: {
        'git-origin:github.com/Whamp/successor': [prototypeRoot],
        [localRepositoryIdentity]: [localPrototypeRoot],
      },
    }),
  );

  const config = await loadRecallConversationConfig({
    homeDirectory: directory,
    configPath,
    environment: {},
  });

  assert.deepEqual(Array.from(config.projectLineages.entries()), [
    [localRepositoryIdentity, [localPrototypeRoot]],
    ['git-origin:github.com/Whamp/successor', [prototypeRoot]],
  ]);
});

void test('recall config rejects noncanonical targets and relative lineage roots', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-invalid-lineage-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');

  await writeFile(
    configPath,
    JSON.stringify({
      projectLineages: {
        'https://github.com/Whamp/successor': ['/absolute/prototype'],
      },
    }),
  );
  await assert.rejects(
    () => loadRecallConversationConfig({ homeDirectory: directory, configPath, environment: {} }),
    /project lineage target must be a canonical repository identity.*https:\/\/github\.com\/Whamp\/successor/,
  );

  await writeFile(
    configPath,
    JSON.stringify({
      projectLineages: {
        'git-origin:github.com/Whamp/successor': ['relative/prototype'],
      },
    }),
  );
  await assert.rejects(
    () => loadRecallConversationConfig({ homeDirectory: directory, configPath, environment: {} }),
    /project lineage root must be absolute.*relative\/prototype/,
  );
});

void test('recall config rejects lineage roots that overlap across repository identities', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-conflicting-lineage-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');
  const prototypeRoot = join(directory, 'prototype');
  const nestedRoot = join(prototypeRoot, 'nested');
  await writeFile(
    configPath,
    JSON.stringify({
      projectLineages: {
        'git-origin:github.com/Whamp/first': [prototypeRoot],
        'git-origin:github.com/Whamp/second': [nestedRoot],
      },
    }),
  );

  await assert.rejects(
    () => loadRecallConversationConfig({ homeDirectory: directory, configPath, environment: {} }),
    new RegExp(
      `project lineage roots conflict.*${prototypeRoot.replaceAll('/', '\\/')}.*${nestedRoot.replaceAll('/', '\\/')}.*assign them to one repository identity`,
      's',
    ),
  );
});

void test('recall config defaults to the deployed local Qwen reranker', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-defaults-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({
    homeDirectory: directory,
    configPath: join(directory, 'missing-recall.json'),
    environment: {},
  });

  assert.equal(config.rerankerBaseUrl, 'http://192.168.0.67:8091/v1');
  assert.equal(config.rerankerModel, 'qwen3-rerank');
  assert.deepEqual(Array.from(config.projectLineages.entries()), []);
});

void test('recall config accepts exactly the three file-only diagnostics modes', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');

  for (const diagnosticsMode of [
    RecallDiagnosticsMode.SLOW,
    RecallDiagnosticsMode.ALL,
    RecallDiagnosticsMode.OFF,
  ]) {
    await writeFile(configPath, JSON.stringify({ diagnostics: diagnosticsMode }));
    const config = await loadRecallConversationConfig({
      homeDirectory: directory,
      configPath,
      environment: { PI_RECALL_DIAGNOSTICS: 'off' },
    });
    assert.equal(config.diagnosticsMode, diagnosticsMode);
  }
});

void test('recall config rejects invalid diagnostics modes and additional properties', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-config-invalid-diagnostics-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const configPath = join(directory, 'recall.json');

  await writeFile(configPath, JSON.stringify({ diagnostics: 'verbose' }));
  await assert.rejects(
    () => loadRecallConversationConfig({ homeDirectory: directory, configPath, environment: {} }),
    /Recall configuration invalid/u,
  );

  await writeFile(configPath, JSON.stringify({ diagnostics: 'slow', diagnosticThreshold: 50 }));
  await assert.rejects(
    () => loadRecallConversationConfig({ homeDirectory: directory, configPath, environment: {} }),
    /Recall configuration invalid/u,
  );
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
  await assert.rejects(
    () =>
      loadRecallConversationConfig({
        homeDirectory: '/tmp',
        configPath: '/missing',
        environment: { PI_RECALL_IDENTIFIER_CANDIDATE_LIMIT: '201' },
      }),
    /candidate limit.*PI_RECALL_IDENTIFIER_CANDIDATE_LIMIT.*200/,
  );
});

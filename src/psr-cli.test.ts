import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  RecallConversationConfig,
  RecallConversationService,
} from './recall-conversation-service.js';
import { runPsrCli } from './psr-cli.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createPsrCliFixture() {
  const calls: unknown[] = [];
  const output: string[] = [];
  const config: RecallConversationConfig = {
    sessionsDirectory: '/sessions',
    databasePath: '/recall/zvec',
    statePath: '/recall/index-state.json',
    manifestPath: '/recall/index-manifest.json',
    tokenizerCacheDirectory: '/recall/tokenizers',
    lockPath: '/recall/operation.lock',
    embeddingBaseUrl: 'http://127.0.0.1:8090/v1',
    embeddingModel: 'octen-embed',
    embeddingServedModelId: 'Octen/Octen-Embedding-4B',
    embeddingNativeDimensions: 2_560,
    embeddingStoredDimensions: 1_024,
    embeddingBatchSize: 16,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  };
  const service = {
    async search() {
      throw new Error('psr must not search');
    },
    async index(options) {
      calls.push(options);
      return {
        totalChunks: 7,
        indexSummary: {
          scannedSessions: 3,
          indexedSessions: 2,
          removedSessions: 1,
          reusedVectors: 4,
          newlyEmbeddedChunks: 5,
          embeddingRequestCount: 1,
          deletedChunks: 2,
          failedSessions: [],
        },
      };
    },
  } satisfies RecallConversationService;
  return {
    calls,
    output,
    dependencies: {
      loadConfig: async () => config,
      createService(receivedConfig: RecallConversationConfig) {
        assert.equal(receivedConfig, config);
        return service;
      },
      writeOutput(text: string) {
        output.push(text);
      },
    },
  };
}

void test('psr index runs explicit incremental maintenance', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.deepEqual(fixture.calls, [{ rebuild: false, optimize: true }]);
  assert.match(fixture.output.join(''), /Indexed 2 of 3 sessions/);
  assert.match(fixture.output.join(''), /7 searchable documents/);
});

void test('psr index --rebuild explicitly replaces the index', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['index', '--rebuild'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.deepEqual(fixture.calls, [{ rebuild: true, optimize: true }]);
});

void test('psr rejects every command surface other than manual index and rebuild', async () => {
  const fixture = createPsrCliFixture();

  await assert.rejects(
    runPsrCli(['resume'], fixture.dependencies),
    /psr usage: psr index \[--rebuild\]/,
  );
  assert.deepEqual(fixture.calls, []);
});

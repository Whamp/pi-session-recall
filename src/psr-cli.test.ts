import assert from 'node:assert/strict';
import test from 'node:test';

import type {
  RecallConversationConfig,
  RecallConversationIndexOptions,
  RecallConversationService,
} from './recall-conversation-service.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import { runPsrCli } from './psr-cli.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createPsrCliFixture(
  progressEvents: readonly RecallIndexProgressEvent[] = [],
  options: {
    monotonicTimes?: number[];
    failedSessions?: Array<{ sessionPath: string; error: string }>;
    fatalError?: Error;
  } = {},
) {
  const calls: RecallConversationIndexOptions[] = [];
  const output: string[] = [];
  const progressOutput: string[] = [];
  const executionLog: string[] = [];
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
    async index(indexOptions) {
      calls.push(indexOptions ?? {});
      for (const event of progressEvents) {
        indexOptions?.onProgress?.(event);
      }
      if (options.fatalError) {
        throw options.fatalError;
      }
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
          failedSessions: options.failedSessions ?? [],
        },
      };
    },
  } satisfies RecallConversationService;
  return {
    calls,
    output,
    progressOutput,
    executionLog,
    dependencies: {
      loadConfig: async () => {
        executionLog.push('load config');
        return config;
      },
      createService(receivedConfig: RecallConversationConfig) {
        assert.equal(receivedConfig, config);
        return service;
      },
      writeOutput(text: string) {
        output.push(text);
      },
      writeProgress(text: string) {
        executionLog.push('write progress');
        progressOutput.push(text);
      },
      getMonotonicTimeMs() {
        return options.monotonicTimes?.shift() ?? 0;
      },
    },
  };
}

void test('psr index keeps progress on stderr and the completed summary on stdout', async () => {
  const fixture = createPsrCliFixture([
    { kind: 'discovering-physical-session-files' },
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 3,
      newFiles: 1,
      changedFiles: 1,
      missingFiles: 0,
      rebuild: false,
    },
    { kind: 'optimizing-collection' },
    { kind: 'completed' },
  ]);

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: false, optimize: true, onProgress: 'function' },
  );
  assert.equal(fixture.executionLog[0], 'write progress');
  assert.equal(fixture.executionLog[1], 'load config');
  assert.match(fixture.progressOutput.join(''), /Preparing/i);
  assert.match(fixture.progressOutput.join(''), /Discovering physical session files/i);
  assert.doesNotMatch(fixture.output.join(''), /Preparing|Discovering/iu);
  assert.match(fixture.output.join(''), /Indexed 2 of 3 sessions/);
  assert.match(fixture.output.join(''), /7 searchable documents/);
  assert.doesNotMatch(fixture.progressOutput.join(''), /7 searchable documents/iu);
});

void test('psr index prints planning and indexing phase transitions in event order', async () => {
  const fixture = createPsrCliFixture([
    { kind: 'planning-maintenance-workset' },
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 1,
      newFiles: 1,
      changedFiles: 0,
      missingFiles: 0,
      rebuild: false,
    },
    { kind: 'indexing-changed-physical-session-files' },
  ]);

  await runPsrCli(['index'], fixture.dependencies);

  assert.deepEqual(fixture.progressOutput.slice(1, 4), [
    'Planning maintenance workset...\n',
    'Maintenance workset: 1 physical session files discovered; 1 new, 0 changed, 0 missing.\n',
    'Indexing changed physical session files...\n',
  ]);
});

void test('psr index describes empty and rebuild maintenance worksets in plain text', async () => {
  const emptyFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 4,
      newFiles: 0,
      changedFiles: 0,
      missingFiles: 0,
      rebuild: false,
    },
  ]);
  await runPsrCli(['index'], emptyFixture.dependencies);
  assert.match(emptyFixture.progressOutput.join(''), /no files require indexing or removal/iu);

  const rebuildFixture = createPsrCliFixture([
    {
      kind: 'maintenance-workset-planned',
      discoveredFiles: 4,
      newFiles: 4,
      changedFiles: 0,
      missingFiles: 0,
      rebuild: true,
    },
  ]);
  await runPsrCli(['index', '--rebuild'], rebuildFixture.dependencies);
  assert.match(rebuildFixture.progressOutput.join(''), /all 4 scheduled for indexing/iu);
  const rebuildProgress = rebuildFixture.progressOutput.join('');
  assert.ok(!rebuildProgress.includes('\u001b'));
  assert.ok(!rebuildProgress.includes('\r'));
  assert.ok(!rebuildProgress.includes('%'));
  assert.ok(rebuildFixture.progressOutput.every((line) => line.endsWith('\n')));
});

void test('psr index throttles routine updates but emits phase changes and warnings immediately', async () => {
  const failedPath = '/sessions/damaged.jsonl';
  const fixture = createPsrCliFixture(
    [
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 2,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 128,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 2,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 256,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      { kind: 'physical-session-file-failed', sessionPath: failedPath },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 2,
        sessionPath: failedPath,
        indexedSessions: 0,
        newlyEmbeddedDocuments: 256,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 2,
        sessionPath: '/sessions/two.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 300,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 2,
        totalFiles: 2,
        sessionPath: '/sessions/two.jsonl',
        indexedSessions: 1,
        newlyEmbeddedDocuments: 301,
        reusedVectors: 0,
        deletedDocuments: 1,
        failedSessions: 1,
      },
      { kind: 'optimizing-collection' },
    ],
    { monotonicTimes: [0, 10, 500, 600, 700, 800, 1_700, 1_710] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  assert.equal(fixture.progressOutput.filter((line) => line.startsWith('Indexing ')).length, 3);
  assert.match(fixture.progressOutput.join(''), new RegExp(failedPath, 'u'));
  assert.match(fixture.progressOutput.join(''), /Optimizing recall collection/iu);
});

void test('psr index prints terminal file progress before the throttle interval elapses', async () => {
  const fixture = createPsrCliFixture(
    [
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 0,
        totalFiles: 1,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 0,
        newlyEmbeddedDocuments: 1,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
      {
        kind: 'indexing-maintenance-workset',
        completedFiles: 1,
        totalFiles: 1,
        sessionPath: '/sessions/one.jsonl',
        indexedSessions: 1,
        newlyEmbeddedDocuments: 1,
        reusedVectors: 0,
        deletedDocuments: 0,
        failedSessions: 0,
      },
    ],
    { monotonicTimes: [0, 0, 0] },
  );

  await runPsrCli(['index'], fixture.dependencies);

  assert.deepEqual(
    fixture.progressOutput
      .filter((line) => line.startsWith('Indexing '))
      .map((line) => line.match(/^Indexing (\d+\/\d+) files/u)?.[1]),
    ['0/1', '1/1'],
  );
});

void test('psr index warns immediately while retaining complete failure details on stdout', async () => {
  const sessionPath = '/sessions/damaged.jsonl';
  const error = 'Session graph invalid: missing parent entry user-9';
  const fixture = createPsrCliFixture([{ kind: 'physical-session-file-failed', sessionPath }], {
    failedSessions: [{ sessionPath, error }],
  });

  const exitCode = await runPsrCli(['index'], fixture.dependencies);

  assert.equal(exitCode, 1);
  assert.match(fixture.progressOutput.join(''), new RegExp(sessionPath, 'u'));
  assert.doesNotMatch(fixture.progressOutput.join(''), new RegExp(error, 'u'));
  assert.match(fixture.output.join(''), new RegExp(`Failed: ${sessionPath}: ${error}`, 'u'));
});

void test('psr index lets fatal service errors reach the standalone process handler', async () => {
  const fixture = createPsrCliFixture([], { fatalError: new Error('Store setup failed') });

  await assert.rejects(runPsrCli(['index'], fixture.dependencies), /Store setup failed/u);

  assert.equal(fixture.output.join(''), '');
  assert.match(fixture.progressOutput.join(''), /Preparing recall index/iu);
});

void test('psr index --rebuild explicitly replaces the index', async () => {
  const fixture = createPsrCliFixture();

  const exitCode = await runPsrCli(['index', '--rebuild'], fixture.dependencies);

  assert.equal(exitCode, 0);
  assert.equal(fixture.calls.length, 1);
  assert.deepEqual(
    { ...fixture.calls[0], onProgress: typeof fixture.calls[0]?.onProgress },
    { rebuild: true, optimize: true, onProgress: 'function' },
  );
});

void test('psr rejects every command surface other than manual index and rebuild', async () => {
  const fixture = createPsrCliFixture();

  await assert.rejects(
    runPsrCli(['resume'], fixture.dependencies),
    /psr usage: psr index \[--rebuild\]/,
  );
  assert.deepEqual(fixture.calls, []);
});

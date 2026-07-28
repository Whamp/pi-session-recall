import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallManualMaintenanceTrigger } from './enums.js';
import { runRecallIndexCommand } from './recall-index-command.js';
import type { RecallConversationIndexOptions } from './recall-conversation-service.js';

void test('recall index command blocks before indexing when quality evidence failed', async () => {
  let indexCalls = 0;
  const statusUpdates: Array<string | undefined> = [];

  await assert.rejects(
    () =>
      runRecallIndexCommand({
        argumentsText: '',
        qualityGateDecision: {
          automatedGatePassed: false,
          selectedPolicy: null,
          blockers: ['512-64: query p95 exceeds 2000 ms'],
        },
        service: {
          async index() {
            indexCalls += 1;
            throw new Error('index must remain blocked');
          },
        },
        ui: {
          setStatus(status) {
            statusUpdates.push(status);
          },
          notify() {},
        },
      }),
    /Recall full backfill blocked.*query p95 exceeds 2000 ms.*npm run evaluate:recall/s,
  );

  assert.equal(indexCalls, 0);
  assert.deepEqual(statusUpdates, []);
});

void test('recall index command attributes explicit incremental maintenance', async () => {
  let receivedOptions: RecallConversationIndexOptions | undefined;

  await runRecallIndexCommand({
    argumentsText: '',
    qualityGateDecision: {
      automatedGatePassed: true,
      selectedPolicy: {
        chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
        candidateCount: 4,
        finalCount: 3,
      },
      blockers: [],
    },
    service: {
      async index(options) {
        receivedOptions = options;
        return {
          totalChunks: 0,
          indexSummary: {
            scannedSessions: 0,
            indexedSessions: 0,
            removedSessions: 0,
            cacheHits: 0,
            newlyEmbeddedChunks: 0,
            embeddingRequestCount: 0,
            deletedChunks: 0,
            failedSessions: [],
          },
        };
      },
    },
    ui: {
      setStatus() {},
      notify() {},
    },
  });

  assert.equal(receivedOptions?.optimize, true);
  assert.equal(
    receivedOptions?.manualMaintenanceTrigger,
    RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  );
});

void test('recall index command forwards explicit rebuild after a clean measured gate pass', async () => {
  let receivedOptions: RecallConversationIndexOptions | undefined;
  const statusUpdates: Array<string | undefined> = [];
  const notifications: string[] = [];

  await runRecallIndexCommand({
    argumentsText: ' --rebuild ',
    qualityGateDecision: {
      automatedGatePassed: true,
      selectedPolicy: {
        chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
        candidateCount: 4,
        finalCount: 3,
      },
      blockers: [],
    },
    service: {
      async index(options) {
        receivedOptions = options;
        options?.onProgress?.({
          scannedSessions: 2,
          totalSessions: 8,
          sessionPath: '/sessions/two.jsonl',
        });
        return {
          totalChunks: 42,
          indexSummary: {
            scannedSessions: 8,
            indexedSessions: 8,
            removedSessions: 0,
            cacheHits: 10,
            newlyEmbeddedChunks: 2,
            embeddingRequestCount: 1,
            deletedChunks: 3,
            failedSessions: [],
          },
        };
      },
    },
    ui: {
      setStatus(status) {
        statusUpdates.push(status);
      },
      notify(message) {
        notifications.push(message);
      },
    },
  });

  assert.equal(receivedOptions?.rebuild, true);
  assert.equal(receivedOptions?.optimize, true);
  assert.equal(
    receivedOptions?.manualMaintenanceTrigger,
    RecallManualMaintenanceTrigger.MANUAL_REBUILD,
  );
  assert.deepEqual(statusUpdates, ['rebuilding conversations…', 'rebuilding 2/8', undefined]);
  assert.deepEqual(notifications, [
    'Recall index ready: 42 chunks · 10 cache hits · 2 newly embedded · 1 embedding requests · 3 removed',
  ]);
});

void test('recall index command performs explicit rollback without rerunning the backfill gate', async () => {
  let rollbackCalls = 0;
  let indexCalls = 0;
  const statusUpdates: Array<string | undefined> = [];
  const notifications: string[] = [];

  await runRecallIndexCommand({
    argumentsText: '--rollback',
    qualityGateDecision: {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: ['quality evidence is irrelevant to pointer rollback'],
    },
    service: {
      async index() {
        indexCalls += 1;
        throw new Error('rollback must not index');
      },
      async rollback() {
        rollbackCalls += 1;
      },
    },
    ui: {
      setStatus(status) {
        statusUpdates.push(status);
      },
      notify(message) {
        notifications.push(message);
      },
    },
  });

  assert.equal(rollbackCalls, 1);
  assert.equal(indexCalls, 0);
  assert.deepEqual(statusUpdates, ['rolling back recall generation…', undefined]);
  assert.deepEqual(notifications, [
    'Recall generation rolled back; retained markers are pending replay',
  ]);
});

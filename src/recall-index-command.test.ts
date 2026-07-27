import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallBackgroundIndexProcessState, RecallManualMaintenanceTrigger } from './enums.js';
import { runRecallIndexCommand } from './recall-index-command.js';
import type { RecallConversationIndexOptions } from './recall-conversation-service.js';

const backgroundStartingStatus = {
  version: 1 as const,
  buildId: 'build-1',
  generationId: null,
  embeddingProfileId: 'embedding-profile-test',
  processId: 4321,
  processState: RecallBackgroundIndexProcessState.STARTING,
  startedAt: '2026-08-01T10:00:00.000Z',
  updatedAt: '2026-08-01T10:00:00.000Z',
  completedAt: null,
  progress: null,
  latestCheckpoint: null,
  latestActionableError: null,
};

const unusedBackgroundIndexControls = {
  async startBackgroundIndexGeneration() {
    throw new Error('background start not expected');
  },
  async resumeBackgroundIndexGeneration() {
    throw new Error('background resume not expected');
  },
  async readBackgroundIndexGenerationStatus() {
    return null;
  },
  async stopBackgroundIndexGeneration() {
    throw new Error('background stop not expected');
  },
  async discardStagingIndexGeneration() {
    return false;
  },
};

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
          ...unusedBackgroundIndexControls,
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
      ...unusedBackgroundIndexControls,
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

  assert.equal(
    receivedOptions?.manualMaintenanceTrigger,
    RecallManualMaintenanceTrigger.MANUAL_INCREMENTAL_INDEX,
  );
});

void test('recall index command starts an explicit rebuild in a detached worker', async () => {
  let backgroundStartCalls = 0;
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
      ...unusedBackgroundIndexControls,
      async index() {
        throw new Error('detached rebuild must not index in the invoking process');
      },
      async startBackgroundIndexGeneration() {
        backgroundStartCalls += 1;
        return backgroundStartingStatus;
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

  assert.equal(backgroundStartCalls, 1);
  assert.deepEqual(statusUpdates, ['starting background rebuild…', undefined]);
  assert.deepEqual(notifications, [
    'Recall background index starting · generation pending · process 4321',
  ]);
});

void test('recall index status remains available when the quality gate is blocked', async () => {
  const notifications: Array<{ message: string; level: string }> = [];

  await runRecallIndexCommand({
    argumentsText: '--status',
    qualityGateDecision: {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: ['quality evidence unavailable'],
    },
    service: {
      ...unusedBackgroundIndexControls,
      async index() {
        throw new Error('status must not index');
      },
      async readBackgroundIndexGenerationStatus() {
        return {
          ...backgroundStartingStatus,
          generationId: 'generation-test',
          processState: RecallBackgroundIndexProcessState.FAILED,
          completedAt: '2026-08-01T10:01:00.000Z',
          latestActionableError: 'embedding endpoint timed out',
        };
      },
    },
    ui: {
      setStatus() {},
      notify(message, level) {
        notifications.push({ message, level });
      },
    },
  });

  assert.deepEqual(notifications, [
    {
      message:
        'Recall background index failed · generation generation-test · process 4321 · embedding endpoint timed out',
      level: 'warning',
    },
  ]);
});

void test('recall index command routes stop, resume, and discard explicitly', async () => {
  const calls: string[] = [];
  const service = {
    ...unusedBackgroundIndexControls,
    async index() {
      throw new Error('control commands must not index');
    },
    async stopBackgroundIndexGeneration() {
      calls.push('stop');
      return {
        ...backgroundStartingStatus,
        processState: RecallBackgroundIndexProcessState.STOPPING,
      };
    },
    async resumeBackgroundIndexGeneration() {
      calls.push('resume');
      return { ...backgroundStartingStatus, generationId: 'generation-test' };
    },
    async discardStagingIndexGeneration() {
      calls.push('discard');
      return true;
    },
  };
  const qualityGateDecision = {
    automatedGatePassed: true,
    selectedPolicy: {
      chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
      candidateCount: 4,
      finalCount: 3,
    },
    blockers: [],
  };
  const ui = { setStatus() {}, notify() {} };

  await runRecallIndexCommand({ argumentsText: '--stop', qualityGateDecision, service, ui });
  await runRecallIndexCommand({ argumentsText: '--resume', qualityGateDecision, service, ui });
  await runRecallIndexCommand({ argumentsText: '--discard', qualityGateDecision, service, ui });

  assert.deepEqual(calls, ['stop', 'resume', 'discard']);
});

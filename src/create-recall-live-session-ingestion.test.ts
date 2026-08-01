import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecallLiveSessionIngestion } from './create-recall-live-session-ingestion.js';
import { RecallLifecycleTrigger } from './enums.js';
import type {
  RecallConversationIndexResult,
  RecallConversationService,
} from './recall-conversation-service.js';

function createEmptyIndexResult(scannedSessions: number): RecallConversationIndexResult {
  return {
    totalChunks: 0,
    indexSummary: {
      scannedSessions,
      indexedSessions: 0,
      removedSessions: 0,
      cacheHits: 0,
      newlyEmbeddedChunks: 0,
      embeddingRequestCount: 0,
      deletedChunks: 0,
      failedSessions: [],
    },
  };
}

void test('live session ingestion serializes startup catch-up, settled refresh, and shutdown refresh', async () => {
  const calls: string[] = [];
  const service: Pick<RecallConversationService, 'index' | 'reconcileSession'> = {
    async index(options) {
      calls.push(
        `catch-up:${options?.lockWaitMilliseconds ?? 'unbounded'}:${options?.requireExistingGeneration ?? false}`,
      );
      return createEmptyIndexResult(3);
    },
    async reconcileSession(sessionPath, options) {
      calls.push(
        `session:${sessionPath}:${options?.lockWaitMilliseconds ?? 'unbounded'}:${options?.lifecycleTrigger ?? 'missing'}`,
      );
      return createEmptyIndexResult(1);
    },
  };
  const warnings: string[] = [];
  const ingestion = createRecallLiveSessionIngestion(service, (message) => warnings.push(message));

  const startup = ingestion.catchUpSessions();
  const settled = ingestion.reconcileActiveSession('/sessions/active.jsonl');
  await Promise.all([startup, settled]);
  await ingestion.shutdownActiveSession('/sessions/active.jsonl');

  assert.deepEqual(calls, [
    'catch-up:250:true',
    `session:/sessions/active.jsonl:250:${RecallLifecycleTrigger.AGENT_SETTLED}`,
    `session:/sessions/active.jsonl:250:${RecallLifecycleTrigger.SESSION_SHUTDOWN}`,
  ]);
  assert.deepEqual(warnings, []);
});

void test('live session shutdown cancels corpus catch-up before final active-session reconciliation', async () => {
  const calls: string[] = [];
  let catchUpSignal: AbortSignal | undefined;
  const service: Pick<RecallConversationService, 'index' | 'reconcileSession'> = {
    async index(options) {
      calls.push('catch-up-started');
      catchUpSignal = options?.signal;
      await new Promise<void>((resolve, reject) => {
        void resolve;
        if (catchUpSignal?.aborted) {
          reject(new Error('Recall conversation operation cancelled'));
          return;
        }
        catchUpSignal?.addEventListener(
          'abort',
          () => {
            reject(new Error('Recall conversation operation cancelled'));
          },
          { once: true },
        );
      });
      return createEmptyIndexResult(3);
    },
    async reconcileSession(sessionPath) {
      calls.push(`final:${sessionPath}`);
      return createEmptyIndexResult(1);
    },
  };
  const warnings: string[] = [];
  const ingestion = createRecallLiveSessionIngestion(service, (message) => warnings.push(message));

  const catchUp = ingestion.catchUpSessions();
  await new Promise<void>((resolve) => {
    setImmediate(resolve);
  });
  const shutdown = ingestion.shutdownActiveSession('/sessions/active.jsonl');
  await Promise.all([catchUp, shutdown]);

  assert.equal(catchUpSignal?.aborted, true);
  assert.deepEqual(calls, ['catch-up-started', 'final:/sessions/active.jsonl']);
  assert.deepEqual(warnings, []);
});

void test('live session ingestion defers quiet lock contention until the next lifecycle event', async () => {
  const calls: string[] = [];
  const service: Pick<RecallConversationService, 'index' | 'reconcileSession'> = {
    async index() {
      calls.push('locked-catch-up');
      throw new Error('Recall conversation operation cancelled');
    },
    async reconcileSession(sessionPath) {
      calls.push(sessionPath);
      return createEmptyIndexResult(1);
    },
  };
  const warnings: string[] = [];
  const ingestion = createRecallLiveSessionIngestion(service, (message) => warnings.push(message));

  await ingestion.catchUpSessions();
  await ingestion.reconcileActiveSession('/sessions/retry-after-lock.jsonl');

  assert.deepEqual(calls, ['locked-catch-up', '/sessions/retry-after-lock.jsonl']);
  assert.deepEqual(warnings, []);
});

void test('live session ingestion warns and retries after a failed background update', async () => {
  const calls: string[] = [];
  const service: Pick<RecallConversationService, 'index' | 'reconcileSession'> = {
    async index() {
      calls.push('catch-up');
      throw new Error('model unavailable');
    },
    async reconcileSession(sessionPath) {
      calls.push(sessionPath);
      return createEmptyIndexResult(1);
    },
  };
  const warnings: string[] = [];
  const ingestion = createRecallLiveSessionIngestion(service, (message) => warnings.push(message));

  await ingestion.catchUpSessions();
  await ingestion.reconcileActiveSession('/sessions/retry.jsonl');

  assert.deepEqual(calls, ['catch-up', '/sessions/retry.jsonl']);
  assert.deepEqual(warnings, [
    'Recall automatic session update skipped without changing the existing index: model unavailable',
  ]);
});

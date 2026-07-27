import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { access } from 'node:fs/promises';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

import {
  RecallBackgroundIndexProcessState,
  RecallDiagnosticsMode,
  RecallSearchScope,
} from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
} from './recall-conversation-service.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const EXEC_FILE_ASYNC = promisify(execFile);

const TOKENIZER = {
  encodeConversationText(text: string) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createBackgroundIndexTestConfig(
  directory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    embeddingCacheDirectory: join(directory, 'embedding-cache'),
    lockPath: join(directory, 'recall.lock'),
    generationsDirectory: join(directory, 'index-generations'),
    activeGenerationPath: join(directory, 'active-generation.json'),
    stagingGenerationPath: join(directory, 'staging-generation.json'),
    backgroundIndexStatusPath: join(directory, 'background-index-status.json'),
    backgroundIndexRequestPath: join(directory, 'background-index-request.json'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(directory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(directory, 'diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-request-model',
    embeddingServedModelId: 'test-served-model',
    embeddingArtifact: 'test-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker-model',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 5, lexical: 5, identifier: 5 },
  };
}

async function writeBackgroundIndexSession(
  sessionPath: string,
  content: string,
  sessionId = 'background-session',
  cwd = '/project',
): Promise<void> {
  await writeFile(
    sessionPath,
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: sessionId,
        timestamp: '2026-08-01T10:00:00.000Z',
        cwd,
      }),
      JSON.stringify({
        type: 'message',
        id: `${sessionId}-entry`,
        parentId: null,
        timestamp: '2026-08-01T10:00:01.000Z',
        message: { role: 'assistant', content },
      }),
    ].join('\n') + '\n',
  );
}

function parseBackgroundProcessId(json: string): number {
  const parsed: unknown = JSON.parse(json);
  if (!isUnknownRecord(parsed) || typeof parsed.processId !== 'number') {
    throw new Error('Background launcher result missing processId');
  }
  return parsed.processId;
}

function parseJsonString(json: string): string {
  const parsed: unknown = JSON.parse(json);
  if (typeof parsed !== 'string') {
    throw new Error('Background fixture log entry must be a JSON string');
  }
  return parsed;
}

async function waitForPath(path: string, statusPath: string): Promise<void> {
  for (let attempt = 0; attempt < 250; attempt += 1) {
    try {
      await access(path);
      return;
    } catch {
      await sleep(20);
    }
  }
  const status = await readFile(statusPath, 'utf8').catch(() => 'status record missing');
  assert.fail(`Timed out waiting for ${path}; background status: ${status}`);
}

async function waitForBackgroundProcessState(
  service: ReturnType<typeof createRecallConversationService>,
  expectedState: RecallBackgroundIndexProcessState,
) {
  for (let attempt = 0; attempt < 500; attempt += 1) {
    const status = await service.readBackgroundIndexGenerationStatus();
    if (status?.processState === expectedState) {
      return status;
    }
    await sleep(20);
  }
  assert.fail(`Timed out waiting for background index state ${expectedState}`);
}

void test('background rebuild reports progress while active recall remains searchable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-background-index-'));
  const releasePath = join(directory, 'fixture-embedding-release');
  t.after(async () => {
    await writeFile(releasePath, 'release\n', 'utf8').catch(() => undefined);
    await sleep(100);
    await rm(directory, { recursive: true, force: true });
  });
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'session.jsonl');
  const config = createBackgroundIndexTestConfig(directory, sessionsDirectory);
  const deterministicDependencies = {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents: readonly string[]) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return TOKENIZER;
    },
  };
  const activeService = createRecallConversationService(config, deterministicDependencies);

  await writeBackgroundIndexSession(sessionPath, 'active generation evidence');
  await activeService.index();
  await writeBackgroundIndexSession(sessionPath, 'background replacement evidence');

  const backgroundService = createRecallConversationService(config, {
    ...deterministicDependencies,
    backgroundIndexServiceFactory: {
      moduleUrl: new URL('./createRecallBackgroundIndexWorkerFixtureService.ts', import.meta.url)
        .href,
      exportName: 'createRecallBackgroundIndexWorkerFixtureService',
    },
  });
  const started = await backgroundService.startBackgroundIndexGeneration();

  assert.equal(started.processState, 'starting');
  assert.ok(started.processId > 0);
  await waitForPath(
    join(directory, 'fixture-embedding-started'),
    join(directory, 'background-index-status.json'),
  );
  const running = await backgroundService.readBackgroundIndexGenerationStatus();
  assert.equal(running?.processState, 'running');
  assert.ok(running?.generationId);
  assert.equal(running?.progress?.scannedSessions, 1);
  assert.equal(running?.progress?.totalSessions, 1);
  assert.equal(running?.latestActionableError, null);

  const activeSearch = await activeService.search('active generation evidence', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(activeSearch.results[0]?.content, 'active generation evidence');

  await writeFile(releasePath, 'release\n', 'utf8');
  await waitForBackgroundProcessState(
    backgroundService,
    RecallBackgroundIndexProcessState.SUCCEEDED,
  );
});

void test('detached staging build survives the client process that started it', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-background-survival-'));
  const releasePath = join(directory, 'fixture-embedding-release');
  t.after(async () => {
    await writeFile(releasePath, 'release\n', 'utf8').catch(() => undefined);
    await sleep(100);
    await rm(directory, { recursive: true, force: true });
  });
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeBackgroundIndexSession(
    join(sessionsDirectory, 'session.jsonl'),
    'background replacement evidence',
  );

  const launcherPath = fileURLToPath(
    new URL('./recall-background-index-launcher.test-utils.ts', import.meta.url),
  );
  const launched = await EXEC_FILE_ASYNC(
    process.execPath,
    ['--import', 'tsx', launcherPath, directory, sessionsDirectory],
    { cwd: process.cwd() },
  );
  const startingProcessId = parseBackgroundProcessId(launched.stdout);
  assert.doesNotThrow(() => process.kill(startingProcessId, 0));

  const config = createBackgroundIndexTestConfig(directory, sessionsDirectory);
  const observer = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return TOKENIZER;
    },
    backgroundIndexServiceFactory: {
      moduleUrl: new URL('./createRecallBackgroundIndexWorkerFixtureService.ts', import.meta.url)
        .href,
      exportName: 'createRecallBackgroundIndexWorkerFixtureService',
    },
  });
  await waitForPath(
    join(directory, 'fixture-embedding-started'),
    join(directory, 'background-index-status.json'),
  );
  const running = await waitForBackgroundProcessState(
    observer,
    RecallBackgroundIndexProcessState.RUNNING,
  );
  assert.equal(running.processId, startingProcessId);
  assert.doesNotThrow(() => process.kill(running.processId, 0));

  await writeFile(releasePath, 'release\n', 'utf8');
  await waitForBackgroundProcessState(observer, RecallBackgroundIndexProcessState.SUCCEEDED);
});

void test('stopped background rebuild resumes from its durable session checkpoint', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-background-resume-'));
  const releasePath = join(directory, 'fixture-embedding-release');
  t.after(async () => {
    await writeFile(releasePath, 'release\n', 'utf8').catch(() => undefined);
    await sleep(100);
    await rm(directory, { recursive: true, force: true });
  });
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeBackgroundIndexSession(
    join(sessionsDirectory, 'a.jsonl'),
    'first checkpointed evidence',
    'checkpoint-a',
    '/project/a',
  );
  await writeBackgroundIndexSession(
    join(sessionsDirectory, 'b.jsonl'),
    'background replacement evidence',
    'checkpoint-b',
    '/project/b',
  );
  const config = createBackgroundIndexTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return TOKENIZER;
    },
    backgroundIndexServiceFactory: {
      moduleUrl: new URL('./createRecallBackgroundIndexWorkerFixtureService.ts', import.meta.url)
        .href,
      exportName: 'createRecallBackgroundIndexWorkerFixtureService',
    },
  });

  await service.startBackgroundIndexGeneration();
  await waitForPath(
    join(directory, 'fixture-embedding-started'),
    join(directory, 'background-index-status.json'),
  );
  const running = await waitForBackgroundProcessState(
    service,
    RecallBackgroundIndexProcessState.RUNNING,
  );
  assert.equal(running.latestCheckpoint?.checkpointedSessions, 1);
  assert.match(running.latestCheckpoint?.sessionPath ?? '', /a\.jsonl$/u);
  await assert.rejects(() => service.discardStagingIndexGeneration(), /stop it before discard/u);

  const stopping = await service.stopBackgroundIndexGeneration();
  assert.equal(stopping.processState, 'stopping');
  const stopped = await waitForBackgroundProcessState(
    service,
    RecallBackgroundIndexProcessState.STOPPED,
  );
  assert.equal(stopped.generationId, running.generationId);
  assert.equal(stopped.latestCheckpoint?.checkpointedSessions, 1);
  assert.equal(stopped.latestActionableError, null);

  await writeFile(releasePath, 'release\n', 'utf8');
  const resumed = await service.resumeBackgroundIndexGeneration();
  assert.equal(resumed.generationId, running.generationId);
  const completed = await waitForBackgroundProcessState(
    service,
    RecallBackgroundIndexProcessState.SUCCEEDED,
  );
  assert.equal(completed.generationId, running.generationId);
  assert.equal(completed.latestCheckpoint?.checkpointedSessions, 2);
  const generationStatus = await service.readIndexGenerationStatus();
  assert.equal(generationStatus.active?.generationId, running.generationId);
  assert.equal(generationStatus.staging, null);

  const projectResolutions = (
    await readFile(join(directory, 'fixture-project-resolutions.jsonl'), 'utf8')
  )
    .trim()
    .split('\n')
    .map(parseJsonString);
  assert.equal(projectResolutions.filter((cwd) => cwd === '/project/a').length, 1);
  const search = await service.search('background replacement evidence', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(search.results[0]?.content, 'background replacement evidence');
});

void test('background worker bootstrap failure persists one bounded actionable error', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-background-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createBackgroundIndexTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProvider: {
      async embedQuery() {
        return [1, 0, 0];
      },
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
    },
    async loadTokenizer() {
      return TOKENIZER;
    },
    backgroundIndexServiceFactory: {
      moduleUrl: new URL('./createRecallBackgroundIndexWorkerFixtureService.ts', import.meta.url)
        .href,
      exportName: 'missingRecallBackgroundIndexServiceFactory',
    },
  });

  await service.startBackgroundIndexGeneration();
  const failed = await waitForBackgroundProcessState(
    service,
    RecallBackgroundIndexProcessState.FAILED,
  );
  assert.match(
    failed.latestActionableError ?? '',
    /service factory export missing: missingRecallBackgroundIndexServiceFactory/u,
  );
  assert.ok((failed.latestActionableError?.length ?? 0) <= 4096);
  const statusLines = (await readFile(config.backgroundIndexStatusPath ?? '', 'utf8'))
    .trim()
    .split('\n');
  assert.equal(statusLines.length, 1);
});

void test('crashed workers at every staging phase remain resumable and idempotent', async (t) => {
  const phases = ['parsing', 'embedding', 'store-write', 'optimization', 'pre-activation'] as const;

  for (const phase of phases) {
    await t.test(phase, async (phaseTest) => {
      const directory = await mkdtemp(join(tmpdir(), `recall-background-${phase}-`));
      phaseTest.after(async () => {
        await sleep(100);
        await rm(directory, { recursive: true, force: true });
      });
      const sessionsDirectory = join(directory, 'sessions');
      await mkdir(sessionsDirectory);
      await writeBackgroundIndexSession(
        join(sessionsDirectory, 'session.jsonl'),
        `recoverable ${phase} evidence`,
        `recoverable-${phase}`,
      );
      const triggerPath = join(directory, `fixture-interrupt-${phase}`);
      const markerPath = join(directory, `fixture-interrupted-${phase}`);
      await writeFile(triggerPath, 'interrupt\n', 'utf8');
      const config = createBackgroundIndexTestConfig(directory, sessionsDirectory);
      const service = createRecallConversationService(config, {
        embeddingProvider: {
          async embedQuery() {
            return [1, 0, 0];
          },
          async embedDocuments(documents) {
            return documents.map(() => [1, 0, 0]);
          },
        },
        async loadTokenizer() {
          return TOKENIZER;
        },
        backgroundIndexServiceFactory: {
          moduleUrl: new URL(
            './createRecallBackgroundIndexWorkerFixtureService.ts',
            import.meta.url,
          ).href,
          exportName: 'createRecallBackgroundIndexWorkerFixtureService',
        },
      });

      await service.startBackgroundIndexGeneration();
      await waitForPath(markerPath, join(directory, 'background-index-status.json'));
      const crashed = await waitForBackgroundProcessState(
        service,
        RecallBackgroundIndexProcessState.CRASHED,
      );
      assert.match(crashed.latestActionableError ?? '', /exited without a completion record/u);
      assert.ok(crashed.generationId);
      const interruptedGeneration = await service.readIndexGenerationStatus();
      assert.equal(interruptedGeneration.staging?.generationId, crashed.generationId);
      assert.equal(interruptedGeneration.staging?.status, 'resumable');

      await rm(triggerPath);
      const resumed = await service.resumeBackgroundIndexGeneration();
      assert.equal(resumed.generationId, crashed.generationId);
      const completed = await waitForBackgroundProcessState(
        service,
        RecallBackgroundIndexProcessState.SUCCEEDED,
      );
      assert.equal(completed.generationId, crashed.generationId);
      const generationStatus = await service.readIndexGenerationStatus();
      assert.equal(generationStatus.active?.generationId, crashed.generationId);
      assert.equal(generationStatus.staging, null);

      if (phase === 'store-write') {
        const embeddingRequests = (
          await readFile(join(directory, 'fixture-embeddings.jsonl'), 'utf8')
        )
          .trim()
          .split('\n');
        assert.equal(
          embeddingRequests.length,
          1,
          'resume must reuse vectors cached before the interrupted idempotent upsert',
        );
      }
    });
  }
});

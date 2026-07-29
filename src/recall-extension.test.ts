import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

import { RecallInferenceBackend, RecallInferenceCapability, RecallSearchScope } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import recallExtension, { searchPiRecall } from './recall-extension.js';
import {
  createRecallConversationService,
  type RecallConversationSearchOptions,
  type RecallConversationService,
} from './recall-conversation-service.js';
import { resolveRecallInferenceConfigurationPath } from './configured-recall-inference-runtime.js';
import { resolveRecallFirstIndexSetupStatePath } from './recall-first-index-setup-command.js';
import {
  writeRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
} from './recall-inference-configuration.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

async function createExtensionTestConfig(t: test.TestContext) {
  const homeDirectory = await mkdtemp(join(tmpdir(), 'recall-extension-'));
  t.after(() => rm(homeDirectory, { recursive: true, force: true }));
  return loadRecallConversationConfig({
    homeDirectory,
    configPath: join(homeDirectory, 'missing-recall.json'),
    environment: {},
  });
}

function createConfiguredInferenceSelection(
  capability: RecallInferenceCapability,
): NonNullable<RecallInferenceConfiguration['embedding']> {
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const embedding = capability === RecallInferenceCapability.EMBEDDING;
  const reranking = capability === RecallInferenceCapability.RERANKING;
  const profileId = embedding
    ? embeddingProfile.profileId
    : reranking
      ? rerankingProfile.profileId
      : queryPlanningProfile.profileId;
  return {
    capability,
    candidateId: embedding
      ? 'recommended-embeddinggemma-http'
      : reranking
        ? 'recommended-qwen-reranker-http'
        : 'recommended-qmd-query-planner-http',
    profileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: embedding
      ? 'llama-cpp-http-embedding-v1'
      : reranking
        ? 'llama-cpp-http-reranking-v1'
        : 'llama-cpp-http-query-planning-v1',
    endpoint: embedding
      ? 'http://127.0.0.1:8080'
      : reranking
        ? 'http://127.0.0.1:8081'
        : 'http://127.0.0.1:8082',
    device: null,
    artifact: null,
    conformance: {
      verifiedAt: '2026-01-01T00:00:00.000Z',
      cacheIdentity: embedding
        ? 'embedding-cache-v1'
        : reranking
          ? 'reranking-cache-v1'
          : 'query-planning-cache-v1',
      embeddingProfileId: embedding ? embeddingProfile.profileId : null,
      measurement: { verificationOperations: 1 },
    },
  };
}

function createDeferred(): { promise: Promise<void>; resolve(): void } {
  let resolvePromise = (): void => {};
  const promise = new Promise<void>((resolve) => {
    resolvePromise = resolve;
  });
  return { promise, resolve: resolvePromise };
}

async function writeRecommendedFirstIndexSetup(
  config: Awaited<ReturnType<typeof createExtensionTestConfig>>,
): Promise<void> {
  const statePath = resolveRecallFirstIndexSetupStatePath(config);
  await mkdir(dirname(statePath), { recursive: true });
  await writeFile(
    statePath,
    `${JSON.stringify({
      version: 1,
      embedding: {
        profileId: createRecommendedEmbeddingGemmaModelProfile().profileId,
        backend: 'embedded',
        adapterId: 'node-llama-cpp-embedded-v2',
        devicePolicy: 'auto',
        verifiedAt: '2026-01-01T00:00:00.000Z',
      },
      lastEstimate: null,
    })}\n`,
    'utf8',
  );
}

function captureIndexCommand() {
  let command: Parameters<ExtensionAPI['registerCommand']>[1] | undefined;
  const registrar: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'> = {
    on() {},
    registerTool() {},
    registerCommand(_name, definition) {
      command = definition;
    },
  };
  async function invokeIndexCommand(): Promise<void> {
    assert.ok(command);
    // oxlint-disable-next-line typescript/consistent-type-assertions -- invalid arguments fail before the command reads any other context fields
    const context = {
      ui: { setStatus() {}, notify() {} },
    } as unknown as Parameters<typeof command.handler>[1];
    await command.handler('invalid', context);
  }
  return {
    registrar,
    invokeIndexCommand,
    async resolveService() {
      await assert.rejects(invokeIndexCommand(), /Recall index command arguments invalid/);
    },
  };
}

void test('Pi recall invalidates its runtime cache for every effective inference setting', async (t) => {
  const config = await createExtensionTestConfig(t);
  const inferenceConfigurationPath = resolveRecallInferenceConfigurationPath(config);
  const embedding = createConfiguredInferenceSelection(RecallInferenceCapability.EMBEDDING);
  let inferenceConfiguration: RecallInferenceConfiguration = {
    version: 2,
    embedding,
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  const createdConfigurations: RecallInferenceConfiguration[] = [];
  const command = captureIndexCommand();

  await recallExtension(command.registrar, {
    config,
    createServiceRuntime(configuration) {
      createdConfigurations.push(configuration);
      return {
        service: createRecallConversationService(config),
        async dispose() {},
      };
    },
  });

  await command.resolveService();
  await command.resolveService();
  assert.equal(createdConfigurations.length, 1);

  inferenceConfiguration = {
    ...inferenceConfiguration,
    reranking: createConfiguredInferenceSelection(RecallInferenceCapability.RERANKING),
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  inferenceConfiguration = {
    ...inferenceConfiguration,
    queryPlanning: createConfiguredInferenceSelection(RecallInferenceCapability.QUERY_PLANNING),
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  const queryPlanning = inferenceConfiguration.queryPlanning;
  assert.ok(queryPlanning);
  inferenceConfiguration = {
    ...inferenceConfiguration,
    queryPlanning: { ...queryPlanning, endpoint: 'http://127.0.0.1:9082' },
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  inferenceConfiguration = {
    ...inferenceConfiguration,
    queryPlanning: { ...queryPlanning, adapterId: 'custom-query-planning-v2' },
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  inferenceConfiguration = {
    ...inferenceConfiguration,
    queryPlanning: {
      ...queryPlanning,
      device: { policy: 'gpu', computeBackend: 'cuda', names: ['test-gpu'] },
    },
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  inferenceConfiguration = {
    ...inferenceConfiguration,
    embedding: { ...embedding, endpoint: 'http://127.0.0.1:9080' },
  };
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, inferenceConfiguration);
  await command.resolveService();

  assert.equal(createdConfigurations.length, 7);
});

void test('Pi recall disposes the recommended runtime when configured inference replaces it', async (t) => {
  const config = await createExtensionTestConfig(t);
  await writeRecommendedFirstIndexSetup(config);
  const disposedRuntimeKinds: string[] = [];
  const createdRuntimeKinds: string[] = [];
  const command = captureIndexCommand();

  await recallExtension(command.registrar, {
    config,
    createServiceRuntime(inferenceConfiguration: RecallInferenceConfiguration) {
      const runtimeKind = inferenceConfiguration.embedding ? 'configured' : 'recommended';
      createdRuntimeKinds.push(runtimeKind);
      return {
        service: createRecallConversationService(config),
        async dispose() {
          disposedRuntimeKinds.push(runtimeKind);
        },
      };
    },
  });

  await command.resolveService();
  await writeRecallInferenceConfiguration(resolveRecallInferenceConfigurationPath(config), {
    version: 2,
    embedding: createConfiguredInferenceSelection(RecallInferenceCapability.EMBEDDING),
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  await command.resolveService();

  assert.deepEqual(createdRuntimeKinds, ['recommended', 'configured']);
  assert.deepEqual(disposedRuntimeKinds, ['recommended']);
});

void test('Pi recall creates one runtime for concurrent cache misses', async (t) => {
  const config = await createExtensionTestConfig(t);
  const inferenceConfigurationPath = resolveRecallInferenceConfigurationPath(config);
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, {
    version: 2,
    embedding: createConfiguredInferenceSelection(RecallInferenceCapability.EMBEDDING),
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  const creationStarted = createDeferred();
  const releaseCreation = createDeferred();
  let creationCount = 0;
  const command = captureIndexCommand();
  await recallExtension(command.registrar, {
    config,
    async createServiceRuntime() {
      creationCount += 1;
      creationStarted.resolve();
      await releaseCreation.promise;
      return {
        service: createRecallConversationService(config),
        async dispose() {},
      };
    },
  });

  const firstResolution = command.resolveService();
  const secondResolution = command.resolveService();
  await creationStarted.promise;
  await sleep(20);
  assert.equal(creationCount, 1);
  releaseCreation.resolve();
  await Promise.all([firstResolution, secondResolution]);
  assert.equal(creationCount, 1);
});

void test('Pi recall retains runtime ownership when replacement disposal fails', async (t) => {
  const config = await createExtensionTestConfig(t);
  const inferenceConfigurationPath = resolveRecallInferenceConfigurationPath(config);
  const embedding = createConfiguredInferenceSelection(RecallInferenceCapability.EMBEDDING);
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, {
    version: 2,
    embedding,
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  let creationCount = 0;
  let firstRuntimeDisposalAttemptCount = 0;
  const command = captureIndexCommand();
  await recallExtension(command.registrar, {
    config,
    createServiceRuntime() {
      creationCount += 1;
      const runtimeNumber = creationCount;
      return {
        service: createRecallConversationService(config),
        async dispose() {
          if (runtimeNumber === 1) {
            firstRuntimeDisposalAttemptCount += 1;
            if (firstRuntimeDisposalAttemptCount === 1) {
              throw new Error('intentional runtime disposal failure');
            }
          }
        },
      };
    },
  });

  await command.resolveService();
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, {
    version: 2,
    embedding,
    reranking: createConfiguredInferenceSelection(RecallInferenceCapability.RERANKING),
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  await assert.rejects(command.invokeIndexCommand(), /intentional runtime disposal failure/u);
  await command.resolveService();

  assert.equal(firstRuntimeDisposalAttemptCount, 2);
  assert.equal(creationCount, 2);
});

void test('Pi recall retries cleanly after replacement runtime creation fails', async (t) => {
  const config = await createExtensionTestConfig(t);
  const inferenceConfigurationPath = resolveRecallInferenceConfigurationPath(config);
  const embedding = createConfiguredInferenceSelection(RecallInferenceCapability.EMBEDDING);
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, {
    version: 2,
    embedding,
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  let creationAttemptCount = 0;
  let firstRuntimeDisposalCount = 0;
  const command = captureIndexCommand();
  await recallExtension(command.registrar, {
    config,
    createServiceRuntime() {
      creationAttemptCount += 1;
      if (creationAttemptCount === 2) {
        throw new Error('intentional runtime creation failure');
      }
      const runtimeNumber = creationAttemptCount;
      return {
        service: createRecallConversationService(config),
        async dispose() {
          if (runtimeNumber === 1) {
            firstRuntimeDisposalCount += 1;
          }
        },
      };
    },
  });

  await command.resolveService();
  await writeRecallInferenceConfiguration(inferenceConfigurationPath, {
    version: 2,
    embedding,
    reranking: createConfiguredInferenceSelection(RecallInferenceCapability.RERANKING),
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  await assert.rejects(command.invokeIndexCommand(), /intentional runtime creation failure/u);
  await command.resolveService();

  assert.equal(creationAttemptCount, 3);
  assert.equal(firstRuntimeDisposalCount, 1);
});

void test('Pi recall disposes its current runtime exactly once on shutdown', async (t) => {
  const config = await createExtensionTestConfig(t);
  await writeRecommendedFirstIndexSetup(config);
  let shutdownRuntime: (() => Promise<void>) | undefined;
  let disposalCount = 0;
  const command = captureIndexCommand();
  await recallExtension(command.registrar, {
    config,
    registerServiceRuntimeShutdown(disposeRuntime) {
      shutdownRuntime = disposeRuntime;
    },
    createServiceRuntime() {
      return {
        service: createRecallConversationService(config),
        async dispose() {
          disposalCount += 1;
        },
      };
    },
  });

  await command.resolveService();
  assert.ok(shutdownRuntime);
  await shutdownRuntime();
  await shutdownRuntime();

  assert.equal(disposalCount, 1);
});

void test('Pi session recall registers collision-free tool guidance and index command', async (t) => {
  const toolNames: string[] = [];
  const toolDescriptions: string[] = [];
  const toolGuidelines: string[] = [];
  const commandNames: string[] = [];
  const commandDescriptions: string[] = [];
  const toolParameterSchemas: string[] = [];
  const registrar: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'> = {
    on() {},
    registerTool(definition) {
      toolNames.push(definition.name);
      toolDescriptions.push(definition.description);
      toolGuidelines.push(...(definition.promptGuidelines ?? []));
      toolParameterSchemas.push(JSON.stringify(definition.parameters));
    },
    registerCommand(name, definition) {
      commandNames.push(name);
      commandDescriptions.push(definition.description ?? '');
    },
  };

  await recallExtension(registrar, { config: await createExtensionTestConfig(t) });

  assert.deepEqual(toolNames, ['pi-session-recall']);
  assert.deepEqual(commandNames, ['pi-session-recall-index']);
  assert.ok(!toolNames.includes('recall'));
  assert.ok(!commandNames.includes('recall-index'));
  assert.match(commandDescriptions[0] ?? '', /quality gate/);
  assert.match(commandDescriptions[0] ?? '', /--rebuild/);
  assert.match(
    toolDescriptions[0] ?? '',
    /dense, lexical, and case-preserving identifier retrieval/,
  );
  assert.match(toolDescriptions[0] ?? '', /defaults to project scope/);
  assert.match(toolDescriptions[0] ?? '', /defaults to deterministic hybrid ranking/);
  assert.match(toolDescriptions[0] ?? '', /deep-rerank.*Qwen/);
  assert.match(toolDescriptions[0] ?? '', /labels active and abandoned branches/);
  assert.match(toolDescriptions[0] ?? '', /valid same-run atomic neighbors/);
  assert.match(toolParameterSchemas[0] ?? '', /project/);
  assert.match(toolParameterSchemas[0] ?? '', /global/);
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('projectPath'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('invocationDirectory'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('activeSessionPath'));
  assert.ok(!(toolParameterSchemas[0] ?? '').includes('lifecycleTrigger'));
  assert.ok(
    toolGuidelines.some(
      (guideline) =>
        guideline.includes('Use pi-session-recall') &&
        guideline.includes('conversation or detail from a past session'),
    ),
  );
});

void test('Pi recall registers five marker events plus runtime shutdown cleanup', async (t) => {
  const lifecycleEvents: string[] = [];
  const registrar: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'> = {
    on(event) {
      lifecycleEvents.push(event);
    },
    registerTool() {},
    registerCommand() {},
  };

  await recallExtension(registrar, { config: await createExtensionTestConfig(t) });

  assert.deepEqual(lifecycleEvents, [
    'agent_settled',
    'session_compact',
    'session_tree',
    'session_shutdown',
    'session_start',
    'session_shutdown',
  ]);
});

void test('Pi recall tool adapter propagates trusted cwd with project default and explicit global scope', async () => {
  const calls: Array<{
    query: string;
    limit: number;
    options: RecallConversationSearchOptions;
  }> = [];
  const service: RecallConversationService = {
    async verifyEmbeddingCapability() {
      throw new Error('Pi recall adapter test does not verify embeddings');
    },
    async inspectConversationCorpus() {
      throw new Error('Pi recall adapter test does not inspect the corpus');
    },
    async measureFirstIndexSample() {
      throw new Error('Pi recall adapter test does not measure indexing');
    },
    async verifyRerankingCapability() {
      throw new Error('Pi recall adapter test does not configure reranking');
    },
    async verifyQueryPlanningCapability() {
      throw new Error('Pi recall adapter test does not configure query planning');
    },
    async search(query, limit, options) {
      if (!options) {
        throw new Error('Pi recall adapter test expected search options');
      }
      calls.push({ query, limit, options });
      return {
        totalChunks: 0,
        results: [],
        searchPolicy: {
          scope: options?.scope ?? RecallSearchScope.PROJECT,
          invocationProjectIdentity: null,
          rankingMode: options?.mode ?? 'hybrid',
          rankFusionVersion: 1,
          reciprocalRankConstant: 60,
          rerankPolicyVersion: null,
          rerankerModel: null,
          rerankerIdentity: null,
          activeBranchPrior: 0.01,
          candidateLimits: { dense: 8, lexical: 8, identifier: 8 },
        },
      };
    },
    async index() {
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
    async startBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not start background indexing');
    },
    async resumeBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not resume background indexing');
    },
    async readBackgroundIndexGenerationStatus() {
      return null;
    },
    async stopBackgroundIndexGeneration() {
      throw new Error('Pi recall adapter test does not stop background indexing');
    },
    async readIndexGenerationStatus() {
      return { active: null, staging: null };
    },
    async discardStagingIndexGeneration() {
      return false;
    },
    async rollback() {},
    async adoptLegacy() {},
    async collectRetired() {},
  };
  const context = {
    cwd: '/trusted/invocation',
    sessionManager: {
      getSessionFile() {
        return '/sessions/active.jsonl';
      },
    },
  };

  await searchPiRecall(service, { query: 'project query', mode: 'hybrid' }, context, 5);
  await searchPiRecall(
    service,
    { query: 'global query', mode: 'deep-rerank', scope: 'global', limit: 2 },
    context,
    5,
  );

  assert.deepEqual(calls, [
    {
      query: 'project query',
      limit: 5,
      options: {
        mode: 'hybrid',
        scope: RecallSearchScope.PROJECT,
        invocationDirectory: '/trusted/invocation',
      },
    },
    {
      query: 'global query',
      limit: 2,
      options: {
        mode: 'deep-rerank',
        scope: RecallSearchScope.GLOBAL,
        invocationDirectory: '/trusted/invocation',
      },
    },
  ]);
});

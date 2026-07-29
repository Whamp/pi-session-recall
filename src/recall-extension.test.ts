import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';

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
  const embedding = capability === RecallInferenceCapability.EMBEDDING;
  return {
    capability,
    candidateId: embedding ? 'recommended-embeddinggemma-http' : 'recommended-qwen-reranker-http',
    profileId: embedding ? embeddingProfile.profileId : rerankingProfile.profileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: embedding ? 'llama-cpp-http-embedding-v1' : 'llama-cpp-http-reranking-v1',
    endpoint: embedding ? 'http://127.0.0.1:8080' : 'http://127.0.0.1:8081',
    device: null,
    artifact: null,
    conformance: {
      verifiedAt: '2026-01-01T00:00:00.000Z',
      cacheIdentity: embedding ? 'embedding-cache-v1' : 'reranking-cache-v1',
      embeddingProfileId: embedding ? embeddingProfile.profileId : null,
      measurement: { verificationOperations: 1 },
    },
  };
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
  return {
    registrar,
    async resolveService() {
      assert.ok(command);
      // oxlint-disable-next-line typescript/consistent-type-assertions -- invalid arguments fail before the command reads any other context fields
      const context = {
        ui: { setStatus() {}, notify() {} },
      } as unknown as Parameters<typeof command.handler>[1];
      await assert.rejects(
        command.handler('invalid', context),
        /Recall index command arguments invalid/,
      );
    },
  };
}

void test('Pi recall replaces its cached runtime when reranking configuration changes', async (t) => {
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
  const createdConfigurations: RecallInferenceConfiguration[] = [];
  const command = captureIndexCommand();

  await recallExtension(command.registrar, {
    config,
    createServiceRuntime(inferenceConfiguration: RecallInferenceConfiguration) {
      createdConfigurations.push(inferenceConfiguration);
      return {
        service: createRecallConversationService(config),
        async dispose() {},
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
  await command.resolveService();

  assert.equal(createdConfigurations.length, 2);
  assert.equal(createdConfigurations[0]?.reranking, null);
  assert.equal(createdConfigurations[1]?.reranking?.candidateId, 'recommended-qwen-reranker-http');
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

void test('Pi recall runtime registers exactly the five marker-only lifecycle events', async (t) => {
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

import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createConfiguredRecallInferenceRuntime,
  resolveRecallInferenceConfigurationPath,
  type RecallInferenceAdapterRegistration,
} from './configured-recall-inference-runtime.js';
import { createRecommendedOptionalInferenceCandidates } from './createRecommendedOptionalInferenceCandidates.js';
import {
  createRecommendedEmbeddingGemmaHttpInferenceCandidate,
  createRecommendedEmbeddingGemmaInferenceCandidate,
} from './recommended-embeddinggemma-inference-candidate.js';
import { createEmbeddingGemmaTokenizerManifestIdentity } from './embedded-embeddinggemma-provider.js';
import {
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallRerankingExecutionIdentity } from './recall-inference-capabilities.js';
import {
  writeRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const FIXED_CONVERSATION_TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createConfiguredSelection(
  candidate: RecallInferenceConfigurationCandidate,
): NonNullable<RecallInferenceConfiguration['embedding']> {
  return {
    capability: candidate.capability,
    candidateId: candidate.candidateId,
    profileId: candidate.profileId,
    backend: candidate.backend,
    adapterId: candidate.adapterId,
    endpoint: candidate.endpoint,
    device: candidate.device
      ? {
          policy: candidate.device.policy,
          computeBackend: candidate.device.computeBackend,
          names: [...candidate.device.names],
        }
      : null,
    artifact: candidate.artifact
      ? { ...candidate.artifact, state: RecallInferenceArtifactState.VALID }
      : null,
    conformance: {
      verifiedAt: '2026-07-29T00:00:00.000Z',
      cacheIdentity: `${candidate.profileId}:${candidate.adapterId}`,
      embeddingProfileId:
        candidate.capability === RecallInferenceCapability.EMBEDDING ? candidate.profileId : null,
      measurement: { fixtureOperations: 1 },
    },
  };
}

function createCustomEmbeddingFixture(expectedManifestPath: string): {
  candidate: RecallInferenceConfigurationCandidate;
  registration: RecallInferenceAdapterRegistration;
} {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const candidate: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.EMBEDDING,
    candidateId: 'project-custom-embedding',
    profileId: profile.profileId,
    backend: RecallInferenceBackend.CUSTOM,
    adapterId: 'project-custom-adapter-v1',
    endpoint: null,
    device: null,
    artifact: null,
    async inspectHealth() {
      return {
        artifactState: RecallInferenceArtifactState.NOT_REQUIRED,
        requiredRepair: null,
      };
    },
    async verifyCapabilityConformance() {
      return {
        profileId: profile.profileId,
        adapterId: 'project-custom-adapter-v1',
        backend: RecallInferenceBackend.CUSTOM,
        cacheIdentity: 'project-custom-embedding-v1',
        embeddingProfileId: profile.profileId,
        measurement: { fixtureOperations: 1 },
      };
    },
  };
  return {
    candidate,
    registration: {
      candidate,
      createConfiguredCapability({ config, selection }) {
        assert.equal(config.manifestPath, expectedManifestPath);
        assert.equal(selection.candidateId, candidate.candidateId);
        const provider = {
          async embedQuery() {
            return Array<number>(profile.identity.dimensions).fill(0.02);
          },
          async embedDocuments(documents: readonly string[]) {
            return documents.map(() => Array<number>(profile.identity.dimensions).fill(0.02));
          },
        };
        return {
          capability: RecallInferenceCapability.EMBEDDING,
          profile,
          provider,
          tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
          async loadTokenizer() {
            return FIXED_CONVERSATION_TOKENIZER;
          },
          embeddingDimensions: profile.identity.dimensions,
          async dispose() {},
        };
      },
    },
  };
}

void test('configured runtime resolves inference state beside the index manifest', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: { PI_RECALL_DATA_DIRECTORY: '/recall/data' },
  });

  assert.equal(
    resolveRecallInferenceConfigurationPath(config),
    '/recall/data/inference-configuration.json',
  );
});

void test('configured runtime composes both built-in embeddings with custom reranking', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-runtime-embedding-registry-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const customReranking: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.RERANKING,
    candidateId: 'project-custom-reranking',
    profileId: rerankingProfile.profileId,
    backend: RecallInferenceBackend.CUSTOM,
    adapterId: 'project-custom-reranking-v1',
    endpoint: null,
    device: null,
    artifact: null,
    async inspectHealth() {
      return {
        artifactState: RecallInferenceArtifactState.NOT_REQUIRED,
        requiredRepair: null,
      };
    },
    async verifyCapabilityConformance() {
      return {
        profileId: rerankingProfile.profileId,
        adapterId: 'project-custom-reranking-v1',
        backend: RecallInferenceBackend.CUSTOM,
        cacheIdentity: 'project-custom-reranking-v1',
        embeddingProfileId: null,
        measurement: { fixtureOperations: 1 },
      };
    },
  };
  const customRerankingRegistration: RecallInferenceAdapterRegistration = {
    candidate: customReranking,
    createConfiguredCapability({ config: runtimeConfig, selection }) {
      assert.equal(runtimeConfig.manifestPath, config.manifestPath);
      assert.equal(selection.candidateId, customReranking.candidateId);
      return {
        capability: RecallInferenceCapability.RERANKING,
        profile: rerankingProfile,
        provider: {
          executionIdentity: createRecallRerankingExecutionIdentity(
            rerankingProfile,
            customReranking.adapterId,
            'project-custom-reranking-config-v1',
            RecallInferenceBackend.CUSTOM,
            60_000,
          ),
          async rerankDocuments(query, documents) {
            assert.ok(query.length > 0);
            return documents.map(() => 0.5);
          },
        },
        async dispose() {},
      };
    },
  };
  const embeddingCandidates = [
    createRecommendedEmbeddingGemmaInferenceCandidate(config),
    createRecommendedEmbeddingGemmaInferenceCandidate(config, 512),
    createRecommendedEmbeddingGemmaHttpInferenceCandidate(config),
    createRecommendedEmbeddingGemmaHttpInferenceCandidate(config, 512),
  ];

  for (const embedding of embeddingCandidates) {
    const statePath = join(root, `${embedding.candidateId}.json`);
    await writeRecallInferenceConfiguration(statePath, {
      version: 2,
      embedding: createConfiguredSelection(embedding),
      reranking: createConfiguredSelection(customReranking),
      queryPlanning: null,
      pendingEmbeddingReplacement: null,
    });
    const runtime = await createConfiguredRecallInferenceRuntime(config, {
      inferenceConfigurationPath: statePath,
      adapterRegistries: [{ registrations: [customRerankingRegistration] }],
    });
    assert.equal(runtime.embeddingDimensions, 768);
    await runtime.dispose();
  }
});

void test('configured runtime rejects mismatched built-in identities without substitution', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-runtime-exact-registry-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const httpCandidate = createRecommendedEmbeddingGemmaHttpInferenceCandidate(config);
  const embeddedCandidate = createRecommendedEmbeddingGemmaInferenceCandidate(config);

  async function assertEmbeddingSelectionRejected(
    name: string,
    selection: NonNullable<RecallInferenceConfiguration['embedding']>,
    expected: RegExp,
  ): Promise<void> {
    const statePath = join(root, `${name}.json`);
    await writeRecallInferenceConfiguration(statePath, {
      version: 2,
      embedding: selection,
      reranking: null,
      queryPlanning: null,
      pendingEmbeddingReplacement: null,
    });
    await assert.rejects(
      () =>
        createConfiguredRecallInferenceRuntime(config, {
          inferenceConfigurationPath: statePath,
        }),
      expected,
    );
  }

  await assertEmbeddingSelectionRejected(
    'profile-mismatch',
    { ...createConfiguredSelection(httpCandidate), profileId: 'unsupported-profile' },
    /configured embedding identity unsupported.*no model or adapter was substituted/u,
  );
  await assertEmbeddingSelectionRejected(
    'adapter-mismatch',
    { ...createConfiguredSelection(httpCandidate), adapterId: 'unsupported-adapter' },
    /configured embedding identity unsupported.*no model or adapter was substituted/u,
  );
  await assertEmbeddingSelectionRejected(
    'backend-mismatch',
    { ...createConfiguredSelection(httpCandidate), backend: RecallInferenceBackend.CUSTOM },
    /configured embedding backend mismatch.*custom/u,
  );
  await assertEmbeddingSelectionRejected(
    'endpoint-missing',
    { ...createConfiguredSelection(httpCandidate), endpoint: null },
    /configured embedding HTTP endpoint missing.*no backend was substituted/u,
  );
  await assertEmbeddingSelectionRejected(
    'device-policy-unsupported',
    {
      ...createConfiguredSelection(embeddedCandidate),
      device: { policy: 'unsupported-device', computeBackend: 'pending', names: [] },
    },
    /configured embedding device policy unsupported.*no device was substituted/u,
  );

  const slotMismatchPath = join(root, 'capability-slot-mismatch.json');
  const embeddingSelection = createConfiguredSelection(httpCandidate);
  await writeRecallInferenceConfiguration(slotMismatchPath, {
    version: 2,
    embedding: embeddingSelection,
    reranking: embeddingSelection,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  await assert.rejects(
    () =>
      createConfiguredRecallInferenceRuntime(config, {
        inferenceConfigurationPath: slotMismatchPath,
      }),
    /configured reranking capability mismatch: received embedding/u,
  );
});

void test('configured runtime composes custom embedding with built-in embedded optional capabilities', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-runtime-embedded-optional-registry-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const statePath = join(root, 'inference-configuration.json');
  const customEmbedding = createCustomEmbeddingFixture(config.manifestPath);
  const optionalCandidates = createRecommendedOptionalInferenceCandidates(config);
  const reranking = optionalCandidates.find(
    ({ capability, backend }) =>
      capability === RecallInferenceCapability.RERANKING &&
      backend === RecallInferenceBackend.EMBEDDED,
  );
  const queryPlanning = optionalCandidates.find(
    ({ capability, backend }) =>
      capability === RecallInferenceCapability.QUERY_PLANNING &&
      backend === RecallInferenceBackend.EMBEDDED,
  );
  assert.ok(reranking);
  assert.ok(queryPlanning);
  await writeRecallInferenceConfiguration(statePath, {
    version: 2,
    embedding: createConfiguredSelection(customEmbedding.candidate),
    reranking: createConfiguredSelection(reranking),
    queryPlanning: createConfiguredSelection(queryPlanning),
    pendingEmbeddingReplacement: null,
  });

  const runtime = await createConfiguredRecallInferenceRuntime(config, {
    inferenceConfigurationPath: statePath,
    adapterRegistries: [{ registrations: [customEmbedding.registration] }],
  });
  assert.equal(runtime.embeddingDimensions, 768);
  await runtime.dispose();
});

void test('configured runtime composes custom embedding with built-in HTTP optional capabilities', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-runtime-registry-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const requestedPaths: string[] = [];
  const server = createServer((request, response) => {
    requestedPaths.push(request.url ?? 'missing');
    response.setHeader('content-type', 'application/json');
    if (request.url === '/v1/rerank') {
      response.end(
        JSON.stringify({
          model: 'qwen3-rerank',
          object: 'list',
          usage: { prompt_tokens: 4, total_tokens: 4 },
          results: [
            { index: 0, relevance_score: 0.9 },
            { index: 1, relevance_score: 0.1 },
          ],
        }),
      );
      return;
    }
    if (request.url === '/v1/chat/completions') {
      response.end(
        JSON.stringify({
          model: 'qmd-query-expansion-1.7B-q4_k_m',
          choices: [
            {
              message: {
                role: 'assistant',
                content: [
                  'lex: copper finch source provenance evidence',
                  'vec: copper finch source provenance in recalled conversations',
                ].join('\n'),
              },
            },
          ],
        }),
      );
      return;
    }
    response.statusCode = 404;
    response.end('{}');
  });
  await new Promise<void>((resolve) => {
    server.listen(0, '127.0.0.1', resolve);
  });
  t.after(
    () =>
      new Promise<void>((resolve, reject) => {
        server.close((error) => {
          if (error) {
            reject(error);
          } else {
            resolve();
          }
        });
      }),
  );
  const address = server.address();
  assert.ok(address && typeof address === 'object');
  const baseUrl = `http://127.0.0.1:${address.port}/v1`;
  const config = await loadRecallConversationConfig({
    homeDirectory: root,
    environment: {
      PI_RECALL_RERANKER_BASE_URL: baseUrl,
      PI_RECALL_QUERY_PLANNER_BASE_URL: baseUrl,
    },
  });
  const statePath = join(root, 'inference-configuration.json');
  const customEmbedding = createCustomEmbeddingFixture(config.manifestPath);
  const optionalCandidates = createRecommendedOptionalInferenceCandidates(config, {
    queryPlanningBaseUrl: baseUrl,
  });
  const reranking = optionalCandidates.find(
    ({ candidateId }) => candidateId === 'recommended-qwen-reranker-http',
  );
  const queryPlanning = optionalCandidates.find(
    ({ candidateId }) => candidateId === 'recommended-qmd-query-planner-http',
  );
  assert.ok(reranking);
  assert.ok(queryPlanning);
  await writeRecallInferenceConfiguration(statePath, {
    version: 2,
    embedding: createConfiguredSelection(customEmbedding.candidate),
    reranking: createConfiguredSelection(reranking),
    queryPlanning: createConfiguredSelection(queryPlanning),
    pendingEmbeddingReplacement: null,
  });

  const runtime = await createConfiguredRecallInferenceRuntime(config, {
    inferenceConfigurationPath: statePath,
    adapterRegistries: [{ registrations: [customEmbedding.registration] }],
  });
  const rerankingVerification = await runtime.service.verifyRerankingCapability({
    query: 'source provenance',
    documents: ['Preserve source provenance.', 'Change the navigation color.'],
    expectedScores: [0.9, 0.1],
  });
  const queryPlanningVerification = await runtime.service.verifyQueryPlanningCapability({
    expectedPlan: [
      { type: 'lex', query: 'copper finch source provenance evidence' },
      { type: 'vec', query: 'copper finch source provenance in recalled conversations' },
    ],
  });
  await runtime.dispose();

  assert.equal(rerankingVerification.executionIdentity.adapterId, 'llama-cpp-http-reranking-v1');
  assert.equal(
    queryPlanningVerification.executionIdentity.adapterId,
    'llama-cpp-http-query-planning-v1',
  );
  assert.deepEqual(requestedPaths, ['/v1/rerank', '/v1/chat/completions']);
});

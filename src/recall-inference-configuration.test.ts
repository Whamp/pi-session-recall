import assert from 'node:assert/strict';
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';

import {
  createConfiguredRecallInferenceRuntime,
  type RecallInferenceAdapterRegistration,
} from './configured-recall-inference-runtime.js';
import { createEmbeddingGemmaTokenizerManifestIdentity } from './embedded-embeddinggemma-provider.js';
import {
  RecallGenerationCutoverState,
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import { tryAcquireRecallRebuildOwnershipLock } from './recall-rebuild-ownership-lock.js';
import { runRecallInferenceSetupCommand } from './runRecallInferenceSetupCommand.js';
import {
  clearPendingRecallEmbeddingReplacement,
  configureRecallInferenceCapability,
  inspectRecallInferenceConfiguration,
  readRecallInferenceConfiguration,
  removeRecallInferenceCapability,
  repairRecallInferenceCapability,
  writeRecallInferenceConfiguration,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const FIXED_CONVERSATION_TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createConformingCandidate(
  capability: RecallInferenceCapability,
  candidateId: string,
  profileId: string,
  backend: RecallInferenceBackend,
  adapterId: string,
): RecallInferenceConfigurationCandidate {
  return {
    capability,
    candidateId,
    profileId,
    backend,
    adapterId,
    endpoint:
      backend === RecallInferenceBackend.LLAMA_CPP_HTTP ? `http://${capability}.test/v1` : null,
    device:
      backend === RecallInferenceBackend.EMBEDDED
        ? {
            policy: 'auto',
            computeBackend: 'cpu',
            names: ['Fixture CPU'],
          }
        : null,
    artifact:
      backend === RecallInferenceBackend.EMBEDDED
        ? {
            path: `/models/${candidateId}.gguf`,
            repository: 'fixtures/models',
            revision: 'a'.repeat(40),
            sha256: 'b'.repeat(64),
            byteSize: 1234,
          }
        : null,
    async inspectHealth() {
      return {
        artifactState:
          backend === RecallInferenceBackend.EMBEDDED
            ? RecallInferenceArtifactState.VALID
            : RecallInferenceArtifactState.NOT_REQUIRED,
        requiredRepair: null,
      };
    },
    async verifyCapabilityConformance() {
      return {
        profileId,
        adapterId,
        backend,
        cacheIdentity: `${profileId}:${adapterId}:fixture-policy-v1`,
        embeddingProfileId: capability === RecallInferenceCapability.EMBEDDING ? profileId : null,
        measurement: { fixtureOperations: 1 },
      };
    },
  };
}

void test('mixed inference configuration requires only embeddings and verifies each selected capability independently', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-configuration-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedding = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-embedded',
    'embedding-profile-v1',
    RecallInferenceBackend.EMBEDDED,
    'embedding-adapter-v1',
  );
  const reranking = createConformingCandidate(
    RecallInferenceCapability.RERANKING,
    'reranking-http',
    'reranking-profile-v1',
    RecallInferenceBackend.LLAMA_CPP_HTTP,
    'reranking-http-v1',
  );
  const queryPlanning = createConformingCandidate(
    RecallInferenceCapability.QUERY_PLANNING,
    'planning-custom',
    'planning-profile-v1',
    RecallInferenceBackend.CUSTOM,
    'planning-custom-v1',
  );

  const initialStatus = await inspectRecallInferenceConfiguration(statePath, [], {
    verifyConformance: false,
  });

  assert.equal(initialStatus.ready, false);
  assert.deepEqual(
    initialStatus.capabilities.map(({ capability, required, configured }) => ({
      capability,
      required,
      configured,
    })),
    [
      { capability: RecallInferenceCapability.EMBEDDING, required: true, configured: false },
      { capability: RecallInferenceCapability.RERANKING, required: false, configured: false },
      { capability: RecallInferenceCapability.QUERY_PLANNING, required: false, configured: false },
    ],
  );

  await configureRecallInferenceCapability(statePath, embedding, {
    nowIsoTimestamp: () => '2026-08-02T10:00:00.000Z',
  });
  await configureRecallInferenceCapability(statePath, reranking, {
    nowIsoTimestamp: () => '2026-08-02T10:01:00.000Z',
  });
  await configureRecallInferenceCapability(statePath, queryPlanning, {
    nowIsoTimestamp: () => '2026-08-02T10:02:00.000Z',
  });

  const configuration = await readRecallInferenceConfiguration(statePath);
  assert.equal(configuration.embedding?.backend, RecallInferenceBackend.EMBEDDED);
  assert.equal(configuration.reranking?.backend, RecallInferenceBackend.LLAMA_CPP_HTTP);
  assert.equal(configuration.queryPlanning?.backend, RecallInferenceBackend.CUSTOM);
  assert.equal(
    configuration.embedding?.conformance.cacheIdentity,
    'embedding-profile-v1:embedding-adapter-v1:fixture-policy-v1',
  );
  assert.equal(configuration.embedding?.conformance.embeddingProfileId, 'embedding-profile-v1');
  assert.equal(
    configuration.reranking?.conformance.cacheIdentity,
    'reranking-profile-v1:reranking-http-v1:fixture-policy-v1',
  );
  assert.equal(
    configuration.queryPlanning?.conformance.cacheIdentity,
    'planning-profile-v1:planning-custom-v1:fixture-policy-v1',
  );
  assert.equal(configuration.embedding?.artifact?.state, RecallInferenceArtifactState.VALID);
  assert.deepEqual(configuration.embedding?.device?.names, ['Fixture CPU']);
});

void test('version 1 inference configuration migrates embedding semantic identity without changing selection', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-v1-migration-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  await writeFile(
    statePath,
    `${JSON.stringify({
      version: 1,
      embedding: {
        capability: RecallInferenceCapability.EMBEDDING,
        candidateId: 'legacy-embedding',
        profileId: 'legacy-model-profile',
        backend: RecallInferenceBackend.CUSTOM,
        adapterId: 'legacy-adapter',
        endpoint: null,
        device: null,
        artifact: null,
        conformance: {
          verifiedAt: '2026-08-02T10:00:00.000Z',
          cacheIdentity: 'legacy-embedding-semantic-identity',
          measurement: { fixtureOperations: 1 },
        },
      },
      reranking: null,
      queryPlanning: null,
    })}\n`,
    'utf8',
  );

  const migrated = await readRecallInferenceConfiguration(statePath);
  assert.equal(migrated.version, 2);
  assert.equal(migrated.embedding?.candidateId, 'legacy-embedding');
  assert.equal(
    migrated.embedding?.conformance.embeddingProfileId,
    'legacy-embedding-semantic-identity',
  );
  assert.equal(migrated.pendingEmbeddingReplacement, null);
});

void test('embedding backend changes reuse vectors while profile changes launch approved staging work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-replacement-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  let backgroundStartCount = 0;
  let backgroundResumeCount = 0;
  const embedded = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-embedded',
    'embedding-profile-v1',
    RecallInferenceBackend.EMBEDDED,
    'embedding-embedded-v1',
  );
  const http = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-http',
    'embedding-profile-v1',
    RecallInferenceBackend.LLAMA_CPP_HTTP,
    'embedding-http-v1',
  );
  const replacement = {
    ...createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-replacement',
      'embedding-profile-v2',
      RecallInferenceBackend.CUSTOM,
      'embedding-custom-v2',
    ),
    generationService: {
      async readIndexGenerationStatus() {
        return { active: null, staging: null };
      },
      async startBackgroundIndexGeneration() {
        backgroundStartCount += 1;
        const persisted: unknown = JSON.parse(await readFile(statePath, 'utf8'));
        assert.ok(persisted && typeof persisted === 'object');
        const persistedEmbedding: unknown = Reflect.get(persisted, 'embedding');
        assert.ok(persistedEmbedding && typeof persistedEmbedding === 'object');
        assert.equal(Reflect.get(persistedEmbedding, 'profileId'), 'embedding-profile-v1');
        const pending: unknown = Reflect.get(persisted, 'pendingEmbeddingReplacement');
        assert.ok(pending && typeof pending === 'object');
        const pendingSelection: unknown = Reflect.get(pending, 'selection');
        assert.ok(pendingSelection && typeof pendingSelection === 'object');
        assert.equal(Reflect.get(pendingSelection, 'profileId'), 'embedding-profile-v2');
        return { generationId: null };
      },
      async resumeBackgroundIndexGeneration() {
        throw new Error('no resumable replacement expected');
      },
    },
  } satisfies RecallInferenceConfigurationCandidate;

  await configureRecallInferenceCapability(statePath, embedded);
  await configureRecallInferenceCapability(statePath, http);
  assert.equal(backgroundStartCount, 0);

  await assert.rejects(
    () => configureRecallInferenceCapability(statePath, replacement),
    /explicit embedding replacement approval/u,
  );
  assert.equal(backgroundStartCount, 0);
  assert.equal(
    (await readRecallInferenceConfiguration(statePath)).embedding?.profileId,
    'embedding-profile-v1',
  );

  await configureRecallInferenceCapability(statePath, replacement, {
    approvedEmbeddingReplacement: true,
  });
  assert.equal(backgroundStartCount, 1);
  const pendingConfiguration = await readRecallInferenceConfiguration(statePath);
  assert.equal(pendingConfiguration.embedding?.profileId, 'embedding-profile-v1');
  assert.equal(
    pendingConfiguration.pendingEmbeddingReplacement?.selection.profileId,
    'embedding-profile-v2',
  );

  const generationRegistryPath = join(root, 'generation-registry.json');
  const pointer = createRecallActiveGenerationPointer('generation-fixture');
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: 1,
    activeGenerationId: 'generation-fixture',
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId: 'generation-fixture',
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: 'embedding-profile-v2',
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'a'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        rebuildMarkerWatermark: [],
        validatedAtEpochMilliseconds: 2,
        retireAfterEpochMilliseconds: null,
      },
    ],
  });
  const activatedConfiguration = await readRecallInferenceConfiguration(statePath);
  assert.equal(activatedConfiguration.embedding?.profileId, 'embedding-profile-v2');
  assert.equal(activatedConfiguration.pendingEmbeddingReplacement, null);

  await rm(generationRegistryPath);
  const resumedReplacement = {
    ...replacement,
    generationService: {
      async readIndexGenerationStatus() {
        return {
          active: null,
          staging: { embeddingProfileId: 'embedding-profile-v2' },
        };
      },
      async startBackgroundIndexGeneration() {
        throw new Error('existing semantic-compatible staging should resume');
      },
      async resumeBackgroundIndexGeneration() {
        backgroundResumeCount += 1;
        return { generationId: null };
      },
    },
  } satisfies RecallInferenceConfigurationCandidate;
  await configureRecallInferenceCapability(statePath, resumedReplacement, {
    approvedEmbeddingReplacement: true,
  });
  assert.equal(backgroundResumeCount, 1);

  const beforeFailedLaunch = await readRecallInferenceConfiguration(statePath);
  const failedReplacement = {
    ...createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-failed-replacement',
      'embedding-profile-v3',
      RecallInferenceBackend.CUSTOM,
      'embedding-custom-v3',
    ),
    generationService: {
      async readIndexGenerationStatus() {
        return { active: null, staging: null };
      },
      async startBackgroundIndexGeneration() {
        throw new Error('fixture worker launch failed');
      },
      async resumeBackgroundIndexGeneration() {
        throw new Error('fixture resume not expected');
      },
    },
  } satisfies RecallInferenceConfigurationCandidate;
  await assert.rejects(
    () =>
      configureRecallInferenceCapability(statePath, failedReplacement, {
        approvedEmbeddingReplacement: true,
      }),
    /fixture worker launch failed/u,
  );
  assert.deepEqual(await readRecallInferenceConfiguration(statePath), beforeFailedLaunch);

  await assert.rejects(
    () => removeRecallInferenceCapability(statePath, RecallInferenceCapability.EMBEDDING),
    /required embedding capability/u,
  );
});

void test('embedding replacement validates and persists pending selection before worker launch', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-pending-write-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const current = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-current',
    'embedding-profile-v1',
    RecallInferenceBackend.CUSTOM,
    'embedding-current-v1',
  );
  await configureRecallInferenceCapability(statePath, current);

  let backgroundStartCount = 0;
  const invalidReplacement = {
    ...createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-invalid',
      'embedding-profile-v2',
      RecallInferenceBackend.CUSTOM,
      'embedding-invalid-v2',
    ),
    endpoint: '',
    generationService: {
      async readIndexGenerationStatus() {
        return { active: null, staging: null };
      },
      async startBackgroundIndexGeneration() {
        backgroundStartCount += 1;
      },
      async resumeBackgroundIndexGeneration() {
        throw new Error('invalid replacement should not resume');
      },
    },
  } satisfies RecallInferenceConfigurationCandidate;
  await assert.rejects(
    () =>
      configureRecallInferenceCapability(statePath, invalidReplacement, {
        approvedEmbeddingReplacement: true,
      }),
    /Parse/u,
  );
  assert.equal(backgroundStartCount, 0);

  const validReplacement = {
    ...createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-valid',
      'embedding-profile-v2',
      RecallInferenceBackend.CUSTOM,
      'embedding-valid-v2',
    ),
    generationService: invalidReplacement.generationService,
  } satisfies RecallInferenceConfigurationCandidate;
  await chmod(root, 0o500);
  try {
    await assert.rejects(
      () =>
        configureRecallInferenceCapability(statePath, validReplacement, {
          approvedEmbeddingReplacement: true,
        }),
      /EACCES|permission denied/u,
    );
  } finally {
    await chmod(root, 0o700);
  }
  assert.equal(backgroundStartCount, 0);
});

void test('inference configuration changes wait for flock ownership', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-flock-contention-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  await writeRecallInferenceConfiguration(statePath, {
    version: 2,
    embedding: null,
    reranking: null,
    queryPlanning: null,
    pendingEmbeddingReplacement: null,
  });
  const owner = await tryAcquireRecallRebuildOwnershipLock(`${statePath}.configuration-lock`);
  assert.ok(owner);
  t.after(() => owner.release());

  const change = removeRecallInferenceCapability(statePath, RecallInferenceCapability.RERANKING);
  const completedWhileContended = await Promise.race([
    change.then(() => true),
    sleep(200).then(() => false),
  ]);
  assert.equal(completedWhileContended, false);

  await owner.release();
  await change;
});

void test('concurrent embedding replacements cannot roll back a later successful launch', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-concurrent-replacement-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  await configureRecallInferenceCapability(
    statePath,
    createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-current',
      'embedding-profile-v1',
      RecallInferenceBackend.CUSTOM,
      'embedding-current-v1',
    ),
  );

  const firstLaunchStarted = Promise.withResolvers<void>();
  const releaseFirstLaunch = Promise.withResolvers<void>();
  const secondLaunchStarted = Promise.withResolvers<void>();
  function createReplacementCandidate(
    candidateId: string,
    profileId: string,
    startBackgroundIndexGeneration: () => Promise<void>,
  ): RecallInferenceConfigurationCandidate {
    return {
      ...createConformingCandidate(
        RecallInferenceCapability.EMBEDDING,
        candidateId,
        profileId,
        RecallInferenceBackend.CUSTOM,
        `${candidateId}-adapter`,
      ),
      generationService: {
        async readIndexGenerationStatus() {
          return { active: null, staging: null };
        },
        startBackgroundIndexGeneration,
        async resumeBackgroundIndexGeneration() {
          throw new Error('concurrent fixture has no staging generation');
        },
      },
    };
  }
  const firstReplacement = createReplacementCandidate(
    'embedding-first',
    'embedding-profile-v2',
    async () => {
      firstLaunchStarted.resolve();
      await releaseFirstLaunch.promise;
      throw new Error('first fixture launch failed');
    },
  );
  const secondReplacement = createReplacementCandidate(
    'embedding-second',
    'embedding-profile-v3',
    async () => {
      secondLaunchStarted.resolve();
    },
  );

  const firstConfiguration = configureRecallInferenceCapability(statePath, firstReplacement, {
    approvedEmbeddingReplacement: true,
  });
  await firstLaunchStarted.promise;
  const secondConfiguration = configureRecallInferenceCapability(statePath, secondReplacement, {
    approvedEmbeddingReplacement: true,
  });
  const secondStartedBeforeFirstSettled = await Promise.race([
    secondLaunchStarted.promise.then(() => true),
    sleep(25).then(() => false),
  ]);
  assert.equal(secondStartedBeforeFirstSettled, false);

  releaseFirstLaunch.resolve();
  await assert.rejects(firstConfiguration, /first fixture launch failed/u);
  await secondConfiguration;
  const configured = await readRecallInferenceConfiguration(statePath);
  assert.equal(configured.pendingEmbeddingReplacement?.selection.profileId, 'embedding-profile-v3');
});

void test('repair and doctor preserve valid sibling capabilities and never accept failed custom conformance', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-repair-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedding = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-custom',
    'embedding-profile-v1',
    RecallInferenceBackend.CUSTOM,
    'embedding-custom-v1',
  );
  let rerankingArtifactState: RecallInferenceArtifactState = RecallInferenceArtifactState.VALID;
  let rerankingRepairCount = 0;
  const reranking = {
    ...createConformingCandidate(
      RecallInferenceCapability.RERANKING,
      'reranking-embedded',
      'reranking-profile-v1',
      RecallInferenceBackend.EMBEDDED,
      'reranking-embedded-v1',
    ),
    async inspectHealth() {
      return {
        artifactState: rerankingArtifactState,
        requiredRepair:
          rerankingArtifactState === RecallInferenceArtifactState.VALID
            ? null
            : 'replace the checksum-mismatched artifact',
      } as const;
    },
    async repairArtifact(approved: boolean) {
      assert.equal(approved, true);
      rerankingRepairCount += 1;
      rerankingArtifactState = RecallInferenceArtifactState.VALID;
    },
  } satisfies RecallInferenceConfigurationCandidate;
  let plannerConformanceFails = false;
  const plannerBase = createConformingCandidate(
    RecallInferenceCapability.QUERY_PLANNING,
    'planner-http',
    'planner-profile-v1',
    RecallInferenceBackend.LLAMA_CPP_HTTP,
    'planner-http-v1',
  );
  const planner = {
    ...plannerBase,
    async verifyCapabilityConformance() {
      if (plannerConformanceFails) {
        throw new Error('planner fixture grammar mismatch');
      }
      return plannerBase.verifyCapabilityConformance();
    },
  } satisfies RecallInferenceConfigurationCandidate;

  await configureRecallInferenceCapability(statePath, embedding);
  await configureRecallInferenceCapability(statePath, reranking);
  await configureRecallInferenceCapability(statePath, planner);
  const beforeRepair = await readRecallInferenceConfiguration(statePath);
  rerankingArtifactState = RecallInferenceArtifactState.CORRUPT;

  const status = await inspectRecallInferenceConfiguration(
    statePath,
    [embedding, reranking, planner],
    { verifyConformance: false },
  );
  assert.equal(status.ready, false);
  assert.match(
    status.capabilities.find(({ capability }) => capability === RecallInferenceCapability.RERANKING)
      ?.requiredRepair ?? '',
    /checksum-mismatched/u,
  );

  await assert.rejects(
    () =>
      repairRecallInferenceCapability(statePath, RecallInferenceCapability.RERANKING, reranking),
    /explicit artifact repair approval/u,
  );
  await repairRecallInferenceCapability(statePath, RecallInferenceCapability.RERANKING, reranking, {
    approvedArtifactRepair: true,
    nowIsoTimestamp: () => '2026-08-02T12:00:00.000Z',
  });
  const repaired = await readRecallInferenceConfiguration(statePath);
  assert.equal(rerankingRepairCount, 1);
  assert.deepEqual(repaired.embedding, beforeRepair.embedding);
  assert.deepEqual(repaired.queryPlanning, beforeRepair.queryPlanning);
  assert.equal(repaired.reranking?.artifact?.state, RecallInferenceArtifactState.VALID);
  assert.equal(repaired.reranking?.conformance.verifiedAt, '2026-08-02T12:00:00.000Z');

  plannerConformanceFails = true;
  const doctor = await inspectRecallInferenceConfiguration(
    statePath,
    [embedding, reranking, planner],
    { verifyConformance: true },
  );
  assert.equal(doctor.ready, false);
  assert.match(
    doctor.capabilities.find(
      ({ capability }) => capability === RecallInferenceCapability.QUERY_PLANNING,
    )?.requiredRepair ?? '',
    /planner fixture grammar mismatch/u,
  );

  await assert.rejects(
    () => configureRecallInferenceCapability(statePath, planner),
    /planner fixture grammar mismatch/u,
  );
  assert.deepEqual(await readRecallInferenceConfiguration(statePath), repaired);

  const withoutPlanner = await removeRecallInferenceCapability(
    statePath,
    RecallInferenceCapability.QUERY_PLANNING,
  );
  assert.equal(withoutPlanner.queryPlanning, null);
  assert.deepEqual(withoutPlanner.embedding, repaired.embedding);
  assert.deepEqual(withoutPlanner.reranking, repaired.reranking);
});

void test('inference setup command emits inspectable JSON for later mixed configuration runs', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedding = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-http',
    'embedding-profile-v1',
    RecallInferenceBackend.LLAMA_CPP_HTTP,
    'embedding-http-v1',
  );
  const reranking = createConformingCandidate(
    RecallInferenceCapability.RERANKING,
    'reranking-custom',
    'reranking-profile-v1',
    RecallInferenceBackend.CUSTOM,
    'reranking-custom-v1',
  );
  const outputs: unknown[] = [];
  const options = {
    statePath,
    candidates: [embedding, reranking],
    nowIsoTimestamp: () => '2026-08-02T13:00:00.000Z',
    writeOutput(value: string) {
      outputs.push(JSON.parse(value));
    },
  };

  await runRecallInferenceSetupCommand(['configure', 'embedding', 'embedding-http'], options);
  await runRecallInferenceSetupCommand(['configure', 'reranking', 'reranking-custom'], options);
  await runRecallInferenceSetupCommand(['doctor'], options);

  const doctor = outputs.at(-1);
  assert.ok(doctor && typeof doctor === 'object');
  assert.equal(Reflect.get(doctor, 'ready'), true);
  const capabilities: unknown = Reflect.get(doctor, 'capabilities');
  assert.ok(Array.isArray(capabilities));
  assert.equal(capabilities.length, 3);

  await runRecallInferenceSetupCommand(['remove', 'reranking'], options);
  assert.equal((await readRecallInferenceConfiguration(statePath)).reranking, null);
});

void test('configured runtime reconstructs registered custom adapters and rejects unavailable ones', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-runtime-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const statePath = join(root, 'inference-configuration.json');
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const customEmbedding = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'project-custom-embedding',
    embeddingProfile.profileId,
    RecallInferenceBackend.CUSTOM,
    'project-custom-adapter-v1',
  );
  await configureRecallInferenceCapability(statePath, customEmbedding);

  await assert.rejects(
    () =>
      createConfiguredRecallInferenceRuntime(config, {
        inferenceConfigurationPath: statePath,
      }),
    /configured embedding adapter unavailable.*no adapter was substituted/u,
  );

  let customEmbeddingOperationCount = 0;
  let customTokenizerOperationCount = 0;
  const runtime = await createConfiguredRecallInferenceRuntime(config, {
    inferenceConfigurationPath: statePath,
    adapterRegistries: [
      {
        registrations: [
          {
            candidate: customEmbedding,
            async createConfiguredCapability({ config: runtimeConfig, selection }) {
              assert.equal(runtimeConfig.manifestPath, config.manifestPath);
              assert.equal(selection.candidateId, 'project-custom-embedding');
              const embeddingProvider = {
                async embedQuery() {
                  customEmbeddingOperationCount += 1;
                  return Array<number>(embeddingProfile.identity.dimensions).fill(0.02);
                },
                async embedDocuments(documents: readonly string[]) {
                  customEmbeddingOperationCount += documents.length;
                  return documents.map(() =>
                    Array<number>(embeddingProfile.identity.dimensions).fill(0.02),
                  );
                },
              };
              const loadTokenizer = async () => {
                customTokenizerOperationCount += 1;
                return FIXED_CONVERSATION_TOKENIZER;
              };
              return {
                capability: RecallInferenceCapability.EMBEDDING,
                profile: embeddingProfile,
                provider: embeddingProvider,
                tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(embeddingProfile),
                loadTokenizer,
                embeddingDimensions: embeddingProfile.identity.dimensions,
                async dispose() {},
              };
            },
          },
        ],
      },
    ],
  });
  await runtime.service.verifyEmbeddingCapability();
  await runtime.dispose();

  assert.equal(customEmbeddingOperationCount, 2);
  assert.equal(customTokenizerOperationCount, 1);
});

void test('background runtime prefers pending embedding replacement over active embedding', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-pending-runtime-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const statePath = join(dirname(config.manifestPath), 'inference-configuration.json');
  const active = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'project-active-embedding',
    'project-embedding-profile-v1',
    RecallInferenceBackend.CUSTOM,
    'project-active-adapter-v1',
  );
  const pending = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'project-pending-embedding',
    'project-embedding-profile-v2',
    RecallInferenceBackend.CUSTOM,
    'project-pending-adapter-v1',
  );
  await configureRecallInferenceCapability(statePath, active);
  await writeRecallInferenceConfiguration(statePath, {
    ...(await readRecallInferenceConfiguration(statePath)),
    pendingEmbeddingReplacement: {
      embeddingProfileId: 'embedding-profile-v2',
      selection: {
        capability: RecallInferenceCapability.EMBEDDING,
        candidateId: pending.candidateId,
        profileId: pending.profileId,
        backend: pending.backend,
        adapterId: pending.adapterId,
        endpoint: pending.endpoint,
        device: pending.device
          ? {
              policy: pending.device.policy,
              computeBackend: pending.device.computeBackend,
              names: [...pending.device.names],
            }
          : null,
        artifact: pending.artifact
          ? { ...pending.artifact, state: RecallInferenceArtifactState.NOT_REQUIRED }
          : null,
        conformance: {
          verifiedAt: '2026-01-01T00:00:00.000Z',
          cacheIdentity: 'embedding-profile-v2',
          embeddingProfileId: 'embedding-profile-v2',
          measurement: { verificationOperations: 1 },
        },
      },
    },
  });

  let reconstructedCandidateId: string | null = null;
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const createRegistration = (
    candidate: RecallInferenceConfigurationCandidate,
  ): RecallInferenceAdapterRegistration => ({
    candidate,
    async createConfiguredCapability({ config: runtimeConfig, selection }) {
      assert.equal(runtimeConfig.manifestPath, config.manifestPath);
      reconstructedCandidateId = selection.candidateId;
      const embeddingProvider = {
        async embedQuery() {
          return Array<number>(embeddingProfile.identity.dimensions).fill(0.03);
        },
        async embedDocuments(documents: readonly string[]) {
          return documents.map(() =>
            Array<number>(embeddingProfile.identity.dimensions).fill(0.03),
          );
        },
      };
      const loadTokenizer = async () => FIXED_CONVERSATION_TOKENIZER;
      return {
        capability: RecallInferenceCapability.EMBEDDING,
        profile: embeddingProfile,
        provider: embeddingProvider,
        tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(embeddingProfile),
        loadTokenizer,
        embeddingDimensions: embeddingProfile.identity.dimensions,
        async dispose() {},
      };
    },
  });
  const registry = { registrations: [createRegistration(active), createRegistration(pending)] };

  const activeRuntime = await createConfiguredRecallInferenceRuntime(config, {
    inferenceConfigurationPath: statePath,
    adapterRegistries: [registry],
  });
  assert.equal(reconstructedCandidateId, 'project-active-embedding');
  await activeRuntime.dispose();

  reconstructedCandidateId = null;
  const pendingRuntime = await createConfiguredRecallInferenceRuntime(config, {
    inferenceConfigurationPath: statePath,
    preferPendingEmbeddingReplacement: true,
    adapterRegistries: [registry],
  });
  assert.equal(reconstructedCandidateId, 'project-pending-embedding');
  await pendingRuntime.dispose();
});

void test('clearing pending embedding replacement cancels a discarded profile change', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-clear-pending-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedded = createConformingCandidate(
    RecallInferenceCapability.EMBEDDING,
    'embedding-embedded',
    'embedding-profile-v1',
    RecallInferenceBackend.EMBEDDED,
    'embedding-embedded-v1',
  );
  const replacement = {
    ...createConformingCandidate(
      RecallInferenceCapability.EMBEDDING,
      'embedding-replacement',
      'embedding-profile-v2',
      RecallInferenceBackend.CUSTOM,
      'embedding-custom-v2',
    ),
    generationService: {
      async readIndexGenerationStatus() {
        return { active: null, staging: null };
      },
      async startBackgroundIndexGeneration() {
        return { generationId: 'generation-staging' };
      },
      async resumeBackgroundIndexGeneration() {
        throw new Error('no resumable replacement expected');
      },
    },
  } satisfies RecallInferenceConfigurationCandidate;

  await configureRecallInferenceCapability(statePath, embedded);
  await configureRecallInferenceCapability(statePath, replacement, {
    approvedEmbeddingReplacement: true,
  });
  assert.equal(
    (await readRecallInferenceConfiguration(statePath)).pendingEmbeddingReplacement?.selection
      .profileId,
    'embedding-profile-v2',
  );

  assert.equal(await clearPendingRecallEmbeddingReplacement(statePath), true);
  const cleared = await readRecallInferenceConfiguration(statePath);
  assert.equal(cleared.embedding?.profileId, 'embedding-profile-v1');
  assert.equal(cleared.pendingEmbeddingReplacement, null);
  assert.equal(await clearPendingRecallEmbeddingReplacement(statePath), false);
});

import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createConfiguredRecallInferenceRuntime } from './configured-recall-inference-runtime.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { runRecallInferenceSetupCommand } from './recall-inference-setup-command.js';
import {
  configureRecallInferenceCapability,
  inspectRecallInferenceConfiguration,
  readRecallInferenceConfiguration,
  removeRecallInferenceCapability,
  repairRecallInferenceCapability,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';

function createConformingCandidate(
  capability: 'embedding' | 'reranking' | 'query-planning',
  candidateId: string,
  profileId: string,
  backend: 'embedded' | 'llama-cpp-http' | 'custom',
  adapterId: string,
): RecallInferenceConfigurationCandidate {
  return {
    capability,
    candidateId,
    profileId,
    backend,
    adapterId,
    endpoint: backend === 'llama-cpp-http' ? `http://${capability}.test/v1` : null,
    device:
      backend === 'embedded'
        ? {
            policy: 'auto',
            computeBackend: 'cpu',
            names: ['Fixture CPU'],
          }
        : null,
    artifact:
      backend === 'embedded'
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
        artifactState: backend === 'embedded' ? 'valid' : 'not-required',
        requiredRepair: null,
      };
    },
    async verifyCapabilityConformance() {
      return {
        profileId,
        adapterId,
        backend,
        cacheIdentity:
          capability === 'embedding' ? profileId : `${profileId}:${adapterId}:fixture-policy-v1`,
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
    'embedding',
    'embedding-embedded',
    'embedding-profile-v1',
    'embedded',
    'embedding-adapter-v1',
  );
  const reranking = createConformingCandidate(
    'reranking',
    'reranking-http',
    'reranking-profile-v1',
    'llama-cpp-http',
    'reranking-http-v1',
  );
  const queryPlanning = createConformingCandidate(
    'query-planning',
    'planning-custom',
    'planning-profile-v1',
    'custom',
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
      { capability: 'embedding', required: true, configured: false },
      { capability: 'reranking', required: false, configured: false },
      { capability: 'query-planning', required: false, configured: false },
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
  assert.equal(configuration.embedding?.backend, 'embedded');
  assert.equal(configuration.reranking?.backend, 'llama-cpp-http');
  assert.equal(configuration.queryPlanning?.backend, 'custom');
  assert.equal(configuration.embedding?.conformance.cacheIdentity, 'embedding-profile-v1');
  assert.equal(
    configuration.reranking?.conformance.cacheIdentity,
    'reranking-profile-v1:reranking-http-v1:fixture-policy-v1',
  );
  assert.equal(
    configuration.queryPlanning?.conformance.cacheIdentity,
    'planning-profile-v1:planning-custom-v1:fixture-policy-v1',
  );
  assert.equal(configuration.embedding?.artifact?.state, 'valid');
  assert.deepEqual(configuration.embedding?.device?.names, ['Fixture CPU']);
});

void test('embedding backend changes reuse vectors while profile changes launch approved staging work', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-replacement-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  let backgroundStartCount = 0;
  const embedded = createConformingCandidate(
    'embedding',
    'embedding-embedded',
    'embedding-profile-v1',
    'embedded',
    'embedding-embedded-v1',
  );
  const http = createConformingCandidate(
    'embedding',
    'embedding-http',
    'embedding-profile-v1',
    'llama-cpp-http',
    'embedding-http-v1',
  );
  const replacement = {
    ...createConformingCandidate(
      'embedding',
      'embedding-replacement',
      'embedding-profile-v2',
      'custom',
      'embedding-custom-v2',
    ),
    generationService: {
      async readIndexGenerationStatus() {
        return { active: null, staging: null };
      },
      async startBackgroundIndexGeneration() {
        backgroundStartCount += 1;
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
  assert.equal(
    (await readRecallInferenceConfiguration(statePath)).embedding?.profileId,
    'embedding-profile-v2',
  );

  const failedReplacement = {
    ...createConformingCandidate(
      'embedding',
      'embedding-failed-replacement',
      'embedding-profile-v3',
      'custom',
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
  assert.equal(
    (await readRecallInferenceConfiguration(statePath)).embedding?.profileId,
    'embedding-profile-v2',
  );

  await assert.rejects(
    () => removeRecallInferenceCapability(statePath, 'embedding'),
    /required embedding capability/u,
  );
});

void test('repair and doctor preserve valid sibling capabilities and never accept failed custom conformance', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-repair-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedding = createConformingCandidate(
    'embedding',
    'embedding-custom',
    'embedding-profile-v1',
    'custom',
    'embedding-custom-v1',
  );
  let rerankingArtifactState: 'valid' | 'corrupt' = 'valid';
  let rerankingRepairCount = 0;
  const reranking = {
    ...createConformingCandidate(
      'reranking',
      'reranking-embedded',
      'reranking-profile-v1',
      'embedded',
      'reranking-embedded-v1',
    ),
    async inspectHealth() {
      return {
        artifactState: rerankingArtifactState,
        requiredRepair:
          rerankingArtifactState === 'valid' ? null : 'replace the checksum-mismatched artifact',
      } as const;
    },
    async repairArtifact(approved: boolean) {
      assert.equal(approved, true);
      rerankingRepairCount += 1;
      rerankingArtifactState = 'valid';
    },
  } satisfies RecallInferenceConfigurationCandidate;
  let plannerConformanceFails = false;
  const plannerBase = createConformingCandidate(
    'query-planning',
    'planner-http',
    'planner-profile-v1',
    'llama-cpp-http',
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
  rerankingArtifactState = 'corrupt';

  const status = await inspectRecallInferenceConfiguration(
    statePath,
    [embedding, reranking, planner],
    { verifyConformance: false },
  );
  assert.equal(status.ready, false);
  assert.match(
    status.capabilities.find(({ capability }) => capability === 'reranking')?.requiredRepair ?? '',
    /checksum-mismatched/u,
  );

  await assert.rejects(
    () => repairRecallInferenceCapability(statePath, 'reranking', reranking),
    /explicit artifact repair approval/u,
  );
  await repairRecallInferenceCapability(statePath, 'reranking', reranking, {
    approvedArtifactRepair: true,
    nowIsoTimestamp: () => '2026-08-02T12:00:00.000Z',
  });
  const repaired = await readRecallInferenceConfiguration(statePath);
  assert.equal(rerankingRepairCount, 1);
  assert.deepEqual(repaired.embedding, beforeRepair.embedding);
  assert.deepEqual(repaired.queryPlanning, beforeRepair.queryPlanning);
  assert.equal(repaired.reranking?.artifact?.state, 'valid');
  assert.equal(repaired.reranking?.conformance.verifiedAt, '2026-08-02T12:00:00.000Z');

  plannerConformanceFails = true;
  const doctor = await inspectRecallInferenceConfiguration(
    statePath,
    [embedding, reranking, planner],
    { verifyConformance: true },
  );
  assert.equal(doctor.ready, false);
  assert.match(
    doctor.capabilities.find(({ capability }) => capability === 'query-planning')?.requiredRepair ??
      '',
    /planner fixture grammar mismatch/u,
  );

  await assert.rejects(
    () => configureRecallInferenceCapability(statePath, planner),
    /planner fixture grammar mismatch/u,
  );
  assert.deepEqual(await readRecallInferenceConfiguration(statePath), repaired);

  const withoutPlanner = await removeRecallInferenceCapability(statePath, 'query-planning');
  assert.equal(withoutPlanner.queryPlanning, null);
  assert.deepEqual(withoutPlanner.embedding, repaired.embedding);
  assert.deepEqual(withoutPlanner.reranking, repaired.reranking);
});

void test('inference setup command emits inspectable JSON for later mixed configuration runs', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'inference-configuration.json');
  const embedding = createConformingCandidate(
    'embedding',
    'embedding-http',
    'embedding-profile-v1',
    'llama-cpp-http',
    'embedding-http-v1',
  );
  const reranking = createConformingCandidate(
    'reranking',
    'reranking-custom',
    'reranking-profile-v1',
    'custom',
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

void test('configured runtime fails closed instead of substituting an unavailable custom adapter', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-inference-runtime-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = await loadRecallConversationConfig({ homeDirectory: root, environment: {} });
  const statePath = join(root, 'inference-configuration.json');
  const customEmbedding = createConformingCandidate(
    'embedding',
    'project-custom-embedding',
    'project-embedding-profile-v1',
    'custom',
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
});

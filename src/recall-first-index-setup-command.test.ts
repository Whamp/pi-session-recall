import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  EmbeddedInferenceComputeBackend,
  EmbeddedInferenceDevicePolicy,
  RecallBackgroundIndexProcessState,
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import {
  configureRecallInferenceCapability,
  readRecallInferenceConfiguration,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import {
  runRecallFirstIndexSetupCommand,
  type RecallFirstIndexSetupCommandService,
} from './recall-first-index-setup-command.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import { RecallDiagnosticsMode } from './enums.js';
import type { RecallConversationConfig } from './recall-conversation-service.js';

function readLastSetupOutputProperty(outputs: readonly unknown[], property: string): unknown {
  const output = outputs.at(-1);
  assert.ok(output && typeof output === 'object');
  return Reflect.get(output, property);
}

function createSetupCommandTestConfig(root: string): RecallConversationConfig {
  const dataDirectory = join(root, 'data');
  return {
    sessionsDirectory: join(root, 'sessions'),
    dataDirectory,
    databasePath: join(dataDirectory, 'zvec'),
    projectionDatabasePath: join(dataDirectory, 'session-projections'),
    statePath: join(dataDirectory, 'index-state.json'),
    manifestPath: join(dataDirectory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'embedding-cache'),
    lockPath: join(dataDirectory, 'operation.lock'),
    markerSpoolDirectory: join(dataDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(dataDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(dataDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(dataDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(dataDirectory, 'generations'),
    activeGenerationPointerPath: join(dataDirectory, 'active-generation.json'),
    generationRegistryPath: join(dataDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(dataDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(dataDirectory, 'incremental-diagnostics.jsonl'),
    backgroundIndexStatusPath: join(dataDirectory, 'background-index-status.json'),
    backgroundIndexRequestPath: join(dataDirectory, 'background-index-request.json'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'unused',
    embeddingServedModelId: 'unused',
    embeddingArtifact: 'unused.gguf',
    embeddingQuantization: 'unused',
    embeddingPooling: 'mean',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused.test/v1',
    rerankerModel: 'unused',
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
  };
}

void test('first-index setup requires consent and uses the authoritative configured runtime after selection', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-first-index-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createSetupCommandTestConfig(root);
  const statePath = join(root, 'data', 'first-index-setup.json');
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const artifactPath = join(root, 'models', profile.source.artifact);
  let downloadCount = 0;
  let verificationCount = 0;
  let backgroundStartCount = 0;
  let backgroundResumeCount = 0;
  let measuredSampleBound: number | undefined;
  let stagingAvailable = false;
  let selectedServiceCreationCount = 0;
  let configuredServiceCreationCount = 0;
  let disposeCount = 0;
  const outputs: unknown[] = [];

  const backgroundStatus = {
    version: 1 as const,
    buildId: 'build-setup',
    generationId: null,
    embeddingProfileId: 'embedding-profile-fixture',
    processId: 4321,
    processState: RecallBackgroundIndexProcessState.STARTING,
    startedAt: '2026-08-01T10:00:00.000Z',
    updatedAt: '2026-08-01T10:00:00.000Z',
    completedAt: null,
    progress: null,
    latestCheckpoint: null,
    latestActionableError: null,
  };
  const service: RecallFirstIndexSetupCommandService = {
    async verifyEmbeddingCapability() {
      verificationCount += 1;
      return {
        embeddingProfileId: 'embedding-profile-fixture',
        model: profile.identity.requestModel,
        dimensions: 768,
        normalization: 'l2',
        tokenizerModel: profile.tokenizer.model,
      };
    },
    async inspectConversationCorpus() {
      return { sessionCount: 4, sourceByteSize: 1_000 };
    },
    async measureFirstIndexSample(measurementOptions) {
      measuredSampleBound = measurementOptions?.maximumSessionCount;
      stagingAvailable = true;
      return {
        corpus: { sessionCount: 4, sourceByteSize: 1_000 },
        sampledSessionCount: 2,
        sampledSourceByteSize: 400,
        sampledDenseDocumentCount: 20,
        coldStartMilliseconds: 100,
        measuredSampleMilliseconds: 200,
        sourceBytesPerSecond: 2_000,
        denseDocumentsPerSecond: 100,
        cacheHitCount: 0,
        newlyEmbeddedDocumentCount: 20,
        embeddingRequestCount: 3,
        estimatedDurationMilliseconds: { minimum: 480, maximum: 750 },
      };
    },
    async readIndexGenerationStatus() {
      return {
        active: null,
        staging: stagingAvailable
          ? {
              generationId: 'sample-generation',
              embeddingProfileId: 'embedding-profile-fixture',
              manifestPath: join(root, 'sample-manifest.json'),
              status: 'resumable' as const,
            }
          : null,
      };
    },
    async startBackgroundIndexGeneration() {
      backgroundStartCount += 1;
      return backgroundStatus;
    },
    async resumeBackgroundIndexGeneration() {
      backgroundResumeCount += 1;
      return { ...backgroundStatus, generationId: 'sample-generation' };
    },
  };
  const artifactCache = {
    async inspectArtifact() {
      return {
        profile,
        status: {
          state: 'missing' as const,
          artifactPath,
          partialPaths: [],
          repair: 'download with explicit approval',
        },
      };
    },
    async verifyArtifact() {
      return {
        state: 'valid' as const,
        artifactPath,
        partialPaths: [],
        repair: 'none',
      };
    },
    async diagnoseArtifact() {
      throw new Error('doctor not expected');
    },
    async downloadArtifact() {
      downloadCount += 1;
      return {
        state: 'valid' as const,
        artifactPath,
        partialPaths: [],
        repair: 'none',
      };
    },
    async repairArtifact() {
      throw new Error('repair not expected');
    },
    async removeArtifact() {
      throw new Error('remove not expected');
    },
  };
  const options = {
    config,
    statePath,
    profile,
    artifactCache,
    qualityGateDecision: {
      automatedGatePassed: true,
      selectedPolicy: {
        chunkPolicy: { id: '512-64', maxTokens: 512, overlapTokens: 64 },
        candidateCount: 8,
        finalCount: 5,
      },
      blockers: [],
    },
    metadataService: service,
    async createSelectedServiceRuntime() {
      selectedServiceCreationCount += 1;
      return {
        service,
        executionIdentity: {
          adapter: 'node-llama-cpp-embedded-v2' as const,
          computeBackend: EmbeddedInferenceComputeBackend.CPU,
          deviceNames: ['CPU'],
          devicePolicy: EmbeddedInferenceDevicePolicy.AUTO,
        },
        async dispose() {
          disposeCount += 1;
        },
      };
    },
    async createConfiguredServiceRuntime() {
      configuredServiceCreationCount += 1;
      return {
        service,
        async dispose() {
          disposeCount += 1;
        },
      };
    },
    nowIsoTimestamp: () => '2026-08-01T10:00:00.000Z',
    writeOutput(value: string) {
      outputs.push(JSON.parse(value));
    },
  };

  await runRecallFirstIndexSetupCommand([], options);

  assert.equal(selectedServiceCreationCount, 0);
  assert.equal(downloadCount, 0);
  assert.deepEqual(outputs.pop(), {
    action: 'status',
    configuration: { state: 'unconfigured', embedding: null },
    recommendation: {
      profileId: profile.profileId,
      purpose: profile.purpose,
      source: profile.source,
      license: profile.license,
      exactSizeBytes: profile.source.byteSize,
      cachePath: artifactPath,
      devicePolicy: 'auto',
      selected: false,
    },
    artifactStatus: 'missing',
    corpusEstimate: null,
    recallReady: false,
  });

  await assert.rejects(
    () => runRecallFirstIndexSetupCommand(['select-embeddinggemma'], options),
    /explicit --approve-download/u,
  );
  assert.equal(downloadCount, 0);

  await runRecallFirstIndexSetupCommand(['select-embeddinggemma', '--approve-download'], options);
  assert.equal(downloadCount, 1);
  assert.equal(verificationCount, 1);
  assert.equal(disposeCount, 1);
  const inferenceConfiguration = await readRecallInferenceConfiguration(
    join(root, 'data', 'inference-configuration.json'),
  );
  assert.equal(inferenceConfiguration.embedding?.profileId, profile.profileId);
  assert.equal(inferenceConfiguration.embedding?.backend, 'embedded');
  assert.equal(inferenceConfiguration.embedding?.artifact?.state, 'valid');
  assert.equal(inferenceConfiguration.reranking, null);
  assert.equal(inferenceConfiguration.queryPlanning, null);

  await runRecallFirstIndexSetupCommand(['estimate'], options);
  assert.deepEqual(readLastSetupOutputProperty(outputs, 'estimate'), {
    kind: 'metadata',
    measuredAt: '2026-08-01T10:00:00.000Z',
    corpus: { sessionCount: 4, sourceByteSize: 1_000 },
  });

  const blockedOptions = {
    ...options,
    qualityGateDecision: {
      automatedGatePassed: false,
      selectedPolicy: null,
      blockers: ['measured policy is stale'],
    },
  };
  await runRecallFirstIndexSetupCommand(
    ['estimate', '--measure', '--sample-sessions', '2'],
    blockedOptions,
  );
  assert.equal(measuredSampleBound, 2);
  assert.equal(selectedServiceCreationCount, 1);
  assert.equal(configuredServiceCreationCount, 1);
  assert.deepEqual(readLastSetupOutputProperty(outputs, 'estimate'), {
    kind: 'measured',
    measuredAt: '2026-08-01T10:00:00.000Z',
    measurement: {
      corpus: { sessionCount: 4, sourceByteSize: 1_000 },
      sampledSessionCount: 2,
      sampledSourceByteSize: 400,
      sampledDenseDocumentCount: 20,
      coldStartMilliseconds: 100,
      measuredSampleMilliseconds: 200,
      sourceBytesPerSecond: 2_000,
      denseDocumentsPerSecond: 100,
      cacheHitCount: 0,
      newlyEmbeddedDocumentCount: 20,
      embeddingRequestCount: 3,
      estimatedDurationMilliseconds: { minimum: 480, maximum: 750 },
    },
  });

  await runRecallFirstIndexSetupCommand(['defer'], options);
  assert.deepEqual(outputs.at(-1), {
    action: 'defer',
    configuration: {
      state: 'configured',
      embedding: {
        profileId: profile.profileId,
        backend: 'embedded',
        adapterId: 'node-llama-cpp-embedded-v2',
        devicePolicy: 'auto',
        verifiedAt: '2026-08-01T10:00:00.000Z',
      },
    },
    recallReady: false,
    message:
      'Recall configuration retained; recall is not ready until the first index generation activates.',
  });

  await assert.rejects(
    () => runRecallFirstIndexSetupCommand(['start'], options),
    /explicit --approve-build/u,
  );
  assert.equal(backgroundStartCount, 0);

  await runRecallFirstIndexSetupCommand(['start', '--approve-build'], blockedOptions);
  assert.equal(backgroundStartCount, 0);
  assert.equal(backgroundResumeCount, 1);
  assert.equal(selectedServiceCreationCount, 1);
  assert.equal(configuredServiceCreationCount, 2);
  assert.equal(disposeCount, 3);
  assert.deepEqual(outputs.at(-1), {
    action: 'start',
    recallReady: false,
    backgroundBuild: { ...backgroundStatus, generationId: 'sample-generation' },
  });
});

void test('first-index measurement accepts an independently configured HTTP embedding', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-first-index-http-command-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createSetupCommandTestConfig(root);
  const inferenceConfigurationPath = join(root, 'data', 'inference-configuration.json');
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const configuredProfileId = 'independent-http-embedding-profile-v1';
  const httpCandidate: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.EMBEDDING,
    candidateId: 'independent-http-embedding',
    profileId: configuredProfileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: 'independent-http-embedding-v1',
    endpoint: 'http://embedding.test/v1',
    device: null,
    artifact: null,
    async inspectHealth() {
      return { artifactState: RecallInferenceArtifactState.NOT_REQUIRED, requiredRepair: null };
    },
    async verifyCapabilityConformance() {
      return {
        profileId: configuredProfileId,
        adapterId: 'independent-http-embedding-v1',
        backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
        cacheIdentity: `${configuredProfileId}:independent-http-embedding-v1`,
        embeddingProfileId: configuredProfileId,
        measurement: { fixtureOperations: 1 },
      };
    },
  };
  await configureRecallInferenceCapability(inferenceConfigurationPath, httpCandidate);

  let measuredSampleCount = 0;
  const service: RecallFirstIndexSetupCommandService = {
    async verifyEmbeddingCapability() {
      throw new Error('HTTP setup fixture verification already completed');
    },
    async inspectConversationCorpus() {
      return { sessionCount: 1, sourceByteSize: 100 };
    },
    async measureFirstIndexSample() {
      measuredSampleCount += 1;
      return {
        corpus: { sessionCount: 1, sourceByteSize: 100 },
        sampledSessionCount: 1,
        sampledSourceByteSize: 100,
        sampledDenseDocumentCount: 2,
        coldStartMilliseconds: 10,
        measuredSampleMilliseconds: 20,
        sourceBytesPerSecond: 5_000,
        denseDocumentsPerSecond: 100,
        cacheHitCount: 0,
        newlyEmbeddedDocumentCount: 2,
        embeddingRequestCount: 1,
        estimatedDurationMilliseconds: { minimum: 20, maximum: 30 },
      };
    },
    async readIndexGenerationStatus() {
      return { active: null, staging: null };
    },
    async startBackgroundIndexGeneration() {
      throw new Error('HTTP setup fixture does not start a build');
    },
    async resumeBackgroundIndexGeneration() {
      throw new Error('HTTP setup fixture does not resume a build');
    },
  };
  const artifactPath = join(root, 'models', profile.source.artifact);
  let artifactDownloadCount = 0;
  const artifactCache = {
    async inspectArtifact() {
      return {
        profile,
        status: {
          state: 'missing' as const,
          artifactPath,
          partialPaths: [],
          repair: 'not required for this HTTP setup fixture',
        },
      };
    },
    async verifyArtifact() {
      throw new Error('HTTP setup fixture does not verify a local artifact');
    },
    async diagnoseArtifact() {
      throw new Error('HTTP setup fixture does not diagnose a local artifact');
    },
    async downloadArtifact() {
      artifactDownloadCount += 1;
      throw new Error('HTTP setup fixture does not download a local artifact');
    },
    async repairArtifact() {
      throw new Error('HTTP setup fixture does not repair a local artifact');
    },
    async removeArtifact() {
      throw new Error('HTTP setup fixture does not remove a local artifact');
    },
  };
  await assert.rejects(
    () =>
      runRecallFirstIndexSetupCommand(['select-embeddinggemma', '--approve-download'], {
        config,
        inferenceConfigurationPath,
        artifactCache,
        writeOutput() {},
      }),
    /different embedding profile.*inference configure/u,
  );
  assert.equal(artifactDownloadCount, 0);

  await runRecallFirstIndexSetupCommand(['estimate', '--measure'], {
    config,
    inferenceConfigurationPath,
    artifactCache,
    metadataService: service,
    createConfiguredServiceRuntime() {
      return { service, async dispose() {} };
    },
    writeOutput() {},
  });

  assert.equal(measuredSampleCount, 1);
});

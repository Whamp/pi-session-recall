import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  RecallDiagnosticsMode,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallSearchScope,
  RecallValidatedGenerationActivationStage,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
} from './recall-generation-state.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createOctenEmbeddingModelProfile } from './recall-model-profiles.js';
import {
  createRecallWorkMarkerId,
  encodeRecallWorkMarker,
  type RecallWorkMarker,
} from './recall-work-marker.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import { runRecallIncrementalWorker } from './run-recall-incremental-worker.js';

function createActivationTestConfig(root: string): RecallConversationConfig {
  const dataDirectory = join(root, 'recall');
  return {
    sessionsDirectory: join(root, 'sessions'),
    dataDirectory,
    databasePath: join(dataDirectory, 'legacy-zvec'),
    projectionDatabasePath: join(dataDirectory, 'legacy-projections'),
    statePath: join(dataDirectory, 'legacy-state.json'),
    manifestPath: join(dataDirectory, 'legacy-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    lockPath: join(dataDirectory, 'operation.lock'),
    diagnosticsMode: RecallDiagnosticsMode.OFF,
    diagnosticLogPath: join(dataDirectory, 'diagnostics.jsonl'),
    retainedDiagnosticLogPath: join(dataDirectory, 'diagnostics.previous.jsonl'),
    markerSpoolDirectory: join(dataDirectory, 'markers', 'pending'),
    markerQuarantineDirectory: join(dataDirectory, 'markers', 'quarantine'),
    markerControlDirectory: join(dataDirectory, 'markers', 'control'),
    workerOwnershipLockPath: join(dataDirectory, 'incremental-worker.lock'),
    generationRootDirectory: join(dataDirectory, 'generations'),
    activeGenerationPointerPath: join(dataDirectory, 'active-generation.json'),
    generationRegistryPath: join(dataDirectory, 'generation-registry.json'),
    backlogSummaryPath: join(dataDirectory, 'backlog-summary.json'),
    incrementalDiagnosticLogPath: join(dataDirectory, 'incremental-diagnostics.jsonl'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'fixture-model',
    embeddingServedModelId: 'fixture/model',
    embeddingArtifact: 'fixture-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused.test/v1',
    rerankerModel: 'fixture-reranker',
    projectLineages: normalizeRecallProjectLineages({}),
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 10,
    confirmedDeletionMaxMissingSourceRatio: 0.5,
  };
}

function createActivationTestService(
  config: RecallConversationConfig,
  activationFault?: (stage: RecallValidatedGenerationActivationStage) => void,
) {
  const profile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-model',
      servedModelId: 'fixture/model',
      artifact: 'fixture-model.fp32',
      artifactSha256: 'a'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  return createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        return documents.map(() => [3, 0, 4]);
      },
      async embedQuery() {
        return [3, 0, 4];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'tokenizer.json', sha256: 'b'.repeat(64) }],
    },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return {
            ids: text
              .split(/\s+/u)
              .filter(Boolean)
              .map((word) => word.length),
          };
        },
      };
    },
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
    ...(activationFault ? { activationFault } : {}),
  });
}

void test('configured service activates one validated target with a fixed replay snapshot and serves reads during replay', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-activation-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
    mkdir(join(config.markerQuarantineDirectory, 'corrupt'), { recursive: true }),
  ]);
  const sourcePath = join(config.sessionsDirectory, 'activation.jsonl');
  await writeFile(
    sourcePath,
    `${[
      {
        type: 'session',
        version: 3,
        id: 'activation-session',
        timestamp: '2026-08-09T00:00:00.000Z',
        cwd: root,
      },
      {
        type: 'message',
        id: 'activation-message',
        parentId: null,
        timestamp: '2026-08-09T00:00:01.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'TARGET_ACTIVATION_NEEDLE remains searchable.' }],
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  const service = createActivationTestService(config);
  const generationId = 'generation_activation_target';
  await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  await writeFile(join(config.markerSpoolDirectory, 'marker_pending.json'), '{}\n');
  await writeFile(
    join(config.markerQuarantineDirectory, 'corrupt', 'marker_quarantined.json.quarantined'),
    '{}\n',
  );

  const activation = await service.activateValidatedRecallGeneration(generationId);

  assert.deepEqual(activation, {
    activeGenerationId: generationId,
    replayPendingMarkerCount: 1,
    replayQuarantinedMarkerCount: 1,
  });
  assert.equal(
    (await readRecallActiveGenerationPointer(config.activeGenerationPointerPath))
      ?.activeGenerationId,
    generationId,
  );
  const registry = await readRecallGenerationRegistry(config.generationRegistryPath);
  assert.equal(registry?.rollbackGenerationId, null);
  assert.equal(registry?.generations[0]?.state, RecallGenerationCutoverState.REPLAY_PENDING);
  const replaySnapshot: unknown = JSON.parse(
    await readFile(
      join(config.generationRootDirectory, generationId, 'generation-replay-snapshot.json'),
      'utf8',
    ),
  );
  assert.ok(isUnknownRecord(replaySnapshot));
  assert.equal(replaySnapshot.snapshotVersion, 1);
  assert.equal(replaySnapshot.generationId, generationId);
  assert.deepEqual(replaySnapshot.pendingMarkerIds, ['marker_pending']);
  assert.deepEqual(replaySnapshot.quarantinedMarkerIds, ['marker_quarantined']);
  assert.equal(typeof replaySnapshot.capturedAtEpochMilliseconds, 'number');

  const search = await service.search('TARGET_ACTIVATION_NEEDLE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  const result = search.results[0];
  assert.ok(result);
  assert.equal(result.content, 'TARGET_ACTIVATION_NEEDLE remains searchable.');
  const neighborhood = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: result.id,
    previousEntryCount: 0,
    nextEntryCount: 0,
  });
  assert.equal(neighborhood.anchorEvidenceOccurrenceId, result.id);
});

void test('configured service completes fixed replay from physical projection coverage while later markers remain ordinary backlog', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-fixed-replay-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
  ]);
  const physicalSessionId = 'fixed-replay-session';
  const sourcePath = join(config.sessionsDirectory, 'fixed-replay.jsonl');
  await writeFile(
    sourcePath,
    `${[
      {
        type: 'session',
        version: 3,
        id: physicalSessionId,
        timestamp: '2026-08-09T01:00:00.000Z',
        cwd: root,
      },
      {
        type: 'message',
        id: 'fixed-replay-message',
        parentId: null,
        timestamp: '2026-08-09T01:00:01.000Z',
        message: { role: 'assistant', content: 'Fixed replay source evidence.' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  await utimes(sourcePath, 1, 1);
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sourcePath,
    runtimeInstanceId: 'runtime-fixed-replay',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'fixed-replay-message',
    },
  } as const;
  const capturedMarker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  const capturedMarkerPath = join(config.markerSpoolDirectory, `${capturedMarker.markerId}.json`);
  await writeFile(capturedMarkerPath, '{}\n');
  const service = createActivationTestService(config);
  const generationId = 'generation_fixed_replay';
  await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  await service.activateValidatedRecallGeneration(generationId);
  const laterMarkerPath = join(config.markerSpoolDirectory, 'marker_later.json');
  await writeFile(laterMarkerPath, '{}\n');

  const transfer = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [capturedMarker.markerId],
    workItems: [{ marker: capturedMarker, coveredMarkerIds: [capturedMarker.markerId] }],
    quarantineDiagnostics: [],
  });
  assert.equal(transfer.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  const completed = await service.completeRecallGenerationReplay();

  assert.equal(completed, true);
  assert.equal(
    (await readRecallGenerationRegistry(config.generationRegistryPath))?.generations.find(
      ({ generationId: candidateId }) => candidateId === generationId,
    )?.state,
    RecallGenerationCutoverState.ACTIVE,
  );
  await assert.rejects(() => access(capturedMarkerPath), { code: 'ENOENT' });
  await access(laterMarkerPath);
});

void test('fixed replay completion waits only for quarantined marker IDs captured at activation', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-quarantine-replay-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  const quarantineCategoryDirectory = join(config.markerQuarantineDirectory, 'corrupt');
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.dataDirectory, { recursive: true }),
    mkdir(quarantineCategoryDirectory, { recursive: true }),
  ]);
  const capturedPath = join(quarantineCategoryDirectory, 'marker_captured.json.first');
  await writeFile(capturedPath, '{}\n');
  const service = createActivationTestService(config);
  const generationId = 'generation_quarantine_replay';
  await service.createEmptyRecallGeneration({ generationId });
  await service.activateValidatedRecallGeneration(generationId);
  const laterPath = join(quarantineCategoryDirectory, 'marker_later.json.second');
  await writeFile(laterPath, '{}\n');

  assert.equal(await service.completeRecallGenerationReplay(), false);
  await rm(capturedPath);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await access(laterPath);
});

void test('configured service recovers failed replay-backlog publication after pointer cutover', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-backlog-recovery-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.dataDirectory, { recursive: true }),
    mkdir(config.backlogSummaryPath, { recursive: true }),
  ]);
  const service = createActivationTestService(config);
  const generationId = 'generation_backlog_recovery';
  await service.createEmptyRecallGeneration({ generationId });

  await assert.rejects(
    () => service.activateValidatedRecallGeneration(generationId),
    /EISDIR|directory/iu,
  );
  assert.equal(
    (await readRecallActiveGenerationPointer(config.activeGenerationPointerPath))
      ?.activeGenerationId,
    generationId,
  );
  await rm(config.backlogSummaryPath, { recursive: true });

  assert.equal(await service.recoverRecallGenerationCutover(), true);
  const backlog: unknown = JSON.parse(await readFile(config.backlogSummaryPath, 'utf8'));
  assert.ok(isUnknownRecord(backlog));
  assert.equal(backlog.activeGenerationId, generationId);
  assert.equal(backlog.generationState, RecallGenerationCutoverState.REPLAY_PENDING);
});

void test('configured service recovers a READY publication fault without selecting a latest directory', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-activation-recovery-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.dataDirectory, { recursive: true }),
  ]);
  let faultEnabled = false;
  const service = createActivationTestService(config, (stage) => {
    if (faultEnabled && stage === RecallValidatedGenerationActivationStage.AFTER_READY_REGISTRY) {
      throw new Error('injected activation fault after READY registry');
    }
  });
  const previousGenerationId = 'generation_previous_target';
  await service.createEmptyRecallGeneration({ generationId: previousGenerationId });
  await service.activateValidatedRecallGeneration(previousGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  const generationId = 'generation_ready_recovery';
  await service.createEmptyRecallGeneration({ generationId });
  faultEnabled = true;

  await assert.rejects(
    () => service.activateValidatedRecallGeneration(generationId),
    /injected activation fault after READY registry/u,
  );
  assert.equal(
    (await readRecallActiveGenerationPointer(config.activeGenerationPointerPath))
      ?.activeGenerationId,
    previousGenerationId,
  );
  assert.equal(
    (await readRecallGenerationRegistry(config.generationRegistryPath))?.generations.find(
      ({ generationId: candidateId }) => candidateId === generationId,
    )?.state,
    RecallGenerationCutoverState.READY,
  );

  assert.equal(await service.recoverRecallGenerationCutover(), true);
  assert.equal(
    (await readRecallActiveGenerationPointer(config.activeGenerationPointerPath))
      ?.activeGenerationId,
    generationId,
  );
  const recoveredRegistry = await readRecallGenerationRegistry(config.generationRegistryPath);
  assert.equal(
    recoveredRegistry?.generations.find(
      ({ generationId: candidateId }) => candidateId === generationId,
    )?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
  assert.equal(recoveredRegistry?.rollbackGenerationId, previousGenerationId);
});

void test('configured service recovers pointer and activated-registry publication faults', async (t) => {
  for (const stage of [
    RecallValidatedGenerationActivationStage.AFTER_POINTER_SWAP,
    RecallValidatedGenerationActivationStage.AFTER_ACTIVATED_REGISTRY,
  ]) {
    await t.test(stage, async (stageTest) => {
      const root = await mkdtemp(join(tmpdir(), `recall-target-${stage}-`));
      stageTest.after(() => rm(root, { recursive: true, force: true }));
      const config = createActivationTestConfig(root);
      await Promise.all([
        mkdir(config.sessionsDirectory, { recursive: true }),
        mkdir(config.dataDirectory, { recursive: true }),
      ]);
      const service = createActivationTestService(config, (observedStage) => {
        if (observedStage === stage) {
          throw new Error(`injected activation fault at ${stage}`);
        }
      });
      const generationId = `generation_${stage}`;
      await service.createEmptyRecallGeneration({ generationId });

      await assert.rejects(
        () => service.activateValidatedRecallGeneration(generationId),
        new RegExp(`injected activation fault at ${stage}`, 'u'),
      );
      assert.equal(
        (await readRecallActiveGenerationPointer(config.activeGenerationPointerPath))
          ?.activeGenerationId,
        generationId,
      );
      assert.equal(await service.recoverRecallGenerationCutover(), true);
      assert.equal(
        (await readRecallGenerationRegistry(config.generationRegistryPath))?.generations.find(
          ({ generationId: candidateId }) => candidateId === generationId,
        )?.state,
        RecallGenerationCutoverState.REPLAY_PENDING,
      );
      const backlog: unknown = JSON.parse(await readFile(config.backlogSummaryPath, 'utf8'));
      assert.ok(isUnknownRecord(backlog));
      assert.equal(backlog.generationState, RecallGenerationCutoverState.REPLAY_PENDING);
    });
  }
});

void test('configured service rejects a target whose validation receipt no longer matches', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-receipt-rejection-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.dataDirectory, { recursive: true }),
  ]);
  const service = createActivationTestService(config);
  const generationId = 'generation_receipt_rejection';
  const generation = await service.createEmptyRecallGeneration({ generationId });
  const receipt: unknown = JSON.parse(await readFile(generation.validationReceiptPath, 'utf8'));
  assert.ok(isUnknownRecord(receipt));
  receipt.manifestFingerprint = 'f'.repeat(64);
  await writeFile(generation.validationReceiptPath, `${JSON.stringify(receipt)}\n`);

  await assert.rejects(
    () => service.activateValidatedRecallGeneration(generationId),
    /validation receipt (?:identity )?mismatch/u,
  );
  assert.equal(await readRecallActiveGenerationPointer(config.activeGenerationPointerPath), null);
  await assert.rejects(
    () =>
      access(join(config.generationRootDirectory, generationId, 'generation-replay-snapshot.json')),
    { code: 'ENOENT' },
  );
});

void test('ordinary worker transfers only fixed replay markers before scheduling later backlog', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-worker-fixed-replay-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createActivationTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
    mkdir(config.markerControlDirectory, { recursive: true }),
  ]);
  const physicalSessionId = 'worker-replay-session';
  const sourcePath = join(config.sessionsDirectory, 'worker-replay.jsonl');
  await writeFile(
    sourcePath,
    `${[
      {
        type: 'session',
        version: 3,
        id: physicalSessionId,
        timestamp: '2026-08-09T02:00:00.000Z',
        cwd: root,
      },
      {
        type: 'message',
        id: 'worker-replay-message',
        parentId: null,
        timestamp: '2026-08-09T02:00:01.000Z',
        message: { role: 'assistant', content: 'Worker fixed replay evidence.' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  await utimes(sourcePath, 1, 1);
  const capturedIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sourcePath,
    runtimeInstanceId: 'runtime-worker-replay',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'worker-replay-message',
    },
  } as const;
  const capturedMarker: RecallWorkMarker = {
    ...capturedIdentity,
    markerId: createRecallWorkMarkerId(capturedIdentity),
  };
  await writeFile(
    join(config.markerSpoolDirectory, `${capturedMarker.markerId}.json`),
    await encodeRecallWorkMarker(capturedMarker, {
      trustedSessionRoots: [config.sessionsDirectory],
    }),
  );
  const service = createActivationTestService(config);
  const generationId = 'generation_worker_fixed_replay';
  await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  await service.activateValidatedRecallGeneration(generationId);
  const laterIdentity = {
    ...capturedIdentity,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: 2,
  };
  const laterMarker: RecallWorkMarker = {
    ...laterIdentity,
    markerId: createRecallWorkMarkerId(laterIdentity),
  };
  const laterMarkerPath = join(config.markerSpoolDirectory, `${laterMarker.markerId}.json`);
  await writeFile(
    laterMarkerPath,
    await encodeRecallWorkMarker(laterMarker, {
      trustedSessionRoots: [config.sessionsDirectory],
    }),
  );
  const transferredMarkerIds: string[][] = [];

  const result = await runRecallIncrementalWorker({
    markerSpoolDirectory: config.markerSpoolDirectory,
    markerQuarantineDirectory: config.markerQuarantineDirectory,
    controlDirectory: config.markerControlDirectory,
    targetGenerationId: generationId,
    generationRegistryPath: config.generationRegistryPath,
    generationReplayCompletion: {
      activeGenerationPointerPath: config.activeGenerationPointerPath,
      generationRegistryPath: config.generationRegistryPath,
      backlogSummaryPath: config.backlogSummaryPath,
      generationRootDirectory: config.generationRootDirectory,
      lockPath: config.lockPath,
    },
    trustedSessionRoots: [config.sessionsDirectory],
    nowEpochMilliseconds: () => 100_000,
    monotonicMilliseconds: () => 100_000,
    async loadHeavyDependencies() {},
    async transferWorkPlan(workPlan) {
      transferredMarkerIds.push([...workPlan.sourceMarkerIds]);
      return service.transferIncrementalRecallWorkPlan(workPlan);
    },
  });

  assert.deepEqual(transferredMarkerIds, [[capturedMarker.markerId]]);
  assert.equal(result.generationReplayCompleted, true);
  assert.equal(result.nextWakeAtEpochMilliseconds, 100_000);
  await access(laterMarkerPath);
});

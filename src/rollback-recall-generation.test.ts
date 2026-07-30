import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  RecallDiagnosticsMode,
  RecallGenerationCutoverState,
  RecallSearchScope,
  RecallTargetGenerationRollbackStage,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  readRecallActiveGenerationPointer,
  readRecallGenerationRegistry,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { readRecallGenerationReplaySnapshot } from './recall-generation-replay-snapshot.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { createOctenEmbeddingModelProfile } from './recall-model-profiles.js';
import {
  createRecallWorkMarkerId,
  encodeRecallWorkMarker,
  type RecallWorkMarker,
} from './recall-work-marker.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import { runRecallIncrementalWorker } from './run-recall-incremental-worker.js';

function createRollbackTestConfig(root: string): RecallConversationConfig {
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

type GenerationCollectionFault = (
  stage: 'after_generation_directory_delete',
  generationId: string,
) => void;

function createRollbackTestService(
  config: RecallConversationConfig,
  rollbackFault?: (stage: RecallTargetGenerationRollbackStage) => void,
  generationCollectionFault?: GenerationCollectionFault,
) {
  const embeddingProfile = createOctenEmbeddingModelProfile(
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
    embeddingProfile,
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
    ...(rollbackFault ? { rollbackFault } : {}),
    ...(generationCollectionFault ? { generationCollectionFault } : {}),
  });
}

async function writeRollbackSource(
  sourcePath: string,
  sessionId: string,
  messageId: string,
  content: string,
  cwd: string,
): Promise<void> {
  await writeFile(
    sourcePath,
    `${[
      {
        type: 'session',
        version: 3,
        id: sessionId,
        timestamp: '2026-08-10T00:00:00.000Z',
        cwd,
      },
      {
        type: 'message',
        id: messageId,
        parentId: null,
        timestamp: '2026-08-10T00:00:01.000Z',
        message: { role: 'assistant', content },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
}

interface ActivatedRollbackPair {
  root: string;
  config: RecallConversationConfig;
  service: ReturnType<typeof createRollbackTestService>;
  firstGenerationId: string;
  secondGenerationId: string;
  firstSourcePath: string;
}

async function createActivatedRollbackPair(
  suffix: string,
  rollbackFault?: (stage: RecallTargetGenerationRollbackStage) => void,
  generationCollectionFault?: GenerationCollectionFault,
): Promise<ActivatedRollbackPair> {
  const root = await mkdtemp(join(tmpdir(), `recall-target-rollback-${suffix}-`));
  const config = createRollbackTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
  ]);
  const firstSourcePath = join(config.sessionsDirectory, 'first.jsonl');
  const secondSourcePath = join(config.sessionsDirectory, 'second.jsonl');
  await Promise.all([
    writeRollbackSource(
      firstSourcePath,
      `first-session-${suffix}`,
      `first-message-${suffix}`,
      `ROLLBACK_FIRST_${suffix} belongs to generation A.`,
      root,
    ),
    writeRollbackSource(
      secondSourcePath,
      `second-session-${suffix}`,
      `second-message-${suffix}`,
      `ROLLBACK_SECOND_${suffix} belongs to generation B.`,
      root,
    ),
  ]);
  await Promise.all([utimes(firstSourcePath, 1, 1), utimes(secondSourcePath, 1, 1)]);
  const service = createRollbackTestService(config, rollbackFault, generationCollectionFault);
  const firstGenerationId = `generation_rollback_a_${suffix}`;
  const secondGenerationId = `generation_rollback_b_${suffix}`;
  await service.createRecallGenerationFromPhysicalSources({
    generationId: firstGenerationId,
    physicalSessionPaths: [firstSourcePath],
  });
  await service.activateValidatedRecallGeneration(firstGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await service.createRecallGenerationFromPhysicalSources({
    generationId: secondGenerationId,
    physicalSessionPaths: [secondSourcePath],
  });
  await service.activateValidatedRecallGeneration(secondGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  return { root, config, service, firstGenerationId, secondGenerationId, firstSourcePath };
}

void test('configured service rolls back between two target generations and can switch back', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-rollback-switch-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createRollbackTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
  ]);
  const firstSourcePath = join(config.sessionsDirectory, 'first.jsonl');
  const secondSourcePath = join(config.sessionsDirectory, 'second.jsonl');
  await Promise.all([
    writeRollbackSource(
      firstSourcePath,
      'first-session',
      'first-message',
      'ROLLBACK_FIRST_NEEDLE belongs to generation A.',
      root,
    ),
    writeRollbackSource(
      secondSourcePath,
      'second-session',
      'second-message',
      'ROLLBACK_SECOND_NEEDLE belongs to generation B.',
      root,
    ),
  ]);
  const service = createRollbackTestService(config);
  const firstGenerationId = 'generation_rollback_a';
  const secondGenerationId = 'generation_rollback_b';
  await service.createRecallGenerationFromPhysicalSources({
    generationId: firstGenerationId,
    physicalSessionPaths: [firstSourcePath],
  });
  await service.activateValidatedRecallGeneration(firstGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await service.createRecallGenerationFromPhysicalSources({
    generationId: secondGenerationId,
    physicalSessionPaths: [secondSourcePath],
  });
  await service.activateValidatedRecallGeneration(secondGenerationId);
  assert.equal(await service.completeRecallGenerationReplay(), true);
  await Promise.all([rm(firstSourcePath), rm(secondSourcePath)]);

  await service.rollback();

  const rolledBackRegistry = await readRecallGenerationRegistry(config.generationRegistryPath);
  const restoredEntry = rolledBackRegistry?.generations.find(
    ({ generationId }) => generationId === firstGenerationId,
  );
  assert.equal(rolledBackRegistry?.activeGenerationId, firstGenerationId);
  assert.equal(rolledBackRegistry?.rollbackGenerationId, secondGenerationId);
  assert.equal(
    rolledBackRegistry?.generations.find(({ generationId }) => generationId === firstGenerationId)
      ?.state,
    RecallGenerationCutoverState.REPLAY_PENDING,
  );
  assert.notEqual(restoredEntry?.replaySnapshotFileName, 'generation-replay-snapshot.json');
  assert.match(
    restoredEntry?.replaySnapshotFileName ?? '',
    /^generation-replay-snapshot-[\w-]+\.json$/u,
  );
  await access(
    join(
      config.generationRootDirectory,
      firstGenerationId,
      restoredEntry?.replaySnapshotFileName ?? 'missing-snapshot',
    ),
  );
  const restoredSearch = await service.search('ROLLBACK_FIRST_NEEDLE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    restoredSearch.results[0]?.content,
    'ROLLBACK_FIRST_NEEDLE belongs to generation A.',
  );
  assert.equal(await service.completeRecallGenerationReplay(), true);

  await service.rollback();

  const switchedBackSearch = await service.search('ROLLBACK_SECOND_NEEDLE', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    switchedBackSearch.results[0]?.content,
    'ROLLBACK_SECOND_NEEDLE belongs to generation B.',
  );
});

void test('configured service refuses rollback when the bounded target health check fails', async (t) => {
  const cases: ReadonlyArray<{
    name: string;
    expectedError: RegExp;
    damage(pair: ActivatedRollbackPair): Promise<void>;
  }> = [
    {
      name: 'required validation receipt missing',
      expectedError: /rollback health validation receipt/u,
      async damage({ config, firstGenerationId }) {
        await rm(
          createRecallGenerationComponentPaths(
            join(config.generationRootDirectory, firstGenerationId),
          ).validationReceiptPath,
        );
      },
    },
    {
      name: 'manifest fingerprint mismatch',
      expectedError: /rollback health manifest fingerprint mismatch/u,
      async damage({ config, firstGenerationId }) {
        const manifestPath = createRecallGenerationComponentPaths(
          join(config.generationRootDirectory, firstGenerationId),
        ).manifestPath;
        const manifest: unknown = JSON.parse(await readFile(manifestPath, 'utf8'));
        assert.ok(isUnknownRecord(manifest));
        assert.ok(isUnknownRecord(manifest.embeddingProfile));
        manifest.embeddingProfile.queryInputPrefix = 'damaged-query-prefix:';
        await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
      },
    },
    {
      name: 'dense store open failure',
      expectedError: /rollback health dense-evidence store/u,
      async damage({ config, firstGenerationId }) {
        await rm(
          createRecallGenerationComponentPaths(
            join(config.generationRootDirectory, firstGenerationId),
          ).denseStorePath,
          { recursive: true },
        );
      },
    },
    {
      name: 'lexical count mismatch',
      expectedError: /rollback health lexical-source count mismatch/u,
      async damage({ config, firstGenerationId }) {
        const storePath = createRecallGenerationComponentPaths(
          join(config.generationRootDirectory, firstGenerationId),
        ).lexicalSourceStorePath;
        const lexicalSource = ZVecOpen(storePath);
        try {
          const [record] = lexicalSource.querySync({ topk: 1, outputFields: [] });
          assert.ok(record);
          assert.equal(lexicalSource.deleteSync([record.id])[0]?.ok, true);
        } finally {
          lexicalSource.closeSync();
        }
      },
    },
    {
      name: 'deterministic lexical canary mismatch',
      expectedError: /rollback health lexical-source canary mismatch/u,
      async damage({ config, firstGenerationId }) {
        const storePath = createRecallGenerationComponentPaths(
          join(config.generationRootDirectory, firstGenerationId),
        ).lexicalSourceStorePath;
        const lexicalSource = ZVecOpen(storePath);
        try {
          const records = lexicalSource.querySync({
            topk: lexicalSource.stats.docCount,
            outputFields: lexicalSource.schema.fields().map(({ name }) => name),
            includeVector: false,
          });
          for (const record of records) {
            assert.equal(
              lexicalSource.upsertSync([
                {
                  id: record.id,
                  fields: { ...record.fields, physicalSourceIdentity: 'damaged-canary-source' },
                },
              ])[0]?.ok,
              true,
            );
          }
        } finally {
          lexicalSource.closeSync();
        }
      },
    },
    {
      name: 'deterministic dense canary mismatch',
      expectedError: /rollback health dense-evidence canary mismatch/u,
      async damage({ config, firstGenerationId }) {
        const storePath = createRecallGenerationComponentPaths(
          join(config.generationRootDirectory, firstGenerationId),
        ).denseStorePath;
        const dense = ZVecOpen(storePath);
        try {
          const records = dense.querySync({
            topk: dense.stats.docCount,
            outputFields: dense.schema.fields().map(({ name }) => name),
            includeVector: true,
          });
          for (const record of records) {
            assert.equal(
              dense.upsertSync([
                {
                  id: record.id,
                  fields: { ...record.fields, embeddingProfileId: 'damaged-profile' },
                  vectors: record.vectors,
                },
              ])[0]?.ok,
              true,
            );
          }
        } finally {
          dense.closeSync();
        }
      },
    },
    {
      name: 'deterministic projection canary mismatch',
      expectedError: /rollback health session-projection canary mismatch/u,
      async damage({ config, firstGenerationId }) {
        const storePath = createRecallGenerationComponentPaths(
          join(config.generationRootDirectory, firstGenerationId),
        ).sessionProjectionStorePath;
        const projections = ZVecOpen(storePath);
        try {
          const records = projections.querySync({
            topk: projections.stats.docCount,
            outputFields: projections.schema.fields().map(({ name }) => name),
            includeVector: false,
          });
          for (const record of records) {
            assert.equal(
              projections.upsertSync([
                {
                  id: record.id,
                  fields: { ...record.fields, physicalSourceIdentity: 'damaged-canary-source' },
                },
              ])[0]?.ok,
              true,
            );
          }
        } finally {
          projections.closeSync();
        }
      },
    },
  ];

  for (const healthCase of cases) {
    await t.test(healthCase.name, async (caseTest) => {
      const pair = await createActivatedRollbackPair(healthCase.name.replaceAll(/[^a-z]+/gu, '_'));
      caseTest.after(() => rm(pair.root, { recursive: true, force: true }));
      await healthCase.damage(pair);

      await assert.rejects(() => pair.service.rollback(), healthCase.expectedError);

      assert.equal(
        (await readRecallActiveGenerationPointer(pair.config.activeGenerationPointerPath))
          ?.activeGenerationId,
        pair.secondGenerationId,
      );
      assert.equal(
        (await readRecallGenerationRegistry(pair.config.generationRegistryPath))
          ?.activeGenerationId,
        pair.secondGenerationId,
      );
    });
  }
});

void test('configured service recovers every interrupted rollback publication boundary', async (t) => {
  for (const stage of [
    RecallTargetGenerationRollbackStage.AFTER_REGISTRY,
    RecallTargetGenerationRollbackStage.AFTER_POINTER,
    RecallTargetGenerationRollbackStage.AFTER_BACKLOG,
  ]) {
    await t.test(stage, async (stageTest) => {
      let faultEnabled = false;
      const pair = await createActivatedRollbackPair(stage, (observedStage) => {
        if (faultEnabled && observedStage === stage) {
          throw new Error(`injected rollback fault at ${stage}`);
        }
      });
      stageTest.after(() => rm(pair.root, { recursive: true, force: true }));
      faultEnabled = true;

      await assert.rejects(
        () => pair.service.rollback(),
        new RegExp(`injected rollback fault at ${stage}`, 'u'),
      );
      faultEnabled = false;

      assert.equal(await pair.service.recoverRecallGenerationCutover(), true);
      assert.equal(
        (await readRecallActiveGenerationPointer(pair.config.activeGenerationPointerPath))
          ?.activeGenerationId,
        pair.firstGenerationId,
      );
      const registry = await readRecallGenerationRegistry(pair.config.generationRegistryPath);
      assert.equal(registry?.activeGenerationId, pair.firstGenerationId);
      assert.equal(registry?.rollbackGenerationId, pair.secondGenerationId);
      assert.equal(
        registry?.generations.find(({ generationId }) => generationId === pair.firstGenerationId)
          ?.state,
        RecallGenerationCutoverState.REPLAY_PENDING,
      );
      const search = await pair.service.search(`ROLLBACK_FIRST_${stage}`, 5, {
        scope: RecallSearchScope.GLOBAL,
      });
      assert.equal(search.results[0]?.content, `ROLLBACK_FIRST_${stage} belongs to generation A.`);
      assert.equal(await pair.service.completeRecallGenerationReplay(), true);
    });
  }
});

void test('configured rollback fixes retained and pending marker replay without dual writes', async (t) => {
  const pair = await createActivatedRollbackPair('fixed_marker_replay');
  t.after(() => rm(pair.root, { recursive: true, force: true }));
  const retainedMarkerDirectory = join(pair.config.markerControlDirectory, 'rollback-retained');
  await mkdir(retainedMarkerDirectory, { recursive: true });
  const markerBase = {
    version: 1,
    physicalSessionId: 'first-session-fixed_marker_replay',
    physicalSessionPath: pair.firstSourcePath,
    runtimeInstanceId: 'runtime-rollback-replay',
    createdAtEpochMilliseconds: 1,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: 'first-session-fixed_marker_replay',
      leafEntryId: 'first-message-fixed_marker_replay',
    },
  } as const;
  const pendingIdentity = { ...markerBase, runtimeSequence: 1 };
  const retainedIdentity = { ...markerBase, runtimeSequence: 2, createdAtEpochMilliseconds: 2 };
  const pendingMarker: RecallWorkMarker = {
    ...pendingIdentity,
    markerId: createRecallWorkMarkerId(pendingIdentity),
  };
  const retainedMarker: RecallWorkMarker = {
    ...retainedIdentity,
    markerId: createRecallWorkMarkerId(retainedIdentity),
  };
  await Promise.all([
    writeFile(
      join(pair.config.markerSpoolDirectory, `${pendingMarker.markerId}.json`),
      await encodeRecallWorkMarker(pendingMarker, {
        trustedSessionRoots: [pair.config.sessionsDirectory],
      }),
    ),
    writeFile(
      join(retainedMarkerDirectory, `${retainedMarker.markerId}.json`),
      await encodeRecallWorkMarker(retainedMarker, {
        trustedSessionRoots: [pair.config.sessionsDirectory],
      }),
    ),
  ]);

  const rollback = await pair.service.rollback();

  assert.deepEqual(rollback, {
    activeGenerationId: pair.firstGenerationId,
    rollbackGenerationId: pair.secondGenerationId,
    restoredMarkerCount: 1,
  });
  const registry = await readRecallGenerationRegistry(pair.config.generationRegistryPath);
  const activeEntry = registry?.generations.find(
    ({ generationId }) => generationId === pair.firstGenerationId,
  );
  assert.ok(activeEntry?.replaySnapshotFileName);
  const snapshot = await readRecallGenerationReplaySnapshot(
    join(
      pair.config.generationRootDirectory,
      pair.firstGenerationId,
      activeEntry.replaySnapshotFileName,
    ),
  );
  assert.deepEqual(
    snapshot.pendingMarkerIds,
    [pendingMarker.markerId, retainedMarker.markerId].toSorted(),
  );
  const searchDuringReplay = await pair.service.search('ROLLBACK_FIRST_fixed_marker_replay', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    searchDuringReplay.results[0]?.content,
    'ROLLBACK_FIRST_fixed_marker_replay belongs to generation A.',
  );

  await mkdir(pair.config.markerControlDirectory, { recursive: true });
  const workerResult = await runRecallIncrementalWorker({
    markerSpoolDirectory: pair.config.markerSpoolDirectory,
    markerQuarantineDirectory: pair.config.markerQuarantineDirectory,
    controlDirectory: pair.config.markerControlDirectory,
    targetGenerationId: pair.firstGenerationId,
    generationRegistryPath: pair.config.generationRegistryPath,
    generationReplayCompletion: {
      activeGenerationPointerPath: pair.config.activeGenerationPointerPath,
      generationRegistryPath: pair.config.generationRegistryPath,
      backlogSummaryPath: pair.config.backlogSummaryPath,
      generationRootDirectory: pair.config.generationRootDirectory,
      lockPath: pair.config.lockPath,
    },
    trustedSessionRoots: [pair.config.sessionsDirectory],
    nowEpochMilliseconds: () => 100_000,
    monotonicMilliseconds: () => 100_000,
    async loadHeavyDependencies() {},
    transferWorkPlan: pair.service.transferIncrementalRecallWorkPlan.bind(pair.service),
  });
  assert.equal(workerResult.generationReplayCompleted, true);
  assert.equal(await pair.service.completeRecallGenerationReplay(), true);
  assert.equal(await pair.service.completeRecallGenerationReplay(), true);
  await pair.service.rollback();
  const switchedBackSearch = await pair.service.search('ROLLBACK_SECOND_fixed_marker_replay', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.equal(
    switchedBackSearch.results[0]?.content,
    'ROLLBACK_SECOND_fixed_marker_replay belongs to generation B.',
  );
});

void test('configured cleanup deletes only prepared collectible generations and resumes after interruption', async (t) => {
  let cleanupFaultEnabled = false;
  const pair = await createActivatedRollbackPair('collectible_cleanup', undefined, (stage) => {
    if (cleanupFaultEnabled && stage === 'after_generation_directory_delete') {
      throw new Error('injected cleanup fault after generation directory delete');
    }
  });
  t.after(() => rm(pair.root, { recursive: true, force: true }));
  const collectibleGenerationId = 'generation_collectible_cleanup';
  const buildingGenerationId = 'generation_building_cleanup';
  const collectible = await pair.service.createEmptyRecallGeneration({
    generationId: collectibleGenerationId,
  });
  const building = await pair.service.createEmptyRecallGeneration({
    generationId: buildingGenerationId,
  });
  const initialRegistry = await readRecallGenerationRegistry(pair.config.generationRegistryPath);
  assert.ok(initialRegistry);
  const embeddingProfileId = initialRegistry.generations[0]?.embeddingProfileId;
  assert.ok(embeddingProfileId);
  const createFixtureEntry = (
    generationId: string,
    manifestFingerprint: string,
    validatedAtEpochMilliseconds: number,
    state: RecallGenerationCutoverState,
    retireAfterEpochMilliseconds: number | null,
  ) => ({
    generationId,
    state,
    embeddingProfileId,
    indexManifestVersion: 6 as const,
    markerSchemaVersion: 1 as const,
    sessionProjectionSchemaVersion: 3 as const,
    indexManifestFingerprint: manifestFingerprint,
    rebuildStartedAtEpochMilliseconds: validatedAtEpochMilliseconds,
    stateChangedAtEpochMilliseconds: validatedAtEpochMilliseconds,
    rebuildStartMarkerId: null,
    rebuildMarkerWatermark: [],
    validatedAtEpochMilliseconds,
    retireAfterEpochMilliseconds,
  });
  const collectibleEntry = createFixtureEntry(
    collectibleGenerationId,
    collectible.manifestFingerprint,
    collectible.validatedAtEpochMilliseconds,
    RecallGenerationCutoverState.RETIRED,
    1,
  );
  const buildingEntry = createFixtureEntry(
    buildingGenerationId,
    building.manifestFingerprint,
    building.validatedAtEpochMilliseconds,
    RecallGenerationCutoverState.BUILDING,
    null,
  );
  await writeRecallGenerationRegistry(pair.config.generationRegistryPath, {
    ...initialRegistry,
    buildingGenerationId,
    generations: [...initialRegistry.generations, collectibleEntry, buildingEntry],
  });

  assert.deepEqual(await pair.service.collectRetired(), { deletedGenerationIds: [] });
  await access(join(pair.config.generationRootDirectory, collectibleGenerationId));
  await access(join(pair.config.generationRootDirectory, buildingGenerationId));
  await writeRecallGenerationRegistry(pair.config.generationRegistryPath, {
    ...initialRegistry,
    generations: [
      ...initialRegistry.generations.map((entry) =>
        entry.generationId === pair.secondGenerationId
          ? { ...entry, state: RecallGenerationCutoverState.REPLAY_PENDING }
          : entry,
      ),
      collectibleEntry,
      { ...buildingEntry, state: RecallGenerationCutoverState.FAILED },
    ],
  });

  assert.deepEqual(await pair.service.collectRetired(), { deletedGenerationIds: [] });
  await access(join(pair.config.generationRootDirectory, collectibleGenerationId));
  await writeRecallGenerationRegistry(pair.config.generationRegistryPath, {
    ...initialRegistry,
    generations: [
      ...initialRegistry.generations,
      collectibleEntry,
      { ...buildingEntry, state: RecallGenerationCutoverState.FAILED },
    ],
  });
  cleanupFaultEnabled = true;

  await assert.rejects(
    () => pair.service.collectRetired(),
    /injected cleanup fault after generation directory delete/u,
  );
  await assert.rejects(() =>
    access(join(pair.config.generationRootDirectory, collectibleGenerationId)),
  );
  await access(join(pair.config.generationRootDirectory, pair.firstGenerationId));
  await access(join(pair.config.generationRootDirectory, pair.secondGenerationId));
  await access(join(pair.config.generationRootDirectory, buildingGenerationId));
  cleanupFaultEnabled = false;

  assert.deepEqual(await pair.service.collectRetired(), {
    deletedGenerationIds: [collectibleGenerationId],
  });
  const completedRegistry = await readRecallGenerationRegistry(pair.config.generationRegistryPath);
  assert.equal(
    completedRegistry?.generations.some(
      ({ generationId }) => generationId === collectibleGenerationId,
    ),
    false,
  );
  assert.equal(completedRegistry?.rollbackGenerationId, pair.firstGenerationId);
});

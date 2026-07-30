import assert from 'node:assert/strict';
import {
  access,
  appendFile,
  mkdir,
  mkdtemp,
  open,
  rm,
  stat,
  utimes,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  RecallDiagnosticsMode,
  RecallGenerationCutoverState,
  RecallIncrementalTransferOutcomeKind,
  RecallSearchScope,
  RecallWorkMarkerTrigger,
} from './enums.js';
import {
  createRecallActiveGenerationPointer,
  RECALL_GENERATION_REGISTRY_VERSION,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import {
  createOctenEmbeddingModelProfile,
  createRecallEmbeddingProfileIdentity,
} from './recall-model-profiles.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createTransferTestConfig(root: string): RecallConversationConfig {
  const dataDirectory = join(root, 'recall');
  return {
    sessionsDirectory: join(root, 'sessions'),
    dataDirectory,
    databasePath: join(dataDirectory, 'legacy-zvec'),
    projectionDatabasePath: join(dataDirectory, 'legacy-projections'),
    statePath: join(dataDirectory, 'legacy-state.json'),
    manifestPath: join(dataDirectory, 'legacy-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'legacy-cache'),
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

void test('configured transfer appends target evidence before acknowledging its marker', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-target-transfer-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const config = createTransferTestConfig(root);
  await Promise.all([
    mkdir(config.sessionsDirectory, { recursive: true }),
    mkdir(config.markerSpoolDirectory, { recursive: true }),
  ]);
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
  const embeddedDocuments: string[][] = [];
  const readRanges: Array<{ startByte: number; endByteExclusive: number }> = [];
  const transferEvents: string[] = [];
  const transferWindowDocumentCounts: number[] = [];
  let rejectedTransferStage: string | null = null;
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        embeddedDocuments.push([...documents]);
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
    async *incrementalTransferReadRange(sourcePath, startByte, endByteExclusive) {
      readRanges.push({ startByte, endByteExclusive });
      const handle = await open(sourcePath, 'r');
      try {
        const bytes = Buffer.alloc(endByteExclusive - startByte);
        const result = await handle.read(bytes, 0, bytes.length, startByte);
        yield bytes.subarray(0, result.bytesRead);
      } finally {
        await handle.close();
      }
    },
    incrementalTransferFault(stage, context) {
      transferEvents.push(stage);
      if (stage === 'after-recovery-record') {
        transferWindowDocumentCounts.push(context.evidenceDocumentCount);
      }
      if (stage === rejectedTransferStage) {
        throw new Error(`fixture transfer interruption: ${stage}`);
      }
    },
    workerSignal: { signalDetachedWorker() {} },
  });
  const generationId = 'generation_incremental_target';
  const opened = await service.createEmptyRecallGeneration({ generationId });
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: createRecallEmbeddingProfileIdentity(profile),
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: opened.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
    ],
  });
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);
  const physicalSessionId = 'session-incremental';
  const physicalSessionPath = join(config.sessionsDirectory, 'incremental.jsonl');
  await writeFile(
    physicalSessionPath,
    `${[
      {
        type: 'session',
        version: 3,
        id: physicalSessionId,
        timestamp: '2026-08-04T00:00:00.000Z',
        cwd: '/fixture/project',
      },
      {
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-08-04T00:00:01.000Z',
        message: { role: 'assistant', content: 'incremental target seam evidence' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
  await utimes(physicalSessionPath, 1, 1);
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath,
    runtimeInstanceId: 'runtime-incremental',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'entry-1',
    },
  } as const;
  const marker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  const markerPath = join(config.markerSpoolDirectory, `${marker.markerId}.json`);
  await writeFile(markerPath, '{}\n');

  const outcome = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [marker.markerId],
    workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
    quarantineDiagnostics: [],
  });

  assert.deepEqual(outcome, {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 1,
  });
  assert.deepEqual(embeddedDocuments, [['incremental target seam evidence']]);
  const lexical = await service.searchRecallGenerationLexical(
    generationId,
    'incremental target seam',
    5,
  );
  assert.equal(lexical.length, 1);
  assert.equal(lexical[0]?.content, 'incremental target seam evidence');
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });

  await writeFile(markerPath, '{}\n');
  const duplicateOutcome = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [marker.markerId],
    workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
    quarantineDiagnostics: [],
  });
  assert.deepEqual(duplicateOutcome, {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 0,
  });
  assert.deepEqual(embeddedDocuments, [['incremental target seam evidence'], []]);

  await appendFile(
    physicalSessionPath,
    `${JSON.stringify({
      type: 'message',
      id: 'entry-2',
      parentId: 'entry-1',
      timestamp: '2026-08-04T00:00:02.000Z',
      message: { role: 'assistant', content: 'bounded append replay evidence' },
    })}\n`,
  );
  await utimes(physicalSessionPath, 2, 2);
  const secondMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: 2_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'entry-2',
    },
  } as const;
  const secondMarker: RecallWorkMarker = {
    ...secondMarkerIdentity,
    markerId: createRecallWorkMarkerId(secondMarkerIdentity),
  };
  await writeFile(join(config.markerSpoolDirectory, `${secondMarker.markerId}.json`), '{}\n');
  const appendedOutcome = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [secondMarker.markerId],
    workItems: [{ marker: secondMarker, coveredMarkerIds: [secondMarker.markerId] }],
    quarantineDiagnostics: [],
  });
  assert.equal(appendedOutcome.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  assert.ok(appendedOutcome.committedDocumentCount > 0);
  const finalSourceByteSize = Number((await stat(physicalSessionPath)).size);
  assert.ok(readRanges.length > 0);
  assert.equal(
    readRanges.some(
      ({ startByte, endByteExclusive }) =>
        startByte === 0 && endByteExclusive === finalSourceByteSize,
    ),
    false,
  );
  const appendedLexical = await service.searchRecallGenerationLexical(
    generationId,
    'bounded append replay',
    5,
  );
  assert.equal(appendedLexical[0]?.content, 'bounded append replay evidence');

  await appendFile(
    physicalSessionPath,
    `${JSON.stringify({
      type: 'message',
      id: 'entry-3',
      parentId: 'entry-2',
      timestamp: '2026-08-04T00:00:03.000Z',
      message: { role: 'assistant', content: 'recovered ordered target evidence' },
    })}\n`,
  );
  await utimes(physicalSessionPath, 3, 3);
  const interruptedMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 3,
    createdAtEpochMilliseconds: 3_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'entry-3',
    },
  } as const;
  const interruptedMarker: RecallWorkMarker = {
    ...interruptedMarkerIdentity,
    markerId: createRecallWorkMarkerId(interruptedMarkerIdentity),
  };
  const interruptedMarkerPath = join(
    config.markerSpoolDirectory,
    `${interruptedMarker.markerId}.json`,
  );
  await writeFile(interruptedMarkerPath, '{}\n');
  const interruptedWorkPlan = {
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [interruptedMarker.markerId],
    workItems: [{ marker: interruptedMarker, coveredMarkerIds: [interruptedMarker.markerId] }],
    quarantineDiagnostics: [],
  };
  rejectedTransferStage = 'after-lexical-source-write';
  await assert.rejects(
    () => service.transferIncrementalRecallWorkPlan(interruptedWorkPlan),
    /Recall target incremental write or close failed/u,
  );
  assert.ok(transferEvents.includes('after-lexical-source-write'));
  const recoveryRecordPath = join(
    config.generationRootDirectory,
    generationId,
    'write-recovery.json',
  );
  await access(recoveryRecordPath);
  await access(interruptedMarkerPath);
  await assert.rejects(
    () => service.searchRecallGenerationLexical(generationId, 'recovered ordered', 5),
    /recovery required/u,
  );

  rejectedTransferStage = null;
  transferEvents.length = 0;
  const recoveredOutcome = await service.transferIncrementalRecallWorkPlan(interruptedWorkPlan);
  assert.equal(recoveredOutcome.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  assert.equal(transferEvents.length, 0);
  await assert.rejects(() => access(recoveryRecordPath), { code: 'ENOENT' });
  await assert.rejects(() => access(interruptedMarkerPath), { code: 'ENOENT' });
  const recoveredLexical = await service.searchRecallGenerationLexical(
    generationId,
    'recovered ordered target',
    5,
  );
  assert.equal(recoveredLexical[0]?.content, 'recovered ordered target evidence');

  const largeAppendRecords: Array<Record<string, unknown>> = [];
  let parentId = 'entry-3';
  for (let turn = 1; turn <= 11; turn += 1) {
    const userEntryId = `large-user-${turn}`;
    largeAppendRecords.push({
      type: 'message',
      id: userEntryId,
      parentId,
      timestamp: `2026-08-04T00:01:${String(turn * 2 - 1).padStart(2, '0')}.000Z`,
      message: { role: 'user', content: `large bounded user ${turn}` },
    });
    const assistantEntryId = `large-assistant-${turn}`;
    largeAppendRecords.push({
      type: 'message',
      id: assistantEntryId,
      parentId: userEntryId,
      timestamp: `2026-08-04T00:01:${String(turn * 2).padStart(2, '0')}.000Z`,
      message: { role: 'assistant', content: `large bounded assistant ${turn}` },
    });
    parentId = assistantEntryId;
  }
  await appendFile(
    physicalSessionPath,
    `${largeAppendRecords.map((record) => JSON.stringify(record)).join('\n')}\n`,
  );
  await utimes(physicalSessionPath, 4, 4);
  const largeMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 4,
    createdAtEpochMilliseconds: 4_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: parentId,
    },
  } as const;
  const largeMarker: RecallWorkMarker = {
    ...largeMarkerIdentity,
    markerId: createRecallWorkMarkerId(largeMarkerIdentity),
  };
  await writeFile(join(config.markerSpoolDirectory, `${largeMarker.markerId}.json`), '{}\n');
  transferEvents.length = 0;
  transferWindowDocumentCounts.length = 0;
  const largeOutcome = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: generationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [largeMarker.markerId],
    workItems: [{ marker: largeMarker, coveredMarkerIds: [largeMarker.markerId] }],
    quarantineDiagnostics: [],
  });
  assert.deepEqual(largeOutcome, {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 33,
  });
  assert.deepEqual(transferWindowDocumentCounts, [32, 1]);
  assert.ok(transferWindowDocumentCounts.every((count) => count <= 32));
  assert.ok(
    transferEvents.lastIndexOf('after-physical-projection-write') <
      transferEvents.lastIndexOf('after-marker-acknowledgement'),
  );
  const activeSearch = await service.search('large bounded assistant', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
  assert.ok(activeSearch.results.length > 0);
  const originalNeighborhood = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: lexical[0]?.evidenceOccurrenceId ?? '',
    previousEntryCount: 0,
    nextEntryCount: 0,
  });
  assert.equal(
    originalNeighborhood.entries[0]?.evidence[0]?.content,
    'incremental target seam evidence',
  );

  transferEvents.length = 0;
  const physicalSourceIdentity = resolveRecallPhysicalSourceIdentity(
    config.sessionsDirectory,
    physicalSessionPath,
  ).physicalSourceIdentity;
  const deletionOutcome = await service.transferIncrementalRecallWorkPlan({
    confirmedPhysicalSourceDeletion: {
      targetGenerationId: generationId,
      physicalSourceIdentity,
    },
  });
  assert.deepEqual(deletionOutcome, {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 0,
  });
  assert.deepEqual(transferEvents, [
    'after-recovery-record',
    'after-dense-delete',
    'after-lexical-source-delete',
    'after-projection-delete',
    'after-store-close',
    'after-reopened-verification',
    'after-recovery-clear',
  ]);
  const generationDirectory = join(config.generationRootDirectory, generationId);
  for (const storeDirectory of ['lexical-source', 'dense', 'session-projections']) {
    const collection = ZVecOpen(join(generationDirectory, storeDirectory), { readOnly: true });
    try {
      assert.equal(collection.stats.docCount, 0);
    } finally {
      collection.closeSync();
    }
  }

  const builtGenerationId = 'generation_fixed_snapshot_then_incremental';
  const builtGeneration = await service.createRecallGenerationFromPhysicalSources({
    generationId: builtGenerationId,
    physicalSessionPaths: [physicalSessionPath],
  });
  const builtPointer = createRecallActiveGenerationPointer(builtGenerationId);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: builtGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: generationId,
    activePointerChecksum: builtPointer.checksum,
    generations: [
      {
        generationId: builtGenerationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: createRecallEmbeddingProfileIdentity(profile),
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: builtGeneration.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 3,
        stateChangedAtEpochMilliseconds: 4,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 4,
      },
      {
        generationId,
        state: RecallGenerationCutoverState.ROLLBACK,
        embeddingProfileId: createRecallEmbeddingProfileIdentity(profile),
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: opened.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 4,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
    ],
  });
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, builtPointer);
  await appendFile(
    physicalSessionPath,
    `${JSON.stringify({
      type: 'message',
      id: 'post-build-entry',
      parentId,
      timestamp: '2026-08-04T00:02:00.000Z',
      message: { role: 'assistant', content: 'post build incremental evidence' },
    })}\n`,
  );
  await utimes(physicalSessionPath, 5, 5);
  const postBuildMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 5,
    createdAtEpochMilliseconds: 5_000,
    trigger: {
      kind: RecallWorkMarkerTrigger.DEPARTURE,
      logicalSessionId: physicalSessionId,
      leafEntryId: 'post-build-entry',
    },
  } as const;
  const postBuildMarker: RecallWorkMarker = {
    ...postBuildMarkerIdentity,
    markerId: createRecallWorkMarkerId(postBuildMarkerIdentity),
  };
  await writeFile(join(config.markerSpoolDirectory, `${postBuildMarker.markerId}.json`), '{}\n');
  const postBuildOutcome = await service.transferIncrementalRecallWorkPlan({
    targetGenerationId: builtGenerationId,
    markerSpoolDirectory: config.markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [postBuildMarker.markerId],
    workItems: [{ marker: postBuildMarker, coveredMarkerIds: [postBuildMarker.markerId] }],
    quarantineDiagnostics: [],
  });
  assert.equal(postBuildOutcome.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  const postBuildLexical = await service.searchRecallGenerationLexical(
    builtGenerationId,
    'post build incremental',
    5,
  );
  assert.equal(postBuildLexical[0]?.content, 'post build incremental evidence');

  const builtRecoveryRecordPath = join(
    config.generationRootDirectory,
    builtGenerationId,
    'write-recovery.json',
  );
  const interruptionStages = [
    'after-recovery-record',
    'after-dense-write',
    'after-logical-projection-write',
    'after-physical-projection-write',
    'after-store-close',
    'after-reopened-verification',
    'after-recovery-clear',
    'after-marker-acknowledgement',
  ] as const;
  let interruptionParentEntryId = 'post-build-entry';
  for (const [index, interruptionStage] of interruptionStages.entries()) {
    const entryId = `interruption-entry-${index}`;
    await appendFile(
      physicalSessionPath,
      `${JSON.stringify({
        type: 'message',
        id: entryId,
        parentId: interruptionParentEntryId,
        timestamp: `2026-08-04T00:03:${String(index).padStart(2, '0')}.000Z`,
        message: { role: 'assistant', content: `interruption evidence ${index}` },
      })}\n`,
    );
    interruptionParentEntryId = entryId;
    await utimes(physicalSessionPath, 6 + index, 6 + index);
    const interruptedIdentity = {
      ...markerIdentity,
      runtimeSequence: 6 + index,
      createdAtEpochMilliseconds: (6 + index) * 1_000,
      trigger: {
        kind: RecallWorkMarkerTrigger.DEPARTURE,
        logicalSessionId: physicalSessionId,
        leafEntryId: entryId,
      },
    } as const;
    const stagedMarker: RecallWorkMarker = {
      ...interruptedIdentity,
      markerId: createRecallWorkMarkerId(interruptedIdentity),
    };
    const stagedMarkerPath = join(config.markerSpoolDirectory, `${stagedMarker.markerId}.json`);
    await writeFile(stagedMarkerPath, '{}\n');
    const stagedWorkPlan = {
      targetGenerationId: builtGenerationId,
      markerSpoolDirectory: config.markerSpoolDirectory,
      discoveredMarkerCount: 1,
      sourceMarkerIds: [stagedMarker.markerId],
      workItems: [{ marker: stagedMarker, coveredMarkerIds: [stagedMarker.markerId] }],
      quarantineDiagnostics: [],
    };
    rejectedTransferStage = interruptionStage;
    await assert.rejects(() => service.transferIncrementalRecallWorkPlan(stagedWorkPlan));
    rejectedTransferStage = null;
    const recoveryExistsAfterFault = ![
      'after-recovery-clear',
      'after-marker-acknowledgement',
    ].includes(interruptionStage);
    if (recoveryExistsAfterFault) {
      await access(builtRecoveryRecordPath);
    } else {
      await assert.rejects(() => access(builtRecoveryRecordPath), { code: 'ENOENT' });
    }
    if (interruptionStage === 'after-marker-acknowledgement') {
      await assert.rejects(() => access(stagedMarkerPath), { code: 'ENOENT' });
    } else {
      await access(stagedMarkerPath);
      try {
        await service.transferIncrementalRecallWorkPlan(stagedWorkPlan);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`fixture recovery failed after ${interruptionStage}: ${message}`, {
          cause: error,
        });
      }
      await assert.rejects(() => access(stagedMarkerPath), { code: 'ENOENT' });
    }
    await assert.rejects(() => access(builtRecoveryRecordPath), { code: 'ENOENT' });
  }

  const builtPhysicalSourceIdentity = resolveRecallPhysicalSourceIdentity(
    config.sessionsDirectory,
    physicalSessionPath,
  ).physicalSourceIdentity;
  const interruptedDeletion = {
    confirmedPhysicalSourceDeletion: {
      targetGenerationId: builtGenerationId,
      physicalSourceIdentity: builtPhysicalSourceIdentity,
    },
  };
  rejectedTransferStage = 'after-dense-delete';
  await assert.rejects(
    () => service.transferIncrementalRecallWorkPlan(interruptedDeletion),
    /Recall target incremental deletion or close failed/u,
  );
  await access(builtRecoveryRecordPath);
  rejectedTransferStage = null;
  await service.transferIncrementalRecallWorkPlan(interruptedDeletion);
  await assert.rejects(() => access(builtRecoveryRecordPath), { code: 'ENOENT' });
  for (const storeDirectory of ['lexical-source', 'dense', 'session-projections']) {
    const collection = ZVecOpen(
      join(config.generationRootDirectory, builtGenerationId, storeDirectory),
      { readOnly: true },
    );
    try {
      assert.equal(collection.stats.docCount, 0);
    } finally {
      collection.closeSync();
    }
  }
});

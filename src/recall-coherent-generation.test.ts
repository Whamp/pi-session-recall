import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { RecallDiagnosticsMode, RecallGenerationCutoverState } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  createRecallActiveGenerationPointer,
  encodeRecallActiveGenerationPointer,
  RECALL_GENERATION_REGISTRY_VERSION,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

function createCoherentGenerationTestConfig(
  dataDirectory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    dataDirectory,
    databasePath: join(dataDirectory, 'legacy-zvec'),
    projectionDatabasePath: join(dataDirectory, 'legacy-session-projections'),
    statePath: join(dataDirectory, 'legacy-index-state.json'),
    manifestPath: join(dataDirectory, 'legacy-index-manifest.json'),
    tokenizerCacheDirectory: join(dataDirectory, 'tokenizers'),
    embeddingCacheDirectory: join(dataDirectory, 'legacy-embedding-cache'),
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
    embeddingModel: 'test-embedding-model',
    embeddingServedModelId: 'test/embedding-model',
    embeddingArtifact: 'test-embedding-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker-model',
    projectLineages: normalizeRecallProjectLineages({}),
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
}

void test('configured service creates, reopens, and deletes an empty coherent recall generation', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-empty-coherent-generation-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory)]);
  const config = createCoherentGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });

  const generationId = 'generation_empty_coherent';
  const created = await service.createEmptyRecallGeneration({ generationId });
  assert.deepEqual(created.storeCounts, {
    lexicalSource: 0,
    dense: 0,
    sessionProjection: 0,
  });
  assert.equal(created.generationId, generationId);
  assert.equal(existsSync(config.activeGenerationPointerPath), false);

  const manifest: unknown = JSON.parse(await readFile(created.manifestPath, 'utf8'));
  assert.ok(isUnknownRecord(manifest));
  assert.equal(manifest.generationId, generationId);
  assert.equal(manifest.generationFormatVersion, 1);
  assert.deepEqual(Object.keys(manifest).toSorted(), [
    'chunkPolicy',
    'embeddingProfile',
    'generationFormatVersion',
    'generationId',
    'importPolicy',
    'projectIdentityPolicy',
    'provenancePolicy',
    'sourceAnchorPolicy',
    'stores',
    'validationPolicy',
  ]);

  const generationDirectory = join(config.generationRootDirectory, generationId);
  const lexicalSource = ZVecOpen(join(generationDirectory, 'lexical-source'), { readOnly: true });
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  const sessionProjection = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(lexicalSource.schema.vectors().length, 0);
    assert.equal(dense.schema.vectors().length, 1);
    assert.equal(sessionProjection.schema.vectors().length, 0);
    assert.equal(lexicalSource.stats.docCount, 0);
    assert.equal(dense.stats.docCount, 0);
    assert.equal(sessionProjection.stats.docCount, 0);
    assert.notEqual(lexicalSource.schema.name, dense.schema.name);
    assert.notEqual(dense.schema.name, sessionProjection.schema.name);
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
    sessionProjection.closeSync();
  }

  await rm(sessionsDirectory, { recursive: true });
  const reopened = await service.openValidatedRecallGeneration(generationId);
  assert.deepEqual(reopened, created);
  assert.equal(existsSync(join(generationDirectory, 'validation-receipt.json')), true);

  await service.deleteUnprotectedRecallGeneration(generationId);
  assert.equal(existsSync(generationDirectory), false);
});

void test('validated generation open and deletion fail closed for every incoherent or protected state', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-coherent-generation-faults-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory)]);
  const config = createCoherentGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });

  const missingManifest = await service.createEmptyRecallGeneration({
    generationId: 'generation_missing_manifest',
  });
  await rm(missingManifest.manifestPath);
  await assert.rejects(
    service.openValidatedRecallGeneration(missingManifest.generationId),
    /Recall coherent generation manifest unreadable/u,
  );

  const corruptManifest = await service.createEmptyRecallGeneration({
    generationId: 'generation_corrupt_manifest',
  });
  await writeFile(corruptManifest.manifestPath, '{not-json');
  await assert.rejects(
    service.openValidatedRecallGeneration(corruptManifest.generationId),
    /Recall coherent generation manifest invalid/u,
  );

  const incompatibleManifest = await service.createEmptyRecallGeneration({
    generationId: 'generation_incompatible_manifest',
  });
  const incompatibleManifestValue: unknown = JSON.parse(
    await readFile(incompatibleManifest.manifestPath, 'utf8'),
  );
  assert.ok(isUnknownRecord(incompatibleManifestValue));
  incompatibleManifestValue.generationFormatVersion = 2;
  await writeFile(
    incompatibleManifest.manifestPath,
    `${JSON.stringify(incompatibleManifestValue, null, 2)}\n`,
  );
  await assert.rejects(
    service.openValidatedRecallGeneration(incompatibleManifest.generationId),
    /Recall coherent generation manifest invalid/u,
  );

  const missingReceipt = await service.createEmptyRecallGeneration({
    generationId: 'generation_missing_receipt',
  });
  await rm(missingReceipt.validationReceiptPath);
  await assert.rejects(
    service.openValidatedRecallGeneration(missingReceipt.generationId),
    /Recall coherent generation validation receipt unreadable/u,
  );

  const corruptReceipt = await service.createEmptyRecallGeneration({
    generationId: 'generation_corrupt_receipt',
  });
  await writeFile(corruptReceipt.validationReceiptPath, '{not-json');
  await assert.rejects(
    service.openValidatedRecallGeneration(corruptReceipt.generationId),
    /Recall coherent generation validation receipt invalid/u,
  );

  const missingLexicalSource = await service.createEmptyRecallGeneration({
    generationId: 'generation_missing_lexical_source',
  });
  await rm(join(missingLexicalSource.generationDirectory, 'lexical-source'), {
    recursive: true,
  });
  await assert.rejects(
    service.openValidatedRecallGeneration(missingLexicalSource.generationId),
    /Recall coherent generation lexical-source store missing/u,
  );

  const corruptDense = await service.createEmptyRecallGeneration({
    generationId: 'generation_corrupt_dense',
  });
  const corruptDensePath = join(corruptDense.generationDirectory, 'dense');
  await rm(corruptDensePath, { recursive: true });
  await writeFile(corruptDensePath, 'not a zvec collection');
  await assert.rejects(
    service.openValidatedRecallGeneration(corruptDense.generationId),
    /Recall coherent generation dense-evidence store open failed/u,
  );

  const unexpectedProjection = await service.createEmptyRecallGeneration({
    generationId: 'generation_unexpected_projection',
  });
  const projectionStore = ZVecOpen(
    join(unexpectedProjection.generationDirectory, 'session-projections'),
  );
  try {
    const [status] = projectionStore.upsertSync([
      {
        id: 'unexpected_projection',
        fields: {
          schemaVersion: 1,
          generationId: unexpectedProjection.generationId,
          projectionKind: 'physical-session',
          physicalSourceIdentity: 'unexpected-source',
          logicalSessionOccurrenceId: '',
          projectionJson: '{}',
        },
      },
    ]);
    assert.equal(status?.ok, true);
  } finally {
    projectionStore.closeSync();
  }
  await assert.rejects(
    service.openValidatedRecallGeneration(unexpectedProjection.generationId),
    /Recall coherent generation session-projection membership mismatch: expected 0 rows, received 1/u,
  );

  const crossGenerationLeft = await service.createEmptyRecallGeneration({
    generationId: 'generation_cross_left',
  });
  const crossGenerationRight = await service.createEmptyRecallGeneration({
    generationId: 'generation_cross_right',
  });
  const leftLexicalSourcePath = join(crossGenerationLeft.generationDirectory, 'lexical-source');
  const rightLexicalSourcePath = join(crossGenerationRight.generationDirectory, 'lexical-source');
  assert.notEqual(leftLexicalSourcePath, rightLexicalSourcePath);
  await rm(leftLexicalSourcePath, { recursive: true });
  await cp(rightLexicalSourcePath, leftLexicalSourcePath, { recursive: true });
  await assert.rejects(
    service.openValidatedRecallGeneration(crossGenerationLeft.generationId),
    /Recall coherent generation lexical-source identity mismatch/u,
  );

  const recoveryRequired = await service.createEmptyRecallGeneration({
    generationId: 'generation_recovery_required',
  });
  await writeFile(
    join(recoveryRequired.generationDirectory, 'write-recovery.json'),
    '{"version":1}\n',
  );
  await assert.rejects(
    service.openValidatedRecallGeneration(recoveryRequired.generationId),
    /Recall coherent generation recovery required/u,
  );

  const registryProtected = await service.createEmptyRecallGeneration({
    generationId: 'generation_registry_protected',
  });
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: null,
    buildingGenerationId: registryProtected.generationId,
    rollbackGenerationId: null,
    activePointerChecksum: null,
    generations: [
      {
        generationId: registryProtected.generationId,
        state: RecallGenerationCutoverState.BUILDING,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: registryProtected.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 1,
        rebuildStartMarkerId: null,
      },
    ],
  });
  await assert.rejects(
    service.deleteUnprotectedRecallGeneration(registryProtected.generationId),
    /Recall coherent generation deletion refused for registry-protected generation/u,
  );
  assert.equal(existsSync(registryProtected.generationDirectory), true);
  await rm(config.generationRegistryPath);

  const pointerProtected = await service.createEmptyRecallGeneration({
    generationId: 'generation_pointer_protected',
  });
  await writeFile(
    config.activeGenerationPointerPath,
    encodeRecallActiveGenerationPointer(
      createRecallActiveGenerationPointer(pointerProtected.generationId),
    ),
  );
  await assert.rejects(
    service.deleteUnprotectedRecallGeneration(pointerProtected.generationId),
    /Recall coherent generation deletion refused for protected active generation/u,
  );
  assert.equal(existsSync(pointerProtected.generationDirectory), true);
});

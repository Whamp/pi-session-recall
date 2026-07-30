import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { createPiRecallToolResponse, executePiRecallRequest } from './recall-extension.js';
import {
  RecallBacklogFailureCategory,
  RecallDiagnosticsMode,
  RecallGenerationCutoverState,
  RecallRankedListSource,
  RecallSearchScope,
} from './enums.js';
import {
  createRecallActiveGenerationPointer,
  RECALL_BACKLOG_SUMMARY_VERSION,
  RECALL_GENERATION_REGISTRY_VERSION,
  writeRecallActiveGenerationPointer,
  writeRecallBacklogSummary,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
import {
  createRecallIndexManifest,
  type RecallTokenizerManifestIdentity,
} from './recall-index-manifest.js';
import {
  createOctenEmbeddingModelProfile,
  createRecallEmbeddingProfileIdentity,
  type RecallEmbeddingModelProfile,
} from './recall-model-profiles.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: text.trim() ? Array.from(text.trim().split(/\s+/u).keys(), (index) => index + 1) : [],
    };
  },
};

function createTargetSearchTestConfig(
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
    embeddingModel: 'fixture-native-model',
    embeddingServedModelId: 'fixture/native-model',
    embeddingArtifact: 'fixture-native-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'fixture-reranker',
    projectLineages: normalizeRecallProjectLineages({}),
    chunkPolicy: { maxTokens: 4, overlapTokens: 1 },
    searchCandidateLimits: { dense: 1, lexical: 1, identifier: 1 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
}

function createTargetSearchEmbeddingProfile(): RecallEmbeddingModelProfile {
  const profile = createOctenEmbeddingModelProfile(
    {
      requestModel: 'fixture-native-model',
      servedModelId: 'fixture/native-model',
      artifact: 'fixture-native-model.fp32',
      artifactSha256: 'a'.repeat(64),
      dimensions: 3,
      quantization: 'fp32',
      pooling: 'last',
      normalization: 'l2',
    },
    2,
  );
  return Object.freeze({
    ...profile,
    canary: Object.freeze({
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'fixture target search canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    }),
  });
}

async function writeSession(
  path: string,
  sessionId: string,
  cwd: string,
  entryId: string,
  content: string,
): Promise<void> {
  await writeFile(
    path,
    `${[
      {
        type: 'session',
        version: 3,
        id: sessionId,
        timestamp: '2026-08-03T00:00:00.000Z',
        cwd,
      },
      {
        type: 'message',
        id: entryId,
        parentId: null,
        timestamp: '2026-08-03T00:00:01.000Z',
        message: { role: 'assistant', content },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );
}

void test('configured service refuses legacy active storage without opening it', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-target-only-search-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory)]);

  const config = createTargetSearchTestConfig(dataDirectory, sessionsDirectory);
  const profile = createTargetSearchEmbeddingProfile();
  const generationId = 'generation_legacy_storage_refused';
  const generationDirectory = join(config.generationRootDirectory, generationId);
  await mkdir(generationDirectory, { recursive: true });
  const tokenizerIdentity: RecallTokenizerManifestIdentity = {
    model: 'fixture-tokenizer',
    revision: 'fixture-revision',
    library: { name: 'fixture-tokenizer', version: '1' },
    encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
    assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'b'.repeat(64) }],
  };
  const currentManifest = createRecallIndexManifest({
    embeddingIdentity: profile.identity,
    ...(profile.canary ? { embeddingCanary: profile.canary } : {}),
    canaryEmbedding: [1, 0, 0],
    tokenizerIdentity,
    ...(config.chunkPolicy ? { chunkPolicy: config.chunkPolicy } : {}),
    projectLineages: config.projectLineages,
  });
  const legacyManifest = Object.fromEntries(
    Object.entries(currentManifest).filter(
      ([key]) => key !== 'markerSchemaVersion' && key !== 'sessionProjectionSchemaVersion',
    ),
  );
  await writeFile(
    join(generationDirectory, 'index-manifest.json'),
    `${JSON.stringify({ ...legacyManifest, manifestVersion: 5 })}\n`,
  );
  const pointer = createRecallActiveGenerationPointer(generationId);
  await writeFile(
    config.generationRegistryPath,
    `${JSON.stringify({
      version: RECALL_GENERATION_REGISTRY_VERSION,
      activeGenerationId: generationId,
      buildingGenerationId: null,
      rollbackGenerationId: null,
      activePointerChecksum: pointer.checksum,
      generations: [
        {
          generationId,
          state: 'legacy_read_only',
          indexManifestVersion: 5,
          markerSchemaVersion: null,
          sessionProjectionSchemaVersion: null,
          indexManifestFingerprint: 'a'.repeat(64),
          rebuildStartedAtEpochMilliseconds: 1,
          stateChangedAtEpochMilliseconds: 1,
          rebuildStartMarkerId: null,
          validatedAtEpochMilliseconds: 1,
        },
      ],
    })}\n`,
  );
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);

  let legacyStoreOpenCount = 0;
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        return documents.map(() => [1, 0, 0]);
      },
      async embedQuery() {
        return [1, 0, 0];
      },
    },
    tokenizerIdentity,
    loadTokenizer: async () => tokenizer,
    openStore() {
      legacyStoreOpenCount += 1;
      throw new Error('Legacy recall store opener invoked');
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  await assert.rejects(service.readOperatorStatus(), /Recall generation registry invalid/u);
  await assert.rejects(
    service.search('target only', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall generation registry invalid/u,
  );
  assert.equal(legacyStoreOpenCount, 0);
});

void test('configured service serves every existing search mode from the active validated target generation', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-active-target-search-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const selectedProjectDirectory = join(disposableRoot, 'selected-project');
  const unrelatedProjectDirectory = join(disposableRoot, 'unrelated-project');
  await Promise.all([
    mkdir(sessionsDirectory),
    mkdir(dataDirectory),
    mkdir(selectedProjectDirectory),
    mkdir(unrelatedProjectDirectory),
  ]);
  const selectedSourcePath = join(sessionsDirectory, 'selected.jsonl');
  const unrelatedSourcePath = join(sessionsDirectory, 'unrelated.jsonl');
  await Promise.all([
    writeSession(
      selectedSourcePath,
      'selected-session',
      selectedProjectDirectory,
      'selected-entry',
      'TargetModeNeedle alpha beta gamma delta epsilon retained decision',
    ),
    writeSession(
      unrelatedSourcePath,
      'unrelated-session',
      unrelatedProjectDirectory,
      'unrelated-entry',
      'TargetModeNeedle alpha beta gamma delta epsilon unrelated decision',
    ),
  ]);

  const profile = createTargetSearchEmbeddingProfile();
  const queryInputs: string[] = [];
  const warnings: string[] = [];
  const config = createTargetSearchTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        return documents.map((document) =>
          document.includes('retained decision') ? [4, 0, 100] : [0, 4, 100],
        );
      },
      async embedQuery(query) {
        queryInputs.push(query);
        return [5, 0, 200];
      },
    },
    tokenizerIdentity: {
      model: 'fixture-tokenizer',
      revision: 'fixture-revision',
      library: { name: 'fixture-tokenizer', version: '1' },
      encodeOptions: { addSpecialTokens: false, returnTokenTypeIds: false },
      assets: [{ fileName: 'fixture-tokenizer.json', sha256: 'b'.repeat(64) }],
    },
    loadTokenizer: async () => tokenizer,
    rerankingProfile: {
      profileId: 'fixture-reranker-profile',
      model: 'fixture-reranker',
      scorePolicy: 'higher-is-better-v1',
      scoreMeaning: 'higher-is-more-relevant',
      scoreRange: { minimum: 0, maximum: 1 },
    },
    reranker: {
      async rerankDocuments(query, documents) {
        void query;
        return documents.map((document) => (document.includes('retained decision') ? 0.9 : 0.1));
      },
    },
    notifyWarning(message) {
      warnings.push(message);
    },
    workerSignal: { signalDetachedWorker() {} },
  });

  const generationId = 'generation_active_target_search';
  const opened = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [selectedSourcePath, unrelatedSourcePath],
  });
  const pointer = createRecallActiveGenerationPointer(generationId);
  const targetEmbeddingProfileId = createRecallEmbeddingProfileIdentity(profile);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: 'generation_replacement_building',
    rollbackGenerationId: null,
    activePointerChecksum: pointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: targetEmbeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: opened.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
      {
        generationId: 'generation_replacement_building',
        state: RecallGenerationCutoverState.BUILDING,
        embeddingProfileId: targetEmbeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'c'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 3,
        stateChangedAtEpochMilliseconds: 3,
        rebuildStartMarkerId: null,
      },
    ],
  });
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);
  await writeRecallBacklogSummary(config.backlogSummaryPath, {
    version: RECALL_BACKLOG_SUMMARY_VERSION,
    pendingEligibleSessionCount: 3,
    oldestEligibleMarkerAgeMilliseconds: 5_000,
    activeGenerationId: generationId,
    buildingGenerationId: 'generation_replacement_building',
    generationState: RecallGenerationCutoverState.ACTIVE,
    activeGenerationAgeMilliseconds: 10_000,
    rebuildAgeMilliseconds: 2_000,
    lastFailureCategory: RecallBacklogFailureCategory.WRITE_FAILED,
    observedAtEpochMilliseconds: 20_000,
  });
  await rm(sessionsDirectory, { recursive: true });

  const hybrid = await service.search('TargetModeNeedle', 5, {
    scope: RecallSearchScope.PROJECT,
    invocationDirectory: selectedProjectDirectory,
    mode: 'hybrid',
  });
  assert.ok(hybrid.results.length > 0);
  assert.ok(hybrid.results.every(({ sessionId }) => sessionId.value === 'selected-session'));
  assert.ok(hybrid.results.every(({ id }) => id.startsWith('occurrence_')));
  assert.ok(hybrid.results.some(({ identifier }) => identifier !== null));
  assert.ok(hybrid.results.some(({ neighborContext }) => neighborContext !== null));
  assert.equal(hybrid.searchPolicy.scope, RecallSearchScope.PROJECT);
  assert.deepEqual(hybrid.searchPolicy.candidateLimits, { dense: 1, lexical: 1, identifier: 1 });

  const modelSearch = await executePiRecallRequest(
    service,
    { query: 'TargetModeNeedle', limit: 5 },
    { cwd: selectedProjectDirectory },
    5,
  );
  assert.equal(modelSearch.operation, 'search');
  if (modelSearch.operation !== 'search') {
    throw new Error('Target generation model-facing search returned the wrong operation');
  }
  assert.ok(modelSearch.search.results.every(({ id }) => id.startsWith('occurrence_')));
  const modelSearchResponse = createPiRecallToolResponse(modelSearch);
  assert.match(modelSearchResponse.text, /Evidence occurrence ID: occurrence_/u);
  assert.ok('sources' in modelSearchResponse.details);
  assert.ok(
    'sources' in modelSearchResponse.details && modelSearchResponse.details.sources.length > 0,
  );

  const anchorOccurrenceId = modelSearch.search.results[0]?.id;
  assert.ok(anchorOccurrenceId);
  const modelExpansion = await executePiRecallRequest(
    service,
    {
      expandSourceNeighborhood: {
        evidenceOccurrenceId: anchorOccurrenceId,
        previousEntryCount: 0,
        nextEntryCount: 0,
      },
    },
    { cwd: selectedProjectDirectory },
    5,
  );
  assert.equal(modelExpansion.operation, 'expansion');
  const modelExpansionResponse = createPiRecallToolResponse(modelExpansion);
  assert.match(modelExpansionResponse.text, /TargetModeNeedle/u);

  const deepRerank = await service.search('retained decision', 5, {
    scope: RecallSearchScope.GLOBAL,
    mode: 'deep-rerank',
    intent: 'recover the retained choice',
  });
  assert.equal(deepRerank.searchPolicy.rankingMode, 'deep-rerank');
  assert.ok(deepRerank.results.some(({ rerankerScore }) => rerankerScore !== null));

  const queryPlanned = await service.search('retained decision', 5, {
    scope: RecallSearchScope.GLOBAL,
    mode: 'query-planned',
    plan: [
      { type: 'lex', query: 'TargetModeNeedle' },
      { type: 'vec', query: 'retained architecture choice' },
      { type: 'hyde', query: 'The retained decision used the target generation.' },
    ],
    intent: 'recover the retained choice',
  });
  assert.equal(queryPlanned.searchPolicy.rankingMode, 'query-planned');
  assert.deepEqual(
    queryPlanned.searchPolicy.queryPlan?.rankedLists.map(({ source }) => source),
    [
      RecallRankedListSource.DENSE,
      RecallRankedListSource.LEXICAL,
      RecallRankedListSource.IDENTIFIER,
      RecallRankedListSource.PLANNED_LEX,
      RecallRankedListSource.PLANNED_VEC,
      RecallRankedListSource.PLANNED_HYDE,
    ],
  );
  assert.ok(queryInputs.includes('retained architecture choice'));
  assert.equal(warnings.length, 4);
  assert.ok(warnings.every((warning) => warning.includes('pendingEligibleSessionCount=3')));
  assert.equal(existsSync(config.markerSpoolDirectory), false);

  const cancelled = new AbortController();
  cancelled.abort();
  await assert.rejects(
    service.search('cancelled target search', 1, {
      scope: RecallSearchScope.GLOBAL,
      signal: cancelled.signal,
    }),
    (error: unknown) => error instanceof DOMException && error.name === 'AbortError',
  );

  const recoveryRecordPath = join(opened.generationDirectory, 'write-recovery.json');
  await writeFile(recoveryRecordPath, '{"version":1,"state":"recovery_required"}\n');
  await assert.rejects(
    service.search('recovery target search', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall coherent generation recovery required/u,
  );
  await rm(recoveryRecordPath);

  const validationReceipt = await readFile(opened.validationReceiptPath, 'utf8');
  await writeFile(opened.validationReceiptPath, '{}\n');
  await assert.rejects(
    service.search('invalid receipt target search', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall coherent generation validation receipt invalid/u,
  );
  await writeFile(opened.validationReceiptPath, validationReceipt);

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
        embeddingProfileId: targetEmbeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'e'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
    ],
  });
  await assert.rejects(
    service.search('registry fingerprint target search', 1, {
      scope: RecallSearchScope.GLOBAL,
    }),
    /Recall target generation registry manifest fingerprint mismatch/u,
  );

  const otherGenerationId = 'generation_registry_selected_elsewhere';
  const otherPointer = createRecallActiveGenerationPointer(otherGenerationId);
  await writeRecallGenerationRegistry(config.generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: otherGenerationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: otherPointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.RETIRED,
        embeddingProfileId: targetEmbeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: opened.manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 30_000,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
      {
        generationId: otherGenerationId,
        state: RecallGenerationCutoverState.ACTIVE,
        embeddingProfileId: targetEmbeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: 'd'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 4,
        stateChangedAtEpochMilliseconds: 30_000,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 5,
      },
    ],
  });
  await assert.rejects(
    service.search('disagreed target search', 1, { scope: RecallSearchScope.GLOBAL }),
    /Recall target generation pointer and registry disagree/u,
  );
});

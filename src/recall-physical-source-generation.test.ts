import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { RecallDiagnosticsMode } from './enums.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import {
  createOctenEmbeddingModelProfile,
  type RecallEmbeddingModelProfile,
} from './recall-model-profiles.js';
import { resolveRecallPhysicalSourceIdentity } from './recall-source-identity.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const tokenizer: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return {
      ids: text
        .split(/\s+/u)
        .filter(Boolean)
        .map((_, index) => index),
    };
  },
};

function createPhysicalSourceGenerationTestConfig(
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

function createToolOnlyLogicalSession(
  rawSessionId: string,
  entryPrefix: string,
  cwd: string,
  searchableToken: string,
): Record<string, unknown>[] {
  const callId = `${entryPrefix}-call`;
  return [
    {
      type: 'session',
      version: 3,
      id: rawSessionId,
      timestamp: '2026-08-01T00:00:00.000Z',
      cwd,
    },
    {
      type: 'message',
      id: `${entryPrefix}-assistant`,
      parentId: null,
      timestamp: '2026-08-01T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [
          {
            type: 'toolCall',
            id: callId,
            name: 'read',
            arguments: { path: `/tmp/${searchableToken}.txt` },
          },
        ],
      },
    },
    {
      type: 'message',
      id: `${entryPrefix}-result`,
      parentId: `${entryPrefix}-assistant`,
      timestamp: '2026-08-01T00:00:02.000Z',
      message: {
        role: 'toolResult',
        toolCallId: callId,
        toolName: 'read',
        isError: false,
        content: [{ type: 'text', text: `${searchableToken} source evidence` }],
      },
    },
  ];
}

async function writeJsonl(
  path: string,
  records: readonly Record<string, unknown>[],
): Promise<void> {
  await writeFile(path, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`);
}

void test('configured service builds and searches a stored-width dense subset beside lexical-only evidence', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-dense-source-generation-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const sourcePath = join(sessionsDirectory, 'mixed.jsonl');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  await writeJsonl(sourcePath, [
    {
      type: 'session',
      version: 3,
      id: 'mixed-session',
      timestamp: '2026-08-02T00:00:00.000Z',
      cwd: projectDirectory,
    },
    {
      type: 'message',
      id: 'mixed-assistant',
      parentId: null,
      timestamp: '2026-08-02T00:00:01.000Z',
      message: {
        role: 'assistant',
        content: [
          { type: 'text', text: 'The dense constellation records the retained design decision.' },
          {
            type: 'toolCall',
            id: 'mixed-call',
            name: 'read',
            arguments: { path: '/tmp/LEXICAL_ONLY_NEEDLE.txt' },
          },
        ],
      },
    },
    {
      type: 'message',
      id: 'mixed-result',
      parentId: 'mixed-assistant',
      timestamp: '2026-08-02T00:00:02.000Z',
      message: {
        role: 'toolResult',
        toolCallId: 'mixed-call',
        toolName: 'read',
        isError: false,
        content: [{ type: 'text', text: 'LEXICAL_ONLY_NEEDLE source evidence' }],
      },
    },
  ]);

  const baseProfile = createOctenEmbeddingModelProfile(
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
  const profile: RecallEmbeddingModelProfile = Object.freeze({
    ...baseProfile,
    canary: Object.freeze({
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'fixture stored-width canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    }),
  });
  const documentInputs: string[] = [];
  const queryInputs: string[] = [];
  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        documentInputs.push(...documents);
        return documents.map((document) =>
          document.includes('dense constellation') ? [0, 4, 100] : [4, 0, 100],
        );
      },
      async embedQuery(query) {
        queryInputs.push(query);
        return [0, 5, 200];
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
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });

  const generationId = 'generation_stored_width';
  const created = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  assert.ok(created.storeCounts.dense > 0);
  assert.ok(documentInputs.some((input) => input.includes('dense constellation')));
  assert.ok(documentInputs.every((input) => !input.includes('LEXICAL_ONLY_NEEDLE')));
  assert.equal(existsSync(config.embeddingCacheDirectory), false);

  const generationDirectory = join(config.generationRootDirectory, generationId);
  const manifest: unknown = JSON.parse(
    await readFile(join(generationDirectory, 'index-manifest.json'), 'utf8'),
  );
  assert.ok(isUnknownRecord(manifest));
  const embeddingProfile: unknown = manifest.embeddingProfile;
  assert.ok(isUnknownRecord(embeddingProfile));
  assert.equal(embeddingProfile.nativeDimensions, 3);
  assert.equal(embeddingProfile.storedDimensions, 2);
  assert.equal(embeddingProfile.reduction, 'first-n-then-l2');
  assert.ok(isUnknownRecord(embeddingProfile.canary));
  assert.equal(embeddingProfile.canary.expectedNativeDimensions, 3);
  assert.equal(embeddingProfile.canary.expectedStoredDimensions, 2);

  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  try {
    assert.equal(dense.schema.vectors()[0]?.dimension, 2);
    assert.equal(dense.stats.docCount, created.storeCounts.dense);
    const denseRows = await dense.query({
      topk: dense.stats.docCount,
      outputFields: [
        'evidenceOccurrenceId',
        'embeddingProfileId',
        'storedDimensions',
        'evidenceChecksum',
        'embeddingInputChecksum',
        'vectorChecksum',
      ],
      includeVector: true,
    });
    assert.ok(denseRows.every((row) => Object.keys(row.vectors.embedding ?? {}).length === 2));
    assert.ok(denseRows.every((row) => row.fields.storedDimensions === 2));
    assert.ok(denseRows.every((row) => row.fields.evidenceOccurrenceId === row.id));
    assert.ok(denseRows.every((row) => String(row.fields.evidenceChecksum).length === 64));
    assert.ok(denseRows.every((row) => String(row.fields.embeddingInputChecksum).length === 64));
    assert.ok(denseRows.every((row) => String(row.fields.vectorChecksum).length === 64));
  } finally {
    dense.closeSync();
  }

  const results = await service.searchRecallGenerationHybrid(
    generationId,
    'LEXICAL_ONLY_NEEDLE constellation',
    10,
  );
  assert.deepEqual(queryInputs, [
    'fixture stored-width canary',
    'fixture stored-width canary',
    'LEXICAL_ONLY_NEEDLE constellation',
  ]);
  assert.ok(results.some((result) => result.denseRank !== null));
  assert.ok(
    results.some(
      (result) => result.lexicalRank !== null && result.evidence.isDenseSearchable === false,
    ),
  );
  assert.ok(
    results.every((result) => result.evidence.evidenceOccurrenceId.startsWith('occurrence_')),
  );
  assert.ok(results.every((result) => result.evidence.sessionsRootRelativePath === 'mixed.jsonl'));

  const damagedDense = ZVecOpen(join(generationDirectory, 'dense'));
  try {
    const [row] = await damagedDense.query({
      topk: 1,
      outputFields: [
        'schemaVersion',
        'generationId',
        'evidenceOccurrenceId',
        'physicalSourceIdentity',
        'logicalSessionOccurrenceId',
        'embeddingProfileId',
        'storedDimensions',
        'evidenceChecksum',
        'embeddingInputChecksum',
        'vectorChecksum',
        'projectIdentity',
      ],
      includeVector: true,
    });
    assert.ok(row);
    const [status] = damagedDense.upsertSync([
      {
        id: row.id,
        fields: { ...row.fields, evidenceChecksum: '0'.repeat(64) },
        vectors: row.vectors,
      },
    ]);
    assert.equal(status?.ok, true);
  } finally {
    damagedDense.closeSync();
  }
  await assert.rejects(
    service.openValidatedRecallGeneration(generationId),
    /Recall coherent generation dense evidence checksum mismatch/u,
  );
});

void test('configured service keys lexical evidence, anchors, and projections by physical source identity', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-physical-source-generation-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const relocatedSessionsDirectory = join(disposableRoot, 'relocated-sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  const firstSourcePath = join(sessionsDirectory, 'team', 'session.jsonl');
  const reusedSourcePath = join(sessionsDirectory, 'archive', 'session.jsonl');
  await Promise.all([
    mkdir(join(sessionsDirectory, 'team'), { recursive: true }),
    mkdir(join(sessionsDirectory, 'archive'), { recursive: true }),
    mkdir(relocatedSessionsDirectory),
    mkdir(dataDirectory),
    mkdir(projectDirectory),
  ]);
  await writeJsonl(
    firstSourcePath,
    createToolOnlyLogicalSession('colliding-raw-session', 'first', projectDirectory, 'alpha_token'),
  );
  await writeJsonl(reusedSourcePath, [
    ...createToolOnlyLogicalSession(
      'colliding-raw-session',
      'reused-first',
      projectDirectory,
      'beta_token',
    ),
    ...createToolOnlyLogicalSession(
      'colliding-raw-session',
      'reused-second',
      projectDirectory,
      'gamma_token',
    ),
  ]);

  const config = createPhysicalSourceGenerationTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    loadTokenizer: async () => tokenizer,
    rerankingProfile: null,
    reranker: null,
    workerSignal: { signalDetachedWorker() {} },
  });
  const generationId = 'generation_physical_sources';
  const created = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [firstSourcePath, reusedSourcePath],
  });

  assert.deepEqual(created.storeCounts, {
    lexicalSource: 15,
    dense: 0,
    sessionProjection: 5,
  });
  assert.deepEqual(await service.openValidatedRecallGeneration(generationId), created);
  const firstSource = resolveRecallPhysicalSourceIdentity(sessionsDirectory, firstSourcePath);
  const reusedSource = resolveRecallPhysicalSourceIdentity(sessionsDirectory, reusedSourcePath);
  assert.notEqual(firstSource.physicalSourceIdentity, reusedSource.physicalSourceIdentity);
  assert.deepEqual(
    resolveRecallPhysicalSourceIdentity(
      relocatedSessionsDirectory,
      join(relocatedSessionsDirectory, 'team', 'session.jsonl'),
    ),
    firstSource,
  );

  const generationDirectory = join(config.generationRootDirectory, generationId);
  const recoveryRecordPath = join(generationDirectory, 'write-recovery.json');
  await writeFile(
    recoveryRecordPath,
    `${JSON.stringify({
      version: 1,
      generationId,
      operation: 'delete-physical-source',
      physicalSourceIdentity: firstSource.physicalSourceIdentity,
    })}\n`,
  );
  await assert.rejects(
    service.searchRecallGenerationLexical(generationId, 'alpha_token', 10),
    /Recall coherent generation recovery required/u,
  );
  await rm(recoveryRecordPath);

  const alphaMatches = await service.searchRecallGenerationLexical(generationId, 'alpha_token', 10);
  const betaMatches = await service.searchRecallGenerationLexical(generationId, 'beta_token', 10);
  const gammaMatches = await service.searchRecallGenerationLexical(generationId, 'gamma_token', 10);
  assert.ok(alphaMatches.length >= 1);
  assert.ok(betaMatches.length >= 1);
  assert.ok(gammaMatches.length >= 1);
  assert.ok(alphaMatches.every((match) => match.isDenseSearchable === false));
  assert.ok(alphaMatches.every((match) => match.rawSessionId === 'colliding-raw-session'));
  assert.ok(
    alphaMatches.every(
      (match) => match.physicalSourceIdentity === firstSource.physicalSourceIdentity,
    ),
  );
  assert.ok(
    betaMatches.every(
      (match) => match.physicalSourceIdentity === reusedSource.physicalSourceIdentity,
    ),
  );
  assert.notEqual(
    betaMatches[0]?.logicalSessionOccurrenceId,
    gammaMatches[0]?.logicalSessionOccurrenceId,
  );
  assert.ok(alphaMatches.every((match) => match.projectIdentity !== ''));
  assert.ok(alphaMatches.every((match) => match.sourceLineStart >= 2));
  assert.ok(alphaMatches.every((match) => match.evidenceOccurrenceId.startsWith('occurrence_')));

  const lexicalSource = ZVecOpen(join(generationDirectory, 'lexical-source'), { readOnly: true });
  const dense = ZVecOpen(join(generationDirectory, 'dense'), { readOnly: true });
  const projections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(lexicalSource.schema.vectors().length, 0);
    assert.equal(dense.stats.docCount, 0);
    assert.equal(dense.schema.vectors().length, 1);
    assert.equal(projections.stats.docCount, 5);
    const anchorRows = await lexicalSource.query({
      filter: `recordKind = 'entry-anchor' AND physicalSourceIdentity = '${firstSource.physicalSourceIdentity}'`,
      topk: 10,
      outputFields: [
        'entryAnchorId',
        'entryId',
        'parentEntryId',
        'sourceOrder',
        'entryStartByte',
        'entryEndByte',
        'branchPathLeafIds',
        'projectIdentity',
      ],
      includeVector: false,
    });
    assert.equal(anchorRows.length, 2);
    const resultAnchor = anchorRows.find((row) => row.fields.entryId === 'first-result');
    assert.ok(resultAnchor);
    assert.equal(resultAnchor.fields.parentEntryId, 'first-assistant');
    assert.equal(resultAnchor.fields.sourceOrder, 3);
    assert.ok(
      Number(resultAnchor.fields.entryEndByte) > Number(resultAnchor.fields.entryStartByte),
    );
    assert.deepEqual(resultAnchor.fields.branchPathLeafIds, ['first-result']);
    assert.notEqual(resultAnchor.fields.projectIdentity, '');

    const sourceProjectionRows = await projections.query({
      filter: `physicalSourceIdentity = '${reusedSource.physicalSourceIdentity}'`,
      topk: 10,
      outputFields: [
        'projectionKind',
        'physicalSourceIdentity',
        'logicalSessionOccurrenceId',
        'projectionJson',
      ],
      includeVector: false,
    });
    assert.equal(sourceProjectionRows.length, 3);
    assert.equal(
      sourceProjectionRows.filter((row) => row.fields.projectionKind === 'physical_session').length,
      1,
    );
    assert.equal(
      sourceProjectionRows.filter((row) => row.fields.projectionKind === 'logical_session').length,
      2,
    );
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
    projections.closeSync();
  }

  await service.deleteRecallGenerationPhysicalSource(
    generationId,
    firstSource.physicalSourceIdentity,
  );
  assert.deepEqual(
    await service.searchRecallGenerationLexical(generationId, 'alpha_token', 10),
    [],
  );
  assert.ok(
    (await service.searchRecallGenerationLexical(generationId, 'beta_token', 10)).length >= 1,
  );

  const afterDeletionLexical = ZVecOpen(join(generationDirectory, 'lexical-source'), {
    readOnly: true,
  });
  const afterDeletionProjections = ZVecOpen(join(generationDirectory, 'session-projections'), {
    readOnly: true,
  });
  try {
    assert.equal(afterDeletionLexical.stats.docCount, 10);
    assert.equal(afterDeletionProjections.stats.docCount, 3);
  } finally {
    afterDeletionLexical.closeSync();
    afterDeletionProjections.closeSync();
  }
});

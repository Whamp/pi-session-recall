import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { RecallDiagnosticsMode } from './enums.js';
import { createRecallConversationService } from './recall-conversation-service.js';
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

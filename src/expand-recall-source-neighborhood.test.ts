import assert from 'node:assert/strict';
import { copyFile, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { ZVecOpen } from '@zvec/zvec';

import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import { RecallDiagnosticsMode, RecallGenerationCutoverState } from './enums.js';
import {
  createRecallActiveGenerationPointer,
  RECALL_GENERATION_REGISTRY_VERSION,
  writeRecallActiveGenerationPointer,
  writeRecallGenerationRegistry,
} from './recall-generation-state.js';
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
      ids: Array.from(text.split(/\s+/u).filter(Boolean).keys(), (index) => index + 1),
    };
  },
};

function createExpansionTestConfig(
  dataDirectory: string,
  sessionsDirectory: string,
): RecallConversationConfig {
  return {
    sessionsDirectory,
    dataDirectory,
    databasePath: join(dataDirectory, 'unused-legacy-zvec'),
    projectionDatabasePath: join(dataDirectory, 'unused-legacy-projections'),
    statePath: join(dataDirectory, 'unused-legacy-state.json'),
    manifestPath: join(dataDirectory, 'unused-legacy-manifest.json'),
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
    chunkPolicy: { maxTokens: 6, overlapTokens: 1 },
    searchCandidateLimits: { dense: 8, lexical: 8, identifier: 8 },
    searchWriteWindowWaitMilliseconds: 500,
    confirmedDeletionMaxMissingSourceCount: 1,
    confirmedDeletionMaxMissingSourceRatio: 0.1,
  };
}

function createExpansionTestProfile(): RecallEmbeddingModelProfile {
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
      query: 'fixture source neighborhood canary',
      expectedDimensions: 3,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    }),
  });
}

async function activateExpansionTestGeneration(
  config: RecallConversationConfig,
  generationId: string,
  embeddingProfileId: string,
  manifestFingerprint: string,
): Promise<void> {
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
        embeddingProfileId,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: 3,
        indexManifestFingerprint: manifestFingerprint,
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 2,
        rebuildStartMarkerId: null,
        validatedAtEpochMilliseconds: 2,
      },
    ],
  });
  await writeRecallActiveGenerationPointer(config.activeGenerationPointerPath, pointer);
}

void test('configured service expands one exact active source neighborhood without source, inference, or projection reads', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-source-neighborhood-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));
  const sessionsDirectory = join(disposableRoot, 'sessions');
  const dataDirectory = join(disposableRoot, 'recall');
  const projectDirectory = join(disposableRoot, 'project');
  await Promise.all([mkdir(sessionsDirectory), mkdir(dataDirectory), mkdir(projectDirectory)]);
  const sourcePath = join(sessionsDirectory, 'neighborhood.jsonl');
  const replacementSourcePath = join(sessionsDirectory, 'replacement-neighborhood.jsonl');
  await writeFile(
    sourcePath,
    `${[
      {
        type: 'session',
        version: 3,
        id: 'logical-session',
        timestamp: '2026-08-04T00:00:00.000Z',
        cwd: projectDirectory,
      },
      {
        type: 'message',
        id: 'root-entry',
        parentId: null,
        timestamp: '2026-08-04T00:00:01.000Z',
        message: { role: 'user', content: 'root source message' },
      },
      {
        type: 'message',
        id: 'image-only-entry',
        parentId: 'root-entry',
        timestamp: '2026-08-04T00:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'image', data: 'excluded', mimeType: 'image/png' }],
        },
      },
      {
        type: 'message',
        id: 'anchor-entry',
        parentId: 'image-only-entry',
        timestamp: '2026-08-04T00:00:03.000Z',
        message: {
          role: 'assistant',
          content:
            'ANCHOR_NEEDLE alpha beta gamma delta epsilon zeta eta theta iota source conclusion',
        },
      },
      {
        type: 'message',
        id: 'tool-call-entry',
        parentId: 'anchor-entry',
        timestamp: '2026-08-04T00:00:04.000Z',
        message: {
          role: 'assistant',
          content: [
            {
              type: 'toolCall',
              id: 'call-1',
              name: 'read',
              arguments: { path: '/tmp/source.txt' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-result-entry',
        parentId: 'tool-call-entry',
        timestamp: '2026-08-04T00:00:05.000Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-1',
          toolName: 'read',
          isError: false,
          content: [{ type: 'text', text: 'source tool result' }],
        },
      },
      {
        type: 'message',
        id: 'branch-a-entry',
        parentId: 'tool-result-entry',
        timestamp: '2026-08-04T00:00:06.000Z',
        message: { role: 'assistant', content: 'branch A source evidence' },
      },
      {
        type: 'message',
        id: 'branch-b-entry',
        parentId: 'tool-result-entry',
        timestamp: '2026-08-04T00:00:07.000Z',
        message: { role: 'assistant', content: 'branch B source evidence' },
      },
      {
        type: 'branch_summary',
        id: 'branch-summary-entry',
        parentId: 'branch-b-entry',
        timestamp: '2026-08-04T00:00:08.000Z',
        fromId: 'branch-b-entry',
        summary: 'Branch summary source evidence.',
      },
      {
        type: 'message',
        id: 'bash-entry',
        parentId: 'branch-summary-entry',
        timestamp: '2026-08-04T00:00:09.000Z',
        message: {
          role: 'bashExecution',
          command: 'printf source-bash-command',
          output: 'source-bash-output',
          exitCode: 0,
          cancelled: false,
          truncated: false,
          excludeFromContext: true,
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n')}\n`,
  );

  await copyFile(sourcePath, replacementSourcePath);

  const profile = createExpansionTestProfile();
  let expansionStarted = false;
  let expansionQueryEmbeddingCount = 0;
  const config = createExpansionTestConfig(dataDirectory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: {
      async embedDocuments(documents) {
        return documents.map(() => [3, 4, 100]);
      },
      async embedQuery() {
        if (expansionStarted) {
          expansionQueryEmbeddingCount += 1;
          throw new Error('Source neighborhood expansion must not embed a query');
        }
        return [3, 4, 100];
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

  const generationId = 'generation_source_neighborhood';
  const opened = await service.createRecallGenerationFromPhysicalSources({
    generationId,
    physicalSessionPaths: [sourcePath],
  });
  const replacementGenerationId = 'generation_replacement_source_neighborhood';
  const replacementOpened = await service.createRecallGenerationFromPhysicalSources({
    generationId: replacementGenerationId,
    physicalSessionPaths: [replacementSourcePath],
  });
  const anchorSearch = await service.searchRecallGenerationLexical(
    generationId,
    'ANCHOR_NEEDLE',
    5,
  );
  const anchorOccurrence = anchorSearch.find(({ entryId }) => entryId === 'anchor-entry');
  assert.ok(anchorOccurrence);
  await activateExpansionTestGeneration(
    config,
    generationId,
    createRecallEmbeddingProfileIdentity(profile),
    opened.manifestFingerprint,
  );

  await Promise.all([
    rm(sourcePath),
    rm(replacementSourcePath),
    rm(join(config.generationRootDirectory, generationId, 'session-projections'), {
      recursive: true,
      force: true,
    }),
  ]);

  expansionStarted = true;
  const expansion = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
  });

  assert.equal(expansionQueryEmbeddingCount, 0);
  assert.equal(expansion.anchorEvidenceOccurrenceId, anchorOccurrence.evidenceOccurrenceId);
  assert.equal(expansion.logicalSessionOccurrenceId, anchorOccurrence.logicalSessionOccurrenceId);
  assert.equal(expansion.physicalSessionPath, sourcePath);
  assert.deepEqual(expansion.requestedEntryCounts, { previous: 2, next: 2 });
  assert.deepEqual(expansion.returnedEntryCounts, { previous: 2, next: 2 });
  assert.deepEqual(
    expansion.entries.map(({ entryId, placeholder }) => ({ entryId, placeholder })),
    [
      { entryId: 'root-entry', placeholder: false },
      { entryId: 'image-only-entry', placeholder: true },
      { entryId: 'anchor-entry', placeholder: false },
      { entryId: 'tool-call-entry', placeholder: false },
      { entryId: 'tool-result-entry', placeholder: false },
    ],
  );
  const anchorEvidence = expansion.entries[2]?.evidence;
  assert.equal(anchorEvidence?.length, 1);
  assert.equal(
    anchorEvidence?.[0]?.content,
    'ANCHOR_NEEDLE alpha beta gamma delta epsilon zeta eta theta iota source conclusion',
  );
  assert.ok((anchorEvidence?.[0]?.occurrences.length ?? 0) > 1);
  assert.ok(
    anchorEvidence?.[0]?.occurrences.every(
      ({ evidenceOccurrenceId }) => evidenceOccurrenceId.length > 0,
    ),
  );
  assert.deepEqual(
    expansion.entries[3]?.evidence.map(({ evidenceKind, evidencePart }) => ({
      evidenceKind,
      evidencePart,
    })),
    [
      { evidenceKind: 'tool_call', evidencePart: 'name' },
      { evidenceKind: 'tool_call', evidencePart: 'arguments' },
    ],
  );

  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
      previousEntryCount: 0,
      nextEntryCount: 3,
    }),
    /ambiguous.*bash-entry, branch-a-entry/,
  );
  const selectedBranch = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
    previousEntryCount: 0,
    nextEntryCount: 5,
    branchPathLeafEntryId: 'bash-entry',
  });
  assert.deepEqual(
    selectedBranch.entries.map(({ entryId }) => entryId),
    [
      'anchor-entry',
      'tool-call-entry',
      'tool-result-entry',
      'branch-b-entry',
      'branch-summary-entry',
      'bash-entry',
    ],
  );
  assert.equal(selectedBranch.entries[0]?.evidence[0]?.currentLeafEntryId, 'bash-entry');
  assert.deepEqual(selectedBranch.entries[0]?.evidence[0]?.compactedByEntryIds, []);
  assert.deepEqual(
    selectedBranch.entries[4]?.evidence.map(({ evidenceKind }) => evidenceKind),
    ['branch_summary'],
  );
  assert.deepEqual(
    selectedBranch.entries[5]?.evidence.map(({ evidenceKind, evidencePart }) => ({
      evidenceKind,
      evidencePart,
    })),
    [
      { evidenceKind: 'bash_execution', evidencePart: 'command' },
      { evidenceKind: 'bash_execution', evidencePart: 'output' },
    ],
  );
  const maximumRange = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
    previousEntryCount: 10,
    nextEntryCount: 10,
    branchPathLeafEntryId: 'bash-entry',
  });
  assert.deepEqual(maximumRange.returnedEntryCounts, { previous: 2, next: 5 });
  assert.equal(maximumRange.entries.length, 8);
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
      previousEntryCount: 0,
      nextEntryCount: 3,
      branchPathLeafEntryId: 'unrelated-leaf',
    }),
    /branch-path leaf entry ID does not contain anchor/,
  );
  const anchorOnly = await service.expandSourceNeighborhood({
    evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
    previousEntryCount: 0,
    nextEntryCount: 0,
  });
  assert.deepEqual(
    anchorOnly.entries.map(({ entryId }) => entryId),
    ['anchor-entry'],
  );
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: ` ${anchorOccurrence.evidenceOccurrenceId} `,
      previousEntryCount: 0,
      nextEntryCount: 0,
    }),
    /evidence occurrence ID not found in active generation/,
  );
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
      previousEntryCount: 11,
    }),
    /previous entry count invalid/,
  );

  const lexicalSourcePath = join(config.generationRootDirectory, generationId, 'lexical-source');
  const lexicalSource = ZVecOpen(lexicalSourcePath);
  try {
    const toolCallAnchorId = expansion.entries[3]?.entryAnchorId;
    assert.ok(toolCallAnchorId);
    const anchorDeletion = lexicalSource.deleteSync([toolCallAnchorId]);
    assert.equal(anchorDeletion[0]?.ok, true);
  } finally {
    lexicalSource.closeSync();
  }
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
      previousEntryCount: 0,
      nextEntryCount: 1,
    }),
    /incoherent entry anchor missing: tool-call-entry/,
  );

  const anchorChunkToDelete = anchorEvidence?.[0]?.occurrences[1]?.evidenceOccurrenceId;
  assert.ok(anchorChunkToDelete);
  const damagedLexicalSource = ZVecOpen(lexicalSourcePath);
  try {
    const evidenceDeletion = damagedLexicalSource.deleteSync([anchorChunkToDelete]);
    assert.equal(evidenceDeletion[0]?.ok, true);
  } finally {
    damagedLexicalSource.closeSync();
  }
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
      previousEntryCount: 0,
      nextEntryCount: 0,
    }),
    /incoherent named evidence occurrence missing/,
  );
  await activateExpansionTestGeneration(
    config,
    replacementGenerationId,
    createRecallEmbeddingProfileIdentity(profile),
    replacementOpened.manifestFingerprint,
  );
  await assert.rejects(
    service.expandSourceNeighborhood({
      evidenceOccurrenceId: anchorOccurrence.evidenceOccurrenceId,
    }),
    /evidence occurrence ID not found in active generation/,
  );
});

import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import type { LocalEmbeddingClient } from './local-embedding-client.js';
import type { LocalRerankerClient } from './local-reranker-client.js';
import {
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  RECALL_EMBEDDING_CANARY_TEXT,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';
import { createRecallConversationService as createProductionRecallConversationService } from './recall-conversation-service.js';
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

function createTestConfig(directory: string, sessionsDirectory: string) {
  return {
    sessionsDirectory,
    databasePath: join(directory, 'zvec'),
    statePath: join(directory, 'index-state.json'),
    manifestPath: join(directory, 'index-manifest.json'),
    tokenizerCacheDirectory: join(directory, 'tokenizers'),
    embeddingCacheDirectory: join(directory, 'embedding-cache'),
    lockPath: join(directory, 'recall.lock'),
    embeddingBaseUrl: 'http://unused.test/v1',
    embeddingModel: 'test-request-model',
    embeddingServedModelId: 'test-served-model',
    embeddingArtifact: 'test-model.fp32',
    embeddingQuantization: 'fp32',
    embeddingPooling: 'last',
    embeddingDimensions: 3,
    embeddingBatchSize: 8,
    rerankerBaseUrl: 'http://unused-reranker.test/v1',
    rerankerModel: 'test-reranker-model',
    searchCandidateLimits: { dense: 1, lexical: 1, identifier: 1 },
  };
}

const PRESERVE_FUSION_ORDER_RERANKER: LocalRerankerClient = {
  async rerankDocuments(query, documents) {
    void query;
    const scores: number[] = [];
    for (let index = 0; index < documents.length; index += 1) {
      scores.push(1 - index / (documents.length + 1));
    }
    return scores;
  },
};

function createRecallConversationService(
  config: Parameters<typeof createProductionRecallConversationService>[0],
  dependencies: NonNullable<Parameters<typeof createProductionRecallConversationService>[1]>,
) {
  return createProductionRecallConversationService(config, {
    reranker: PRESERVE_FUSION_ORDER_RERANKER,
    ...dependencies,
  });
}

void test('recall service indexes explicitly and searches a compatible index without updating it', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  const entries: object[] = [
    {
      type: 'session',
      version: 3,
      id: 'session-1',
      timestamp: '2026-07-24T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'queue-entry',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'We chose a durable queue for job delivery.' }],
      },
    },
    {
      type: 'message',
      id: 'ui-entry',
      parentId: 'queue-entry',
      timestamp: '2026-07-24T10:02:00Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'The navigation bar is blue.' }],
      },
    },
  ];
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');

  const embeddedInputs: string[] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddedInputs.push(...texts);
      return texts.map((text) => {
        if (text === RECALL_EMBEDDING_CANARY_TEXT) {
          return [0, 0, 1];
        }
        return text.toLowerCase().includes('queue') ? [1, 0, 0] : [0, 1, 0];
      });
    },
  };
  let tokenizerLoads = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
  });

  const indexed = await service.index();
  assert.equal(indexed.indexSummary.cacheHits, 0);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 2);
  assert.equal(indexed.indexSummary.embeddingRequestCount, 1);
  assert.equal(indexed.totalChunks, 2);

  const first = await service.search('What did we decide about job queues?', 1);
  assert.equal(first.results[0]?.entryId.value, 'queue-entry');
  assert.equal(first.results[0]?.sessionPath, sessionPath);
  assert.equal(first.totalChunks, 2);

  entries.push({
    type: 'message',
    id: 'unindexed-entry',
    parentId: 'ui-entry',
    timestamp: '2026-07-24T10:03:00Z',
    message: { role: 'assistant', content: 'This must not be indexed by search.' },
  });
  await writeFile(sessionPath, entries.map((entry) => JSON.stringify(entry)).join('\n') + '\n');
  const second = await service.search('queue decision', 1);
  assert.equal(second.totalChunks, 2);
  assert.equal(tokenizerLoads, 1);
  assert.deepEqual(embeddedInputs, [
    RECALL_EMBEDDING_CANARY_TEXT,
    'We chose a durable queue for job delivery.',
    'The navigation bar is blue.',
    'What did we decide about job queues?',
    'queue decision',
  ]);

  const lockOwner = `${JSON.stringify({ pid: 999_999_999 })}\n`;
  await mkdir(join(directory, 'recall.lock'));
  await writeFile(join(directory, 'recall.lock', 'owner.json'), lockOwner);
  await assert.rejects(
    () => service.search('must not clear a stale lock', 1),
    /read-only search did not remove the lock/,
  );
  assert.equal(await readFile(join(directory, 'recall.lock', 'owner.json'), 'utf8'), lockOwner);
});

void test('recall service fuses bounded dense, lexical, and identifier candidates with component scores', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-hybrid-search-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'hybrid.jsonl');
  const denseContent =
    'An append-only queue prevents jobs from disappearing after process restarts.';
  const identifierContent =
    'The process probe uses readNodeErrorCode() when permission checks fail with EPERM.';
  const quotedPhraseContent = 'The release marker contains alpha beta as one exact phrase.';
  const separatedPhraseContent =
    'The release marker contains alpha with unrelated words before beta.';
  await writeFile(
    sessionPath,
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'hybrid-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/hybrid-project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'dense-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'assistant', content: denseContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'identifier-entry',
        parentId: 'dense-entry',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'assistant', content: identifierContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'quoted-phrase-entry',
        parentId: 'identifier-entry',
        timestamp: '2026-07-24T10:03:00Z',
        message: { role: 'assistant', content: quotedPhraseContent },
      }),
      JSON.stringify({
        type: 'message',
        id: 'separated-phrase-entry',
        parentId: 'quoted-phrase-entry',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: separatedPhraseContent },
      }),
    ].join('\n') + '\n',
  );

  const semanticParaphraseQuery = 'How did we make background task delivery resilient?';
  const identifierQuery = 'readNodeErrorCode';
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          if (
            text === identifierContent ||
            text === quotedPhraseContent ||
            text === separatedPhraseContent
          ) {
            return [0, 1, 0];
          }
          return [1, 0, 0];
        });
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  const denseResult = await service.search(semanticParaphraseQuery, 1);
  assert.deepEqual(denseResult.searchPolicy, {
    rankFusionVersion: 1,
    reciprocalRankConstant: 60,
    rerankPolicyVersion: 1,
    rerankerModel: 'test-reranker-model',
    activeBranchPrior: 0.01,
    candidateLimits: { dense: 1, lexical: 1, identifier: 1 },
  });
  assert.equal(denseResult.results[0]?.entryId.value, 'dense-entry');
  assert.equal(denseResult.results[0]?.sessionPath, sessionPath);
  assert.deepEqual(denseResult.results[0]?.dense, { rank: 1, cosineDistance: 0 });
  assert.equal(denseResult.results[0]?.lexical, null);
  assert.equal(denseResult.results[0]?.identifier, null);
  assert.equal(denseResult.results[0]?.fusedScore, 0.01639344262295082);

  const exactIdentifier = await service.search(identifierQuery, 2);
  assert.equal(exactIdentifier.results[0]?.entryId.value, 'identifier-entry');
  assert.equal(exactIdentifier.results[0]?.sessionPath, sessionPath);
  assert.equal(exactIdentifier.results[0]?.dense, null);
  assert.equal(exactIdentifier.results[0]?.lexical?.rank, 1);
  assert.ok((exactIdentifier.results[0]?.lexical?.fullTextScore ?? 0) > 0);
  assert.equal(exactIdentifier.results[0]?.identifier?.rank, 1);
  assert.ok((exactIdentifier.results[0]?.identifier?.fullTextScore ?? 0) > 0);
  assert.equal(exactIdentifier.results[0]?.fusedScore, 0.03278688524590164);
  assert.equal(new Set(exactIdentifier.results.map((result) => result.id)).size, 2);

  const wrongCase = await service.search('readnodeerrorcode', 2);
  const wrongCaseIdentifier = wrongCase.results.find(
    (result) => result.entryId.value === 'identifier-entry',
  );
  assert.equal(wrongCaseIdentifier?.lexical?.rank, 1);
  assert.equal(wrongCaseIdentifier?.identifier, null);
  assert.equal(wrongCase.results[0]?.fusedScore, wrongCase.results[1]?.fusedScore);
  assert.deepEqual(
    wrongCase.results.map((result) => result.id),
    wrongCase.results.map((result) => result.id).toSorted(),
  );

  const quotedPhrase = await service.search('Where did we write "alpha beta"?', 2);
  assert.equal(quotedPhrase.results[0]?.entryId.value, 'quoted-phrase-entry');
  assert.equal(quotedPhrase.results[0]?.lexical?.rank, 1);
  assert.equal(quotedPhrase.results[0]?.identifier?.rank, 1);
  assert.ok(
    !quotedPhrase.results.some((result) => result.entryId.value === 'separated-phrase-entry'),
  );
});

void test('recall service reranks the full fused pool before applying the final result limit', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-reranked-pool-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const fusionFavorite = 'Exact fusion favorite evidence.';
  const rerankerFavorite = 'Semantically strongest Qwen evidence.';
  await writeFile(
    join(sessionsDirectory, 'reranked.jsonl'),
    [
      {
        type: 'session',
        version: 3,
        id: 'reranked-session',
        timestamp: '2026-07-25T10:00:00Z',
        cwd: '/reranked-project',
      },
      {
        type: 'message',
        id: 'fusion-favorite',
        parentId: null,
        timestamp: '2026-07-25T10:01:00Z',
        message: { role: 'assistant', content: fusionFavorite },
      },
      {
        type: 'message',
        id: 'reranker-favorite',
        parentId: 'fusion-favorite',
        timestamp: '2026-07-25T10:02:00Z',
        message: { role: 'assistant', content: rerankerFavorite },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const query = 'fusion favorite';
  const rerankerInputs: string[][] = [];
  const reranker: LocalRerankerClient = {
    async rerankDocuments(receivedQuery, documents) {
      assert.equal(receivedQuery, query);
      rerankerInputs.push([...documents]);
      return [0.1, 0.9];
    },
  };
  const config = {
    ...createTestConfig(directory, sessionsDirectory),
    searchCandidateLimits: { dense: 2, lexical: 2, identifier: 2 },
  };
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => {
          if (text === RECALL_EMBEDDING_CANARY_TEXT) {
            return [0, 0, 1];
          }
          if (text === rerankerFavorite) {
            return [0.9, 0.1, 0];
          }
          return [1, 0, 0];
        });
      },
    },
    reranker,
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  const search = await service.search(query, 1);

  assert.deepEqual(rerankerInputs, [[fusionFavorite, rerankerFavorite]]);
  assert.equal(search.results.length, 1);
  assert.equal(search.results[0]?.entryId.value, 'reranker-favorite');
  assert.equal(search.results[0]?.rerankerScore, 0.9);
  assert.equal(search.results[0]?.dense?.rank, 2);
  assert.ok(Number.isFinite(search.results[0]?.dense?.cosineDistance));
  assert.equal(search.results[0]?.lexical, null);
  assert.equal(search.results[0]?.identifier, null);
});

void test('recall service fails clearly when Qwen reranking is unavailable', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-reranker-failure-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  await writeFile(
    join(sessionsDirectory, 'failure.jsonl'),
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'reranker-failure-session',
        timestamp: '2026-07-25T10:00:00Z',
        cwd: '/failure-project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'failure-entry',
        parentId: null,
        timestamp: '2026-07-25T10:01:00Z',
        message: { role: 'assistant', content: 'Evidence that must not bypass reranking.' },
      }),
    ].join('\n') + '\n',
  );
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    reranker: {
      async rerankDocuments() {
        throw new Error(
          'Recall reranker request failed at http://reranker.test/v1/rerank: unavailable',
        );
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });
  await service.index();

  await assert.rejects(
    () => service.search('must use Qwen', 1),
    /Recall reranker request failed at http:\/\/reranker\.test\/v1\/rerank: unavailable/,
  );
});

void test('recall service retrieves context-dependent replies and reuses turn-context vectors', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-turn-context-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'turn-context.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'turn-context-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/release-project',
      },
      {
        type: 'message',
        id: 'user-request',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Ship release Atlas to edge nodes.' },
      },
      {
        type: 'message',
        id: 'assistant-call',
        parentId: 'user-request',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'private deployment plan' },
            {
              type: 'toolCall',
              id: 'call-release',
              name: 'read',
              arguments: { path: 'release.json' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-result',
        parentId: 'assistant-call',
        timestamp: '2026-07-24T10:03:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-release',
          toolName: 'read',
          content: [{ type: 'text', text: 'RAW_RELEASE_TOOL_OUTPUT' }],
          isError: false,
        },
      },
      {
        type: 'message',
        id: 'assistant-reply',
        parentId: 'tool-result',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: 'Yes, do it.' },
      },
      {
        type: 'message',
        id: 'next-user',
        parentId: 'assistant-reply',
        timestamp: '2026-07-24T10:05:00Z',
        message: { role: 'user', content: 'Report when deployment finishes.' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const embeddingInputs: string[][] = [];
  const query = 'Atlas Yes';
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddingInputs.push([...texts]);
      return texts.map((text) => {
        if (text === RECALL_EMBEDDING_CANARY_TEXT) {
          return [0, 0, 1];
        }
        if (text === query || text.startsWith('User:\nShip release Atlas')) {
          return [1, 0, 0];
        }
        return [0, 1, 0];
      });
    },
  };
  const config = createTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const indexed = await service.index();
  const recalled = await service.search(query, 1);
  const turnContext = recalled.results[0];

  assert.equal(indexed.totalChunks, 7);
  assert.equal(indexed.indexSummary.cacheHits, 0);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 4);
  assert.equal(indexed.indexSummary.embeddingRequestCount, 1);
  assert.equal(turnContext?.documentKind, 'turn_context');
  assert.equal(turnContext?.evidenceKind, 'turn_context');
  assert.equal(turnContext?.role, 'turn');
  assert.deepEqual(
    turnContext?.contributingEntryIds.map((id) => id.value),
    ['user-request', 'assistant-reply'],
  );
  assert.equal(turnContext?.dense?.rank, 1);
  assert.equal(turnContext?.lexical?.rank, 1);
  assert.equal(turnContext?.identifier?.rank, 1);
  assert.ok(turnContext?.content.includes('Ship release Atlas'));
  assert.ok(turnContext?.content.includes('Yes, do it.'));
  assert.ok(!turnContext?.content.includes('RAW_RELEASE_TOOL_OUTPUT'));
  assert.ok(!turnContext?.content.includes('private deployment plan'));

  const requestsBeforeRebuild = embeddingInputs.length;
  await rm(config.databasePath, { recursive: true });
  await rm(config.statePath);
  const rebuilt = await service.index();

  assert.equal(rebuilt.totalChunks, 7);
  assert.equal(rebuilt.indexSummary.cacheHits, 4);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(embeddingInputs.length, requestsBeforeRebuild);
});

void test('recall service fuses lexical-only tool evidence with dense conversation results', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-tool-evidence-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'tools.jsonl');
  const conversationContent = 'We diagnosed a file permission failure.';
  const toolCommand = 'cat /tmp/locked-file';
  const toolResult =
    'EPERM readNodeErrorCode /tmp/locked-file https://example.test/failure?id=EPERM';
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'tool-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/tool-project',
      },
      {
        type: 'message',
        id: 'assistant-tools',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: conversationContent },
            { type: 'thinking', thinking: 'never retrieve this private plan' },
            {
              type: 'toolCall',
              id: 'call-tools',
              name: 'bash',
              arguments: { command: toolCommand },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'result-tools',
        parentId: 'assistant-tools',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-tools',
          toolName: 'bash',
          content: [{ type: 'text', text: toolResult }],
          isError: true,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const embeddedInputs: string[] = [];
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts(texts) {
        embeddedInputs.push(...texts);
        return texts.map((text) => (text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [1, 0, 0]));
      },
    },
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const indexed = await service.index();
  const exactError = await service.search('EPERM readNodeErrorCode', 2);
  const toolEvidence = exactError.results.find((result) => result.documentKind === 'tool');
  const conversation = exactError.results.find((result) => result.documentKind === 'conversation');

  assert.equal(indexed.totalChunks, 4);
  assert.equal(indexed.indexSummary.newlyEmbeddedChunks, 1);
  assert.ok(toolEvidence);
  assert.equal(toolEvidence.entryId.value, 'result-tools');
  assert.equal(toolEvidence.evidenceKind, 'tool_result');
  assert.equal(toolEvidence.evidencePart, 'result');
  assert.equal(toolEvidence.toolCallId, 'call-tools');
  assert.equal(toolEvidence.toolName, 'bash');
  assert.equal(toolEvidence.toolCallEntryId?.value, 'assistant-tools');
  assert.equal(toolEvidence.toolResultEntryId?.value, 'result-tools');
  assert.equal(toolEvidence.toolError, true);
  assert.equal(toolEvidence.sessionPath, sessionPath);
  assert.equal(toolEvidence.dense, null);
  assert.equal(toolEvidence.lexical?.rank, 1);
  assert.equal(toolEvidence.identifier?.rank, 1);
  assert.ok(conversation?.dense);
  assert.equal(conversation?.documentKind, 'conversation');
  assert.deepEqual(embeddedInputs, [
    RECALL_EMBEDDING_CANARY_TEXT,
    conversationContent,
    'EPERM readNodeErrorCode',
  ]);

  const exactCommand = await service.search(toolCommand, 2);
  const callArguments = exactCommand.results.find((result) => result.evidencePart === 'arguments');
  assert.equal(callArguments?.content, `{"command":"${toolCommand}"}`);
  assert.equal(callArguments?.toolResultEntryId?.value, 'result-tools');
  assert.ok(exactCommand.results.every((result) => !result.content.includes('private plan')));

  const exactUrl = await service.search('https://example.test/failure?id=EPERM', 2);
  assert.equal(
    exactUrl.results.find((result) => result.evidencePart === 'result')?.content,
    toolResult,
  );
  const exactToolName = await service.search('bash', 2);
  assert.equal(
    exactToolName.results.find((result) => result.evidencePart === 'name')?.content,
    'bash',
  );
});

void test('fresh zvec rebuild reuses unchanged cached chunk vectors without embedding requests', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-cache-rebuild-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const sessionPath = join(sessionsDirectory, 'one.jsonl');
  await writeFile(
    sessionPath,
    [
      JSON.stringify({
        type: 'session',
        version: 3,
        id: 'session-1',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      }),
      JSON.stringify({
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Reuse this durable vector.' },
      }),
    ].join('\n') + '\n',
  );

  const embeddingInputs: string[][] = [];
  const embeddings: LocalEmbeddingClient = {
    async embedTexts(texts) {
      embeddingInputs.push([...texts]);
      return texts.map((text) =>
        text === RECALL_EMBEDDING_CANARY_TEXT ? [0, 0, 1] : [text.length, 1, 0],
      );
    },
  };
  const config = createTestConfig(directory, sessionsDirectory);
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      return tokenizer;
    },
  });

  const first = await service.index();
  assert.equal(first.indexSummary.cacheHits, 0);
  assert.equal(first.indexSummary.newlyEmbeddedChunks, 1);
  assert.equal(first.indexSummary.embeddingRequestCount, 1);
  assert.equal(embeddingInputs.length, 2);

  await rm(config.databasePath, { recursive: true });
  await rm(config.statePath);

  const rebuilt = await service.index();
  assert.equal(rebuilt.totalChunks, 1);
  assert.equal(rebuilt.indexSummary.cacheHits, 1);
  assert.equal(rebuilt.indexSummary.newlyEmbeddedChunks, 0);
  assert.equal(rebuilt.indexSummary.embeddingRequestCount, 0);
  assert.equal(embeddingInputs.length, 2);
});

void test('explicit indexing retries a transient embedding-canary failure in the same process', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-canary-retry-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let canaryRequests = 0;
  let tokenizerLoads = 0;
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings: {
      async embedTexts() {
        canaryRequests += 1;
        if (canaryRequests === 1) {
          throw new Error('temporary canary failure');
        }
        return [[0, 0, 1]];
      },
    },
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
  });

  await assert.rejects(() => service.index(), /temporary canary failure/);
  const retried = await service.index();

  assert.equal(retried.totalChunks, 0);
  assert.equal(canaryRequests, 2);
  assert.equal(tokenizerLoads, 1);
});

void test('recall search refuses a missing manifest before opening or mutating index state', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-missing-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  let embeddingRequests = 0;
  let tokenizerLoads = 0;
  let storeOpens = 0;
  const embeddings: LocalEmbeddingClient = {
    async embedTexts() {
      embeddingRequests += 1;
      return [[0, 0, 1]];
    },
  };
  const service = createRecallConversationService(createTestConfig(directory, sessionsDirectory), {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('store must not open without a manifest');
    },
  });

  await assert.rejects(
    () => service.search('must remain read only', 1),
    /Recall index manifest missing.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(embeddingRequests, 0);
  assert.equal(tokenizerLoads, 0);
  assert.equal(storeOpens, 0);
});

void test('recall search reports an incompatible manifest before opening zvec', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-incompatible-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const actualManifest = createRecallIndexManifest({
    embeddingIdentity: {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: 'mean',
    },
    canaryFingerprint: createRecallEmbeddingCanaryFingerprint([0, 0, 1], 3),
  });
  await writeRecallIndexManifest(config.manifestPath, actualManifest);
  let storeOpens = 0;
  let tokenizerLoads = 0;
  const embeddings: LocalEmbeddingClient = {
    async embedTexts() {
      return [[0, 0, 1]];
    },
  };
  const service = createRecallConversationService(config, {
    embeddings,
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('incompatible index must not open');
    },
  });

  await assert.rejects(
    () => service.search('incompatible', 1),
    /embedding\.pooling.*expected "last", received "mean".*\/pi-session-recall-index --rebuild/s,
  );
  assert.equal(storeOpens, 0);
  assert.equal(tokenizerLoads, 0);
});

void test('explicit indexing refuses unmanifested legacy state before tokenizer or zvec access', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-service-legacy-state-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  await mkdir(sessionsDirectory);
  const config = createTestConfig(directory, sessionsDirectory);
  const legacyState = '{"version":1,"sessions":{}}\n';
  await writeFile(config.statePath, legacyState);
  let embeddingRequests = 0;
  let tokenizerLoads = 0;
  let storeOpens = 0;
  const service = createRecallConversationService(config, {
    embeddings: {
      async embedTexts() {
        embeddingRequests += 1;
        return [[0, 0, 1]];
      },
    },
    async loadTokenizer() {
      tokenizerLoads += 1;
      return tokenizer;
    },
    openStore() {
      storeOpens += 1;
      throw new Error('legacy index must not open');
    },
  });

  await assert.rejects(
    () => service.index(),
    /manifest missing.*existing index data.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(await readFile(config.statePath, 'utf8'), legacyState);
  assert.equal(embeddingRequests, 0);
  assert.equal(tokenizerLoads, 0);
  assert.equal(storeOpens, 0);
});

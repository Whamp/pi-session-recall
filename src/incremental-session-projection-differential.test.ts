import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallProjectIdentitySource } from './enums.js';
import { indexChangedConversationSessions } from './incremental-session-indexer.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import {
  openSqliteRecallDatabase,
  SQLITE_RECALL_EMBEDDING_DIMENSIONS,
} from './sqlite-recall-database.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';

const DIFFERENTIAL_CHUNK_POLICY = { maxTokens: 32, overlapTokens: 4 } as const;
const DIFFERENTIAL_PROJECT_ATTRIBUTION = {
  projectIdentity: parseRepositoryIdentity(
    'git-origin:github.com/Whamp/incremental-projection-differential',
  ),
  identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
} as const;
const DIFFERENTIAL_TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

function createDifferentialEmbedding(): number[] {
  const embedding = new Array<number>(SQLITE_RECALL_EMBEDDING_DIMENSIONS).fill(0);
  embedding[0] = 1;
  return embedding;
}

function createDifferentialEmbeddingProvider(): RecallEmbeddingProvider {
  return {
    async embedQuery() {
      return createDifferentialEmbedding();
    },
    async embedDocuments(documents) {
      return documents.map(() => createDifferentialEmbedding());
    },
  };
}

function serializeSessionRecords(records: readonly Record<string, unknown>[]): string {
  return `${records.map((record) => JSON.stringify(record)).join('\n')}\n`;
}

function createSessionHeader(sessionId: string, cwd = '/projects/differential') {
  return {
    type: 'session',
    version: 3,
    id: sessionId,
    timestamp: '2026-08-15T12:00:00Z',
    cwd,
  };
}

function createMessage(
  id: string,
  parentId: string | null,
  role: 'assistant' | 'user',
  content: unknown,
) {
  return {
    type: 'message',
    id,
    parentId,
    timestamp: '2026-08-15T12:01:00Z',
    message: { role, content },
  };
}

interface DifferentialScenario {
  name: string;
  initialRecords: Array<Record<string, unknown>>;
  updatedSource: string;
  invalid?: boolean;
}

function createDifferentialScenarios(): DifferentialScenario[] {
  const linearInitial = [
    createSessionHeader('linear'),
    createMessage('linear-user', null, 'user', 'linear user evidence'),
  ];
  const branchInitial = [
    createSessionHeader('branch'),
    createMessage('branch-root', null, 'user', 'shared branch root'),
    createMessage('branch-first', 'branch-root', 'assistant', 'first branch response'),
  ];
  const leafBranches = [
    createSessionHeader('leaf'),
    createMessage('leaf-root', null, 'user', 'leaf root'),
    createMessage('leaf-first', 'leaf-root', 'assistant', 'first leaf'),
    createMessage('leaf-second', 'leaf-root', 'assistant', 'second leaf'),
  ];
  const compactInitial = [
    createSessionHeader('compaction'),
    createMessage('compact-user', null, 'user', 'compaction user evidence'),
    createMessage('compact-assistant', 'compact-user', 'assistant', 'compaction assistant'),
  ];
  const toolInitial = [
    createSessionHeader('tool'),
    createMessage('tool-user', null, 'user', 'inspect the tool source'),
    createMessage('tool-call', 'tool-user', 'assistant', [
      { type: 'text', text: 'reading source' },
      { type: 'toolCall', id: 'differential-call', name: 'read', arguments: { path: 'a.ts' } },
    ]),
  ];
  const turnInitial = [
    createSessionHeader('turn'),
    createMessage('turn-user', null, 'user', 'open turn user'),
    createMessage('turn-assistant-one', 'turn-user', 'assistant', 'first open turn response'),
  ];
  const namedInitial = [
    createSessionHeader('named'),
    createMessage('named-user', null, 'user', 'session naming evidence'),
  ];
  const reuseInitial = [
    createSessionHeader('reuse-one', '/projects/reuse-one'),
    createMessage('reuse-one-user', null, 'user', 'first physical segment'),
  ];
  const rewriteInitial = [
    createSessionHeader('rewrite'),
    createMessage('rewrite-user', null, 'user', 'before same ID rewrite'),
  ];

  return [
    {
      name: 'linear assistant append',
      initialRecords: linearInitial,
      updatedSource: serializeSessionRecords([
        ...linearInitial,
        createMessage('linear-assistant', 'linear-user', 'assistant', 'linear appended response'),
      ]),
    },
    {
      name: 'branch from an old ancestor',
      initialRecords: branchInitial,
      updatedSource: serializeSessionRecords([
        ...branchInitial,
        createMessage('branch-second', 'branch-root', 'assistant', 'second branch response'),
        { type: 'leaf', targetId: 'branch-second' },
      ]),
    },
    {
      name: 'explicit leaf movement without a new entry',
      initialRecords: [...leafBranches, { type: 'leaf', targetId: 'leaf-first' }],
      updatedSource: serializeSessionRecords([
        ...leafBranches,
        { type: 'leaf', targetId: 'leaf-second' },
      ]),
    },
    {
      name: 'compaction with first kept entry',
      initialRecords: compactInitial,
      updatedSource: serializeSessionRecords([
        ...compactInitial,
        {
          type: 'compaction',
          id: 'compaction-entry',
          parentId: 'compact-assistant',
          timestamp: '2026-08-15T12:02:00Z',
          summary: 'compaction summary evidence',
          firstKeptEntryId: 'compact-user',
          tokensBefore: 100,
        },
      ]),
    },
    {
      name: 'retained-tail compaction',
      initialRecords: compactInitial,
      updatedSource: serializeSessionRecords([
        ...compactInitial,
        {
          type: 'compaction',
          id: 'retained-tail-entry',
          parentId: 'compact-assistant',
          timestamp: '2026-08-15T12:02:00Z',
          summary: 'retained tail summary evidence',
          retainedTail: [{ role: 'user', content: 'materialized retained tail' }],
          tokensBefore: 100,
        },
      ]),
    },
    {
      name: 'tool result completes an old call',
      initialRecords: toolInitial,
      updatedSource: serializeSessionRecords([
        ...toolInitial,
        {
          type: 'message',
          id: 'tool-result',
          parentId: 'tool-call',
          timestamp: '2026-08-15T12:02:00Z',
          message: {
            role: 'toolResult',
            toolCallId: 'differential-call',
            toolName: 'read',
            content: [{ type: 'text', text: 'private raw tool result' }],
            isError: false,
          },
        },
      ]),
    },
    {
      name: 'assistant extends an open turn then a user closes it',
      initialRecords: turnInitial,
      updatedSource: serializeSessionRecords([
        ...turnInitial,
        createMessage(
          'turn-assistant-two',
          'turn-assistant-one',
          'assistant',
          'second open turn response',
        ),
        createMessage('turn-user-two', 'turn-assistant-two', 'user', 'next user boundary'),
      ]),
    },
    {
      name: 'session name changes presentation metadata',
      initialRecords: namedInitial,
      updatedSource: serializeSessionRecords([
        ...namedInitial,
        {
          type: 'session_info',
          id: 'session-name',
          parentId: 'named-user',
          timestamp: '2026-08-15T12:02:00Z',
          name: 'Differential session name',
        },
      ]),
    },
    {
      name: 'second header creates reuse history',
      initialRecords: reuseInitial,
      updatedSource: serializeSessionRecords([
        ...reuseInitial,
        createSessionHeader('reuse-two', '/projects/reuse-two'),
        createMessage('reuse-two-user', null, 'user', 'second physical segment'),
      ]),
    },
    {
      name: 'same entry ID rewrites source text',
      initialRecords: rewriteInitial,
      updatedSource: serializeSessionRecords([
        createSessionHeader('rewrite'),
        createMessage('rewrite-user', null, 'user', 'after same ID rewrite'),
      ]),
    },
    {
      name: 'malformed appended JSON removes the physical projection',
      initialRecords: linearInitial,
      updatedSource: `${serializeSessionRecords(linearInitial)}{"type":"message"`,
      invalid: true,
    },
    {
      name: 'empty appended reuse segment removes the physical projection',
      initialRecords: reuseInitial,
      updatedSource: serializeSessionRecords([
        ...reuseInitial,
        createSessionHeader('reuse-empty', '/projects/reuse-empty'),
      ]),
      invalid: true,
    },
  ];
}

function sortConversationDocuments(documents: readonly SessionConversationChunk[]) {
  return [...documents].sort((left, right) => left.id.localeCompare(right.id));
}

function normalizeInvocations(invocations: readonly object[]) {
  return invocations
    .map((invocation) =>
      Object.fromEntries(Object.entries(invocation).filter(([fieldName]) => fieldName !== 'rank')),
    )
    .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
}

void test('cached incremental projection equals full import across graph transitions', async (t) => {
  for (const scenario of createDifferentialScenarios()) {
    await t.test(scenario.name, async (scenarioTest) => {
      const root = await mkdtemp(join(tmpdir(), 'recall-projection-differential-'));
      scenarioTest.after(() => rm(root, { recursive: true, force: true }));
      const sessionsDirectory = join(root, 'sessions');
      const sessionPath = join(sessionsDirectory, 'scenario.jsonl');
      const databasePath = join(root, 'recall.sqlite');
      await mkdir(sessionsDirectory, { recursive: true });
      await writeFile(sessionPath, serializeSessionRecords(scenario.initialRecords));
      const options = {
        sessionsDirectory,
        databasePath,
        embeddingProvider: createDifferentialEmbeddingProvider(),
        tokenizer: DIFFERENTIAL_TOKENIZER,
        chunkPolicy: DIFFERENTIAL_CHUNK_POLICY,
        ignoredPhysicalSessionPaths: new Set<string>(),
        resolveProjectIdentity: async () => DIFFERENTIAL_PROJECT_ATTRIBUTION,
      };
      const initial = await indexChangedConversationSessions(options);
      assert.equal(initial.failedSessions.length, 0);
      const beforeDatabase = openSqliteRecallDatabase(databasePath, { readOnly: true });
      const beforeState = beforeDatabase.readPhysicalSessionState(sessionPath);
      assert.ok(beforeState);
      const beforeDocuments = beforeDatabase.fetchDenseDocuments(beforeState.denseDocumentIds);
      const beforeVectors = beforeDatabase.fetchDenseVectors(beforeState.denseDocumentIds);
      beforeDatabase.close();

      await writeFile(sessionPath, scenario.updatedSource);
      const updated = await indexChangedConversationSessions(options);

      if (scenario.invalid) {
        assert.equal(updated.failedSessions.length, 1);
        await assert.rejects(() =>
          readSessionConversationImport(sessionPath, {
            tokenizer: DIFFERENTIAL_TOKENIZER,
            ...DIFFERENTIAL_CHUNK_POLICY,
            resolveProjectIdentity: async () => DIFFERENTIAL_PROJECT_ATTRIBUTION,
          }),
        );
        const invalidDatabase = openSqliteRecallDatabase(databasePath, { readOnly: true });
        assert.equal(invalidDatabase.readPhysicalSessionState(sessionPath), null);
        assert.equal(invalidDatabase.checkIntegrity().healthy, true);
        invalidDatabase.close();
        return;
      }

      assert.equal(updated.failedSessions.length, 0);
      const fullImport = await readSessionConversationImport(sessionPath, {
        tokenizer: DIFFERENTIAL_TOKENIZER,
        ...DIFFERENTIAL_CHUNK_POLICY,
        resolveProjectIdentity: async () => DIFFERENTIAL_PROJECT_ATTRIBUTION,
      });
      const expectedDocuments = sortConversationDocuments(
        fullImport.chunks
          .filter((chunk) => chunk.isDenseSearchable)
          .map((chunk) => ({
            ...chunk,
            projectAttribution: DIFFERENTIAL_PROJECT_ATTRIBUTION,
          })),
      );
      const optimizedDatabase = openSqliteRecallDatabase(databasePath, { readOnly: true });
      const replacement = optimizedDatabase.readPhysicalSessionReplacement(sessionPath);
      assert.ok(replacement);
      assert.deepEqual(sortConversationDocuments(replacement.denseDocuments), expectedDocuments);
      assert.deepEqual(
        normalizeInvocations(replacement.invocations),
        normalizeInvocations(fullImport.invocations),
      );
      for (const expectedDocument of expectedDocuments) {
        const previousDocument = beforeDocuments.get(expectedDocument.id);
        const previousVector = beforeVectors.get(expectedDocument.id);
        if (previousDocument?.checksum === expectedDocument.checksum && previousVector) {
          assert.deepEqual(
            optimizedDatabase.fetchDenseVectors([expectedDocument.id]).get(expectedDocument.id),
            previousVector,
          );
        }
      }
      const projectCandidates = optimizedDatabase.searchDenseCandidates(
        createDifferentialEmbedding(),
        Math.max(1, expectedDocuments.length),
        DIFFERENTIAL_PROJECT_ATTRIBUTION.projectIdentity,
      );
      assert.deepEqual(
        new Set(projectCandidates.map(({ id }) => id)),
        new Set(expectedDocuments.map(({ id }) => id)),
      );
      assert.equal(optimizedDatabase.checkIntegrity().healthy, true);
      optimizedDatabase.close();
    });
  }
});

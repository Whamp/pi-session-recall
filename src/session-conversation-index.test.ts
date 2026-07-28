import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { SessionImportFormat } from './enums.js';
import { createPhysicalSessionProjectionId } from './recall-session-projection.js';
import {
  readSessionConversationChunks,
  readSessionConversationImport,
  SESSION_CONVERSATION_SCHEMA_VERSION,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';

function createWhitespaceConversationTokenizer(): ConversationTextTokenizer {
  return {
    encodeConversationText(text) {
      return {
        ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()),
      };
    },
  };
}

function summarizeSessionChunkParity(chunks: readonly SessionConversationChunk[]) {
  return chunks.map((chunk) => ({
    id: chunk.id,
    checksum: chunk.checksum,
    documentKind: chunk.documentKind,
    evidenceKind: chunk.evidenceKind,
    evidencePart: chunk.evidencePart,
    content: chunk.content,
    sessionId: chunk.sessionId.value,
    entryId: chunk.entryId.value,
    parentEntryId: chunk.parentEntryId?.value ?? null,
    contributingEntryIds: chunk.contributingEntryIds.map((entryId) => entryId.value),
    sourceLineStart: chunk.sourceLineStart,
    sourceLineEnd: chunk.sourceLineEnd,
  }));
}

void test('rebuild import emits only documents whose contributors are already eligible', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-rebuild-eligibility-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-approved',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'eligible-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'approved historical evidence' },
      },
      {
        type: 'message',
        id: 'active-tail-entry',
        parentId: 'eligible-entry',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'assistant', content: 'unapproved active tail' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    eligibleContributorEntryIdsByLogicalSessionId: new Map([
      ['session-approved', new Set(['eligible-entry'])],
    ]),
  });

  assert.equal(
    imported.chunks.some(({ content }) => content.includes('approved historical')),
    true,
  );
  assert.equal(
    imported.chunks.some(({ content }) => content.includes('unapproved active')),
    false,
  );
  await assert.rejects(
    () =>
      readSessionConversationImport(sessionPath, {
        tokenizer: createWhitespaceConversationTokenizer(),
        eligibleContributorEntryIdsByLogicalSessionId: new Map([
          ['session-approved', new Set(['deleted-approved-entry'])],
        ]),
      }),
    /Recall rebuild approved contributor missing.*deleted-approved-entry/u,
  );
  await assert.rejects(
    () =>
      readSessionConversationImport(sessionPath, {
        tokenizer: createWhitespaceConversationTokenizer(),
        eligibleContributorEntryIdsByLogicalSessionId: new Map([
          ['deleted-logical-session', new Set(['eligible-entry'])],
        ]),
      }),
    /Recall rebuild approved logical session missing.*deleted-logical-session/u,
  );
});

void test('session JSONL becomes searchable conversation chunks with provenance', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-session-'));
  const sessionPath = join(directory, 'session.jsonl');
  const oversized = `${'first '.repeat(30)}needle ${'last '.repeat(30)}`;
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-1',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'user-1',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Where did we put the migration plan?' },
      },
      {
        type: 'message',
        id: 'assistant-1',
        parentId: 'user-1',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'secret reasoning' },
            { type: 'text', text: oversized },
            {
              type: 'toolCall',
              id: 'call-1',
              name: 'pi-session-recall',
              arguments: {},
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'tool-1',
        parentId: 'assistant-1',
        timestamp: '2026-07-24T10:03:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-1',
          toolName: 'pi-session-recall',
          content: [{ type: 'text', text: 'giant private tool output' }],
        },
      },
      {
        type: 'compaction',
        id: 'compact-1',
        parentId: 'tool-1',
        timestamp: '2026-07-24T10:04:00Z',
        summary: 'The migration plan lives in docs/migration.md',
        firstKeptEntryId: 'user-1',
        tokensBefore: 1000,
      },
      {
        type: 'session_info',
        id: 'name-1',
        parentId: 'compact-1',
        timestamp: '2026-07-24T10:05:00Z',
        name: 'Migration planning',
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 20,
    overlapTokens: 4,
  });

  assert.equal(chunks[0]?.sessionId.value, 'session-1');
  assert.equal(
    chunks[0]?.physicalSessionProjectionId,
    createPhysicalSessionProjectionId('session-1'),
  );
  assert.equal(chunks[0]?.cwd, '/project');
  assert.equal(chunks[0]?.sessionName, 'Migration planning');
  assert.equal(chunks[0]?.currentLeafId?.value, 'name-1');
  assert.ok(
    chunks.some((chunk) => chunk.role === 'user' && chunk.content.includes('migration plan')),
  );
  assert.ok(chunks.some((chunk) => chunk.role === 'assistant' && chunk.content.includes('needle')));
  assert.ok(
    chunks.some((chunk) => chunk.role === 'summary' && chunk.entryId.value === 'compact-1'),
  );
  assert.ok(chunks.every((chunk) => chunk.tokenCount <= 20));
  assert.ok(chunks.every((chunk) => chunk.sessionPath === sessionPath));
  assert.ok(chunks.every((chunk) => !chunk.content.includes('secret reasoning')));
  assert.ok(chunks.every((chunk) => !chunk.content.includes('giant private tool output')));
  assert.equal(new Set(chunks.map((chunk) => chunk.id)).size, chunks.length);
  const repeated = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 20,
    overlapTokens: 4,
  });
  assert.deepEqual(
    repeated.map((chunk) => ({ id: chunk.id, textRunId: chunk.textRunId })),
    chunks.map((chunk) => ({ id: chunk.id, textRunId: chunk.textRunId })),
  );
  for (const chunk of chunks) {
    const runChunks = chunks
      .filter((candidate) => candidate.textRunId === chunk.textRunId)
      .sort((left, right) => left.chunkIndex - right.chunkIndex);
    const expectedSiblingIds = [
      runChunks[chunk.chunkIndex - 1]?.id,
      runChunks[chunk.chunkIndex + 1]?.id,
    ].filter((id) => id !== undefined);
    assert.deepEqual(chunk.siblingIds, expectedSiblingIds);
  }
});

void test('session chunking preserves tool boundaries and overlaps only within token-limited text runs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-token-chunks-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-2',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'assistant-2',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: 'one two three four five six seven' },
            { type: 'thinking', thinking: 'private boundary' },
            { type: 'text', text: 'eight nine ten eleven twelve thirteen' },
            { type: 'toolCall', id: 'call-2', name: 'read', arguments: {} },
            { type: 'text', text: 'fourteen fifteen sixteen' },
          ],
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const tokenizer = createWhitespaceConversationTokenizer();

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer,
    maxTokens: 5,
    overlapTokens: 1,
  });
  const conversationChunks = chunks.filter((chunk) => chunk.documentKind === 'conversation');

  assert.deepEqual(
    conversationChunks.map(({ textRunIndex, content }) => ({ textRunIndex, content })),
    [
      { textRunIndex: 0, content: 'one two three four five' },
      { textRunIndex: 0, content: 'five six seven' },
      { textRunIndex: 1, content: 'eight nine ten eleven twelve' },
      { textRunIndex: 1, content: 'twelve thirteen' },
      { textRunIndex: 2, content: 'fourteen fifteen sixteen' },
    ],
  );
  assert.ok(chunks.every((chunk) => chunk.tokenCount <= 5));
  assert.deepEqual(
    conversationChunks.map((chunk) => chunk.overlapTokenCount),
    [0, 1, 0, 1, 0],
  );
  for (const chunk of conversationChunks) {
    assert.equal(chunk.siblingIds.length, chunk.chunkCount - 1);
    assert.ok(chunk.siblingIds.every((id) => id !== chunk.id));
    if (chunk.previousSiblingId) {
      const previous = conversationChunks.find(
        (candidate) => candidate.id === chunk.previousSiblingId,
      );
      assert.equal(chunk.tokenStart, (previous?.tokenEnd ?? 0) - chunk.overlapTokenCount);
    }
  }
});

void test('turn-context documents follow parent paths across tool activity', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-turn-context-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-turn-context',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'next-user',
        parentId: 'assistant-final',
        timestamp: '2026-07-24T10:05:00Z',
        message: { role: 'user', content: 'What happened after deployment?' },
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
          content: [{ type: 'text', text: 'RAW_TOOL_OUTPUT must stay out of turn context' }],
          isError: false,
        },
      },
      {
        type: 'message',
        id: 'assistant-final',
        parentId: 'tool-result',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'Yes, do it.' }] },
      },
      {
        type: 'message',
        id: 'user-request',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'user',
          content: `Ship release Atlas to edge nodes. ${'context '.repeat(24).trim()}`,
        },
      },
      {
        type: 'message',
        id: 'assistant-call',
        parentId: 'user-request',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'private release reasoning' },
            { type: 'text', text: 'I will inspect the manifest.' },
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
        id: 'assistant-abandoned',
        parentId: 'user-request',
        timestamp: '2026-07-24T10:02:30Z',
        message: {
          role: 'assistant',
          content: `Abandoned ${'word '.repeat(24).trim()}`,
        },
      },
      { type: 'leaf', targetId: 'next-user' },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 16,
    overlapTokens: 2,
  });
  const turnContexts = chunks.filter((chunk) => chunk.documentKind === 'turn_context');
  const activeTurnChunks = turnContexts.filter((chunk) =>
    chunk.contributingEntryIds.some((id) => id.value === 'assistant-final'),
  );
  const abandonedTurnChunks = turnContexts.filter((chunk) =>
    chunk.contributingEntryIds.some((id) => id.value === 'assistant-abandoned'),
  );

  assert.equal(
    chunks.find(
      (chunk) => chunk.documentKind === 'conversation' && chunk.entryId.value === 'user-request',
    )?.contributingEntryIds[0]?.value,
    'user-request',
  );
  assert.ok(activeTurnChunks.length > 1);
  assert.ok(abandonedTurnChunks.length > 1);
  assert.deepEqual(
    activeTurnChunks[0]?.contributingEntryIds.map((id) => id.value),
    ['user-request', 'assistant-call', 'assistant-final'],
  );
  assert.equal(activeTurnChunks[0]?.entryId.value, 'user-request');
  assert.equal(activeTurnChunks[0]?.evidenceKind, 'turn_context');
  assert.equal(activeTurnChunks[0]?.role, 'turn');
  assert.deepEqual(
    activeTurnChunks[0]?.branchPathLeafIds.map((id) => id.value),
    ['next-user'],
  );
  assert.equal(activeTurnChunks[0]?.isOnActiveBranch, true);
  assert.equal(activeTurnChunks[0]?.sourceLineStart, 4);
  assert.equal(activeTurnChunks[0]?.sourceLineEnd, 6);
  assert.ok(activeTurnChunks.some((chunk) => chunk.content.includes('Ship release Atlas')));
  assert.ok(activeTurnChunks.every((chunk) => chunk.content.includes('User:')));
  assert.ok(activeTurnChunks.every((chunk) => chunk.content.includes('Assistant:')));
  assert.ok(activeTurnChunks.every((chunk) => chunk.content.includes('Yes, do it.')));
  assert.ok(
    activeTurnChunks.every((chunk) => !chunk.content.includes('What happened after deployment?')),
  );
  assert.ok(turnContexts.every((chunk) => chunk.tokenCount <= 16));
  assert.ok(turnContexts.every((chunk) => chunk.overlapTokenCount <= 2));
  assert.ok(turnContexts.every((chunk) => chunk.isDenseSearchable));
  assert.ok(turnContexts.every((chunk) => !chunk.content.includes('RAW_TOOL_OUTPUT')));
  assert.ok(turnContexts.every((chunk) => !chunk.content.includes('private release reasoning')));
  assert.deepEqual(
    abandonedTurnChunks[0]?.branchPathLeafIds.map((id) => id.value),
    ['assistant-abandoned'],
  );
  assert.equal(abandonedTurnChunks[0]?.isOnActiveBranch, false);
});

void test('turn-context documents reject a token budget that cannot contain both roles', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-turn-context-budget-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-turn-budget',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'user',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'Approve' },
      },
      {
        type: 'message',
        id: 'assistant',
        parentId: 'user',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'assistant', content: 'Done' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  await assert.rejects(
    () =>
      readSessionConversationChunks(sessionPath, {
        tokenizer: createWhitespaceConversationTokenizer(),
        maxTokens: 3,
        overlapTokens: 0,
      }),
    /Recall turn context cannot fit both user and assistant text within maxTokens=3/,
  );
});

void test('session chunks index bounded verbatim tool evidence with exact call provenance', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-tool-evidence-'));
  const sessionPath = join(directory, 'session.jsonl');
  const toolOutput = 'EPERM readNodeErrorCode /tmp/locked-file\nsecond output line';
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-tools',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'assistant-tools',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'thinking', thinking: 'never index this private tool plan' },
            {
              type: 'toolCall',
              id: 'call-tools',
              name: 'bash',
              arguments: {
                command: 'cat /tmp/locked-file',
                url: 'https://example.test/tool-output?id=EPERM',
              },
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
          content: [
            { type: 'text', text: toolOutput },
            { type: 'image', data: 'ignored', mimeType: 'image/png' },
            { type: 'text', text: 'final result URL https://example.test/final' },
          ],
          isError: true,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 4,
    overlapTokens: 1,
  });
  const toolChunks = chunks.filter((chunk) => chunk.documentKind === 'tool');
  const callChunks = toolChunks.filter((chunk) => chunk.evidenceKind === 'tool_call');
  const resultChunks = toolChunks.filter((chunk) => chunk.evidenceKind === 'tool_result');

  assert.deepEqual(
    callChunks.map((chunk) => chunk.evidencePart),
    ['name', 'arguments'],
  );
  assert.equal(callChunks[0]?.content, 'bash');
  assert.equal(
    callChunks[1]?.content,
    '{"command":"cat /tmp/locked-file","url":"https://example.test/tool-output?id=EPERM"}',
  );
  assert.equal(
    resultChunks.map((chunk) => chunk.content).join(''),
    toolOutput + 'final result URL https://example.test/final',
  );
  assert.ok(toolChunks.every((chunk) => chunk.isDenseSearchable === false));
  assert.ok(toolChunks.every((chunk) => chunk.tokenCount <= 4));
  assert.ok(toolChunks.every((chunk) => chunk.overlapTokenCount === 0));
  assert.ok(toolChunks.every((chunk) => chunk.toolCallId === 'call-tools'));
  assert.ok(toolChunks.every((chunk) => chunk.toolName === 'bash'));
  assert.ok(callChunks.every((chunk) => chunk.toolCallEntryId?.value === 'assistant-tools'));
  assert.ok(callChunks.every((chunk) => chunk.toolResultEntryId?.value === 'result-tools'));
  assert.ok(resultChunks.every((chunk) => chunk.toolCallEntryId?.value === 'assistant-tools'));
  assert.ok(resultChunks.every((chunk) => chunk.toolResultEntryId?.value === 'result-tools'));
  assert.ok(callChunks.every((chunk) => chunk.sourceBlockStart === 1));
  assert.deepEqual([...new Set(resultChunks.map((chunk) => chunk.sourceBlockStart))], [0, 2]);
  assert.ok(chunks.every((chunk) => !chunk.content.includes('private tool plan')));
});

void test('session chunks exclude derived pi-session-recall tool calls and results', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-derived-tool-evidence-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-derived-recall',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'assistant-recall',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: 'I will check the prior decision.' },
            {
              type: 'toolCall',
              id: 'call-recall',
              name: 'pi-session-recall',
              arguments: { query: 'recursive evidence marker' },
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'result-recall',
        parentId: 'assistant-recall',
        timestamp: '2026-07-24T10:02:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-recall',
          toolName: 'pi-session-recall',
          content: [{ type: 'text', text: 'recursive evidence marker from derived output' }],
          isError: false,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 512,
    overlapTokens: 64,
  });

  assert.deepEqual(
    chunks.map((chunk) => ({ documentKind: chunk.documentKind, content: chunk.content })),
    [{ documentKind: 'conversation', content: 'I will check the prior decision.' }],
  );
});

void test('session chunks index direct bash commands and outputs as lexical-only evidence', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-bash-evidence-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-bash',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'bash-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: {
          role: 'bashExecution',
          command: 'rg "needle" src/read-node-error-code.ts',
          output: 'src/read-node-error-code.ts: EPERM needle',
          exitCode: 2,
          cancelled: false,
          truncated: false,
          excludeFromContext: true,
        },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
    maxTokens: 3,
    overlapTokens: 1,
  });

  assert.deepEqual(
    chunks.map((chunk) => chunk.evidencePart),
    ['command', 'output'],
  );
  assert.equal(
    chunks
      .filter((chunk) => chunk.evidencePart === 'command')
      .map((chunk) => chunk.content)
      .join(''),
    'rg "needle" src/read-node-error-code.ts',
  );
  assert.equal(
    chunks
      .filter((chunk) => chunk.evidencePart === 'output')
      .map((chunk) => chunk.content)
      .join(''),
    'src/read-node-error-code.ts: EPERM needle',
  );
  assert.ok(chunks.every((chunk) => chunk.documentKind === 'tool'));
  assert.ok(chunks.every((chunk) => chunk.evidenceKind === 'bash_execution'));
  assert.ok(chunks.every((chunk) => chunk.toolName === 'bash'));
  assert.ok(chunks.every((chunk) => chunk.toolCallId === null));
  assert.ok(chunks.every((chunk) => chunk.toolError === true));
  assert.ok(chunks.every((chunk) => chunk.isDenseSearchable === false));
  assert.ok(chunks.every((chunk) => chunk.overlapTokenCount === 0));
});

void test('session chunks expose branch, summary, sibling, and source geometry provenance', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-graph-provenance-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-graph',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
        parentSession: '/sessions/parent.jsonl',
      },
      {
        type: 'message',
        id: 'root',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'shared root' },
      },
      {
        type: 'message',
        id: 'abandoned',
        parentId: 'root',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'abandoned path' }] },
      },
      {
        type: 'branch_summary',
        id: 'branch-summary',
        parentId: 'root',
        timestamp: '2026-07-24T10:03:00Z',
        fromId: 'root',
        summary: 'Branch explored the abandoned approach.',
      },
      {
        type: 'message',
        id: 'active',
        parentId: 'branch-summary',
        timestamp: '2026-07-24T10:04:00Z',
        message: {
          role: 'assistant',
          content: [
            { type: 'text', text: 'alpha' },
            { type: 'text', text: 'beta' },
            { type: 'text', text: '   ' },
            { type: 'text', text: 'gamma' },
            { type: 'image', data: 'ignored', mimeType: 'image/png' },
            { type: 'text', text: 'delta' },
          ],
        },
      },
      { type: 'leaf', targetId: 'active' },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const tokenizer = createWhitespaceConversationTokenizer();

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer,
    maxTokens: 1_024,
    overlapTokens: 128,
  });
  const root = chunks.find((chunk) => chunk.entryId.value === 'root');
  const abandoned = chunks.find((chunk) => chunk.entryId.value === 'abandoned');
  const summary = chunks.find((chunk) => chunk.entryId.value === 'branch-summary');
  const activeRuns = chunks.filter((chunk) => chunk.entryId.value === 'active');

  assert.deepEqual(
    root?.branchPathLeafIds.map((id) => id.value),
    ['abandoned', 'active'],
  );
  assert.equal(root?.currentLeafId?.value, 'active');
  assert.equal(root?.isOnActiveBranch, true);
  assert.equal(root?.isVisibleInActiveContext, true);
  assert.equal(root?.parentSessionPath, '/sessions/parent.jsonl');
  assert.equal(abandoned?.isOnActiveBranch, false);
  assert.equal(abandoned?.isVisibleInActiveContext, false);
  assert.equal(summary?.documentKind, 'summary');
  assert.equal(summary?.summaryKind, 'branch');
  assert.equal(summary?.branchSummaryFromEntryId?.value, 'root');
  assert.deepEqual(
    activeRuns.map((chunk) => ({
      content: chunk.content,
      textRunIndex: chunk.textRunIndex,
      sourceLineStart: chunk.sourceLineStart,
      sourceBlockStart: chunk.sourceBlockStart,
      sourceBlockEnd: chunk.sourceBlockEnd,
    })),
    [
      {
        content: 'alpha\nbeta',
        textRunIndex: 0,
        sourceLineStart: 5,
        sourceBlockStart: 0,
        sourceBlockEnd: 1,
      },
      {
        content: 'gamma',
        textRunIndex: 1,
        sourceLineStart: 5,
        sourceBlockStart: 3,
        sourceBlockEnd: 3,
      },
      {
        content: 'delta',
        textRunIndex: 2,
        sourceLineStart: 5,
        sourceBlockStart: 5,
        sourceBlockEnd: 5,
      },
    ],
  );
  assert.ok(
    activeRuns.every((chunk) => chunk.schemaVersion === SESSION_CONVERSATION_SCHEMA_VERSION),
  );
  assert.ok(activeRuns.every((chunk) => chunk.contributingEntryIds[0]?.value === 'active'));
  assert.ok(activeRuns.every((chunk) => chunk.textRunId.length === 40));
  assert.ok(activeRuns.every((chunk) => chunk.chunkCount === 1));
  assert.ok(activeRuns.every((chunk) => chunk.siblingIds.length === 0));
  assert.ok(activeRuns.every((chunk) => chunk.characterStart === 0));
  assert.ok(activeRuns.every((chunk) => chunk.tokenStart === 0));
  assert.ok(activeRuns.every((chunk) => chunk.tokenCount > 0));
});

void test('session chunking prefers structural text boundaries before hard token cuts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-natural-boundaries-'));
  const cases = [
    {
      name: 'markdown-section',
      maxTokens: 5,
      text: '# Alpha\none two\n\n# Beta\nthree four five',
      expected: ['# Alpha\none two', '# Beta\nthree four five'],
    },
    {
      name: 'paragraph',
      maxTokens: 4,
      text: 'one two three\n\nfour five six',
      expected: ['one two three', 'four five six'],
    },
    {
      name: 'fenced-code',
      maxTokens: 8,
      text: 'one two\n```ts\nthree four five\n```\nsix seven',
      expected: ['one two\n```ts\nthree four five\n```', 'six seven'],
    },
    {
      name: 'line',
      maxTokens: 4,
      text: 'one two three\nfour five six',
      expected: ['one two three', 'four five six'],
    },
    {
      name: 'sentence',
      maxTokens: 4,
      text: 'One two three. Four five six.',
      expected: ['One two three.', 'Four five six.'],
    },
  ];

  for (const fixture of cases) {
    const sessionPath = join(directory, `${fixture.name}.jsonl`);
    await writeFile(
      sessionPath,
      [
        {
          type: 'session',
          version: 3,
          id: fixture.name,
          timestamp: '2026-07-24T10:00:00Z',
          cwd: '/project',
        },
        {
          type: 'message',
          id: 'entry',
          parentId: null,
          timestamp: '2026-07-24T10:01:00Z',
          message: { role: 'assistant', content: fixture.text },
        },
      ]
        .map((entry) => JSON.stringify(entry))
        .join('\n') + '\n',
    );

    const chunks = await readSessionConversationChunks(sessionPath, {
      tokenizer: createWhitespaceConversationTokenizer(),
      maxTokens: fixture.maxTokens,
      overlapTokens: 0,
    });

    assert.deepEqual(
      chunks.map((chunk) => chunk.content),
      fixture.expected,
      fixture.name,
    );
    assert.ok(
      chunks.every((chunk) => chunk.tokenCount <= fixture.maxTokens),
      fixture.name,
    );
  }
});

void test('session chunks distinguish active-branch entries hidden by compaction', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-compaction-visibility-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-compaction',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'summarized',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'old context' },
      },
      {
        type: 'message',
        id: 'kept',
        parentId: 'summarized',
        timestamp: '2026-07-24T10:02:00Z',
        message: { role: 'user', content: 'kept context' },
      },
      {
        type: 'compaction',
        id: 'compaction',
        parentId: 'kept',
        timestamp: '2026-07-24T10:03:00Z',
        summary: 'Summary of old context.',
        firstKeptEntryId: 'kept',
        tokensBefore: 100,
      },
      {
        type: 'message',
        id: 'after',
        parentId: 'compaction',
        timestamp: '2026-07-24T10:04:00Z',
        message: { role: 'assistant', content: [{ type: 'text', text: 'new context' }] },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });
  const summarized = chunks.find((chunk) => chunk.entryId.value === 'summarized');
  const kept = chunks.find((chunk) => chunk.entryId.value === 'kept');
  const compaction = chunks.find((chunk) => chunk.entryId.value === 'compaction');
  const after = chunks.find((chunk) => chunk.entryId.value === 'after');

  assert.equal(summarized?.isOnActiveBranch, true);
  assert.equal(summarized?.isVisibleInActiveContext, false);
  assert.deepEqual(
    summarized?.compactedByEntryIds.map((id) => id.value),
    ['compaction'],
  );
  assert.equal(kept?.isVisibleInActiveContext, true);
  assert.deepEqual(kept?.compactedByEntryIds, []);
  assert.equal(compaction?.isVisibleInActiveContext, true);
  assert.equal(compaction?.compactionFirstKeptEntryId?.value, 'kept');
  assert.equal(after?.isVisibleInActiveContext, true);
});

void test('retained-tail compaction acts as a self-contained active-context checkpoint', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-retained-tail-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'session-retained-tail',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'before',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'source message before checkpoint' },
      },
      {
        type: 'compaction',
        id: 'checkpoint',
        parentId: 'before',
        timestamp: '2026-07-24T10:02:00Z',
        summary: 'Self-contained checkpoint summary.',
        retainedTail: [{ role: 'user', content: 'materialized retained context' }],
        tokensBefore: 100,
      },
      {
        type: 'message',
        id: 'after',
        parentId: 'checkpoint',
        timestamp: '2026-07-24T10:03:00Z',
        message: { role: 'assistant', content: 'source message after checkpoint' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });
  const before = chunks.find((chunk) => chunk.entryId.value === 'before');
  const checkpoint = chunks.find((chunk) => chunk.entryId.value === 'checkpoint');
  const after = chunks.find((chunk) => chunk.entryId.value === 'after');

  assert.equal(before?.isOnActiveBranch, true);
  assert.equal(before?.isVisibleInActiveContext, false);
  assert.deepEqual(
    before?.compactedByEntryIds.map((id) => id.value),
    ['checkpoint'],
  );
  assert.equal(checkpoint?.isVisibleInActiveContext, true);
  assert.equal(checkpoint?.compactionFirstKeptEntryId, null);
  assert.equal(after?.isVisibleInActiveContext, true);
});

void test('session JSONL framing preserves Unicode line separators inside records', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-unicode-line-separators-'));
  const sessionPath = join(directory, 'unicode-line-separators.jsonl');
  const content = 'before\u2028between\u2029after';
  const records = [
    {
      type: 'session',
      version: 3,
      id: 'unicode-line-session',
      timestamp: '2026-07-24T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'unicode-line-message',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content },
    },
  ];
  await writeFile(
    sessionPath,
    `${records.map((record) => JSON.stringify(record)).join('\r\n')}\r\n`,
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });

  assert.deepEqual(
    chunks.filter((chunk) => chunk.documentKind === 'conversation').map((chunk) => chunk.content),
    [content],
  );
});

void test('session JSONL framing remains exact across a stream chunk boundary', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-stream-chunk-boundary-'));
  const sessionPath = join(directory, 'chunk-boundary.jsonl');
  const content = `${'x'.repeat(70_000)}before\u2028between\u2029after`;
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'stream-boundary-session',
        timestamp: '2026-07-24T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'stream-boundary-entry',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  const chunks = await readSessionConversationChunks(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });

  assert.deepEqual(
    chunks.map((chunk) => chunk.content),
    [content],
  );
  assert.equal(chunks[0]?.sourceLineStart, 2);
});

void test('session JSONL framing rejects a genuinely truncated final record', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-truncated-session-record-'));
  const sessionPath = join(directory, 'truncated.jsonl');
  const header = JSON.stringify({
    type: 'session',
    version: 3,
    id: 'truncated-session',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  });
  await writeFile(sessionPath, `${header}\n{"type":"message","id":"unfinished`);

  await assert.rejects(
    () =>
      readSessionConversationChunks(sessionPath, {
        tokenizer: createWhitespaceConversationTokenizer(),
      }),
    /Recall session JSON invalid.*truncated\.jsonl:2.*Unterminated string/u,
  );
});

void test('unversioned Pi v1 sessions convert deterministically through the strict graph boundary', async () => {
  const sessionPath = join(import.meta.dirname, 'fixtures/session-import/pi-v1-linear.jsonl');
  const bytesBefore = await readFile(sessionPath);
  const metadataBefore = await stat(sessionPath);
  const options = { tokenizer: createWhitespaceConversationTokenizer() };

  const imported = await readSessionConversationImport(sessionPath, options);
  const repeated = await readSessionConversationImport(sessionPath, options);
  const canonicalEquivalent = await readSessionConversationImport(
    join(import.meta.dirname, 'fixtures/session-import/pi-v1-canonical-equivalent.jsonl'),
    options,
  );

  assert.equal(imported.format, SessionImportFormat.PI_V1_LINEAR);
  assert.equal(imported.logicalSessions.length, 1);
  assert.deepEqual(imported.logicalSessions[0], {
    sessionId: 'session-v1-fixture',
    sourceLineStart: 1,
    sourceLineEnd: 5,
    entryIds: [
      '3365ad4e8d9190b1c7ab35a854eeacaa8984f8a7',
      '6d35bf551059e2664b2bbefc7b43d63d006b6158',
      '10d74c0b1c52779ebbbdec04453a1bbfc2747aac',
      '6a3eac6379d2e6df8c364330052940c8f84ddc43',
    ],
    parentEntryIds: [
      null,
      '3365ad4e8d9190b1c7ab35a854eeacaa8984f8a7',
      '6d35bf551059e2664b2bbefc7b43d63d006b6158',
      '10d74c0b1c52779ebbbdec04453a1bbfc2747aac',
    ],
  });
  assert.ok(imported.chunks.every((chunk) => chunk.sessionId.value === 'session-v1-fixture'));
  assert.ok(imported.chunks.every((chunk) => chunk.sessionPath === sessionPath));
  assert.ok(imported.chunks.every((chunk) => chunk.cwd === '/legacy/project'));
  assert.ok(imported.chunks.every((chunk) => chunk.parentSessionPath === '/legacy/parent.jsonl'));
  assert.ok(imported.chunks.some((chunk) => chunk.evidenceKind === 'tool_call'));
  assert.ok(imported.chunks.some((chunk) => chunk.evidenceKind === 'tool_result'));
  assert.ok(imported.chunks.some((chunk) => chunk.evidenceKind === 'compaction_summary'));
  const compaction = imported.chunks.find((chunk) => chunk.evidenceKind === 'compaction_summary');
  assert.equal(
    compaction?.compactionFirstKeptEntryId?.value,
    '6d35bf551059e2664b2bbefc7b43d63d006b6158',
  );
  assert.deepEqual(repeated, imported);
  const importedParity = summarizeSessionChunkParity(imported.chunks);
  const canonicalParity = summarizeSessionChunkParity(canonicalEquivalent.chunks);
  assert.deepEqual(
    importedParity.map((chunk) => ({ ...chunk, id: 'physical-occurrence' })),
    canonicalParity.map((chunk) => ({ ...chunk, id: 'physical-occurrence' })),
  );
  assert.notDeepEqual(
    importedParity.map(({ id }) => id),
    canonicalParity.map(({ id }) => id),
  );
  assert.deepEqual(await readFile(sessionPath), bytesBefore);
  const metadataAfter = await stat(sessionPath);
  assert.deepEqual(
    {
      size: metadataAfter.size,
      mode: metadataAfter.mode,
      mtimeMs: metadataAfter.mtimeMs,
      ino: metadataAfter.ino,
    },
    {
      size: metadataBefore.size,
      mode: metadataBefore.mode,
      mtimeMs: metadataBefore.mtimeMs,
      ino: metadataBefore.ino,
    },
  );
});

void test('Pi session-file reuse history becomes independent logical sessions with physical provenance', async () => {
  const sessionPath = join(
    import.meta.dirname,
    'fixtures/session-import/pi-session-reuse-history.jsonl',
  );

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });

  assert.equal(imported.format, SessionImportFormat.PI_SESSION_REUSE_HISTORY);
  assert.deepEqual(
    imported.logicalSessions.map(({ sessionId, sourceLineStart, sourceLineEnd }) => ({
      sessionId,
      sourceLineStart,
      sourceLineEnd,
    })),
    [
      { sessionId: 'reuse-session-one', sourceLineStart: 1, sourceLineEnd: 2 },
      { sessionId: 'reuse-session-two', sourceLineStart: 3, sourceLineEnd: 4 },
    ],
  );
  assert.deepEqual(
    imported.chunks.map(({ sessionId, cwd, sourceLineStart, content }) => ({
      sessionId: sessionId.value,
      cwd,
      sourceLineStart,
      content,
    })),
    [
      {
        sessionId: 'reuse-session-one',
        cwd: '/project/one',
        sourceLineStart: 2,
        content: 'first logical session',
      },
      {
        sessionId: 'reuse-session-two',
        cwd: '/project/two',
        sourceLineStart: 4,
        content: 'second logical session',
      },
    ],
  );
  assert.equal(new Set(imported.chunks.map((chunk) => chunk.id)).size, 2);
  assert.notEqual(imported.chunks[0]?.id, imported.chunks[1]?.id);
  assert.equal(imported.chunks[1]?.parentSessionPath, '/parent/two.jsonl');
});

void test('v1 compaction conversion validates edge indexes and maps compaction references directly', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-v1-compaction-indexes-'));
  const options = { tokenizer: createWhitespaceConversationTokenizer() };
  const header = {
    type: 'session',
    id: 'v1-compaction-indexes',
    timestamp: '2025-12-01T10:00:00.000Z',
    cwd: '/legacy/project',
  };
  const firstMessage = {
    type: 'message',
    timestamp: '2025-12-01T10:00:01.000Z',
    message: {
      role: 'user',
      content: 'kept v1 message',
      timestamp: 1_764_583_201_000,
    },
  };
  const firstCompaction = {
    type: 'compaction',
    timestamp: '2025-12-01T10:00:02.000Z',
    summary: 'first v1 summary',
    firstKeptEntryIndex: 1,
    tokensBefore: 10,
  };
  const secondCompaction = {
    type: 'compaction',
    timestamp: '2025-12-01T10:00:03.000Z',
    summary: 'second v1 summary',
    firstKeptEntryIndex: 2,
    tokensBefore: 20,
  };
  const validPath = join(directory, 'valid.jsonl');
  await writeFile(
    validPath,
    [header, firstMessage, firstCompaction, secondCompaction]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  const imported = await readSessionConversationImport(validPath, options);
  const entryIds = imported.logicalSessions[0]?.entryIds ?? [];
  const summaries = imported.chunks.filter((chunk) => chunk.documentKind === 'summary');
  assert.equal(imported.format, SessionImportFormat.PI_V1_LINEAR);
  assert.equal(summaries[0]?.compactionFirstKeptEntryId?.value, entryIds[0]);
  assert.equal(summaries[1]?.compactionFirstKeptEntryId?.value, entryIds[1]);

  for (const firstKeptEntryIndex of [0, 99]) {
    const invalidPath = join(directory, `invalid-${firstKeptEntryIndex}.jsonl`);
    await writeFile(
      invalidPath,
      [header, firstMessage, { ...firstCompaction, firstKeptEntryIndex }]
        .map((record) => JSON.stringify(record))
        .join('\n'),
    );
    await assert.rejects(
      () => readSessionConversationImport(invalidPath, options),
      /compaction firstKeptEntryIndex does not name an entry/u,
    );
  }
});

void test('reuse histories keep document identities distinct when session and entry IDs repeat', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-repeated-logical-identities-'));
  const sessionPath = join(directory, 'repeated-identities.jsonl');
  const records = [
    {
      type: 'session',
      version: 3,
      id: 'repeated-session',
      timestamp: '2026-01-10T10:00:00Z',
      cwd: '/first',
    },
    {
      type: 'message',
      id: 'repeated-entry',
      parentId: null,
      timestamp: '2026-01-10T10:00:01Z',
      message: { role: 'user', content: 'first repeated identity' },
    },
    {
      type: 'session',
      version: 3,
      id: 'repeated-session',
      timestamp: '2026-01-10T11:00:00Z',
      cwd: '/second',
    },
    {
      type: 'message',
      id: 'repeated-entry',
      parentId: null,
      timestamp: '2026-01-10T11:00:01Z',
      message: { role: 'user', content: 'second repeated identity' },
    },
  ];
  await writeFile(sessionPath, records.map((record) => JSON.stringify(record)).join('\n'));

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });

  assert.equal(imported.format, SessionImportFormat.PI_SESSION_REUSE_HISTORY);
  assert.deepEqual(
    imported.logicalSessions.map((session) => session.sessionId),
    ['repeated-session', 'repeated-session'],
  );
  assert.equal(new Set(imported.chunks.map((chunk) => chunk.id)).size, 2);
});

void test('format detection rejects v1 and reuse-history near misses without heuristic repair', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-import-near-miss-'));
  const options = { tokenizer: createWhitespaceConversationTokenizer() };
  const versionedWithoutEntryIds = join(directory, 'versioned-without-entry-ids.jsonl');
  const v1MissingMessageMetadata = join(directory, 'v1-missing-message-metadata.jsonl');
  const preHeaderRecord = join(directory, 'pre-header-record.jsonl');
  const independentlyInvalidSegment = join(directory, 'invalid-segment.jsonl');
  await writeFile(
    versionedWithoutEntryIds,
    [
      {
        type: 'session',
        version: 3,
        id: 'not-v1',
        timestamp: '2026-01-10T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'missing modern graph identity' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );
  await writeFile(
    v1MissingMessageMetadata,
    [
      {
        type: 'session',
        id: 'incomplete-v1',
        timestamp: '2026-01-10T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'message timestamp is required' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );
  await writeFile(
    preHeaderRecord,
    [
      {
        type: 'message',
        id: 'before',
        parentId: null,
        timestamp: '2026-01-10T10:00:00Z',
        message: { role: 'user', content: 'before header' },
      },
      {
        type: 'session',
        version: 3,
        id: 'one',
        timestamp: '2026-01-10T10:00:01Z',
        cwd: '/project',
      },
      {
        type: 'session',
        version: 3,
        id: 'two',
        timestamp: '2026-01-10T10:00:02Z',
        cwd: '/project',
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );
  await writeFile(
    independentlyInvalidSegment,
    [
      {
        type: 'session',
        version: 3,
        id: 'valid-first',
        timestamp: '2026-01-10T10:00:00Z',
        cwd: '/first',
      },
      {
        type: 'message',
        id: 'first-entry',
        parentId: null,
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'valid first segment' },
      },
      {
        type: 'session',
        version: 3,
        id: 'invalid-second',
        timestamp: '2026-01-10T11:00:00Z',
        cwd: '/second',
      },
      {
        type: 'message',
        id: 'orphan',
        parentId: 'first-entry',
        timestamp: '2026-01-10T11:00:01Z',
        message: { role: 'user', content: 'must not cross the header boundary' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  await assert.rejects(
    () => readSessionConversationImport(versionedWithoutEntryIds, options),
    /entry.id must be a nonempty string/u,
  );
  await assert.rejects(
    () => readSessionConversationImport(v1MissingMessageMetadata, options),
    /unsupported or ambiguous.*canonical session version/u,
  );
  await assert.rejects(
    () => readSessionConversationImport(preHeaderRecord, options),
    /unsupported or ambiguous/u,
  );
  await assert.rejects(
    () => readSessionConversationImport(independentlyInvalidSegment, options),
    /logical session invalid-second.*entry orphan has missing parent first-entry/u,
  );
});

void test('format detection rejects unsupported canonical session versions', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-unsupported-session-version-'));
  const options = { tokenizer: createWhitespaceConversationTokenizer() };

  for (const version of [1, 4, 'future-ish']) {
    const sessionPath = join(directory, `version-${String(version)}.jsonl`);
    await writeFile(
      sessionPath,
      [
        {
          type: 'session',
          version,
          id: `unsupported-${String(version)}`,
          timestamp: '2026-01-10T10:00:00Z',
          cwd: '/project',
        },
        {
          type: 'message',
          id: 'entry',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: { role: 'user', content: 'must not become searchable' },
        },
      ]
        .map((record) => JSON.stringify(record))
        .join('\n'),
    );

    await assert.rejects(
      () => readSessionConversationImport(sessionPath, options),
      /unsupported or ambiguous.*canonical session version/u,
    );
  }
});

void test('strict session graph validation accepts a branch summary from an abandoned entry', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-valid-branch-summary-link-'));
  const sessionPath = join(directory, 'valid-branch-summary.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'valid-branch-summary',
        timestamp: '2026-01-10T10:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'root',
        parentId: null,
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'root' },
      },
      {
        type: 'message',
        id: 'abandoned',
        parentId: 'root',
        timestamp: '2026-01-10T10:00:02Z',
        message: { role: 'assistant', content: 'abandoned path' },
      },
      {
        type: 'branch_summary',
        id: 'branch-summary',
        parentId: 'root',
        timestamp: '2026-01-10T10:00:03Z',
        fromId: 'abandoned',
        summary: 'Summary of the abandoned path.',
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });
  const summary = imported.chunks.find((chunk) => chunk.entryId.value === 'branch-summary');
  assert.equal(summary?.branchSummaryFromEntryId?.value, 'abandoned');
});

void test('session import ignores exact blank tool placeholders without losing conversation evidence', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-blank-tool-placeholder-'));
  const sessionPath = join(directory, 'blank-tool-placeholder.jsonl');
  const records = [
    {
      type: 'session',
      version: 3,
      id: 'blank-tool-placeholder',
      timestamp: '2026-01-10T10:00:00Z',
      cwd: '/project',
    },
    {
      type: 'message',
      id: 'user',
      parentId: null,
      timestamp: '2026-01-10T10:00:01Z',
      message: { role: 'user', content: 'searchable request before blank tool placeholder' },
    },
    {
      type: 'message',
      id: 'blank-call',
      parentId: 'user',
      timestamp: '2026-01-10T10:00:02Z',
      message: {
        role: 'assistant',
        provider: 'zai',
        model: 'glm-4.6v',
        content: [{ type: 'toolCall', id: '', name: '', arguments: {} }],
      },
    },
    {
      type: 'message',
      id: 'blank-result',
      parentId: 'blank-call',
      timestamp: '2026-01-10T10:00:03Z',
      message: {
        role: 'toolResult',
        toolCallId: '',
        toolName: '',
        content: [{ type: 'text', text: 'blank tool placeholder result must not be searchable' }],
        isError: true,
      },
    },
    {
      type: 'message',
      id: 'assistant',
      parentId: 'blank-result',
      timestamp: '2026-01-10T10:00:04Z',
      message: { role: 'assistant', content: 'searchable answer after blank tool placeholder' },
    },
  ];
  await writeFile(sessionPath, records.map((record) => JSON.stringify(record)).join('\n'));

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: createWhitespaceConversationTokenizer(),
  });

  assert.deepEqual(imported.logicalSessions[0]?.entryIds, [
    'user',
    'blank-call',
    'blank-result',
    'assistant',
  ]);
  assert.ok(
    imported.chunks.some((chunk) =>
      chunk.content.includes('searchable request before blank tool placeholder'),
    ),
  );
  assert.ok(
    imported.chunks.some((chunk) =>
      chunk.content.includes('searchable answer after blank tool placeholder'),
    ),
  );
  assert.ok(
    imported.chunks.every(
      (chunk) =>
        chunk.documentKind !== 'tool' &&
        !chunk.content.includes('blank tool placeholder result must not be searchable'),
    ),
  );
});

void test('strict session graph validation rejects invalid compaction, branch, and tool links', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invalid-graph-links-'));
  const options = { tokenizer: createWhitespaceConversationTokenizer() };
  const header = {
    type: 'session',
    version: 3,
    id: 'invalid-links',
    timestamp: '2026-01-10T10:00:00Z',
    cwd: '/project',
  };
  const root = {
    type: 'message',
    id: 'root',
    parentId: null,
    timestamp: '2026-01-10T10:00:01Z',
    message: { role: 'user', content: 'root' },
  };
  const invalidCases = [
    {
      name: 'compaction-missing-reference',
      records: [
        header,
        root,
        {
          type: 'compaction',
          id: 'compaction',
          parentId: 'root',
          timestamp: '2026-01-10T10:00:02Z',
          summary: 'invalid compaction',
          firstKeptEntryId: 'missing',
          tokensBefore: 10,
        },
      ],
      error: /compaction compaction firstKeptEntryId missing is not an ancestor/u,
    },
    {
      name: 'branch-summary-missing-reference',
      records: [
        header,
        root,
        {
          type: 'branch_summary',
          id: 'branch-summary',
          parentId: 'root',
          timestamp: '2026-01-10T10:00:02Z',
          fromId: 'missing',
          summary: 'invalid branch summary',
        },
      ],
      error: /branch summary branch-summary fromId missing does not name an entry or root/u,
    },
    {
      name: 'duplicate-tool-call',
      records: [
        header,
        {
          type: 'message',
          id: 'assistant',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'assistant',
            content: [
              { type: 'toolCall', id: 'duplicate-call', name: 'read', arguments: {} },
              { type: 'toolCall', id: 'duplicate-call', name: 'bash', arguments: {} },
            ],
          },
        },
      ],
      error: /duplicate tool call id duplicate-call/u,
    },
    {
      name: 'unresolved-tool-call-id',
      records: [
        header,
        {
          type: 'message',
          id: 'assistant',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'assistant',
            content: [{ type: 'toolCall', id: '', name: 'read', arguments: {} }],
          },
        },
      ],
      error: /toolCall\.id must be a nonempty string/u,
    },
    {
      name: 'partial-empty-tool-result',
      records: [
        header,
        {
          type: 'message',
          id: 'result',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'toolResult',
            toolCallId: '',
            toolName: 'read',
            content: [],
            isError: false,
          },
        },
      ],
      error: /toolResult\.toolCallId must be a nonempty string/u,
    },
    {
      name: 'unmatched-tool-result',
      records: [
        header,
        {
          type: 'message',
          id: 'result',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'toolResult',
            toolCallId: 'missing-call',
            toolName: 'read',
            content: [{ type: 'text', text: 'must not become searchable' }],
            isError: false,
          },
        },
      ],
      error: /tool result missing-call has no matching tool call/u,
    },
    {
      name: 'duplicate-tool-result',
      records: [
        header,
        {
          type: 'message',
          id: 'assistant',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'assistant',
            content: [{ type: 'toolCall', id: 'call', name: 'read', arguments: {} }],
          },
        },
        {
          type: 'message',
          id: 'result-one',
          parentId: 'assistant',
          timestamp: '2026-01-10T10:00:02Z',
          message: {
            role: 'toolResult',
            toolCallId: 'call',
            toolName: 'read',
            content: [],
            isError: false,
          },
        },
        {
          type: 'message',
          id: 'result-two',
          parentId: 'result-one',
          timestamp: '2026-01-10T10:00:03Z',
          message: {
            role: 'toolResult',
            toolCallId: 'call',
            toolName: 'read',
            content: [],
            isError: false,
          },
        },
      ],
      error: /duplicate tool result id call/u,
    },
    {
      name: 'mismatched-tool-name',
      records: [
        header,
        {
          type: 'message',
          id: 'assistant',
          parentId: null,
          timestamp: '2026-01-10T10:00:01Z',
          message: {
            role: 'assistant',
            content: [{ type: 'toolCall', id: 'call', name: 'read', arguments: {} }],
          },
        },
        {
          type: 'message',
          id: 'result',
          parentId: 'assistant',
          timestamp: '2026-01-10T10:00:02Z',
          message: {
            role: 'toolResult',
            toolCallId: 'call',
            toolName: 'bash',
            content: [],
            isError: false,
          },
        },
      ],
      error: /tool result call names bash, but call names read/u,
    },
  ];

  for (const fixture of invalidCases) {
    const sessionPath = join(directory, `${fixture.name}.jsonl`);
    await writeFile(
      sessionPath,
      fixture.records.map((record) => JSON.stringify(record)).join('\n'),
    );
    await assert.rejects(
      () => readSessionConversationImport(sessionPath, options),
      fixture.error,
      fixture.name,
    );
  }
});

void test('session graph rejects multiple headers and broken parent links', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invalid-graph-'));
  const duplicateHeaderPath = join(directory, 'duplicate-header.jsonl');
  const brokenParentPath = join(directory, 'broken-parent.jsonl');
  const header = {
    type: 'session',
    version: 3,
    id: 'session-invalid',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  };
  await writeFile(
    duplicateHeaderPath,
    `${JSON.stringify(header)}\n${JSON.stringify({ ...header, id: 'second' })}\n`,
  );
  await writeFile(
    brokenParentPath,
    [
      header,
      {
        type: 'message',
        id: 'orphan',
        parentId: 'missing',
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'orphaned text' },
      },
    ]
      .map((entry) => JSON.stringify(entry))
      .join('\n') + '\n',
  );
  const options = { tokenizer: createWhitespaceConversationTokenizer() };

  await assert.rejects(
    () => readSessionConversationChunks(duplicateHeaderPath, options),
    /unsupported or ambiguous/u,
  );
  await assert.rejects(
    () => readSessionConversationChunks(brokenParentPath, options),
    /entry orphan has missing parent missing/,
  );
});

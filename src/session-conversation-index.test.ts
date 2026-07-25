import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

function createWhitespaceConversationTokenizer(): ConversationTextTokenizer {
  return {
    encodeConversationText(text) {
      return {
        ids: text
          .split(/\s+/u)
          .filter(Boolean)
          .map((_, index) => index),
      };
    },
  };
}

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
            { type: 'toolCall', id: 'call-1', name: 'read', arguments: {} },
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
          content: [{ type: 'text', text: 'giant private tool output' }],
        },
      },
      {
        type: 'compaction',
        id: 'compact-1',
        parentId: 'tool-1',
        timestamp: '2026-07-24T10:04:00Z',
        summary: 'The migration plan lives in docs/migration.md',
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
  assert.ok(activeRuns.every((chunk) => chunk.schemaVersion === 3));
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
    /expected exactly one session header, found 2/,
  );
  await assert.rejects(
    () => readSessionConversationChunks(brokenParentPath, options),
    /entry orphan has missing parent missing/,
  );
});

import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallProjectIdentitySource } from './enums.js';
import { resolveProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';

const TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
  },
};

async function writeInvocationSession(
  sessionPath: string,
  sessionOrigin: string,
  argumentsValue: Record<string, unknown>,
): Promise<void> {
  const records = [
    {
      type: 'session',
      version: 3,
      id: 'invocation-session',
      timestamp: '2026-08-10T10:00:00Z',
      cwd: sessionOrigin,
    },
    {
      type: 'message',
      id: 'assistant-entry',
      parentId: null,
      timestamp: '2026-08-10T10:01:00Z',
      message: {
        role: 'assistant',
        content: [
          { type: 'text', text: 'I will inspect the source.' },
          {
            type: 'toolCall',
            id: 'call-read',
            name: 'read',
            arguments: argumentsValue,
          },
          {
            type: 'toolCall',
            id: 'call-recall',
            name: 'pi-session-recall',
            arguments: { query: 'derived result must stay excluded' },
          },
        ],
      },
    },
    {
      type: 'message',
      id: 'read-result',
      parentId: 'assistant-entry',
      timestamp: '2026-08-10T10:02:00Z',
      message: {
        role: 'toolResult',
        toolCallId: 'call-read',
        toolName: 'read',
        content: [
          { type: 'text', text: 'bulky tool result must not enter the Invocation record' },
          { type: 'image', data: 'image bytes must not enter the Invocation record' },
        ],
        isError: true,
      },
    },
    {
      type: 'message',
      id: 'recall-result',
      parentId: 'read-result',
      timestamp: '2026-08-10T10:03:00Z',
      message: {
        role: 'toolResult',
        toolCallId: 'call-recall',
        toolName: 'pi-session-recall',
        content: [{ type: 'text', text: 'derived recall output must stay excluded' }],
        isError: false,
      },
    },
    {
      type: 'message',
      id: 'bash-entry',
      parentId: 'recall-result',
      timestamp: '2026-08-10T10:04:00Z',
      message: {
        role: 'bashExecution',
        command: 'rg "Invocation record" src',
        output: 'bulky bash output must not enter the Invocation record',
        exitCode: 0,
        cancelled: false,
      },
    },
  ];
  await writeFile(sessionPath, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`);
}

void test('session import creates source-located Invocation records without payload copies', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invocation-record-'));
  const sessionPath = join(directory, 'session.jsonl');
  const argumentsValue = {
    path: '/tmp/needle.ts',
    url: 'https://example.test/issues/166',
    query: 'compact Invocation records',
    issueNumber: 166,
    branch: 'feat/issue-165-compact-recall-storage',
    offset: 4,
    limit: 25,
    timestamp: '2026-08-10T10:00:30Z',
    body: 'body payload must remain only in source',
    code: 'code payload must remain only in source',
    content: 'file body must remain only in source',
    fileBody: 'explicit file body must remain only in source',
    image: 'image payload must remain only in source',
    instructions: 'instructions must remain only in source',
    edits: [
      {
        oldText: 'old replacement text',
        newText: 'new replacement text',
        replacementText: 'replacement payload must remain only in source',
      },
    ],
    output: 'output payload must remain only in source',
    prompt: 'prompt payload must remain only in source',
    script: 'script payload must remain only in source',
    task: 'subagent task prompt must remain only in source',
  };
  await writeInvocationSession(sessionPath, directory, argumentsValue);

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: TOKENIZER,
    resolveProjectIdentity,
  });

  assert.equal(imported.invocations.length, 2);
  const toolCall = imported.invocations[0];
  assert.equal(toolCall?.kind, 'tool_call');
  assert.equal(toolCall?.toolName, 'read');
  assert.equal(toolCall?.toolCallId, 'call-read');
  assert.equal(toolCall?.sessionPath, sessionPath);
  assert.equal(toolCall?.sessionId, 'invocation-session');
  assert.equal(toolCall?.entryId, 'assistant-entry');
  assert.equal(toolCall?.sourceLineStart, 2);
  assert.equal(toolCall?.sourceLineEnd, 2);
  assert.equal(toolCall?.sourceBlockIndex, 1);
  assert.equal(toolCall?.timestamp, '2026-08-10T10:01:00Z');
  assert.equal(toolCall?.sessionOrigin, directory);
  assert.equal(
    toolCall?.projectAttribution?.identitySource,
    RecallProjectIdentitySource.NON_GIT_SESSION_ORIGIN,
  );
  assert.equal(toolCall?.isError, true);
  for (const expected of [
    'path="/tmp/needle.ts"',
    'url="https://example.test/issues/166"',
    'query="compact Invocation records"',
    'issueNumber=166',
    'branch="feat/issue-165-compact-recall-storage"',
    'offset=4',
    'limit=25',
    'timestamp="2026-08-10T10:00:30Z"',
    'body=<omitted>',
    'code=<omitted>',
    'content=<omitted>',
    'fileBody=<omitted>',
    'image=<omitted>',
    'instructions=<omitted>',
    'edits[0].oldText=<omitted>',
    'edits[0].newText=<omitted>',
    'edits[0].replacementText=<omitted>',
    'output=<omitted>',
    'prompt=<omitted>',
    'script=<omitted>',
    'task=<omitted>',
  ]) {
    assert.match(
      toolCall?.searchableText ?? '',
      new RegExp(expected.replaceAll(/[.*+?^${}()|[\]\\]/gu, '\\$&'), 'u'),
    );
  }
  for (const excluded of [
    'body payload must remain only in source',
    'code payload must remain only in source',
    'file body must remain only in source',
    'explicit file body must remain only in source',
    'image payload must remain only in source',
    'instructions must remain only in source',
    'old replacement text',
    'new replacement text',
    'replacement payload must remain only in source',
    'output payload must remain only in source',
    'prompt payload must remain only in source',
    'script payload must remain only in source',
    'subagent task prompt must remain only in source',
    'bulky tool result',
    'image bytes',
    'derived recall',
  ]) {
    assert.ok(!toolCall?.searchableText.includes(excluded));
  }

  const bashExecution = imported.invocations[1];
  assert.equal(bashExecution?.kind, 'bash_execution');
  assert.equal(bashExecution?.toolName, 'bash');
  assert.equal(bashExecution?.toolCallId, null);
  assert.equal(bashExecution?.entryId, 'bash-entry');
  assert.equal(bashExecution?.sourceLineStart, 5);
  assert.equal(bashExecution?.sourceBlockIndex, null);
  assert.equal(bashExecution?.isError, false);
  assert.match(bashExecution?.searchableText ?? '', /command="rg \\"Invocation record\\" src"/u);
  assert.ok(!bashExecution?.searchableText.includes('bulky bash output'));

  const source = await readFile(sessionPath, 'utf8');
  assert.ok(source.includes('body payload must remain only in source'));
  assert.ok(source.includes('bulky tool result must not enter the Invocation record'));
  assert.ok(source.includes('bulky bash output must not enter the Invocation record'));
});

void test('Invocation argument projection is deterministic and bounded', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invocation-bounds-'));
  const firstPath = join(directory, 'first.jsonl');
  const secondPath = join(directory, 'second.jsonl');
  const longScalar = 'x'.repeat(1_500);
  const entries = [
    ['zulu', longScalar],
    ['zebra', longScalar],
    ['zephyr', longScalar],
    ['zenith', longScalar],
    ['query', longScalar],
    ['path', longScalar],
    ['aPayloadContent', 'must be omitted'],
  ] as const;
  await writeInvocationSession(firstPath, directory, Object.fromEntries(entries));
  await writeInvocationSession(secondPath, directory, Object.fromEntries([...entries].reverse()));

  const first = await readSessionConversationImport(firstPath, { tokenizer: TOKENIZER });
  const second = await readSessionConversationImport(secondPath, { tokenizer: TOKENIZER });
  const firstText = first.invocations[0]?.searchableText ?? '';
  const secondText = second.invocations[0]?.searchableText ?? '';

  assert.equal(firstText, secondText);
  assert.ok(Array.from(firstText).length <= 4_096);
  assert.match(firstText, /aPayloadContent=<omitted>/u);
  const pathLine = firstText.split('\n').find((line) => line.startsWith('path='));
  assert.ok(pathLine);
  assert.equal(pathLine.slice('path='.length), JSON.stringify('x'.repeat(1_024)));
});

void test('Invocation argument projection reserves capacity for omitted payload markers', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invocation-omission-capacity-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeInvocationSession(sessionPath, directory, {
    aLocator: 'a'.repeat(1_024),
    bLocator: 'b'.repeat(1_024),
    cLocator: 'c'.repeat(1_024),
    dLocator: 'd'.repeat(950),
    zContent: 'payload must remain only in source',
  });

  const imported = await readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER });
  const searchableText = imported.invocations[0]?.searchableText ?? '';

  assert.ok(Array.from(searchableText).length <= 4_096);
  assert.match(searchableText, /zContent=<omitted>/u);
  assert.ok(!searchableText.includes('payload must remain only in source'));
});

void test('session import rejects malformed tool calls before producing Invocation records', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-malformed-invocation-'));
  const sessionPath = join(directory, 'malformed.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'malformed-invocation',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: directory,
      },
      {
        type: 'message',
        id: 'assistant-entry',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: {
          role: 'assistant',
          content: [{ type: 'toolCall', id: 'call-without-name', arguments: { path: '/tmp' } }],
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );

  await assert.rejects(
    () => readSessionConversationImport(sessionPath, { tokenizer: TOKENIZER }),
    /toolCall\.name must be a nonempty string/u,
  );
});

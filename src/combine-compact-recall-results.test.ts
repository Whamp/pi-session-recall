import assert from 'node:assert/strict';
import test from 'node:test';

import {
  combineCompactRecallResults,
  type CompactRecallInvocationResult,
} from './combine-compact-recall-results.js';
import { RecallEvidenceRelation } from './enums.js';
import { createTestRankedRecallSearchResult } from './recall-test-utils.js';

function createInvocationResult(index: number): CompactRecallInvocationResult {
  return {
    resultKind: 'invocation',
    kind: 'tool_call',
    toolName: 'read',
    toolCallId: `call-${index}`,
    sessionPath: `/sessions/invocation-${index}.jsonl`,
    sessionId: `invocation-session-${index}`,
    entryId: `invocation-entry-${index}`,
    sourceLineStart: index + 1,
    sourceLineEnd: index + 1,
    sourceBlockIndex: 0,
    timestamp: '2026-08-10T10:00:00Z',
    sessionOrigin: '/project',
    projectAttribution: null,
    isError: false,
    searchableText: `tool="read"\npath="/project/file-${index}.ts"`,
    content: `tool="read"\npath="/project/file-${index}.ts"`,
    rank: -10 + index,
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
  };
}

void test('mixed compact recall keeps an exact Invocation visible without displacing strong conversations', () => {
  const conversations = Array.from({ length: 5 }, (_, index) =>
    createTestRankedRecallSearchResult({
      id: `conversation-${index}`,
      rankingScore: 1 - index / 10,
    }),
  );
  const invocations = Array.from({ length: 5 }, (_, index) => createInvocationResult(index));

  const results = combineCompactRecallResults(conversations, invocations, 5);

  assert.deepEqual(
    results.map((result) => result.resultKind),
    ['conversation', 'invocation', 'conversation', 'conversation', 'conversation'],
  );
  assert.deepEqual(
    results.filter((result) => result.resultKind === 'conversation').map((result) => result.id),
    ['conversation-0', 'conversation-1', 'conversation-2', 'conversation-3'],
  );
});

void test('compact recall fills the requested limit from either projection when the other has no matches', () => {
  const conversations = [
    createTestRankedRecallSearchResult({ id: 'conversation-0' }),
    createTestRankedRecallSearchResult({ id: 'conversation-1' }),
  ];
  const invocations = [createInvocationResult(0), createInvocationResult(1)];

  assert.deepEqual(
    combineCompactRecallResults(conversations, [], 2).map((result) => result.resultKind),
    ['conversation', 'conversation'],
  );
  assert.deepEqual(
    combineCompactRecallResults([], invocations, 2).map((result) => result.resultKind),
    ['invocation', 'invocation'],
  );
});

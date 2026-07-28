import assert from 'node:assert/strict';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallSessionProjectionKind } from './enums.js';
import { createRecallSessionProjectionBaseline } from './create-recall-session-projection-baseline.js';
import type { LogicalSessionProjection } from './recall-session-projection.js';
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

void test('projection baseline keeps repeated session occurrences independent and size-bounded', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-projection-baseline-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'reused-session.jsonl');
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'reused-session',
        timestamp: '2026-07-28T00:00:00.000Z',
        cwd: '/first',
      },
      {
        type: 'message',
        id: 'first-entry',
        parentId: null,
        timestamp: '2026-07-28T00:00:01.000Z',
        message: { role: 'assistant', content: 'first occurrence evidence' },
      },
      {
        type: 'session',
        version: 3,
        id: 'reused-session',
        timestamp: '2026-07-28T01:00:00.000Z',
        cwd: '/second',
      },
      {
        type: 'message',
        id: 'second-entry',
        parentId: null,
        timestamp: '2026-07-28T01:00:01.000Z',
        message: { role: 'assistant', content: 'second occurrence evidence' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n',
  );
  const sourceByteSize = (await stat(sessionPath)).size;

  const projections = await createRecallSessionProjectionBaseline({
    physicalSessionPath: sessionPath,
    generationId: 'generation-baseline',
    tokenizer,
    expectedSourceByteSize: sourceByteSize,
  });

  const logicalProjections = projections.filter(
    (projection): projection is LogicalSessionProjection =>
      projection.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
  );
  assert.deepEqual(
    logicalProjections.map(({ logicalSessionId, eligibleContributorEntryIds }) => ({
      logicalSessionId,
      eligibleContributorEntryIds,
    })),
    [
      {
        logicalSessionId: 'reused-session@1',
        eligibleContributorEntryIds: ['first-entry'],
      },
      {
        logicalSessionId: 'reused-session@3',
        eligibleContributorEntryIds: ['second-entry'],
      },
    ],
  );
  await assert.rejects(
    () =>
      createRecallSessionProjectionBaseline({
        physicalSessionPath: sessionPath,
        generationId: 'generation-stale-baseline',
        tokenizer,
        expectedSourceByteSize: sourceByteSize - 1,
      }),
    /Recall rebuild source changed while projections were created/u,
  );
});

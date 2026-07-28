import assert from 'node:assert/strict';
import { mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallSessionProjectionKind } from './enums.js';
import { createRecallSessionProjectionBaseline } from './create-recall-session-projection-baseline.js';
import type { LogicalSessionProjection, PhysicalSessionProjection } from './recall-session-projection.js';
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

void test('projection baseline succeeds for unversioned Pi v1 JSONL and produces physical and logical projections', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-projection-baseline-v1-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionPath = join(directory, 'pi-v1-session.jsonl');
  // Pi v1: header has no version field; entries have no id/parentId but have a string timestamp
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        id: 'v1-baseline-session',
        timestamp: '2026-07-28T00:00:00.000Z',
        cwd: '/v1-workspace',
      },
      {
        type: 'message',
        timestamp: '2026-07-28T00:00:01.000Z',
        message: { role: 'user', content: 'pi v1 user message', timestamp: 1753660801000 },
      },
      {
        type: 'message',
        timestamp: '2026-07-28T00:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'pi v1 assistant reply' }],
          api: 'test',
          provider: 'test',
          model: 'test',
          usage: {},
          stopReason: 'stop',
          timestamp: 1753660802000,
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n',
  );
  const sourceByteSize = (await stat(sessionPath)).size;

  const projections = await createRecallSessionProjectionBaseline({
    physicalSessionPath: sessionPath,
    generationId: 'generation-v1-baseline',
    tokenizer,
    expectedSourceByteSize: sourceByteSize,
  });

  const physicalProjections = projections.filter(
    (p): p is PhysicalSessionProjection =>
      p.projectionKind === RecallSessionProjectionKind.PHYSICAL_SESSION,
  );
  const logicalProjections = projections.filter(
    (p): p is LogicalSessionProjection =>
      p.projectionKind === RecallSessionProjectionKind.LOGICAL_SESSION,
  );

  assert.equal(physicalProjections.length, 1);
  assert.equal(logicalProjections.length, 1);

  const physical = physicalProjections[0];
  assert.ok(physical);
  assert.equal(physical.appendCursorBytes, sourceByteSize);
  assert.equal(physical.sourcePath, sessionPath);

  const logical = logicalProjections[0];
  assert.ok(logical);
  assert.equal(logical.logicalSessionId, 'v1-baseline-session@1');
  assert.ok(
    logical.eligibleContributorEntryIds.length > 0,
    'expected at least one eligible contributor entry id',
  );
  assert.ok(
    logical.eligibleSpans.length > 0,
    'expected at least one eligible span with physical byte geometry',
  );
  // Eligible spans must reference physical byte offsets within the file
  for (const span of logical.eligibleSpans) {
    assert.ok(span.startByte >= 0);
    assert.ok(span.endByte > span.startByte);
    assert.ok(span.endByte <= sourceByteSize);
  }
});

import assert from 'node:assert/strict';
import { join, resolve } from 'node:path';
import test from 'node:test';

import fc from 'fast-check';

import {
  createRecallEvidenceOccurrenceId,
  createRecallEntryAnchorId,
  createRecallLogicalSessionOccurrenceId,
  resolveRecallPhysicalSourceIdentity,
} from './recall-source-identity.js';

void test('physical source identity survives root relocation and distinguishes paths inside the root', () => {
  const original = resolveRecallPhysicalSourceIdentity(
    '/old/pi/sessions',
    '/old/pi/sessions/project/session.jsonl',
  );
  const relocated = resolveRecallPhysicalSourceIdentity(
    '/new/pi/sessions',
    '/new/pi/sessions/project/session.jsonl',
  );
  const copied = resolveRecallPhysicalSourceIdentity(
    '/new/pi/sessions',
    '/new/pi/sessions/project/session-copy.jsonl',
  );

  assert.deepEqual(relocated, original);
  assert.notEqual(copied.physicalSourceIdentity, original.physicalSourceIdentity);
  assert.equal(original.sessionsRootRelativePath, 'project/session.jsonl');
  assert.throws(
    () => resolveRecallPhysicalSourceIdentity('/old/pi/sessions', '/old/pi/other/session.jsonl'),
    /Recall physical source path escapes configured sessions root/u,
  );
});

void test('normalized relative paths and complete-header positions produce stable distinct identities', () => {
  fc.assert(
    fc.property(
      fc.array(fc.stringMatching(/^[A-Za-z0-9_-]{1,12}$/u), { minLength: 1, maxLength: 5 }),
      fc.integer({ min: 1, max: 1_000_000 }),
      fc.integer({ min: 1, max: 1_000_000 }),
      (segments, firstHeaderLine, secondHeaderLine) => {
        fc.pre(firstHeaderLine !== secondHeaderLine);
        const root = resolve('/tmp/disposable-sessions');
        const sourcePath = join(root, ...segments, 'session.jsonl');
        const source = resolveRecallPhysicalSourceIdentity(root, sourcePath);
        const normalizedSource = resolveRecallPhysicalSourceIdentity(
          `${root}/nested/..`,
          join(root, '.', ...segments, 'session.jsonl'),
        );
        assert.deepEqual(normalizedSource, source);

        const firstLogicalOccurrenceId = createRecallLogicalSessionOccurrenceId(
          source.physicalSourceIdentity,
          firstHeaderLine,
        );
        const repeatedLogicalOccurrenceId = createRecallLogicalSessionOccurrenceId(
          source.physicalSourceIdentity,
          secondHeaderLine,
        );
        assert.notEqual(firstLogicalOccurrenceId, repeatedLogicalOccurrenceId);

        const firstAnchorId = createRecallEntryAnchorId({
          physicalSourceIdentity: source.physicalSourceIdentity,
          logicalSessionOccurrenceId: firstLogicalOccurrenceId,
          entryId: 'same-entry-id',
          sourceLine: firstHeaderLine + 1,
          startByte: 100,
          endByte: 200,
        });
        const repeatedAnchorId = createRecallEntryAnchorId({
          physicalSourceIdentity: source.physicalSourceIdentity,
          logicalSessionOccurrenceId: repeatedLogicalOccurrenceId,
          entryId: 'same-entry-id',
          sourceLine: secondHeaderLine + 1,
          startByte: 100,
          endByte: 200,
        });
        assert.notEqual(firstAnchorId, repeatedAnchorId);

        const firstOccurrenceId = createRecallEvidenceOccurrenceId({
          physicalSourceIdentity: source.physicalSourceIdentity,
          logicalSessionOccurrenceId: firstLogicalOccurrenceId,
          entryId: 'same-entry-id',
          evidencePart: 'result',
          sourceLineStart: firstHeaderLine + 1,
          sourceLineEnd: firstHeaderLine + 1,
          sourceBlockStart: 0,
          sourceBlockEnd: 0,
          characterStart: 0,
          characterEnd: 12,
          tokenStart: 0,
          tokenEnd: 2,
          textRunIndex: 0,
          chunkIndex: 0,
        });
        const repeatedOccurrenceId = createRecallEvidenceOccurrenceId({
          physicalSourceIdentity: source.physicalSourceIdentity,
          logicalSessionOccurrenceId: repeatedLogicalOccurrenceId,
          entryId: 'same-entry-id',
          evidencePart: 'result',
          sourceLineStart: secondHeaderLine + 1,
          sourceLineEnd: secondHeaderLine + 1,
          sourceBlockStart: 0,
          sourceBlockEnd: 0,
          characterStart: 0,
          characterEnd: 12,
          tokenStart: 0,
          tokenEnd: 2,
          textRunIndex: 0,
          chunkIndex: 0,
        });
        assert.notEqual(firstOccurrenceId, repeatedOccurrenceId);
      },
    ),
  );
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, open, rename, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import fc from 'fast-check';

import { RecallAppendDeltaStatus, RecallProjectionRepairReason } from './enums.js';
import {
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { readRecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import {
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
} from './enums.js';

function fingerprint(source: Buffer, cursor: number): string {
  return createHash('sha256')
    .update(source.subarray(Math.max(0, cursor - 4_096), cursor))
    .digest('hex');
}

async function createProjection(
  sessionPath: string,
  source: Buffer,
  cursor: number,
  lines: number,
): Promise<PhysicalSessionProjection> {
  const metadata = await stat(sessionPath, { bigint: true });
  const physicalSessionId = 'physical-session';
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId: 'generation',
    physicalSessionId,
    sourcePath: sessionPath,
    sourceDevice: metadata.dev.toString(),
    sourceInode: metadata.ino.toString(),
    appendCursorBytes: cursor,
    appendCursorLines: lines,
    boundaryFingerprint: fingerprint(source, cursor),
    lastEntryId: null,
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    markerCheckpoint: {
      generationId: 'generation',
      coveredMarkerIds: [],
      runtimeSequences: [],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function record(value: object, lineEnding = '\n'): Buffer {
  return Buffer.from(`${JSON.stringify(value)}${lineEnding}`);
}

void test('append reader advances through complete LF-framed records and retains a partial final record', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-reader-'));
  const sessionPath = join(directory, 'session.jsonl');
  const header = record({
    type: 'session',
    version: 3,
    id: 'logical-1',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  });
  await writeFile(sessionPath, header);
  const projection = await createProjection(sessionPath, header, header.length, 1);
  const complete = record(
    {
      type: 'message',
      id: 'entry-1',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content: 'complete' },
    },
    '\r\n',
  );
  const partial = Buffer.from('{"type":"message","id":"entry-2"');
  await writeFile(sessionPath, Buffer.concat([header, complete, partial]));

  const delta = await readRecallSessionAppendDelta(sessionPath, projection);

  assert.equal(delta.status, RecallAppendDeltaStatus.APPENDED);
  if (delta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  assert.equal(delta.appendCursorBytes, header.length + complete.length);
  assert.equal(delta.appendCursorLines, 2);
  assert.equal(delta.partialFinalRecordBytes, partial.length);
  assert.deepEqual(delta.records, [
    {
      sourceLine: 2,
      startByte: header.length,
      endByte: header.length + complete.length,
      value: {
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-24T10:01:00Z',
        message: { role: 'user', content: 'complete' },
      },
    },
  ]);
});

void test('partial final records remain at the cursor and commit exactly once after their LF arrives', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-partial-'));
  const sessionPath = join(directory, 'session.jsonl');
  const header = record({
    type: 'session',
    version: 3,
    id: 'logical-partial',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  });
  const partial = Buffer.from(
    JSON.stringify({
      type: 'message',
      id: 'entry-partial',
      parentId: null,
      timestamp: '2026-07-24T10:01:00Z',
      message: { role: 'user', content: 'completed later' },
    }),
  );
  await writeFile(sessionPath, Buffer.concat([header, partial]));
  const projection = await createProjection(sessionPath, header, header.length, 1);
  const beforeLf = await readRecallSessionAppendDelta(sessionPath, projection);
  assert.equal(beforeLf.status, RecallAppendDeltaStatus.APPENDED);
  if (beforeLf.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  assert.equal(beforeLf.appendCursorBytes, header.length);
  assert.equal(beforeLf.records.length, 0);

  await writeFile(sessionPath, Buffer.concat([header, partial, Buffer.from('\n')]));
  const afterLf = await readRecallSessionAppendDelta(sessionPath, projection);
  assert.equal(afterLf.status, RecallAppendDeltaStatus.APPENDED);
  if (afterLf.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  assert.equal(afterLf.appendCursorBytes, header.length + partial.length + 1);
  assert.deepEqual(
    afterLf.records.map(({ value }) => value.id),
    ['entry-partial'],
  );
});

void test('append reader reads only the bounded fingerprint and bytes at or after the cursor', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-bounds-'));
  const sessionPath = join(directory, 'session.jsonl');
  const padding = Buffer.alloc(8_192, 0x20);
  const appended = record({ type: 'leaf', targetId: null });
  await writeFile(sessionPath, Buffer.concat([padding, appended]));
  const projection = await createProjection(sessionPath, padding, padding.length, 4);
  const reads: Array<{ start: number; endExclusive: number }> = [];

  const delta = await readRecallSessionAppendDelta(sessionPath, projection, {
    async *readRange(path, start, endExclusive) {
      reads.push({ start, endExclusive });
      const handle = await open(path, 'r');
      try {
        const bytes = Buffer.alloc(endExclusive - start);
        await handle.read(bytes, 0, bytes.length, start);
        yield bytes;
      } finally {
        await handle.close();
      }
    },
  });

  assert.equal(delta.status, RecallAppendDeltaStatus.APPENDED);
  assert.deepEqual(reads, [
    { start: padding.length - 4_096, endExclusive: padding.length },
    { start: padding.length, endExclusive: padding.length + appended.length },
  ]);
});

void test('arbitrary source byte chunking preserves complete JSONL framing and exact cursor advancement', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-property-'));
  const sessionPath = join(directory, 'session.jsonl');
  await fc.assert(
    fc.asyncProperty(
      fc.array(fc.string({ maxLength: 30 }), { minLength: 1, maxLength: 12 }),
      fc.array(fc.integer({ min: 1, max: 37 }), { minLength: 1, maxLength: 12 }),
      async (texts, chunkSizes) => {
        const source = Buffer.concat([
          record({
            type: 'session',
            version: 3,
            id: 'logical-property',
            timestamp: '2026-07-24T10:00:00Z',
            cwd: '/project',
          }),
          ...texts.map((text, index) =>
            record({
              type: 'message',
              id: `entry-${index}`,
              parentId: index === 0 ? null : `entry-${index - 1}`,
              timestamp: '2026-07-24T10:01:00Z',
              message: { role: 'user', content: text },
            }),
          ),
        ]);
        await writeFile(sessionPath, source);
        const projection = await createProjection(sessionPath, source, 0, 0);
        const expected = await readRecallSessionAppendDelta(sessionPath, projection);
        const chunked = await readRecallSessionAppendDelta(sessionPath, projection, {
          async *readRange(path, start, endExclusive) {
            assert.equal(path, sessionPath);
            let position = start;
            let chunkIndex = 0;
            while (position < endExclusive) {
              const chunkSize = chunkSizes[chunkIndex % chunkSizes.length] ?? 1;
              const next = Math.min(endExclusive, position + chunkSize);
              yield source.subarray(position, next);
              position = next;
              chunkIndex += 1;
            }
          },
        });
        assert.deepEqual(chunked, expected);
        assert.equal(
          chunked.status === RecallAppendDeltaStatus.APPENDED ? chunked.appendCursorBytes : -1,
          source.length,
        );
      },
    ),
    { numRuns: 50 },
  );
});

void test('append reader classifies malformed complete JSON and historical v1 layout for reconciliation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-layout-'));
  const sessionPath = join(directory, 'session.jsonl');
  await writeFile(sessionPath, '{not-json}\n');
  let projection = await createProjection(sessionPath, Buffer.alloc(0), 0, 0);
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.MALFORMED_GRAPH,
  });

  const historical = record({
    type: 'session',
    id: 'v1',
    timestamp: '2026-07-24T10:00:00Z',
    cwd: '/project',
  });
  await writeFile(sessionPath, historical);
  projection = await createProjection(sessionPath, Buffer.alloc(0), 0, 0);
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.UNSUPPORTED_LAYOUT,
  });
});

void test('append reader returns actionable reconciliation for shrink, boundary mismatch, replaced identity, and missing cursor', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-append-reconcile-'));
  const sessionPath = join(directory, 'session.jsonl');
  const source = record({ type: 'leaf', targetId: null });
  await writeFile(sessionPath, source);
  const projection = await createProjection(sessionPath, source, source.length, 1);

  await writeFile(sessionPath, source.subarray(0, source.length - 1));
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.SOURCE_SHRANK,
  });

  await writeFile(sessionPath, Buffer.from('x'.repeat(source.length)));
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.BOUNDARY_MISMATCH,
  });

  const replacementPath = join(directory, 'replacement.jsonl');
  await writeFile(replacementPath, source);
  await rename(replacementPath, sessionPath);
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.SOURCE_IDENTITY_MISMATCH,
  });

  Reflect.deleteProperty(projection, 'appendCursorBytes');
  assert.deepEqual(await readRecallSessionAppendDelta(sessionPath, projection), {
    status: RecallAppendDeltaStatus.REQUIRES_RECONCILIATION,
    repairReason: RecallProjectionRepairReason.APPEND_CURSOR_MISSING,
  });
});

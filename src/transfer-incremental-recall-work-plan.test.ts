import assert from 'node:assert/strict';
import { access, appendFile, mkdir, mkdtemp, open, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallSessionProjectionKind, RecallWorkMarkerTrigger } from './enums.js';
import { createPhysicalSessionProjectionId } from './recall-session-projection.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';
import { transferIncrementalRecallWorkPlan } from './transfer-incremental-recall-work-plan.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

void test('nonzero durable append cursor commits new evidence without a whole-session read', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-transfer-work-plan-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'markers');
  const sessionPath = join(sessionsDirectory, 'session.jsonl');
  const projectionDatabasePath = join(directory, 'projections');
  const evidenceDatabasePath = join(directory, 'evidence');
  const generationId = 'generation_transfer';
  const physicalSessionId = 'session-transfer';
  await mkdir(sessionsDirectory, { recursive: true });
  await mkdir(markerSpoolDirectory, { recursive: true });
  await writeFile(
    sessionPath,
    [
      {
        type: 'session',
        version: 3,
        id: physicalSessionId,
        timestamp: '2026-07-28T00:00:00Z',
        cwd: '/project',
      },
      {
        type: 'message',
        id: 'entry-1',
        parentId: null,
        timestamp: '2026-07-28T00:00:01Z',
        message: {
          role: 'user',
          content: `durable marker searchable evidence ${'old-body-padding '.repeat(400)}`,
        },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n',
  );
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sessionPath,
    runtimeInstanceId: 'runtime-transfer',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: { kind: RecallWorkMarkerTrigger.DEPARTURE },
  } as const;
  const marker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  const markerPath = join(markerSpoolDirectory, `${marker.markerId}.json`);
  await writeFile(markerPath, '{}\n');
  const emptyProjectionStore = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
  });
  emptyProjectionStore.close();

  const transferDependencies = {
    lockPath: join(directory, 'operation.lock'),
    evidenceDatabasePath,
    projectionDatabasePath,
    embeddingDimensions: 3,
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return {
            ids: text
              .split(/\s+/u)
              .filter(Boolean)
              .map((word) => word.length),
          };
        },
      };
    },
    async resolveProjectIdentity() {
      return null;
    },
    embeddingCache: {
      async resolveEmbeddingVectors(texts: readonly string[]) {
        return {
          vectors: texts.map(() => [1, 0, 0]),
          cacheHits: 0,
          newlyEmbeddedChunks: texts.length,
          embeddingRequestCount: texts.length === 0 ? 0 : 1,
          embeddingCacheResolutionMilliseconds: 0,
          embeddingServerRequestMilliseconds: 0,
        };
      },
    },
  };
  await transferIncrementalRecallWorkPlan({
    ...transferDependencies,
    workPlan: {
      targetGenerationId: generationId,
      markerSpoolDirectory,
      discoveredMarkerCount: 1,
      sourceMarkerIds: [marker.markerId],
      workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
      quarantineDiagnostics: [],
    },
  });

  const initialSourceSize = Number((await stat(sessionPath)).size);
  await appendFile(
    sessionPath,
    `${JSON.stringify({
      type: 'message',
      id: 'entry-2',
      parentId: 'entry-1',
      timestamp: '2026-07-28T00:00:02Z',
      message: { role: 'assistant', content: 'new bounded append evidence' },
    })}\n`,
  );
  const secondMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: 2,
  } as const;
  const secondMarker: RecallWorkMarker = {
    ...secondMarkerIdentity,
    markerId: createRecallWorkMarkerId(secondMarkerIdentity),
  };
  const secondMarkerPath = join(markerSpoolDirectory, `${secondMarker.markerId}.json`);
  await writeFile(secondMarkerPath, '{}\n');
  const readRanges: Array<{ startByte: number; endByteExclusive: number }> = [];
  await transferIncrementalRecallWorkPlan({
    ...transferDependencies,
    workPlan: {
      targetGenerationId: generationId,
      markerSpoolDirectory,
      discoveredMarkerCount: 1,
      sourceMarkerIds: [secondMarker.markerId],
      workItems: [{ marker: secondMarker, coveredMarkerIds: [secondMarker.markerId] }],
      quarantineDiagnostics: [],
    },
    async *readRange(sourcePath, startByte, endByteExclusive) {
      readRanges.push({ startByte, endByteExclusive });
      const handle = await open(sourcePath, 'r');
      try {
        const bytes = Buffer.alloc(endByteExclusive - startByte);
        const result = await handle.read(bytes, 0, bytes.length, startByte);
        yield bytes.subarray(0, result.bytesRead);
      } finally {
        await handle.close();
      }
    },
  });
  const finalSourceSize = Number((await stat(sessionPath)).size);
  assert.ok(
    readRanges.some(
      (range) =>
        range.startByte === Math.max(0, initialSourceSize - 4_096) &&
        range.endByteExclusive === initialSourceSize,
    ),
  );
  assert.ok(
    readRanges.some(
      (range) =>
        range.startByte === initialSourceSize && range.endByteExclusive === finalSourceSize,
    ),
  );
  assert.equal(
    readRanges.some(
      ({ startByte, endByteExclusive }) => startByte === 0 && endByteExclusive === finalSourceSize,
    ),
    false,
  );

  const evidenceStore = openZvecConversationStore({
    databasePath: evidenceDatabasePath,
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const initialResult = await evidenceStore.searchLexicalCandidates('searchable evidence', 5);
    assert.equal(
      initialResult.some(({ content }) => content.includes('searchable evidence')),
      true,
    );
    const appendedResult = await evidenceStore.searchLexicalCandidates('bounded append', 5);
    const appendedEvidence = appendedResult.find(({ content }) =>
      content.includes('bounded append evidence'),
    );
    assert.ok(appendedEvidence);
    assert.equal(appendedEvidence.sourceLineStart, 3);
    assert.equal(appendedEvidence.sourceLineEnd, 3);
  } finally {
    evidenceStore.close();
  }
  const projectionStore = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const physicalProjectionId = createPhysicalSessionProjectionId(physicalSessionId);
    const physicalProjection = projectionStore
      .fetchProjections([physicalProjectionId])
      .get(physicalProjectionId);
    assert.equal(
      physicalProjection?.markerCheckpoint.coveredMarkerIds.includes(marker.markerId),
      true,
    );
    assert.equal(physicalProjection?.projectionKind, RecallSessionProjectionKind.PHYSICAL_SESSION);
    if (
      physicalProjection === undefined ||
      physicalProjection.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION
    ) {
      return;
    }
    assert.equal(physicalProjection.appendCursorBytes, finalSourceSize);
    assert.equal(
      physicalProjection.markerCheckpoint.coveredMarkerIds.filter(
        (markerId) => markerId === secondMarker.markerId,
      ).length,
      1,
    );
  } finally {
    projectionStore.close();
  }
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });
  await assert.rejects(() => access(secondMarkerPath), { code: 'ENOENT' });
});

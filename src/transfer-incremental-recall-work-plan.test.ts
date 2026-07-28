import assert from 'node:assert/strict';
import { access, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallWorkMarkerTrigger } from './enums.js';
import { createPhysicalSessionProjectionId } from './recall-session-projection.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';
import { transferIncrementalRecallWorkPlan } from './transfer-incremental-recall-work-plan.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

void test('durable marker becomes searchable evidence before checkpoint acknowledgement', async (t) => {
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
        message: { role: 'user', content: 'durable marker searchable evidence' },
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

  await transferIncrementalRecallWorkPlan({
    workPlan: {
      targetGenerationId: generationId,
      markerSpoolDirectory,
      discoveredMarkerCount: 1,
      sourceMarkerIds: [marker.markerId],
      workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
      quarantineDiagnostics: [],
    },
    lockPath: join(directory, 'operation.lock'),
    evidenceDatabasePath,
    projectionDatabasePath,
    embeddingDimensions: 3,
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    async loadTokenizer() {
      return {
        encodeConversationText(text) {
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
      async resolveEmbeddingVectors(texts) {
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
  });

  const evidenceStore = openZvecConversationStore({
    databasePath: evidenceDatabasePath,
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const result = await evidenceStore.searchLexicalCandidates('searchable evidence', 5);
    assert.equal(
      result.some(({ content }) => content.includes('searchable evidence')),
      true,
    );
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
  } finally {
    projectionStore.close();
  }
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });
});

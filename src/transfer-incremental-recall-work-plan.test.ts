import assert from 'node:assert/strict';
import {
  access,
  appendFile,
  mkdir,
  mkdtemp,
  open,
  rm,
  stat,
  utimes,
  writeFile,
} from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  RecallDiagnosticOperationKind,
  RecallDiagnosticStatus,
  RecallEligibilityThreshold,
  RecallIncrementalTransferOutcomeKind,
  RecallSessionProjectionKind,
  RecallWorkMarkerTrigger,
} from './enums.js';
import type { RecallIncrementalDiagnosticCompletion } from './recall-operation-diagnostics.js';
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
  const initialGrowthAtEpochMilliseconds = 1_753_315_200_000;
  await utimes(
    sessionPath,
    initialGrowthAtEpochMilliseconds / 1_000,
    initialGrowthAtEpochMilliseconds / 1_000,
  );
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sessionPath,
    runtimeInstanceId: 'runtime-transfer',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: initialGrowthAtEpochMilliseconds,
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

  let nowEpochMilliseconds = initialGrowthAtEpochMilliseconds + 60_000;
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
    nowEpochMilliseconds: () => nowEpochMilliseconds,
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
  const initialOutcome = await transferIncrementalRecallWorkPlan({
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
  assert.equal(initialOutcome.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);

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
  const secondGrowthAtEpochMilliseconds = initialGrowthAtEpochMilliseconds + 1_000;
  await utimes(
    sessionPath,
    secondGrowthAtEpochMilliseconds / 1_000,
    secondGrowthAtEpochMilliseconds / 1_000,
  );
  nowEpochMilliseconds = secondGrowthAtEpochMilliseconds + 60_000;
  const secondMarkerIdentity = {
    ...markerIdentity,
    runtimeSequence: 2,
    createdAtEpochMilliseconds: secondGrowthAtEpochMilliseconds,
  } as const;
  const secondMarker: RecallWorkMarker = {
    ...secondMarkerIdentity,
    markerId: createRecallWorkMarkerId(secondMarkerIdentity),
  };
  const secondMarkerPath = join(markerSpoolDirectory, `${secondMarker.markerId}.json`);
  await writeFile(secondMarkerPath, '{}\n');
  const readRanges: Array<{ startByte: number; endByteExclusive: number }> = [];
  const appendedOutcome = await transferIncrementalRecallWorkPlan({
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
  assert.equal(appendedOutcome.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
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

void test('committed transfer forwards write-window phase diagnostics after crash-only quiescence', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-crash-quiescence-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'markers');
  const sessionPath = join(sessionsDirectory, 'session.jsonl');
  const projectionDatabasePath = join(directory, 'projections');
  const evidenceDatabasePath = join(directory, 'evidence');
  const generationId = 'generation_crash_quiescence';
  const physicalSessionId = 'session-crash-quiescence';
  const growthAtEpochMilliseconds = 1_753_315_200_000;
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
        id: 'active-entry',
        parentId: null,
        timestamp: '2026-07-28T00:00:01Z',
        message: { role: 'user', content: 'crash quiescence searchable tail' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n') + '\n',
  );
  await utimes(sessionPath, growthAtEpochMilliseconds / 1_000, growthAtEpochMilliseconds / 1_000);
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sessionPath,
    runtimeInstanceId: 'runtime-crash',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: growthAtEpochMilliseconds,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  const marker: RecallWorkMarker = {
    ...markerIdentity,
    markerId: createRecallWorkMarkerId(markerIdentity),
  };
  const markerPath = join(markerSpoolDirectory, `${marker.markerId}.json`);
  await writeFile(markerPath, '{}\n');
  const projectionStore = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
  });
  projectionStore.close();
  const workPlan = {
    targetGenerationId: generationId,
    markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [marker.markerId],
    workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
    quarantineDiagnostics: [],
  };
  let nowEpochMilliseconds = growthAtEpochMilliseconds + 30 * 60_000 - 1;
  let monotonicMilliseconds = 0;
  let tokenizerLoadCount = 0;
  const readRanges: Array<{ startByte: number; endByteExclusive: number }> = [];
  const incrementalDiagnostics: RecallIncrementalDiagnosticCompletion[] = [];
  const transferDependencies = {
    workPlan,
    lockPath: join(directory, 'operation.lock'),
    evidenceDatabasePath,
    projectionDatabasePath,
    embeddingDimensions: 3,
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    nowEpochMilliseconds: () => nowEpochMilliseconds,
    monotonicMilliseconds: () => {
      monotonicMilliseconds += 1;
      return monotonicMilliseconds;
    },
    async loadTokenizer() {
      tokenizerLoadCount += 1;
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
    operationDiagnostics: {
      recordIncrementalOperation(completion: RecallIncrementalDiagnosticCompletion) {
        incrementalDiagnostics.push(completion);
      },
    },
    async *readRange(sourcePath: string, startByte: number, endByteExclusive: number) {
      readRanges.push({ startByte, endByteExclusive });
      const handle = await open(sourcePath, 'r');
      try {
        const bytes = Buffer.alloc(endByteExclusive - startByte);
        const read = await handle.read(bytes, 0, bytes.length, startByte);
        yield bytes.subarray(0, read.bytesRead);
      } finally {
        await handle.close();
      }
    },
  };

  const deferred = await transferIncrementalRecallWorkPlan(transferDependencies);
  assert.deepEqual(deferred, {
    kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
    threshold: RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
    readyAtEpochMilliseconds: growthAtEpochMilliseconds + 30 * 60_000,
  });
  assert.equal(readRanges.length > 0, true);
  assert.equal(tokenizerLoadCount, 0);
  assert.equal(incrementalDiagnostics.length, 0);
  await access(markerPath);
  const deferredProjectionStore = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    assert.equal(
      deferredProjectionStore.fetchProjections([
        createPhysicalSessionProjectionId(physicalSessionId),
      ]).size,
      0,
    );
  } finally {
    deferredProjectionStore.close();
  }

  nowEpochMilliseconds = growthAtEpochMilliseconds + 30 * 60_000;
  const committed = await transferIncrementalRecallWorkPlan(transferDependencies);
  assert.equal(committed.kind, RecallIncrementalTransferOutcomeKind.COMMITTED);
  assert.equal(tokenizerLoadCount, 1);
  assert.equal(incrementalDiagnostics.length, 2);
  const preparationDiagnostic = incrementalDiagnostics.find(
    ({ operationKind }) => operationKind === RecallDiagnosticOperationKind.INCREMENTAL_WORKER,
  );
  assert.ok(preparationDiagnostic);
  assert.equal(preparationDiagnostic.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(preparationDiagnostic.metrics.appendedByteCount > 0, true);
  assert.equal(preparationDiagnostic.metrics.parsedEntryCount, 2);
  assert.equal(preparationDiagnostic.metrics.tokenizerMilliseconds > 0, true);
  assert.equal(preparationDiagnostic.metrics.generationId, generationId);
  const writeWindowDiagnostic = incrementalDiagnostics.find(
    ({ operationKind }) => operationKind === RecallDiagnosticOperationKind.WRITE_WINDOW,
  );
  assert.ok(writeWindowDiagnostic);
  assert.equal(writeWindowDiagnostic.operationKind, RecallDiagnosticOperationKind.WRITE_WINDOW);
  assert.equal(writeWindowDiagnostic.status, RecallDiagnosticStatus.SUCCEEDED);
  assert.equal(writeWindowDiagnostic.metrics.generationId, generationId);
  assert.equal(
    [
      writeWindowDiagnostic.metrics.lockWaitMilliseconds,
      writeWindowDiagnostic.metrics.evidenceOpenMilliseconds,
      writeWindowDiagnostic.metrics.evidenceWriteMilliseconds,
      writeWindowDiagnostic.metrics.projectionOpenMilliseconds,
      writeWindowDiagnostic.metrics.projectionCommitMilliseconds,
      writeWindowDiagnostic.metrics.closeMilliseconds,
      writeWindowDiagnostic.metrics.checkpointObservationMilliseconds,
      writeWindowDiagnostic.metrics.markerAcknowledgementMilliseconds,
    ].every((phaseMilliseconds) => Number.isFinite(phaseMilliseconds)),
    true,
  );
  const evidenceStore = openZvecConversationStore({
    databasePath: evidenceDatabasePath,
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const results = await evidenceStore.searchLexicalCandidates('quiescence searchable', 5);
    assert.equal(
      results.some(({ content }) => content.includes('crash quiescence searchable tail')),
      true,
    );
  } finally {
    evidenceStore.close();
  }
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });
});

void test('33 actual prepared documents retain the marker until five minutes', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-large-transfer-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'markers');
  const sessionPath = join(sessionsDirectory, 'session.jsonl');
  const projectionDatabasePath = join(directory, 'projections');
  const evidenceDatabasePath = join(directory, 'evidence');
  const generationId = 'generation_large_transfer';
  const physicalSessionId = 'session-large-transfer';
  const growthAtEpochMilliseconds = 1_753_315_200_000;
  await mkdir(sessionsDirectory, { recursive: true });
  await mkdir(markerSpoolDirectory, { recursive: true });
  const records: Array<Record<string, unknown>> = [
    {
      type: 'session',
      version: 3,
      id: physicalSessionId,
      timestamp: '2026-07-28T00:00:00Z',
      cwd: '/project',
    },
  ];
  let parentId: string | null = null;
  for (let turn = 1; turn <= 11; turn += 1) {
    const userEntryId = `user-${turn}`;
    records.push({
      type: 'message',
      id: userEntryId,
      parentId,
      timestamp: `2026-07-28T00:00:${String(turn * 2 - 1).padStart(2, '0')}Z`,
      message: { role: 'user', content: `large transfer user ${turn}` },
    });
    const assistantEntryId = `assistant-${turn}`;
    records.push({
      type: 'message',
      id: assistantEntryId,
      parentId: userEntryId,
      timestamp: `2026-07-28T00:00:${String(turn * 2).padStart(2, '0')}Z`,
      message: { role: 'assistant', content: `large transfer assistant ${turn}` },
    });
    parentId = assistantEntryId;
  }
  await writeFile(sessionPath, records.map((record) => JSON.stringify(record)).join('\n') + '\n');
  await utimes(sessionPath, growthAtEpochMilliseconds / 1_000, growthAtEpochMilliseconds / 1_000);
  const markerIdentity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: sessionPath,
    runtimeInstanceId: 'runtime-large',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: growthAtEpochMilliseconds,
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
  let nowEpochMilliseconds = growthAtEpochMilliseconds + 60_000;
  const options = {
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
    nowEpochMilliseconds: () => nowEpochMilliseconds,
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

  const deferred = await transferIncrementalRecallWorkPlan(options);
  assert.deepEqual(deferred, {
    kind: RecallIncrementalTransferOutcomeKind.DEFERRED,
    threshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
    readyAtEpochMilliseconds: growthAtEpochMilliseconds + 5 * 60_000,
  });
  await access(markerPath);
  const deferredProjectionStore = openZvecSessionProjectionStore({
    databasePath: projectionDatabasePath,
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    assert.equal(
      deferredProjectionStore.fetchProjections([
        createPhysicalSessionProjectionId(physicalSessionId),
      ]).size,
      0,
    );
  } finally {
    deferredProjectionStore.close();
  }

  nowEpochMilliseconds = growthAtEpochMilliseconds + 5 * 60_000;
  const committed = await transferIncrementalRecallWorkPlan(options);
  assert.deepEqual(committed, {
    kind: RecallIncrementalTransferOutcomeKind.COMMITTED,
    committedDocumentCount: 33,
  });
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });
});

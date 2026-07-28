import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { access, mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  commitIncrementalRecallTransfer,
  type IncrementalRecallCommitEvidenceStore,
  type IncrementalRecallCommitProjectionStore,
} from './commit-incremental-recall-transfer.js';
import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import {
  assertRecallWriteWindowAvailableForRead,
  inspectRecallWriteWindow,
} from './coordinate-recall-write-window.js';
import {
  RecallProjectionEncodingStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';
import type { PreparedIncrementalRecallTransfer } from './prepare-incremental-recall-transfer.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';
import {
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';

const generationId = 'generation_commit';
const physicalSessionId = 'physical-session-commit';
const physicalProjectionId = createPhysicalSessionProjectionId(physicalSessionId);

function createWorkPlan(): RecallMarkerReplayWorkPlan {
  const identity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: '/isolated/sessions/commit.jsonl',
    runtimeInstanceId: 'runtime-commit',
    runtimeSequence: 4,
    createdAtEpochMilliseconds: 100,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  const marker: RecallWorkMarker = {
    ...identity,
    markerId: createRecallWorkMarkerId(identity),
  };
  return {
    targetGenerationId: generationId,
    markerSpoolDirectory: '/isolated/markers',
    discoveredMarkerCount: 1,
    sourceMarkerIds: [marker.markerId],
    workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
    quarantineDiagnostics: [],
  };
}

function createPhysicalProjection(workPlan: RecallMarkerReplayWorkPlan): PhysicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: physicalProjectionId,
    generationId,
    physicalSessionId,
    sourcePath: '/isolated/sessions/commit.jsonl',
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 100,
    appendCursorLines: 2,
    boundaryFingerprint: 'b'.repeat(64),
    lastEntryId: 'entry-1',
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: [...workPlan.sourceMarkerIds],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-commit', sequence: 4 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createPreparedTransfer(documentCount: number): PreparedIncrementalRecallTransfer {
  const workPlan = createWorkPlan();
  const physicalProjection = createPhysicalProjection(workPlan);
  return {
    status: RecallProjectionEncodingStatus.ENCODED,
    targetGenerationId: generationId,
    documents: Array.from({ length: documentCount }, (_, index) => ({
      ...createTestSessionConversationChunk({
        id: `commit-document-${index}`,
        physicalSessionProjectionId: physicalProjectionId,
        content: `commit evidence ${index} IDENTIFIER_${index}`,
      }),
      isDenseSearchable: true,
      embedding: [1, index, 0],
    })),
    checkpointIntent: { physicalProjection, logicalProjections: [] },
    workPlan,
    cacheHits: documentCount,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
  };
}

interface FakeCommitStores {
  events: string[];
  openEvidenceStore(): IncrementalRecallCommitEvidenceStore;
  openProjectionStore(mode: 'read' | 'write'): IncrementalRecallCommitProjectionStore;
  acknowledgementCount: number;
  acknowledge(): Promise<number>;
  optimizeCallCount: number;
}

function createFakeCommitStores(options: {
  prepared: PreparedIncrementalRecallTransfer;
  failEvidencePosition?: number;
  failProjection?: boolean;
  failEvidenceClose?: boolean;
  failProjectionCloseMode?: 'read' | 'write';
  omitObservedCheckpoint?: boolean;
}): FakeCommitStores {
  const events: string[] = [];
  let acknowledgementCount = 0;
  let optimizeCallCount = 0;
  return {
    events,
    openEvidenceStore() {
      events.push('evidence-open');
      return {
        async upsertChunks(documents) {
          events.push(`evidence-write:${documents.length}`);
          for (const [index] of documents.entries()) {
            if (index === options.failEvidencePosition) {
              throw new Error(`fake evidence status failed at position ${index}`);
            }
          }
        },
        close() {
          events.push('evidence-close');
          if (options.failEvidenceClose) {
            throw new Error('fake evidence close failed');
          }
        },
        optimize() {
          optimizeCallCount += 1;
        },
      };
    },
    openProjectionStore(mode) {
      events.push(`projection-open:${mode}`);
      return {
        async upsertProjections() {
          events.push('projection-write');
          if (options.failProjection) {
            throw new Error('fake projection status failed');
          }
        },
        fetchProjections() {
          events.push('projection-fetch');
          return options.omitObservedCheckpoint
            ? new Map()
            : new Map([
                [
                  options.prepared.checkpointIntent.physicalProjection.projectionId,
                  options.prepared.checkpointIntent.physicalProjection,
                ],
              ]);
        },
        close() {
          events.push(`projection-close:${mode}`);
          if (options.failProjectionCloseMode === mode) {
            throw new Error(`fake projection ${mode} close failed`);
          }
        },
      };
    },
    get acknowledgementCount() {
      return acknowledgementCount;
    },
    async acknowledge() {
      events.push('marker-acknowledge');
      acknowledgementCount += 1;
      return options.prepared.workPlan.sourceMarkerIds.length;
    },
    get optimizeCallCount() {
      return optimizeCallCount;
    },
  };
}

void test('commit writes at most 32 evidence documents per closed window, then observes checkpoint before acknowledgement', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-commit-window-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const prepared = createPreparedTransfer(33);
  const stores = createFakeCommitStores({ prepared });

  const result = await commitIncrementalRecallTransfer({
    prepared,
    lockPath: join(directory, 'operation.lock'),
    evidenceDatabasePath: join(directory, 'evidence'),
    projectionDatabasePath: join(directory, 'projections'),
    embeddingDimensions: 3,
    openEvidenceStore: () => stores.openEvidenceStore(),
    openProjectionStore: (mode) => stores.openProjectionStore(mode),
    acknowledgeMarkers: () => stores.acknowledge(),
  });

  assert.equal(result.committedDocumentCount, 33);
  assert.equal(result.writeWindowCount, 2);
  assert.equal(result.acknowledgedMarkerCount, 1);
  assert.deepEqual(
    stores.events.filter((event) => event.startsWith('evidence-write')),
    ['evidence-write:32', 'evidence-write:1'],
  );
  assert.equal(stores.events.filter((event) => event === 'projection-write').length, 1);
  assert.ok(
    stores.events.indexOf('projection-write') > stores.events.lastIndexOf('evidence-write:1'),
  );
  assert.ok(
    stores.events.indexOf('projection-fetch') > stores.events.lastIndexOf('projection-close:write'),
  );
  assert.ok(
    stores.events.indexOf('marker-acknowledge') > stores.events.indexOf('projection-fetch'),
  );
  assert.equal(stores.optimizeCallCount, 0);
  assert.equal(result.writeWindowDiagnostics.length, 2);
});

for (const failEvidencePosition of Array.from({ length: 32 }, (_, index) => index)) {
  void test(`evidence status failure at position ${failEvidencePosition} retains marker acknowledgement`, async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'recall-commit-status-'));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const prepared = createPreparedTransfer(32);
    const stores = createFakeCommitStores({ prepared, failEvidencePosition });

    await assert.rejects(
      () =>
        commitIncrementalRecallTransfer({
          prepared,
          lockPath: join(directory, 'operation.lock'),
          evidenceDatabasePath: join(directory, 'evidence'),
          projectionDatabasePath: join(directory, 'projections'),
          embeddingDimensions: 3,
          openEvidenceStore: () => stores.openEvidenceStore(),
          openProjectionStore: (mode) => stores.openProjectionStore(mode),
          acknowledgeMarkers: () => stores.acknowledge(),
        }),
      new RegExp(`status failed at position ${failEvidencePosition}`, 'u'),
    );
    assert.equal(stores.acknowledgementCount, 0);
    assert.equal(stores.events.includes('projection-write'), false);
    assert.ok(stores.events.includes('evidence-close'));
    assert.ok(stores.events.includes('projection-close:write'));
  });
}

for (const failure of [
  'projection',
  'evidence-close',
  'projection-close',
  'observation',
  'observation-close',
] as const) {
  void test(`${failure} failure retains markers and allows deterministic evidence replay`, async (t) => {
    const directory = await mkdtemp(join(tmpdir(), `recall-commit-${failure}-`));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const prepared = createPreparedTransfer(1);
    const failProjectionCloseMode: 'read' | 'write' | undefined =
      failure === 'projection-close'
        ? 'write'
        : failure === 'observation-close'
          ? 'read'
          : undefined;
    const stores = createFakeCommitStores({
      prepared,
      failProjection: failure === 'projection',
      failEvidenceClose: failure === 'evidence-close',
      ...(failProjectionCloseMode === undefined ? {} : { failProjectionCloseMode }),
      omitObservedCheckpoint: failure === 'observation',
    });
    const options = {
      prepared,
      lockPath: join(directory, 'operation.lock'),
      evidenceDatabasePath: join(directory, 'evidence'),
      projectionDatabasePath: join(directory, 'projections'),
      embeddingDimensions: 3,
      openEvidenceStore: () => stores.openEvidenceStore(),
      openProjectionStore: (mode: 'read' | 'write') => stores.openProjectionStore(mode),
      acknowledgeMarkers: () => stores.acknowledge(),
    };

    await assert.rejects(() => commitIncrementalRecallTransfer(options));
    assert.equal(stores.acknowledgementCount, 0);

    const replayStores = createFakeCommitStores({ prepared });
    const replay = await commitIncrementalRecallTransfer({
      ...options,
      openEvidenceStore: () => replayStores.openEvidenceStore(),
      openProjectionStore: (mode) => replayStores.openProjectionStore(mode),
      acknowledgeMarkers: () => replayStores.acknowledge(),
    });
    assert.equal(replay.committedDocumentCount, 1);
    assert.equal(replay.acknowledgedMarkerCount, 1);
  });
}

void test('writer crash after evidence upsert requires write-capable replay before read-only search', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-commit-crash-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const lockPath = join(directory, 'operation.lock');
  const evidenceDatabasePath = join(directory, 'evidence');
  const projectionDatabasePath = join(directory, 'projections');
  const markerSpoolDirectory = join(directory, 'markers');
  await mkdir(markerSpoolDirectory, { recursive: true });
  const basePrepared = createPreparedTransfer(1);
  const prepared: PreparedIncrementalRecallTransfer = {
    ...basePrepared,
    workPlan: { ...basePrepared.workPlan, markerSpoolDirectory },
  };
  const markerId = prepared.workPlan.sourceMarkerIds[0];
  assert.ok(markerId);
  const markerPath = join(markerSpoolDirectory, `${markerId}.json`);
  await writeFile(markerPath, '{}\n');
  const harnessPath = join(directory, 'crash-writer.ts');
  const coordinateModuleUrl = new URL('./coordinate-recall-write-window.js', import.meta.url).href;
  const storeModuleUrl = new URL('./zvec-conversation-store.js', import.meta.url).href;
  await writeFile(
    harnessPath,
    `import { coordinateRecallWriteWindow } from ${JSON.stringify(coordinateModuleUrl)};\n` +
      `import { openZvecConversationStore } from ${JSON.stringify(storeModuleUrl)};\n` +
      `const document = JSON.parse(${JSON.stringify(JSON.stringify(prepared.documents[0]))});\n` +
      `void (async () => {\n` +
      `  await coordinateRecallWriteWindow({ lockPath: ${JSON.stringify(lockPath)}, allowRecovery: false }, async () => {\n` +
      `    const store = openZvecConversationStore({ databasePath: ${JSON.stringify(evidenceDatabasePath)}, dimensions: 3 });\n` +
      `    await store.upsertChunks([document]);\n` +
      `    process.exit(73);\n` +
      `  });\n` +
      `})();\n`,
  );
  const child = spawn(process.execPath, ['--import', 'tsx', harnessPath], {
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let childError = '';
  child.stderr.setEncoding('utf8');
  child.stderr.on('data', (chunk: string) => {
    childError += chunk;
  });
  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
  });
  assert.equal(exitCode, 73, childError);
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: true,
    recoveryRequired: true,
  });
  await assert.rejects(
    () => assertRecallWriteWindowAvailableForRead(lockPath),
    /Recall write recovery required/u,
  );

  const committed = await commitIncrementalRecallTransfer({
    prepared,
    lockPath,
    evidenceDatabasePath,
    projectionDatabasePath,
    embeddingDimensions: 3,
  });
  assert.equal(committed.acknowledgedMarkerCount, 1);
  assert.equal(committed.writeWindowDiagnostics[0]?.recovering, true);
  await assert.rejects(() => access(markerPath), { code: 'ENOENT' });
  assert.deepEqual(await inspectRecallWriteWindow(lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });

  const reader = openZvecConversationStore({
    databasePath: evidenceDatabasePath,
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(
    (await reader.searchLexicalCandidates('commit evidence', 1))[0]?.id,
    prepared.documents[0]?.id,
  );
  reader.close();
});

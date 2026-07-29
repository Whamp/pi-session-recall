import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { appendFile, mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { commitIncrementalRecallTransfer } from './commit-incremental-recall-transfer.js';
import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import {
  coordinateRecallReadWindow,
  inspectRecallWriteWindow,
  recallWriteWindowStatePaths,
} from './coordinate-recall-write-window.js';
import {
  RecallConfirmedDeletionDecisionKind,
  RecallConfirmedDeletionHaltCategory,
  RecallConfirmedDeletionPhase,
  RecallGenerationCutoverState,
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallMetadataSweepStatus,
  RecallProjectionEncodingStatus,
  RecallProjectionRepairState,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  SessionImportFormat,
} from './enums.js';
import type { PreparedIncrementalRecallTransfer } from './prepare-incremental-recall-transfer.js';
import { projectRecallSessionAppend } from './project-recall-session-append.js';
import { readRecallSessionAppendDelta } from './read-recall-session-append-delta.js';
import {
  decideConfirmedSessionDeletion,
  formatConfirmedSessionDeletionResult,
  reconcileConfirmedSessionDeletion,
  type ConfirmedSessionDeletionSourceObservation,
} from './reconcile-confirmed-session-deletion.js';
import {
  createRecallActiveGenerationPointer,
  encodeRecallActiveGenerationPointer,
  readRecallActiveGenerationSelection,
  readRecallGenerationRegistry,
  writeRecallGenerationRegistry,
  RECALL_GENERATION_REGISTRY_VERSION,
} from './recall-generation-state.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import type { RecallSessionMetadataSweepResult } from './scan-recall-session-metadata.js';
import { readSessionConversationImport } from './session-conversation-index.js';
import { createTestSessionConversationChunk } from './recall-test-utils.js';
import { openZvecConversationStore } from './zvec-conversation-store.js';
import { openZvecSessionProjectionStore } from './zvec-session-projection-store.js';

const generationId = 'generation-deletion-policy';
const physicalSessionId = 'physical-session-policy';

function createPhysicalProjection(
  overrides: Partial<PhysicalSessionProjection> = {},
): PhysicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: createPhysicalSessionProjectionId(physicalSessionId),
    generationId,
    physicalSessionId,
    sourcePath: '/isolated/sessions/session.jsonl',
    sourceDevice: '10',
    sourceInode: '20',
    appendCursorBytes: 100,
    appendCursorLines: 2,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: 'entry-1',
    logicalSessionIds: ['logical-session-1'],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: ['marker-1'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 1 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
    ...overrides,
  };
}

const presentObservation: ConfirmedSessionDeletionSourceObservation = {
  sourceDevice: '10',
  sourceInode: '20',
};

void test('confirmed deletion policy requires two distinct healthy missing sweeps', () => {
  const first = decideConfirmedSessionDeletion({
    projection: createPhysicalProjection(),
    sweepId: 'sweep-1',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 100,
    sourceObservation: null,
  });
  assert.equal(first.kind, RecallConfirmedDeletionDecisionKind.RECORD_SOURCE_MISSING);
  assert.equal(first.nextProjection.sourceMissingObservationCount, 1);
  assert.equal(first.nextProjection.sourceMissingSweepId, 'sweep-1');

  const sameSweep = decideConfirmedSessionDeletion({
    projection: first.nextProjection,
    sweepId: 'sweep-1',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 101,
    sourceObservation: null,
  });
  assert.equal(sameSweep.kind, RecallConfirmedDeletionDecisionKind.NO_CHANGE);

  const confirmed = decideConfirmedSessionDeletion({
    projection: first.nextProjection,
    sweepId: 'sweep-2',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 200,
    sourceObservation: null,
  });
  assert.equal(confirmed.kind, RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION);
  assert.equal(
    confirmed.nextProjection.sourceAvailability,
    RecallSourceAvailability.DELETION_CONFIRMED,
  );
  assert.equal(confirmed.nextProjection.sourceMissingObservationCount, 2);
  assert.equal(confirmed.nextProjection.deletionCheckpoint?.confirmedSweepId, 'sweep-2');

  const replay = decideConfirmedSessionDeletion({
    projection: confirmed.nextProjection,
    sweepId: 'sweep-2',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 201,
    sourceObservation: null,
  });
  assert.equal(replay.kind, RecallConfirmedDeletionDecisionKind.RESUME_CONFIRMED_DELETION);
});

void test('confirmed deletion policy clears one missing observation when the same source reappears', () => {
  const missing = createPhysicalProjection({
    sourceAvailability: RecallSourceAvailability.SOURCE_MISSING,
    sourceMissingObservedAtEpochMilliseconds: 100,
    sourceMissingObservationCount: 1,
    sourceMissingSweepId: 'sweep-1',
  });
  const decision = decideConfirmedSessionDeletion({
    projection: missing,
    sweepId: 'sweep-2',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 200,
    sourceObservation: presentObservation,
  });

  assert.equal(decision.kind, RecallConfirmedDeletionDecisionKind.CLEAR_SOURCE_MISSING);
  assert.equal(decision.nextProjection.sourceAvailability, RecallSourceAvailability.PRESENT);
  assert.equal(decision.nextProjection.sourceMissingObservationCount, 0);
  assert.equal(decision.nextProjection.sourceMissingSweepId, null);
  assert.deepEqual(decision.nextProjection.markerCheckpoint, missing.markerCheckpoint);
});

void test('confirmed deletion policy halts on unhealthy, suspicious, or changed source identity observations', () => {
  for (const [sweepStatus, haltCategory] of [
    [
      RecallMetadataSweepStatus.ROOT_UNAVAILABLE,
      RecallConfirmedDeletionHaltCategory.ROOT_UNAVAILABLE,
    ],
    [
      RecallMetadataSweepStatus.PERMISSION_DENIED,
      RecallConfirmedDeletionHaltCategory.PERMISSION_DENIED,
    ],
    [
      RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS,
      RecallConfirmedDeletionHaltCategory.SUSPICIOUS_MASS_LOSS,
    ],
    [
      RecallMetadataSweepStatus.CONTINUATION_REQUIRED,
      RecallConfirmedDeletionHaltCategory.INCOMPLETE_SWEEP,
    ],
  ] as const) {
    const decision = decideConfirmedSessionDeletion({
      projection: createPhysicalProjection(),
      sweepId: 'sweep-halted',
      sweepStatus,
      observedAtEpochMilliseconds: 100,
      sourceObservation: null,
    });
    assert.deepEqual(decision, {
      kind: RecallConfirmedDeletionDecisionKind.HALT,
      haltCategory,
    });
  }

  const identityChanged = decideConfirmedSessionDeletion({
    projection: createPhysicalProjection(),
    sweepId: 'sweep-identity',
    sweepStatus: RecallMetadataSweepStatus.COMPLETE,
    observedAtEpochMilliseconds: 100,
    sourceObservation: { sourceDevice: '10', sourceInode: 'different' },
  });
  assert.deepEqual(identityChanged, {
    kind: RecallConfirmedDeletionDecisionKind.HALT,
    haltCategory: RecallConfirmedDeletionHaltCategory.SOURCE_IDENTITY_CHANGED,
  });
});

function createLogicalProjection(
  physicalProjection: PhysicalSessionProjection,
  logicalSessionId: string,
): LogicalSessionProjection {
  return {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(
      physicalProjection.physicalSessionId,
      logicalSessionId,
    ),
    generationId: physicalProjection.generationId,
    physicalSessionId: physicalProjection.physicalSessionId,
    physicalProjectionId: physicalProjection.projectionId,
    logicalSessionId,
    effectiveLeafEntryId: 'entry-1',
    activeContextBoundary: { firstEntryId: 'entry-1', lastEntryId: 'entry-1' },
    compactionBoundary: null,
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 10,
      sourceFingerprint: 'a'.repeat(64),
      cwd: '/isolated/project',
      parentSessionPath: null,
    },
    entryDescriptors: [
      {
        entryId: 'entry-1',
        parentEntryId: null,
        entryType: 'message',
        timestamp: '2026-01-01T00:00:00Z',
        messageRole: 'user',
        branchSummaryFromEntryId: null,
        sourceLine: 2,
        startByte: 10,
        endByte: 20,
        sourceFingerprint: 'b'.repeat(64),
        firstKeptEntryId: null,
        hasRetainedTail: false,
        toolCalls: [],
        toolResult: null,
      },
    ],
    eligibleContributorEntryIds: ['entry-1'],
    eligibleSpans: [
      {
        startByte: 10,
        endByte: 20,
        startEntryId: 'entry-1',
        endEntryId: 'entry-1',
        contributorEntryIds: ['entry-1'],
      },
    ],
    labels: [],
    markerCheckpoint: physicalProjection.markerCheckpoint,
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
}

function createMetadataSweep(
  sweepId: string,
  sourcePresent: boolean,
): RecallSessionMetadataSweepResult {
  return {
    sweepId,
    status: RecallMetadataSweepStatus.COMPLETE,
    rootHealthy: true,
    deletionConfirmationSuppressed: false,
    scannedFileCount: sourcePresent ? 1 : 0,
    observedSessionFileCount: sourcePresent ? 1 : 0,
    observedSessionMetadata: [],
    observedKnownSourceIdentities: sourcePresent
      ? [{ physicalSessionId, sourceDevice: '10', sourceInode: '20' }]
      : [],
    missingPhysicalSessionIds: sourcePresent ? [] : [physicalSessionId],
    continuationPersisted: false,
    elapsedMilliseconds: 1,
  };
}

async function writeScratchActiveGenerationState(
  activeGenerationPointerPath: string,
  generationRegistryPath: string,
): Promise<void> {
  const activePointer = createRecallActiveGenerationPointer(generationId);
  await writeFile(activeGenerationPointerPath, encodeRecallActiveGenerationPointer(activePointer));
  await writeRecallGenerationRegistry(generationRegistryPath, {
    version: RECALL_GENERATION_REGISTRY_VERSION,
    activeGenerationId: generationId,
    buildingGenerationId: null,
    rollbackGenerationId: null,
    activePointerChecksum: activePointer.checksum,
    generations: [
      {
        generationId,
        state: RecallGenerationCutoverState.ACTIVE,
        indexManifestVersion: 6,
        markerSchemaVersion: 1,
        sessionProjectionSchemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
        indexManifestFingerprint: 'd'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 1,
        stateChangedAtEpochMilliseconds: 1,
        rebuildStartMarkerId: null,
      },
    ],
  });
}

interface ScratchConfirmedDeletionFixture {
  directory: string;
  generationRootDirectory: string;
  generationDirectory: string;
  activeGenerationPointerPath: string;
  generationRegistryPath: string;
  lockPath: string;
  physicalProjection: PhysicalSessionProjection;
  logicalProjections: LogicalSessionProjection[];
  survivingEvidenceId: string;
}

async function createScratchConfirmedDeletionFixture(): Promise<ScratchConfirmedDeletionFixture> {
  const directory = await mkdtemp(join(tmpdir(), 'confirmed-session-deletion-'));
  const generationRootDirectory = join(directory, 'generations');
  const generationDirectory = join(generationRootDirectory, generationId);
  await mkdir(generationDirectory, { recursive: true });
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  await writeScratchActiveGenerationState(activeGenerationPointerPath, generationRegistryPath);
  const physicalProjection = createPhysicalProjection({
    sourcePath: '/isolated/private-path-sentinel/session.jsonl',
    logicalSessionIds: ['logical-session-1', 'logical-session-2'],
  });
  const logicalProjections = physicalProjection.logicalSessionIds.map((logicalSessionId) =>
    createLogicalProjection(physicalProjection, logicalSessionId),
  );
  const evidenceStore = openZvecConversationStore({
    databasePath: join(generationDirectory, 'zvec'),
    dimensions: 3,
  });
  await evidenceStore.upsertChunks([
    {
      ...createTestSessionConversationChunk({
        id: 'deleted-evidence-1',
        physicalSessionProjectionId: physicalProjection.projectionId,
        content: 'private conversation sentinel durableduplicate',
      }),
      isDenseSearchable: true,
      embedding: [1, 0, 0],
    },
    {
      ...createTestSessionConversationChunk({
        id: 'deleted-evidence-2',
        physicalSessionProjectionId: physicalProjection.projectionId,
        content: 'second private conversation sentinel',
      }),
      isDenseSearchable: true,
      embedding: [1, 0, 0],
    },
    {
      ...createTestSessionConversationChunk({
        id: 'surviving-duplicate-evidence',
        physicalSessionProjectionId: 'physical_surviving_source',
        sessionPath: '/isolated/surviving-source.jsonl',
        content: 'private conversation sentinel durableduplicate',
      }),
      isDenseSearchable: true,
      embedding: [1, 0, 0],
    },
  ]);
  evidenceStore.close();
  const projectionStore = openZvecSessionProjectionStore({
    databasePath: join(generationDirectory, 'session-projections'),
    generationId,
  });
  await projectionStore.upsertProjections([physicalProjection, ...logicalProjections]);
  projectionStore.close();
  return {
    directory,
    generationRootDirectory,
    generationDirectory,
    activeGenerationPointerPath,
    generationRegistryPath,
    lockPath: join(directory, 'recall.lock'),
    physicalProjection,
    logicalProjections,
    survivingEvidenceId: 'surviving-duplicate-evidence',
  };
}

function createScratchReconciliationOptions(
  fixture: ScratchConfirmedDeletionFixture,
  metadataSweep: RecallSessionMetadataSweepResult,
  physicalProjection = fixture.physicalProjection,
) {
  return {
    metadataSweep,
    physicalProjections: [physicalProjection],
    activeGenerationPointerPath: fixture.activeGenerationPointerPath,
    generationRegistryPath: fixture.generationRegistryPath,
    generationRootDirectory: fixture.generationRootDirectory,
    lockPath: fixture.lockPath,
    embeddingDimensions: 3,
  };
}

void test('scratch zvec confirmed deletion removes only one physical source and all of its logical sessions', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));
  const acknowledgedSweeps: string[] = [];

  const first = await reconcileConfirmedSessionDeletion({
    ...createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
    async acknowledgeCheckpoint(sweepId) {
      acknowledgedSweeps.push(sweepId);
    },
  });
  assert.equal(first.sourceMissingRecordedCount, 1);
  assert.equal(first.removedEvidenceOccurrenceCount, 0);
  assert.equal(first.removedLogicalProjectionCount, 0);
  assert.equal(first.removedPhysicalProjectionCount, 0);
  assert.equal(acknowledgedSweeps.length, 0);

  const evidenceAfterFirstAbsence = openZvecConversationStore({
    databasePath: join(fixture.generationDirectory, 'zvec'),
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(evidenceAfterFirstAbsence.count(), 3);
  evidenceAfterFirstAbsence.close();
  const projectionsAfterFirstAbsence = openZvecSessionProjectionStore({
    databasePath: join(fixture.generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(
    projectionsAfterFirstAbsence.fetchProjections([
      fixture.physicalProjection.projectionId,
      ...fixture.logicalProjections.map(({ projectionId }) => projectionId),
    ]).size,
    3,
  );
  projectionsAfterFirstAbsence.close();

  let searchDuringDeletionCount = 0;
  const scalarDiagnostics: string[] = [];
  const second = await reconcileConfirmedSessionDeletion({
    ...createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-2', false)),
    async acknowledgeCheckpoint(sweepId) {
      acknowledgedSweeps.push(sweepId);
    },
    onDiagnostic(result) {
      scalarDiagnostics.push(JSON.stringify(result));
    },
    async onProgress() {
      await coordinateRecallReadWindow(
        { lockPath: fixture.lockPath, waitMilliseconds: 500 },
        async () => {
          const selectedGeneration = await readRecallActiveGenerationSelection(
            fixture.activeGenerationPointerPath,
            fixture.generationRootDirectory,
          );
          const reader = openZvecConversationStore({
            databasePath: selectedGeneration.databasePath,
            dimensions: 3,
            createIfMissing: false,
            readOnly: true,
          });
          try {
            const results = await reader.searchLexicalCandidates('durableduplicate', 10);
            assert.equal(results[0]?.id, fixture.survivingEvidenceId);
            searchDuringDeletionCount += 1;
          } finally {
            reader.close();
          }
        },
      );
    },
  });
  assert.deepEqual(second, {
    halted: false,
    consideredPhysicalSessionCount: 1,
    sourceMissingRecordedCount: 0,
    sourceMissingClearedCount: 0,
    confirmedSourceDeletionCount: 1,
    removedEvidenceOccurrenceCount: 2,
    removedLogicalProjectionCount: 2,
    removedPhysicalProjectionCount: 1,
    acknowledgedCheckpointCount: 1,
    haltCategoryCounts: {},
  });
  assert.deepEqual(acknowledgedSweeps, ['sweep-2']);
  assert.equal(searchDuringDeletionCount, 3);
  assert.equal(scalarDiagnostics.length, 1);
  assert.doesNotMatch(
    scalarDiagnostics[0] ?? '',
    /private conversation sentinel|private-path-sentinel|session\.jsonl/u,
  );

  const survivingEvidence = openZvecConversationStore({
    databasePath: join(fixture.generationDirectory, 'zvec'),
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  assert.deepEqual(
    [
      ...survivingEvidence
        .fetchConversationChunks([
          'deleted-evidence-1',
          'deleted-evidence-2',
          fixture.survivingEvidenceId,
        ])
        .keys(),
    ],
    [fixture.survivingEvidenceId],
  );
  const duplicateSearch = await survivingEvidence.searchLexicalCandidates('durableduplicate', 10);
  assert.equal(duplicateSearch[0]?.id, fixture.survivingEvidenceId);
  assert.equal(duplicateSearch[0]?.physicalSessionProjectionId, 'physical_surviving_source');
  survivingEvidence.close();

  const output = formatConfirmedSessionDeletionResult(second);
  assert.match(output, /confirmedSourceDeletions=1/u);
  assert.doesNotMatch(output, /private conversation sentinel|private-path-sentinel/u);
});

void test('incremental repeated-header commits reopen without an orphan and confirmed deletion removes both occurrences', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'confirmed-session-occurrence-identity-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const generationRootDirectory = join(directory, 'generations');
  const generationDirectory = join(generationRootDirectory, generationId);
  await mkdir(generationDirectory, { recursive: true });
  const activeGenerationPointerPath = join(directory, 'active-generation.json');
  const generationRegistryPath = join(directory, 'generation-registry.json');
  await writeScratchActiveGenerationState(activeGenerationPointerPath, generationRegistryPath);
  const sessionPath = join(directory, 'reused-session.jsonl');
  await writeFile(sessionPath, '');
  const metadata = await stat(sessionPath, { bigint: true });
  const initialPhysicalProjection = createPhysicalProjection({
    sourcePath: sessionPath,
    sourceDevice: metadata.dev.toString(),
    sourceInode: metadata.ino.toString(),
    appendCursorBytes: 0,
    appendCursorLines: 0,
    boundaryFingerprint: createHash('sha256').update('').digest('hex'),
    lastEntryId: null,
    logicalSessionIds: [],
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
  });
  const firstRecords = [
    {
      type: 'session',
      version: 3,
      id: 'reused-logical',
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/one',
    },
    {
      type: 'message',
      id: 'first-entry',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'first historical occurrence' },
    },
  ];
  await appendFile(
    sessionPath,
    `${firstRecords.map((record) => JSON.stringify(record)).join('\n')}\n`,
  );
  const firstDelta = await readRecallSessionAppendDelta(sessionPath, initialPhysicalProjection);
  assert.equal(firstDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (firstDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const firstProjection = projectRecallSessionAppend({
    physicalProjection: initialPhysicalProjection,
    logicalProjections: [],
    appendDelta: firstDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(firstProjection.status, RecallAppendProjectionStatus.PROJECTED);
  if (firstProjection.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  const firstLogicalProjection = firstProjection.logicalProjections[0];
  assert.equal(firstLogicalProjection?.logicalSessionId, 'reused-logical@1');
  assert.equal(firstLogicalProjection?.rawSessionId, 'reused-logical');

  const workPlan: RecallMarkerReplayWorkPlan = {
    targetGenerationId: generationId,
    markerSpoolDirectory: join(directory, 'markers'),
    discoveredMarkerCount: 0,
    sourceMarkerIds: [],
    workItems: [],
    quarantineDiagnostics: [],
  };
  const commitProjection = async (
    physicalProjection: PhysicalSessionProjection,
    logicalProjections: readonly LogicalSessionProjection[],
  ): Promise<void> => {
    const prepared: PreparedIncrementalRecallTransfer = {
      status: RecallProjectionEncodingStatus.ENCODED,
      targetGenerationId: generationId,
      documents: [],
      checkpointIntent: { physicalProjection, logicalProjections },
      workPlan,
      cacheHits: 0,
      newlyEmbeddedChunks: 0,
      embeddingRequestCount: 0,
    };
    await commitIncrementalRecallTransfer({
      prepared,
      lockPath: join(directory, 'recall.lock'),
      evidenceDatabasePath: join(generationDirectory, 'zvec'),
      projectionDatabasePath: join(generationDirectory, 'session-projections'),
      embeddingDimensions: 3,
      async acknowledgeMarkers() {
        return 0;
      },
    });
  };
  await commitProjection(firstProjection.physicalProjection, firstProjection.logicalProjections);

  const secondRecords = [
    {
      type: 'session',
      version: 3,
      id: 'reused-logical',
      timestamp: '2026-01-02T00:00:00Z',
      cwd: '/two',
    },
    {
      type: 'message',
      id: 'second-entry',
      parentId: null,
      timestamp: '2026-01-02T00:00:01Z',
      message: { role: 'user', content: 'second historical occurrence' },
    },
  ];
  await appendFile(
    sessionPath,
    `${secondRecords.map((record) => JSON.stringify(record)).join('\n')}\n`,
  );
  const secondDelta = await readRecallSessionAppendDelta(
    sessionPath,
    firstProjection.physicalProjection,
  );
  assert.equal(secondDelta.status, RecallAppendDeltaStatus.APPENDED);
  if (secondDelta.status !== RecallAppendDeltaStatus.APPENDED) {
    return;
  }
  const secondProjection = projectRecallSessionAppend({
    physicalProjection: firstProjection.physicalProjection,
    logicalProjections: firstProjection.logicalProjections,
    appendDelta: secondDelta,
    markers: [],
    quiescenceObserved: false,
  });
  assert.equal(secondProjection.status, RecallAppendProjectionStatus.PROJECTED);
  if (secondProjection.status !== RecallAppendProjectionStatus.PROJECTED) {
    return;
  }
  assert.equal(
    secondProjection.logicalProjections[0]?.projectionId,
    firstLogicalProjection?.projectionId,
  );
  assert.deepEqual(secondProjection.physicalProjection.logicalSessionIds, [
    'reused-logical@1',
    'reused-logical@3',
  ]);
  await commitProjection(secondProjection.physicalProjection, secondProjection.logicalProjections);

  const imported = await readSessionConversationImport(sessionPath, {
    tokenizer: {
      encodeConversationText(text) {
        return {
          ids: text
            .split(/\s+/u)
            .filter(Boolean)
            .map((_, index) => index),
        };
      },
    },
    eligibleContributorEntryIdsByLogicalSessionId: new Map(
      secondProjection.logicalProjections.map((projection) => [
        projection.logicalSessionId,
        new Set(projection.entryDescriptors.map(({ entryId }) => entryId)),
      ]),
    ),
  });
  assert.equal(imported.format, SessionImportFormat.PI_SESSION_REUSE_HISTORY);
  assert.deepEqual(
    imported.chunks.map(({ content }) => content),
    ['first historical occurrence', 'second historical occurrence'],
  );

  const oldRawProjectionId = createLogicalSessionProjectionId(physicalSessionId, 'reused-logical');
  const projectionIds = [
    secondProjection.physicalProjection.projectionId,
    ...secondProjection.logicalProjections.map(({ projectionId }) => projectionId),
    oldRawProjectionId,
  ];
  const reopenedProjectionStore = openZvecSessionProjectionStore({
    databasePath: join(generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(reopenedProjectionStore.fetchProjections(projectionIds).size, 3);
  assert.equal(reopenedProjectionStore.fetchProjections([oldRawProjectionId]).size, 0);
  reopenedProjectionStore.close();

  const deletionOptions = {
    physicalProjections: [secondProjection.physicalProjection],
    activeGenerationPointerPath,
    generationRegistryPath,
    generationRootDirectory,
    lockPath: join(directory, 'recall.lock'),
    embeddingDimensions: 3,
  };
  const firstDeletionSweep = createMetadataSweep('occurrence-sweep-1', false);
  const secondDeletionSweep = createMetadataSweep('occurrence-sweep-2', false);
  const firstDeletion = await reconcileConfirmedSessionDeletion({
    ...deletionOptions,
    metadataSweep: firstDeletionSweep,
  });
  assert.equal(firstDeletion.sourceMissingRecordedCount, 1);
  const confirmedDeletion = await reconcileConfirmedSessionDeletion({
    ...deletionOptions,
    metadataSweep: secondDeletionSweep,
  });
  assert.equal(confirmedDeletion.removedLogicalProjectionCount, 2);
  assert.equal(confirmedDeletion.removedPhysicalProjectionCount, 1);

  const reopenedAfterDeletion = openZvecSessionProjectionStore({
    databasePath: join(generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(reopenedAfterDeletion.fetchProjections(projectionIds).size, 0);
  reopenedAfterDeletion.close();
});

void test('scratch zvec source reappearance clears source_missing without changing eligible evidence', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));
  await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
  );

  const projectionStore = openZvecSessionProjectionStore({
    databasePath: join(fixture.generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  const missingProjection = projectionStore
    .fetchProjections([fixture.physicalProjection.projectionId])
    .get(fixture.physicalProjection.projectionId);
  projectionStore.close();
  assert.equal(missingProjection?.projectionKind, RecallSessionProjectionKind.PHYSICAL_SESSION);
  if (missingProjection?.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
    throw new Error('Expected source-missing physical projection');
  }

  const reappeared = await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(
      fixture,
      createMetadataSweep('sweep-2', true),
      missingProjection,
    ),
  );
  assert.equal(reappeared.sourceMissingClearedCount, 1);
  assert.equal(reappeared.removedEvidenceOccurrenceCount, 0);

  const evidenceStore = openZvecConversationStore({
    databasePath: join(fixture.generationDirectory, 'zvec'),
    dimensions: 3,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(evidenceStore.count(), 3);
  evidenceStore.close();
});

void test('scratch zvec confirmed deletion resumes idempotently after every destructive phase', async (t) => {
  for (const interruptedPhase of [
    RecallConfirmedDeletionPhase.EVIDENCE,
    RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS,
    RecallConfirmedDeletionPhase.PHYSICAL_PROJECTION,
  ]) {
    const fixture = await createScratchConfirmedDeletionFixture();
    t.after(() => rm(fixture.directory, { recursive: true, force: true }));
    await reconcileConfirmedSessionDeletion(
      createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
    );
    let interrupted = false;
    await assert.rejects(
      () =>
        reconcileConfirmedSessionDeletion({
          ...createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-2', false)),
          onProgress(progress) {
            if (!interrupted && progress.phase === interruptedPhase) {
              interrupted = true;
              throw new Error(`intentional interruption ${interruptedPhase}`);
            }
          },
        }),
      /intentional interruption/u,
    );
    assert.equal(interrupted, true);

    const acknowledgements: string[] = [];
    const resumed = await reconcileConfirmedSessionDeletion({
      ...createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-2', false)),
      async acknowledgeCheckpoint(sweepId) {
        acknowledgements.push(sweepId);
      },
    });
    assert.equal(resumed.halted, false);
    assert.equal(resumed.acknowledgedCheckpointCount, 1);
    assert.deepEqual(acknowledgements, ['sweep-2']);

    const evidenceStore = openZvecConversationStore({
      databasePath: join(fixture.generationDirectory, 'zvec'),
      dimensions: 3,
      createIfMissing: false,
      readOnly: true,
    });
    assert.deepEqual(
      [
        ...evidenceStore
          .fetchConversationChunks([
            'deleted-evidence-1',
            'deleted-evidence-2',
            fixture.survivingEvidenceId,
          ])
          .keys(),
      ],
      [fixture.survivingEvidenceId],
    );
    evidenceStore.close();
  }
});

void test('confirmed deletion rechecks the rebuild freeze under the write lock', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));
  await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
  );
  const registry = await readRecallGenerationRegistry(fixture.generationRegistryPath);
  assert.ok(registry);
  const activeEntry = registry.generations[0];
  assert.ok(activeEntry);
  await writeRecallGenerationRegistry(fixture.generationRegistryPath, {
    ...registry,
    buildingGenerationId: 'generation-building',
    generations: [
      activeEntry,
      {
        ...activeEntry,
        generationId: 'generation-building',
        state: RecallGenerationCutoverState.BUILDING,
        indexManifestFingerprint: 'e'.repeat(64),
        rebuildStartedAtEpochMilliseconds: 2,
        stateChangedAtEpochMilliseconds: 2,
      },
    ],
  });

  const result = await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-2', false)),
  );

  assert.equal(result.halted, true);
  assert.deepEqual(result.haltCategoryCounts, {
    [RecallConfirmedDeletionHaltCategory.REBUILD_IN_PROGRESS]: 1,
  });
  assert.equal(readScratchPhysicalProjection(fixture).sourceMissingObservationCount, 1);
});

void test('confirmed deletion halts before mutating a generation that replaces the active pointer', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));
  await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
  );
  const replacementGenerationId = 'replacement-generation';
  await mkdir(join(fixture.generationRootDirectory, replacementGenerationId));

  const result = await reconcileConfirmedSessionDeletion({
    ...createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-2', false)),
    async onProgress(progress) {
      if (progress.phase === RecallConfirmedDeletionPhase.EVIDENCE) {
        await writeFile(
          fixture.activeGenerationPointerPath,
          encodeRecallActiveGenerationPointer(
            createRecallActiveGenerationPointer(replacementGenerationId),
          ),
        );
      }
    },
  });

  assert.equal(result.halted, true);
  assert.deepEqual(result.haltCategoryCounts, {
    [RecallConfirmedDeletionHaltCategory.ACTIVE_GENERATION_CHANGED]: 1,
  });
  const projectionStore = openZvecSessionProjectionStore({
    databasePath: join(fixture.generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  assert.equal(
    projectionStore.fetchProjections([
      fixture.physicalProjection.projectionId,
      ...fixture.logicalProjections.map(({ projectionId }) => projectionId),
    ]).size,
    3,
  );
  projectionStore.close();
});

function readScratchPhysicalProjection(
  fixture: ScratchConfirmedDeletionFixture,
): PhysicalSessionProjection {
  const store = openZvecSessionProjectionStore({
    databasePath: join(fixture.generationDirectory, 'session-projections'),
    generationId,
    createIfMissing: false,
    readOnly: true,
  });
  try {
    const projection = store
      .fetchProjections([fixture.physicalProjection.projectionId])
      .get(fixture.physicalProjection.projectionId);
    if (projection?.projectionKind !== RecallSessionProjectionKind.PHYSICAL_SESSION) {
      throw new Error('Expected scratch physical projection');
    }
    return projection;
  } finally {
    store.close();
  }
}

void test('successful recovering deletion attests normal closure and clears recovery state', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));
  const statePaths = recallWriteWindowStatePaths(fixture.lockPath);
  await writeFile(statePaths.recoveryRequiredPath, 'recovery required\n');

  const result = await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
  );

  assert.equal(result.sourceMissingRecordedCount, 1);
  assert.deepEqual(await inspectRecallWriteWindow(fixture.lockPath), {
    currentWindow: false,
    recoveryRequired: false,
  });
});

void test('confirmed deletion resumes when a crash leaves checkpointed IDs already deleted', async (t) => {
  for (const crashPhase of [
    RecallConfirmedDeletionPhase.EVIDENCE,
    RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS,
  ]) {
    const fixture = await createScratchConfirmedDeletionFixture();
    t.after(() => rm(fixture.directory, { recursive: true, force: true }));
    await reconcileConfirmedSessionDeletion(
      createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-1', false)),
    );
    const missingProjection = readScratchPhysicalProjection(fixture);
    const confirmation = decideConfirmedSessionDeletion({
      projection: missingProjection,
      sweepId: 'sweep-2',
      sweepStatus: RecallMetadataSweepStatus.COMPLETE,
      observedAtEpochMilliseconds: 200,
      sourceObservation: null,
    });
    assert.equal(confirmation.kind, RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION);
    if (confirmation.kind !== RecallConfirmedDeletionDecisionKind.CONFIRM_SOURCE_DELETION) {
      throw new Error('Expected confirmed deletion projection');
    }
    const checkpoint = confirmation.nextProjection.deletionCheckpoint;
    if (checkpoint === null) {
      throw new Error('Expected confirmed deletion checkpoint');
    }
    const logicalProjectionIds = fixture.logicalProjections.map(({ projectionId }) => projectionId);
    const crashedProjection: PhysicalSessionProjection = {
      ...confirmation.nextProjection,
      deletionCheckpoint:
        crashPhase === RecallConfirmedDeletionPhase.EVIDENCE
          ? {
              ...checkpoint,
              pendingEvidenceIds: ['deleted-evidence-1', 'deleted-evidence-2'],
            }
          : {
              ...checkpoint,
              phase: RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS,
              deletedEvidenceCount: 2,
              pendingLogicalProjectionIds: logicalProjectionIds,
            },
    };
    const projectionStore = openZvecSessionProjectionStore({
      databasePath: join(fixture.generationDirectory, 'session-projections'),
      generationId,
      createIfMissing: false,
    });
    await projectionStore.upsertProjections([crashedProjection]);
    if (crashPhase === RecallConfirmedDeletionPhase.LOGICAL_PROJECTIONS) {
      await projectionStore.deleteProjections(logicalProjectionIds);
    }
    projectionStore.close();
    const evidenceStore = openZvecConversationStore({
      databasePath: join(fixture.generationDirectory, 'zvec'),
      dimensions: 3,
      createIfMissing: false,
    });
    await evidenceStore.deleteChunks(['deleted-evidence-1', 'deleted-evidence-2']);
    evidenceStore.close();

    const resumed = await reconcileConfirmedSessionDeletion(
      createScratchReconciliationOptions(
        fixture,
        createMetadataSweep('sweep-2', false),
        crashedProjection,
      ),
    );
    assert.equal(resumed.halted, false);
    assert.equal(resumed.removedEvidenceOccurrenceCount, 2);
    assert.equal(resumed.removedLogicalProjectionCount, 2);
    assert.equal(resumed.removedPhysicalProjectionCount, 1);
  }
});

void test('healthy unchanged present sources require no reconciliation write window', async (t) => {
  const fixture = await createScratchConfirmedDeletionFixture();
  t.after(() => rm(fixture.directory, { recursive: true, force: true }));

  const result = await reconcileConfirmedSessionDeletion(
    createScratchReconciliationOptions(fixture, createMetadataSweep('sweep-present', true)),
  );

  assert.equal(result.halted, false);
  assert.equal(result.consideredPhysicalSessionCount, 0);
  assert.equal(result.sourceMissingRecordedCount, 0);
  assert.equal(result.removedEvidenceOccurrenceCount, 0);
});

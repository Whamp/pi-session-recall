import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
import {
  RecallProjectionEncodingStatus,
  RecallProjectionRepairState,
  RecallProjectIdentitySource,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
  SessionImportFormat,
} from './enums.js';
import {
  parseRecallSessionGraph,
  type CanonicalSessionRepresentation,
} from './parse-recall-session-record.js';
import {
  prepareIncrementalRecallTransfer,
  type IncrementalRecallEligibleLogicalSession,
} from './prepare-incremental-recall-transfer.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';
import {
  createLogicalSessionProjectionId,
  createPhysicalSessionProjectionId,
  RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
  type LogicalSessionProjection,
  type PhysicalSessionProjection,
} from './recall-session-projection.js';
import { createRecallWorkMarkerId, type RecallWorkMarker } from './recall-work-marker.js';

const generationId = 'generation_prepare';
const physicalSessionId = 'physical-session-prepare';
const logicalSessionId = 'logical-session-prepare';
const physicalProjectionId = createPhysicalSessionProjectionId(physicalSessionId);

function createMarker(): RecallWorkMarker {
  const identity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: '/isolated/sessions/prepare.jsonl',
    runtimeInstanceId: 'runtime-prepare',
    runtimeSequence: 3,
    createdAtEpochMilliseconds: 100,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  return { ...identity, markerId: createRecallWorkMarkerId(identity) };
}

function createPreparationFixture(): {
  physicalProjection: PhysicalSessionProjection;
  eligibleSessions: readonly IncrementalRecallEligibleLogicalSession[];
  workPlan: RecallMarkerReplayWorkPlan;
} {
  const records = [
    {
      type: 'session',
      version: 3,
      id: logicalSessionId,
      timestamp: '2026-01-01T00:00:00Z',
      cwd: '/isolated/project',
    },
    {
      type: 'message',
      id: 'user-old',
      parentId: null,
      timestamp: '2026-01-01T00:00:01Z',
      message: { role: 'user', content: 'eligible user evidence' },
    },
    {
      type: 'message',
      id: 'assistant-new',
      parentId: 'user-old',
      timestamp: '2026-01-01T00:00:02Z',
      message: {
        role: 'assistant',
        content: [
          { type: 'text', text: 'eligible assistant evidence CACHE_MISS' },
          {
            type: 'toolCall',
            id: 'call-prepare',
            name: 'bash',
            arguments: { command: 'echo TOOL_PREPARE_ONLY' },
          },
        ],
      },
    },
    {
      type: 'compaction',
      id: 'compact-new',
      parentId: 'assistant-new',
      timestamp: '2026-01-01T00:00:03Z',
      summary: 'eligible compact summary',
      firstKeptEntryId: 'assistant-new',
      tokensBefore: 100,
    },
    {
      type: 'message',
      id: 'active-tail',
      parentId: 'compact-new',
      timestamp: '2026-01-01T00:00:04Z',
      message: { role: 'user', content: 'ACTIVE TAIL MUST NEVER PREPARE' },
    },
  ];
  const canonicalSession: CanonicalSessionRepresentation = {
    format: SessionImportFormat.CANONICAL_JSONL,
    physicalPath: '/isolated/sessions/prepare.jsonl',
    logicalSessionId,
    sourceLineStart: 1,
    sourceLineEnd: records.length,
    records: records.map((value, index) => ({ sourceLine: index + 1, value })),
  };
  const marker = createMarker();
  const physicalProjection: PhysicalSessionProjection = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId: physicalProjectionId,
    generationId,
    physicalSessionId,
    sourcePath: canonicalSession.physicalPath,
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 500,
    appendCursorLines: records.length,
    boundaryFingerprint: 'a'.repeat(64),
    lastEntryId: 'active-tail',
    logicalSessionIds: [logicalSessionId],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
  const logicalProjection: LogicalSessionProjection = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.LOGICAL_SESSION,
    projectionId: createLogicalSessionProjectionId(physicalSessionId, logicalSessionId),
    generationId,
    physicalSessionId,
    physicalProjectionId,
    logicalSessionId,
    effectiveLeafEntryId: 'active-tail',
    activeContextBoundary: { firstEntryId: 'assistant-new', lastEntryId: 'active-tail' },
    compactionBoundary: {
      compactionEntryId: 'compact-new',
      firstRetainedEntryId: 'assistant-new',
    },
    runtimeLeafObservations: [],
    preservedBranchExits: [],
    headerDescriptor: {
      sourceLine: 1,
      startByte: 0,
      endByte: 100,
      sourceFingerprint: createHash('sha256').update(JSON.stringify(records[0])).digest('hex'),
      cwd: '/isolated/project',
      parentSessionPath: null,
    },
    entryDescriptors: records.slice(1).map((value, index) => ({
      entryId: value.id,
      parentEntryId: value.parentId ?? null,
      entryType: value.type,
      timestamp: value.timestamp,
      messageRole:
        value.type === 'message' && 'message' in value && typeof value.message.role === 'string'
          ? value.message.role
          : null,
      branchSummaryFromEntryId:
        value.type === 'branch_summary' && 'fromId' in value && typeof value.fromId === 'string'
          ? value.fromId
          : null,
      sourceLine: index + 2,
      startByte: 100 + index * 100,
      endByte: 200 + index * 100,
      sourceFingerprint: createHash('sha256').update(JSON.stringify(value)).digest('hex'),
      firstKeptEntryId:
        value.type === 'compaction' && typeof value.firstKeptEntryId === 'string'
          ? value.firstKeptEntryId
          : null,
      hasRetainedTail: false,
      toolCalls: [],
      toolResult: null,
    })),
    eligibleContributorEntryIds: ['user-old', 'assistant-new', 'compact-new'],
    eligibleSpans: [
      {
        startByte: 100,
        endByte: 400,
        startEntryId: 'user-old',
        endEntryId: 'compact-new',
        contributorEntryIds: ['user-old', 'assistant-new', 'compact-new'],
      },
    ],
    labels: [],
    markerCheckpoint: { generationId, coveredMarkerIds: [], runtimeSequences: [] },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
  return {
    physicalProjection,
    eligibleSessions: [
      {
        graphView: {
          graph: parseRecallSessionGraph(canonicalSession),
          physicalPath: canonicalSession.physicalPath,
          logicalSessionId: canonicalSession.logicalSessionId,
        },
        logicalProjection,
        newlyEligibleSpans: [
          {
            startByte: 200,
            endByte: 400,
            startEntryId: 'assistant-new',
            endEntryId: 'compact-new',
            contributorEntryIds: ['assistant-new', 'compact-new'],
          },
        ],
      },
    ],
    workPlan: {
      targetGenerationId: generationId,
      markerSpoolDirectory: '/isolated/markers',
      discoveredMarkerCount: 1,
      sourceMarkerIds: [marker.markerId],
      workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
      quarantineDiagnostics: [],
    },
  };
}

void test('incremental preparation finishes tokenizer, attribution, and document embedding before returning immutable evidence', async () => {
  const fixture = createPreparationFixture();
  const tokenizedTexts: string[] = [];
  const embeddedTexts: string[] = [];
  const events: string[] = [];

  const prepared = await prepareIncrementalRecallTransfer({
    ...fixture,
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    async loadTokenizer() {
      events.push('tokenizer-loaded');
      return {
        encodeConversationText(text: string) {
          tokenizedTexts.push(text);
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    async resolveProjectIdentity(sessionOrigin) {
      events.push(`project:${sessionOrigin}`);
      return {
        projectIdentity: parseRepositoryIdentity('git-origin:github.com/Whamp/pi-session-recall'),
        identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
      };
    },
    embeddingProvider: {
      async embedDocuments(documents) {
        embeddedTexts.push(...documents);
        events.push('documents-embedded');
        return documents.map((document) => [document.includes('CACHE_MISS') ? 0 : 1, 1, 0]);
      },
    },
  });

  assert.equal(prepared.status, RecallProjectionEncodingStatus.ENCODED);
  if (prepared.status !== RecallProjectionEncodingStatus.ENCODED) {
    return;
  }
  assert.deepEqual(events, ['tokenizer-loaded', 'project:/isolated/project', 'documents-embedded']);
  assert.ok(prepared.documents.some(({ documentKind }) => documentKind === 'turn_context'));
  assert.ok(prepared.documents.some(({ documentKind }) => documentKind === 'summary'));
  const toolDocuments = prepared.documents.filter(({ documentKind }) => documentKind === 'tool');
  assert.ok(toolDocuments.length > 0);
  assert.ok(toolDocuments.every(({ isDenseSearchable }) => !isDenseSearchable));
  assert.ok(toolDocuments.every(({ content }) => !embeddedTexts.includes(content)));
  assert.ok(
    prepared.documents.every(({ physicalSessionProjectionId: id }) => id === physicalProjectionId),
  );
  assert.ok(prepared.documents.every(({ projectAttribution }) => projectAttribution !== null));
  assert.ok(tokenizedTexts.every((text) => !text.includes('ACTIVE TAIL')));
  assert.ok(embeddedTexts.every((text) => !text.includes('ACTIVE TAIL')));
  assert.ok(prepared.documents.every(({ content }) => !content.includes('ACTIVE TAIL')));
  assert.equal(Object.isFrozen(prepared), true);
  assert.equal(Object.isFrozen(prepared.documents), true);
  assert.deepEqual(
    prepared.checkpointIntent.physicalProjection.markerCheckpoint.coveredMarkerIds,
    fixture.workPlan.sourceMarkerIds,
  );
  assert.ok(
    prepared.checkpointIntent.logicalProjections.every(({ markerCheckpoint }) =>
      fixture.workPlan.sourceMarkerIds.every((markerId) =>
        markerCheckpoint.coveredMarkerIds.includes(markerId),
      ),
    ),
  );
});

void test('incremental preparation accepts logical state larger than one projection record', async () => {
  const fixture = createPreparationFixture();
  const eligibleSession = fixture.eligibleSessions[0];
  assert.ok(eligibleSession);
  const logicalProjection = eligibleSession.logicalProjection;
  let embeddingRequestCount = 0;
  const result = await prepareIncrementalRecallTransfer({
    ...fixture,
    eligibleSessions: [
      {
        ...eligibleSession,
        logicalProjection: { ...logicalProjection, labels: ['x'.repeat(9_000_000)] },
      },
    ],
    chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
    async loadTokenizer() {
      return {
        encodeConversationText(text: string) {
          return { ids: Array.from(text.split(/\s+/u).filter(Boolean).keys()) };
        },
      };
    },
    async resolveProjectIdentity() {
      return null;
    },
    embeddingProvider: {
      async embedDocuments(documents) {
        embeddingRequestCount += 1;
        return documents.map(() => [1, 0, 0]);
      },
    },
  });

  assert.equal(result.status, RecallProjectionEncodingStatus.ENCODED);
  assert.equal(embeddingRequestCount, 1);
  assert.equal(result.checkpointIntent.logicalProjections[0]?.labels[0]?.length, 9_000_000);
});

void test('preparation rejects cross-physical marker intent before tokenizer or inference work', async () => {
  const fixture = createPreparationFixture();
  const firstWorkItem = fixture.workPlan.workItems[0];
  assert.ok(firstWorkItem);
  let forbiddenCallCount = 0;

  await assert.rejects(
    () =>
      prepareIncrementalRecallTransfer({
        ...fixture,
        workPlan: {
          ...fixture.workPlan,
          workItems: [
            {
              ...firstWorkItem,
              marker: { ...firstWorkItem.marker, physicalSessionId: 'different-physical-session' },
            },
          ],
        },
        chunkPolicy: { maxTokens: 64, overlapTokens: 8 },
        async loadTokenizer() {
          forbiddenCallCount += 1;
          throw new Error('tokenizer must not load for mismatched marker intent');
        },
        async resolveProjectIdentity() {
          forbiddenCallCount += 1;
          return null;
        },
        embeddingProvider: {
          async embedDocuments() {
            forbiddenCallCount += 1;
            throw new Error('embedding provider must not load for mismatched marker intent');
          },
        },
      }),
    /marker physical session mismatch/u,
  );
  assert.equal(forbiddenCallCount, 0);
});

import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { commitIncrementalRecallTransfer } from './commit-incremental-recall-transfer.js';
import type { RecallMarkerReplayWorkPlan } from './coordinate-recall-marker-replay.js';
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

const RUN_WRITE_WINDOW_DIAGNOSTIC = process.env.PI_RECALL_RUN_WRITE_WINDOW_DIAGNOSTIC === '1';
const WRITE_WINDOW_SAMPLE_COUNT = 20;
const WRITE_WINDOW_P95_TARGET_MILLISECONDS = 300;

function percentile95(values: readonly number[]): number {
  const ordered = values.toSorted((left, right) => left - right);
  return ordered[Math.max(0, Math.ceil(ordered.length * 0.95) - 1)] ?? 0;
}

function createDiagnosticTransfer(markerSpoolDirectory: string): PreparedIncrementalRecallTransfer {
  const generationId = 'generation_write_window_diagnostic';
  const physicalSessionId = 'physical-session-write-window-diagnostic';
  const projectionId = createPhysicalSessionProjectionId(physicalSessionId);
  const identity = {
    version: 1,
    physicalSessionId,
    physicalSessionPath: '/isolated/diagnostic/session.jsonl',
    runtimeInstanceId: 'runtime-write-window-diagnostic',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  const marker: RecallWorkMarker = {
    ...identity,
    markerId: createRecallWorkMarkerId(identity),
  };
  const workPlan: RecallMarkerReplayWorkPlan = {
    targetGenerationId: generationId,
    markerSpoolDirectory,
    discoveredMarkerCount: 1,
    sourceMarkerIds: [marker.markerId],
    workItems: [{ marker, coveredMarkerIds: [marker.markerId] }],
    quarantineDiagnostics: [],
  };
  const physicalProjection: PhysicalSessionProjection = {
    schemaVersion: RECALL_SESSION_PROJECTION_SCHEMA_VERSION,
    projectionKind: RecallSessionProjectionKind.PHYSICAL_SESSION,
    projectionId,
    generationId,
    physicalSessionId,
    sourcePath: identity.physicalSessionPath,
    sourceDevice: '1',
    sourceInode: '2',
    appendCursorBytes: 1_000,
    appendCursorLines: 33,
    boundaryFingerprint: 'c'.repeat(64),
    lastEntryId: 'entry-31',
    logicalSessionIds: [],
    sourceAvailability: RecallSourceAvailability.PRESENT,
    sourceMissingObservedAtEpochMilliseconds: null,
    sourceMissingObservationCount: 0,
    sourceMissingSweepId: null,
    deletionCheckpoint: null,
    markerCheckpoint: {
      generationId,
      coveredMarkerIds: [marker.markerId],
      runtimeSequences: [{ runtimeInstanceId: identity.runtimeInstanceId, sequence: 1 }],
    },
    repairState: RecallProjectionRepairState.READY,
    repairReason: null,
  };
  return {
    status: RecallProjectionEncodingStatus.ENCODED,
    targetGenerationId: generationId,
    documents: Array.from({ length: 32 }, (_, index) => ({
      ...createTestSessionConversationChunk({
        id: `diagnostic-document-${index}`,
        physicalSessionProjectionId: projectionId,
        content: `diagnostic evidence ${index} DiagnosticIdentifier${index}`,
      }),
      isDenseSearchable: true,
      embedding: [1, index / 32, 0],
    })),
    checkpointIntent: { physicalProjection, logicalProjections: [] },
    workPlan,
    cacheHits: 32,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
  };
}

void test(
  'target-host 32-document recall write-window p95 stays within the measured candidate',
  { skip: !RUN_WRITE_WINDOW_DIAGNOSTIC },
  async (t) => {
    const directory = await mkdtemp(join(tmpdir(), 'recall-write-window-diagnostic-'));
    t.after(() => rm(directory, { recursive: true, force: true }));
    const markerSpoolDirectory = join(directory, 'markers');
    await mkdir(markerSpoolDirectory, { recursive: true });
    const prepared = createDiagnosticTransfer(markerSpoolDirectory);
    const markerId = prepared.workPlan.sourceMarkerIds[0];
    assert.ok(markerId);
    const samples = [];
    for (let index = 0; index < WRITE_WINDOW_SAMPLE_COUNT; index += 1) {
      await writeFile(join(markerSpoolDirectory, `${markerId}.json`), '{}\n');
      const result = await commitIncrementalRecallTransfer({
        prepared,
        lockPath: join(directory, 'operation.lock'),
        evidenceDatabasePath: join(directory, 'evidence'),
        projectionDatabasePath: join(directory, 'projections'),
        embeddingDimensions: 3,
      });
      const writeWindow = result.writeWindowDiagnostics[0];
      assert.ok(writeWindow);
      samples.push({
        lockWaitMilliseconds: writeWindow.lockWaitMilliseconds,
        evidenceOpenMilliseconds: writeWindow.evidenceOpenMilliseconds,
        evidenceWriteMilliseconds: writeWindow.evidenceWriteMilliseconds,
        projectionOpenMilliseconds: writeWindow.projectionOpenMilliseconds,
        projectionWriteMilliseconds: writeWindow.projectionWriteMilliseconds,
        closeMilliseconds: writeWindow.closeMilliseconds,
        checkpointObservationMilliseconds: result.checkpointObservationMilliseconds,
        markerAcknowledgementMilliseconds: result.markerAcknowledgementMilliseconds,
        writeWindowMilliseconds: writeWindow.writeWindowMilliseconds,
      });
    }
    const report = {
      sampleCount: samples.length,
      p95: {
        lockWaitMilliseconds: percentile95(
          samples.map(({ lockWaitMilliseconds }) => lockWaitMilliseconds),
        ),
        evidenceOpenMilliseconds: percentile95(
          samples.map(({ evidenceOpenMilliseconds }) => evidenceOpenMilliseconds),
        ),
        evidenceWriteMilliseconds: percentile95(
          samples.map(({ evidenceWriteMilliseconds }) => evidenceWriteMilliseconds),
        ),
        projectionOpenMilliseconds: percentile95(
          samples.map(({ projectionOpenMilliseconds }) => projectionOpenMilliseconds),
        ),
        projectionWriteMilliseconds: percentile95(
          samples.map(({ projectionWriteMilliseconds }) => projectionWriteMilliseconds),
        ),
        closeMilliseconds: percentile95(samples.map(({ closeMilliseconds }) => closeMilliseconds)),
        checkpointObservationMilliseconds: percentile95(
          samples.map(({ checkpointObservationMilliseconds }) => checkpointObservationMilliseconds),
        ),
        markerAcknowledgementMilliseconds: percentile95(
          samples.map(({ markerAcknowledgementMilliseconds }) => markerAcknowledgementMilliseconds),
        ),
        writeWindowMilliseconds: percentile95(
          samples.map(({ writeWindowMilliseconds }) => writeWindowMilliseconds),
        ),
      },
      targetWriteWindowMilliseconds: WRITE_WINDOW_P95_TARGET_MILLISECONDS,
    };
    process.stdout.write(`${JSON.stringify(report)}\n`);
    assert.ok(
      report.p95.writeWindowMilliseconds <= WRITE_WINDOW_P95_TARGET_MILLISECONDS,
      `Recall write-window p95 ${report.p95.writeWindowMilliseconds} ms exceeds ${WRITE_WINDOW_P95_TARGET_MILLISECONDS} ms; return to design review`,
    );
  },
);

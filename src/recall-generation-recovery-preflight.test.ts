import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { runRecallGenerationRecoveryPreflight } from './recall-generation-recovery-preflight.js';

void test('disposable recovery preflight matches interrupted and uninterrupted generation membership', async (t) => {
  const disposableRoot = await mkdtemp(join(tmpdir(), 'recall-generation-recovery-preflight-'));
  t.after(() => rm(disposableRoot, { recursive: true, force: true }));

  const result = await runRecallGenerationRecoveryPreflight({
    disposableRoot,
    logicalSessionCount: 1_001,
  });

  assert.equal(result.sourceSafety.originalPiSessionFilesAccessed, false);
  assert.equal(result.sourceSafety.liveRecallGenerationAccessed, false);
  assert.equal(result.fixedSnapshot.originalCardinalitySourceRemoved, true);
  assert.equal(result.fixedSnapshot.originalRetainedSourceChanged, true);
  assert.equal(result.fixedSnapshot.retainedOriginalEvidenceFound, true);
  assert.equal(result.fixedSnapshot.changedReplacementEvidenceFound, false);
  assert.equal(result.interruptions.bootstrapSnapshotCapture, 'resumed');
  assert.equal(result.interruptions.physicalSourceCheckpoint, 'resumed');
  assert.equal(result.failureClassification.malformedSourceSkipped, true);
  assert.equal(result.failureClassification.operationalFailureFatal, true);
  assert.equal(result.failureClassification.implementationFailureFatal, true);
  assert.deepEqual(result.interrupted.exactMembership, result.uninterrupted.exactMembership);
  assert.equal(
    result.interrupted.startingSnapshotFingerprint,
    result.uninterrupted.startingSnapshotFingerprint,
  );
  assert.equal(result.interrupted.manifestFingerprint, result.uninterrupted.manifestFingerprint);
  assert.equal(result.interrupted.storeCounts.dense > 1_001, true);
  assert.equal(
    result.interrupted.storeCounts.lexicalSource > result.interrupted.storeCounts.dense,
    true,
  );
  assert.equal(result.interrupted.storeCounts.sessionProjection > result.logicalSessionCount, true);
  assert.equal(result.detached.interruptionSignal, 'SIGKILL');
  assert.equal(result.detached.resumedWorkerReachedTerminalValidation, true);
  assert.equal(result.detached.uninterruptedWorkerReachedTerminalValidation, true);
  assert.equal(result.detached.validationReceiptsEquivalent, true);
  assert.deepEqual(
    result.detached.interrupted.exactMembership,
    result.detached.uninterrupted.exactMembership,
  );
});

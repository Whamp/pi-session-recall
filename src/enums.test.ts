import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PROJECT_SCOPE_POLICY_VERSION,
  RecallAppendDeltaStatus,
  RecallAppendProjectionStatus,
  RecallBacklogFailureCategory,
  RecallEligibilityThreshold,
  RecallEvidenceRelation,
  RecallGenerationCutoverState,
  RecallProjectionEncodingStatus,
  RecallProjectionRepairReason,
  RecallProjectionRepairState,
  RecallProjectIdentitySource,
  RecallSearchScope,
  RecallSessionProjectionKind,
  RecallSourceAvailability,
  RecallWorkMarkerTrigger,
} from './enums.js';

void test('project scope provenance values keep their persisted wire spellings', () => {
  assert.equal(PROJECT_SCOPE_POLICY_VERSION, 1);
  assert.deepEqual(Object.values(RecallProjectIdentitySource), [
    'git_origin',
    'git_common_directory',
    'non_git_session_origin',
    'configured_project_lineage',
  ]);
  assert.deepEqual(Object.values(RecallSearchScope), ['project', 'global']);
  assert.deepEqual(Object.values(RecallEvidenceRelation), [
    'same_repository',
    'configured_project_lineage',
    'same_session_origin',
    'unrestricted_global_evidence',
  ]);
});

void test('incremental recall contract values keep their persisted wire spellings', () => {
  assert.deepEqual(Object.values(RecallWorkMarkerTrigger), [
    'activity',
    'compaction',
    'branch_exit',
    'departure',
    'arrival',
  ]);
  assert.deepEqual(Object.values(RecallSessionProjectionKind), [
    'physical_session',
    'logical_session',
  ]);
  assert.deepEqual(Object.values(RecallProjectionRepairState), [
    'ready',
    'requires_reconciliation',
  ]);
  assert.deepEqual(Object.values(RecallProjectionRepairReason), [
    'append_cursor_missing',
    'source_shrank',
    'source_identity_mismatch',
    'boundary_mismatch',
    'unsupported_layout',
    'malformed_graph',
    'projection_overflow',
  ]);
  assert.deepEqual(Object.values(RecallSourceAvailability), [
    'present',
    'source_missing',
    'deletion_confirmed',
  ]);
  assert.deepEqual(Object.values(RecallProjectionEncodingStatus), [
    'encoded',
    'requires_reconciliation',
  ]);
  assert.deepEqual(Object.values(RecallAppendDeltaStatus), ['appended', 'requires_reconciliation']);
  assert.deepEqual(Object.values(RecallAppendProjectionStatus), [
    'projected',
    'requires_reconciliation',
  ]);
  assert.deepEqual(Object.values(RecallEligibilityThreshold), [
    'explicit_exit_quiet',
    'large_prepared_transfer',
    'crash_only_quiescence',
  ]);
  assert.deepEqual(Object.values(RecallGenerationCutoverState), [
    'building',
    'ready',
    'active',
    'replay_pending',
    'legacy_read_only',
    'rollback',
    'retired',
    'failed',
  ]);
  assert.deepEqual(Object.values(RecallBacklogFailureCategory), [
    'marker_decode_failed',
    'projection_reconciliation_required',
    'write_failed',
    'recovery_required',
    'rebuild_failed',
  ]);
});

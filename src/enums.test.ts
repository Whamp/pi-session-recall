import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PROJECT_SCOPE_POLICY_VERSION,
  RecallEvidenceRelation,
  RecallProjectIdentitySource,
  RecallSearchScope,
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

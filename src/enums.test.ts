import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';

void test('project scope provenance values keep their persisted wire spellings', () => {
  assert.deepEqual(Object.values(RecallProjectIdentitySource), [
    'git_origin',
    'git_common_directory',
    'non_git_session_origin',
  ]);
  assert.deepEqual(Object.values(RecallSearchScope), ['project', 'global']);
  assert.deepEqual(Object.values(RecallEvidenceRelation), [
    'same_repository',
    'same_session_origin',
    'unrestricted_global_evidence',
  ]);
});

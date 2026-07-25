import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEvidenceRelation, RecallProjectIdentitySource, RecallSearchScope } from './enums.js';

void test('project scope provenance values keep their persisted wire spellings', () => {
  assert.deepEqual(Object.values(RecallProjectIdentitySource), [
    'git_origin',
    'git_common_directory',
  ]);
  assert.deepEqual(Object.values(RecallSearchScope), ['project', 'global']);
  assert.deepEqual(Object.values(RecallEvidenceRelation), [
    'same_repository',
    'unrestricted_global_evidence',
  ]);
});

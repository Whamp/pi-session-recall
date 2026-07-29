import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';
import {
  formatQmdQueryPlanningPrompt,
  parseQmdQueryPlanningOutput,
} from './recall-query-planning-policy.js';

void test('QMD planning policy formats intent and parses bounded typed queries', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();

  assert.equal(
    formatQmdQueryPlanningPrompt('source provenance', 'Find retained evidence'),
    '/no_think Expand this search query: source provenance\nQuery intent: Find retained evidence',
  );
  assert.deepEqual(
    parseQmdQueryPlanningOutput(
      'lex: source provenance evidence\nvec: retained source provenance',
      profile,
    ),
    [
      { type: 'lex', query: 'source provenance evidence' },
      { type: 'vec', query: 'retained source provenance' },
    ],
  );
});

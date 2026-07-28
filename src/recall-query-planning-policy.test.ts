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
      'lex: source provenance evidence\nvec: retained source provenance\nhyde: Source provenance connects evidence.</think>',
      profile,
    ),
    [
      { type: 'lex', query: 'source provenance evidence' },
      { type: 'vec', query: 'retained source provenance' },
      { type: 'hyde', query: 'Source provenance connects evidence.' },
    ],
  );
});

void test('QMD planning policy preserves semantic drift across bounded generated lines', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();

  assert.deepEqual(
    parseQmdQueryPlanningOutput(
      [
        'lex: unrelated generic documentation',
        'lex: source provenance evidence',
        'vec: general software architecture',
        'vec: retained provenance records',
      ].join('\n'),
      profile,
    ),
    [
      { type: 'lex', query: 'unrelated generic documentation' },
      { type: 'lex', query: 'source provenance evidence' },
      { type: 'vec', query: 'general software architecture' },
      { type: 'vec', query: 'retained provenance records' },
    ],
  );
});

void test('QMD planning policy removes repeated typed lines before bounded retrieval', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();

  assert.deepEqual(
    parseQmdQueryPlanningOutput(
      [
        'lex: source provenance evidence',
        'lex: source provenance evidence',
        'vec: retained provenance records',
        'vec: retained provenance records',
      ].join('\n'),
      profile,
    ),
    [
      { type: 'lex', query: 'source provenance evidence' },
      { type: 'vec', query: 'retained provenance records' },
    ],
  );
});

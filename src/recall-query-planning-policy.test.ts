import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallPlannedRetrievalQuery } from './recall-inference-capabilities.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';
import {
  formatQmdQueryPlanningPrompt,
  parseQmdQueryPlanningOutput,
  validateQmdQueryPlanningPlan,
} from './recall-query-planning-policy.js';

interface InvalidQmdQueryPlanCase {
  name: string;
  plan: RecallPlannedRetrievalQuery[];
}

const INVALID_QMD_QUERY_PLAN_CASES: readonly InvalidQmdQueryPlanCase[] = [
  {
    name: 'vector before the first lexical query',
    plan: [
      { type: 'vec', query: 'semantic source provenance' },
      { type: 'lex', query: 'source provenance' },
    ],
  },
  {
    name: 'lexical query after a vector query',
    plan: [
      { type: 'lex', query: 'source provenance' },
      { type: 'vec', query: 'semantic source provenance' },
      { type: 'lex', query: 'retained evidence' },
    ],
  },
  {
    name: 'non-final hypothetical-answer query',
    plan: [
      { type: 'lex', query: 'source provenance' },
      { type: 'hyde', query: 'Source provenance connects retained evidence.' },
      { type: 'vec', query: 'semantic source provenance' },
    ],
  },
  {
    name: 'entry after a hypothetical-answer query',
    plan: [
      { type: 'lex', query: 'source provenance' },
      { type: 'vec', query: 'semantic source provenance' },
      { type: 'hyde', query: 'Source provenance connects retained evidence.' },
      { type: 'vec', query: 'where provenance was retained' },
    ],
  },
  {
    name: '513-code-point query content',
    plan: [
      { type: 'lex', query: 'x'.repeat(513) },
      { type: 'vec', query: 'semantic source provenance' },
    ],
  },
];

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

void test('QMD parsed planner output rejects repeated typed lines instead of repairing them', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();

  assert.throws(
    () =>
      parseQmdQueryPlanningOutput(
        [
          'lex: source provenance evidence',
          'lex: source provenance evidence',
          'vec: retained provenance records',
          'vec: retained provenance records',
        ].join('\n'),
        profile,
      ),
    /Recall query planning output invalid: duplicate typed queries/u,
  );
});

void test('QMD planning policy accepts exactly 512 Unicode code points', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const maximumContent = '😀'.repeat(512);
  const plan: RecallPlannedRetrievalQuery[] = [
    { type: 'lex', query: maximumContent },
    { type: 'vec', query: 'semantic source provenance' },
  ];

  assert.deepEqual(validateQmdQueryPlanningPlan(plan, profile), plan);
  assert.deepEqual(
    parseQmdQueryPlanningOutput(`lex: ${maximumContent}\nvec: semantic source provenance`, profile),
    plan,
  );
});

for (const invalidCase of INVALID_QMD_QUERY_PLAN_CASES) {
  void test(`QMD direct plan validation rejects ${invalidCase.name}`, () => {
    assert.throws(
      () =>
        validateQmdQueryPlanningPlan(
          invalidCase.plan,
          createRecommendedQmdQueryPlanningModelProfile(),
        ),
      /Recall query planning output/u,
    );
  });

  void test(`QMD parsed planner output rejects ${invalidCase.name}`, () => {
    const generatedOutput = invalidCase.plan
      .map(({ type, query }) => `${type}: ${query}`)
      .join('\n');

    assert.throws(
      () =>
        parseQmdQueryPlanningOutput(
          generatedOutput,
          createRecommendedQmdQueryPlanningModelProfile(),
        ),
      /Recall query planning output/u,
    );
  });
}

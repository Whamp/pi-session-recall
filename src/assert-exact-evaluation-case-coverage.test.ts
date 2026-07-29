import assert from 'node:assert/strict';
import test from 'node:test';

import { assertExactEvaluationCaseCoverage } from './assert-exact-evaluation-case-coverage.js';

void test('evaluation case coverage requires nonempty, independently unique, exactly equal ID sets', () => {
  assert.doesNotThrow(() =>
    assertExactEvaluationCaseCoverage({
      controls: ['case-001', 'case-002'],
      plans: ['case-002', 'case-001'],
      measurements: ['case-001', 'case-002'],
    }),
  );

  for (const collections of [
    {
      controls: ['case-001', 'case-002'],
      plans: ['case-001', 'case-001'],
      measurements: ['case-001', 'case-002'],
    },
    {
      controls: ['case-001', 'case-002'],
      plans: ['case-001', 'case-002'],
      measurements: ['case-001', 'case-003'],
    },
    {
      controls: ['case-001', 'case-002'],
      plans: ['case-001'],
      measurements: ['case-001', 'case-002'],
    },
    {
      controls: ['case-001'],
      plans: ['case-001'],
      measurements: [],
    },
  ]) {
    assert.throws(
      () => assertExactEvaluationCaseCoverage(collections),
      /Evaluation case coverage invalid/u,
    );
  }
});

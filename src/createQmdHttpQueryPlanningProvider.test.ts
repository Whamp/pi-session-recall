import assert from 'node:assert/strict';
import test from 'node:test';

import { createQmdHttpQueryPlanningProvider } from './createQmdHttpQueryPlanningProvider.js';
import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

void test('QMD HTTP query planning provider rejects a non-HTTP endpoint', () => {
  assert.throws(
    () =>
      createQmdHttpQueryPlanningProvider(createRecommendedQmdQueryPlanningModelProfile(), {
        baseUrl: 'file:///models',
      }),
    /query planner base URL invalid protocol/u,
  );
});

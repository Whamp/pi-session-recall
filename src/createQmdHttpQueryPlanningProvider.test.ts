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

void test('QMD HTTP query planning identity distinguishes endpoint configuration without exposing it', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  const first = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl: 'http://planner-a.example.test/v1',
  });
  const second = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl: 'http://planner-b.example.test/v1',
  });
  const equivalent = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl: 'HTTP://PLANNER-A.EXAMPLE.TEST:80/v1/',
  });
  const changedTimeout = createQmdHttpQueryPlanningProvider(profile, {
    baseUrl: 'http://planner-a.example.test/v1',
    requestTimeoutMilliseconds: 2_000,
  });

  assert.match(
    first.executionIdentity.adapterConfigurationIdentity,
    /^llama-cpp-http-query-planning-config-v1:[a-f0-9]{64}$/u,
  );
  assert.notEqual(
    first.executionIdentity.adapterConfigurationIdentity,
    second.executionIdentity.adapterConfigurationIdentity,
  );
  assert.notEqual(first.executionIdentity.cacheIdentity, second.executionIdentity.cacheIdentity);
  assert.deepEqual(first.executionIdentity, equivalent.executionIdentity);
  assert.notEqual(
    first.executionIdentity.cacheIdentity,
    changedTimeout.executionIdentity.cacheIdentity,
  );
  assert.doesNotMatch(
    first.executionIdentity.adapterConfigurationIdentity,
    /planner-a\.example\.test/u,
  );
});

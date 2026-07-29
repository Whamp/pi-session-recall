import assert from 'node:assert/strict';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  runExactRecallEvaluationSemanticTest,
  verifyRequiredRecallEvaluationSemanticChecks,
} from './recall-evaluation-semantic-checks.js';

const PROJECT_DIRECTORY = dirname(dirname(fileURLToPath(import.meta.url)));

void test('required acceptance semantics execute each stable test identity exactly once', () => {
  assert.deepEqual(verifyRequiredRecallEvaluationSemanticChecks(PROJECT_DIRECTORY), {
    plannerFallbackPublicServicePassed: true,
    rerankerFailurePublicServicePassed: true,
    piToolContractPassed: true,
  });
});

void test('a nonmatching required semantic identity fails despite a successful test process', () => {
  assert.throws(
    () =>
      runExactRecallEvaluationSemanticTest(
        PROJECT_DIRECTORY,
        join('src', 'recall-extension.test.ts'),
        'guaranteed nonmatching acceptance identity',
      ),
    /required test identity did not execute exactly once/u,
  );
});

import assert from 'node:assert/strict';
import test from 'node:test';

import { RecallEligibilityThreshold } from './enums.js';
import { scheduleRecallEligibility } from './schedule-recall-eligibility.js';

const cases = [
  {
    name: 'explicit context exit waits for 60 seconds without growth',
    explicitExitObserved: true,
    preparedDocumentCount: 1,
    expectedThreshold: RecallEligibilityThreshold.EXPLICIT_EXIT_QUIET,
    thresholdMilliseconds: 60_000,
  },
  {
    name: 'prepared work over 32 documents waits for 5 minutes without growth',
    explicitExitObserved: true,
    preparedDocumentCount: 33,
    expectedThreshold: RecallEligibilityThreshold.LARGE_PREPARED_TRANSFER,
    thresholdMilliseconds: 5 * 60_000,
  },
  {
    name: 'crash-only quiescence waits for 30 minutes without growth',
    explicitExitObserved: false,
    preparedDocumentCount: 1,
    expectedThreshold: RecallEligibilityThreshold.CRASH_ONLY_QUIESCENCE,
    thresholdMilliseconds: 30 * 60_000,
  },
];

for (const scenario of cases) {
  void test(scenario.name, () => {
    let now = scenario.thresholdMilliseconds - 1;
    const before = scheduleRecallEligibility({
      lastGrowthAtEpochMilliseconds: 0,
      explicitExitObserved: scenario.explicitExitObserved,
      preparedDocumentCount: scenario.preparedDocumentCount,
      nowEpochMilliseconds: () => now,
    });
    assert.deepEqual(before, {
      ready: false,
      threshold: scenario.expectedThreshold,
      thresholdMilliseconds: scenario.thresholdMilliseconds,
      quietMilliseconds: scenario.thresholdMilliseconds - 1,
    });
    now += 1;
    assert.deepEqual(
      scheduleRecallEligibility({
        lastGrowthAtEpochMilliseconds: 0,
        explicitExitObserved: scenario.explicitExitObserved,
        preparedDocumentCount: scenario.preparedDocumentCount,
        nowEpochMilliseconds: () => now,
      }),
      {
        ready: true,
        threshold: scenario.expectedThreshold,
        thresholdMilliseconds: scenario.thresholdMilliseconds,
        quietMilliseconds: scenario.thresholdMilliseconds,
      },
    );
  });
}

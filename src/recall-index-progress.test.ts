import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallIndexProgressEvent } from './recall-index-progress.js';

const REBUILD_PROGRESS_EVENTS = [
  { kind: 'preparing-rebuild-candidate', staleCandidatesRemoved: 2 },
  { kind: 'resuming-rebuild-candidate' },
  { kind: 'rebuild-candidate-failed' },
  { kind: 'rebuild-candidate-staged', databaseTarget: 'generations/candidate' },
  { kind: 'rebuild-candidate-activated', previousAvailable: true },
] satisfies readonly RecallIndexProgressEvent[];

void test('recall index progress events carry the candidate rebuild lifecycle facts', () => {
  assert.deepEqual(
    REBUILD_PROGRESS_EVENTS.map((event) => event.kind),
    [
      'preparing-rebuild-candidate',
      'resuming-rebuild-candidate',
      'rebuild-candidate-failed',
      'rebuild-candidate-staged',
      'rebuild-candidate-activated',
    ],
  );
  const preparingEvent = REBUILD_PROGRESS_EVENTS.at(0);
  const stagedEvent = REBUILD_PROGRESS_EVENTS.at(3);
  const activatedEvent = REBUILD_PROGRESS_EVENTS.at(4);
  assert.ok(preparingEvent);
  assert.ok(stagedEvent);
  assert.ok(activatedEvent);
  assert.equal(preparingEvent.staleCandidatesRemoved, 2);
  assert.equal(stagedEvent.databaseTarget, 'generations/candidate');
  assert.equal(activatedEvent.previousAvailable, true);
});

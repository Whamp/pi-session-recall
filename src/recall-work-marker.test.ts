import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallWorkMarkerTrigger } from './enums.js';
import {
  createRecallWorkMarkerId,
  decodeRecallWorkMarker,
  encodeRecallWorkMarker,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
} from './recall-work-marker.js';

async function createMarkerFixture(t: test.TestContext): Promise<{
  marker: RecallWorkMarker;
  sessionsDirectory: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), 'recall-work-marker-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const physicalSessionPath = join(sessionsDirectory, '2026-07-24', 'session.jsonl');
  await mkdir(join(sessionsDirectory, '2026-07-24'), { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');
  const markerWithoutId = {
    version: RECALL_WORK_MARKER_VERSION,
    physicalSessionId: 'physical-session-1',
    physicalSessionPath,
    runtimeInstanceId: 'runtime-instance-1',
    runtimeSequence: 7,
    createdAtEpochMilliseconds: 1_753_315_200_000,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  return {
    sessionsDirectory,
    marker: {
      ...markerWithoutId,
      markerId: createRecallWorkMarkerId(markerWithoutId),
    },
  };
}

void test('recall work marker round-trips every strict trigger payload', async (t) => {
  const { marker, sessionsDirectory } = await createMarkerFixture(t);
  const triggers: RecallWorkMarker['trigger'][] = [
    { kind: RecallWorkMarkerTrigger.ACTIVITY },
    { kind: RecallWorkMarkerTrigger.COMPACTION, compactionEntryId: 'compaction-entry-1' },
    {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      oldLeafEntryId: 'old-leaf-1',
      newLeafEntryId: 'new-leaf-1',
      summaryEntryId: 'branch-summary-1',
    },
    { kind: RecallWorkMarkerTrigger.DEPARTURE },
    { kind: RecallWorkMarkerTrigger.ARRIVAL },
  ];

  for (const trigger of triggers) {
    const markerWithoutId = { ...marker, trigger };
    const candidate: RecallWorkMarker = {
      ...markerWithoutId,
      markerId: createRecallWorkMarkerId(markerWithoutId),
    };
    const encoded = await encodeRecallWorkMarker(candidate, {
      trustedSessionRoots: [sessionsDirectory],
    });
    assert.deepEqual(
      await decodeRecallWorkMarker(encoded, { trustedSessionRoots: [sessionsDirectory] }),
      candidate,
    );
    assert.match(candidate.markerId, /^[A-Za-z0-9_-]+$/u);
    assert.equal(candidate.markerId.includes(':'), false);
  }
});

void test('recall work marker ID has a deterministic golden wire value', async (t) => {
  const { marker } = await createMarkerFixture(t);
  const first = createRecallWorkMarkerId(marker);
  assert.equal(createRecallWorkMarkerId(marker), first);
  assert.notEqual(createRecallWorkMarkerId({ ...marker, runtimeSequence: 8 }), first);
  assert.equal(
    createRecallWorkMarkerId({
      version: RECALL_WORK_MARKER_VERSION,
      physicalSessionId: 'physical-session-golden',
      physicalSessionPath: '/trusted/sessions/golden.jsonl',
      runtimeInstanceId: 'runtime-golden',
      runtimeSequence: 11,
      createdAtEpochMilliseconds: 1_753_315_200_000,
      trigger: {
        kind: RecallWorkMarkerTrigger.COMPACTION,
        compactionEntryId: 'compaction-golden',
      },
    }),
    'marker_-8Pep7mmpgob7WlIh9kTX7ixxDILcMQkjKtYM2EmD4w',
  );
});

void test('recall work marker rejects malformed and forward-incompatible values', async (t) => {
  const { marker, sessionsDirectory } = await createMarkerFixture(t);
  const decode = async (value: unknown): Promise<unknown> =>
    decodeRecallWorkMarker(JSON.stringify(value), { trustedSessionRoots: [sessionsDirectory] });

  await assert.rejects(() => decode({ ...marker, version: 2 }), /marker version|invalid/iu);
  await assert.rejects(
    () => decode({ ...marker, trigger: { kind: 'future_trigger' } }),
    /marker|trigger|invalid/iu,
  );
  await assert.rejects(
    () => decode({ ...marker, trigger: { kind: RecallWorkMarkerTrigger.COMPACTION } }),
    /marker|compactionEntryId|invalid/iu,
  );
  await assert.rejects(
    () =>
      decode({
        ...marker,
        trigger: {
          kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
          newLeafEntryId: 'new-leaf-1',
        },
      }),
    /marker|oldLeafEntryId|invalid/iu,
  );
  for (const runtimeSequence of [0, -1, 1.5]) {
    await assert.rejects(
      () => decode({ ...marker, runtimeSequence }),
      /marker|runtimeSequence|invalid/iu,
    );
  }
  await assert.rejects(() => decode({ ...marker, unexpected: true }), /marker|invalid/iu);
  await assert.rejects(
    () => decode({ ...marker, trigger: { ...marker.trigger, unexpected: true } }),
    /marker|invalid/iu,
  );
  await assert.rejects(
    () => decode({ ...marker, markerId: 'forged_marker_id' }),
    /marker ID mismatch/iu,
  );
});

void test('recall work marker accepts a missing source beneath a canonical trusted root', async (t) => {
  const { marker, sessionsDirectory } = await createMarkerFixture(t);
  const physicalSessionPath = join(sessionsDirectory, 'removed-date', 'removed-session.jsonl');
  const candidate = { ...marker, physicalSessionPath };
  candidate.markerId = createRecallWorkMarkerId(candidate);

  assert.deepEqual(
    await decodeRecallWorkMarker(JSON.stringify(candidate), {
      trustedSessionRoots: [sessionsDirectory],
    }),
    candidate,
  );
});

void test('recall work marker rejects relative, outside-root, and symlink-escaped paths', async (t) => {
  const { marker, sessionsDirectory } = await createMarkerFixture(t);
  const directory = await mkdtemp(join(tmpdir(), 'recall-work-marker-escape-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const outsidePath = join(directory, 'outside.jsonl');
  await writeFile(outsidePath, '{}\n');
  const escapePath = join(sessionsDirectory, 'escape.jsonl');
  await symlink(outsidePath, escapePath);

  for (const physicalSessionPath of ['relative/session.jsonl', outsidePath, escapePath]) {
    const candidate = { ...marker, physicalSessionPath };
    candidate.markerId = createRecallWorkMarkerId(candidate);
    await assert.rejects(
      () => encodeRecallWorkMarker(candidate, { trustedSessionRoots: [sessionsDirectory] }),
      /trusted session root|absolute/iu,
    );
  }
});

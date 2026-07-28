import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { coordinateRecallMarkerReplay } from './coordinate-recall-marker-replay.js';
import { RecallMarkerQuarantineCategory, RecallWorkMarkerTrigger } from './enums.js';
import {
  createRecallWorkMarkerId,
  encodeRecallWorkMarker,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
  type RecallWorkMarkerTriggerPayload,
} from './recall-work-marker.js';

interface CoordinatorFixture {
  markerQuarantineDirectory: string;
  markerSpoolDirectory: string;
  sessionsDirectory: string;
  createMarker(
    runtimeInstanceId: string,
    runtimeSequence: number,
    trigger: RecallWorkMarkerTriggerPayload,
  ): RecallWorkMarker;
  publishMarker(marker: RecallWorkMarker): Promise<void>;
}

async function createCoordinatorFixture(t: test.TestContext): Promise<CoordinatorFixture> {
  const directory = await mkdtemp(join(tmpdir(), 'coordinate-recall-marker-replay-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'recall', 'markers', 'pending');
  const markerQuarantineDirectory = join(directory, 'recall', 'markers', 'quarantine');
  const physicalSessionPath = join(sessionsDirectory, 'session.jsonl');
  await mkdir(markerSpoolDirectory, { recursive: true });
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');

  function createMarker(
    runtimeInstanceId: string,
    runtimeSequence: number,
    trigger: RecallWorkMarkerTriggerPayload,
  ): RecallWorkMarker {
    const identity = {
      version: RECALL_WORK_MARKER_VERSION,
      physicalSessionId: 'physical-session-1',
      physicalSessionPath,
      runtimeInstanceId,
      runtimeSequence,
      createdAtEpochMilliseconds: 1_753_315_200_000 + runtimeSequence,
      trigger,
    } as const;
    return { ...identity, markerId: createRecallWorkMarkerId(identity) };
  }

  return {
    markerQuarantineDirectory,
    markerSpoolDirectory,
    sessionsDirectory,
    createMarker,
    async publishMarker(marker) {
      await writeFile(
        join(markerSpoolDirectory, `${marker.markerId}.json`),
        await encodeRecallWorkMarker(marker, { trustedSessionRoots: [sessionsDirectory] }),
      );
    },
  };
}

void test('recall marker replay coalesces activity while preserving divergent compactions and branch exits', async (t) => {
  const fixture = await createCoordinatorFixture(t);
  const markers = [
    fixture.createMarker('runtime-b', 2, { kind: RecallWorkMarkerTrigger.ACTIVITY }),
    fixture.createMarker('runtime-a', 5, { kind: RecallWorkMarkerTrigger.ACTIVITY }),
    fixture.createMarker('runtime-a', 1, { kind: RecallWorkMarkerTrigger.ACTIVITY }),
    fixture.createMarker('runtime-a', 2, { kind: RecallWorkMarkerTrigger.ARRIVAL }),
    fixture.createMarker('runtime-a', 6, { kind: RecallWorkMarkerTrigger.ARRIVAL }),
    fixture.createMarker('runtime-a', 3, {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical-session-1',
      oldLeafEntryId: 'old-1',
      newLeafEntryId: 'new-1',
    }),
    fixture.createMarker('runtime-a', 4, {
      kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
      logicalSessionId: 'logical-session-1',
      oldLeafEntryId: 'old-2',
      newLeafEntryId: 'new-2',
    }),
    fixture.createMarker('runtime-a', 7, { kind: RecallWorkMarkerTrigger.DEPARTURE }),
    fixture.createMarker('runtime-a', 8, { kind: RecallWorkMarkerTrigger.DEPARTURE }),
    fixture.createMarker('runtime-a', 9, {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'logical-session-1',
      compactionEntryId: 'compaction-1',
    }),
    fixture.createMarker('runtime-b', 1, {
      kind: RecallWorkMarkerTrigger.COMPACTION,
      logicalSessionId: 'logical-session-2',
      compactionEntryId: 'compaction-2',
    }),
  ];
  await Promise.all(markers.toReversed().map((marker) => fixture.publishMarker(marker)));

  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => 1_753_315_300_000,
  });

  assert.equal(workPlan.discoveredMarkerCount, markers.length);
  assert.deepEqual(
    workPlan.workItems.map(({ marker }) => [
      marker.runtimeInstanceId,
      marker.runtimeSequence,
      marker.trigger.kind,
    ]),
    [
      ['runtime-a', 3, RecallWorkMarkerTrigger.BRANCH_EXIT],
      ['runtime-a', 4, RecallWorkMarkerTrigger.BRANCH_EXIT],
      ['runtime-a', 5, RecallWorkMarkerTrigger.ACTIVITY],
      ['runtime-a', 6, RecallWorkMarkerTrigger.ARRIVAL],
      ['runtime-a', 8, RecallWorkMarkerTrigger.DEPARTURE],
      ['runtime-a', 9, RecallWorkMarkerTrigger.COMPACTION],
      ['runtime-b', 1, RecallWorkMarkerTrigger.COMPACTION],
      ['runtime-b', 2, RecallWorkMarkerTrigger.ACTIVITY],
    ],
  );
  const compactionItems = workPlan.workItems.filter(
    ({ marker }) => marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION,
  );
  assert.deepEqual(
    compactionItems.map(({ marker, coveredMarkerIds }) => ({
      logicalSessionId:
        marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION
          ? marker.trigger.logicalSessionId
          : null,
      compactionEntryId:
        marker.trigger.kind === RecallWorkMarkerTrigger.COMPACTION
          ? marker.trigger.compactionEntryId
          : null,
      coveredMarkerIds,
    })),
    markers
      .filter(({ trigger }) => trigger.kind === RecallWorkMarkerTrigger.COMPACTION)
      .map(({ markerId, trigger }) => ({
        logicalSessionId:
          trigger.kind === RecallWorkMarkerTrigger.COMPACTION ? trigger.logicalSessionId : null,
        compactionEntryId:
          trigger.kind === RecallWorkMarkerTrigger.COMPACTION ? trigger.compactionEntryId : null,
        coveredMarkerIds: [markerId],
      }))
      .toSorted(
        (left, right) => left.compactionEntryId?.localeCompare(right.compactionEntryId ?? '') ?? 0,
      ),
  );
  assert.deepEqual(workPlan.quarantineDiagnostics, []);
});

void test('recall marker replay quarantines corrupt and unsupported files without exposing their contents', async (t) => {
  const fixture = await createCoordinatorFixture(t);
  const validMarker = fixture.createMarker('runtime-a', 1, {
    kind: RecallWorkMarkerTrigger.ACTIVITY,
  });
  await fixture.publishMarker(validMarker);
  const corruptSource = '{private broken marker';
  const unsupportedSource = JSON.stringify({ ...validMarker, version: 2, privateText: 'secret' });
  const unknownTriggerSource = JSON.stringify({
    ...validMarker,
    trigger: { kind: 'future_trigger' },
  });
  const corruptPath = join(fixture.markerSpoolDirectory, 'corrupt-marker.json');
  const unsupportedPath = join(fixture.markerSpoolDirectory, 'unsupported-marker.json');
  const unknownTriggerPath = join(fixture.markerSpoolDirectory, 'unknown-trigger-marker.json');
  await writeFile(corruptPath, corruptSource);
  await writeFile(unsupportedPath, unsupportedSource);
  await writeFile(unknownTriggerPath, unknownTriggerSource);
  await utimes(corruptPath, 10, 10);
  await utimes(unsupportedPath, 20, 20);
  await utimes(unknownTriggerPath, 15, 15);

  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
    nowEpochMilliseconds: () => 30_000,
  });

  assert.deepEqual(workPlan.sourceMarkerIds, [validMarker.markerId]);
  assert.deepEqual(workPlan.quarantineDiagnostics, [
    {
      category: RecallMarkerQuarantineCategory.CORRUPT,
      count: 1,
      oldestAgeMilliseconds: 20_000,
    },
    {
      category: RecallMarkerQuarantineCategory.UNSUPPORTED,
      count: 2,
      oldestAgeMilliseconds: 15_000,
    },
  ]);
  assert.deepEqual(await readdir(fixture.markerSpoolDirectory), [`${validMarker.markerId}.json`]);
  const quarantinedSources: string[] = [];
  for (const category of [
    RecallMarkerQuarantineCategory.CORRUPT,
    RecallMarkerQuarantineCategory.UNSUPPORTED,
  ]) {
    const categoryDirectory = join(fixture.markerQuarantineDirectory, category);
    for (const name of await readdir(categoryDirectory)) {
      quarantinedSources.push(await readFile(join(categoryDirectory, name), 'utf8'));
    }
  }
  assert.deepEqual(
    quarantinedSources.toSorted(),
    [corruptSource, unknownTriggerSource, unsupportedSource].toSorted(),
  );
  assert.equal(JSON.stringify(workPlan.quarantineDiagnostics).includes('private'), false);
  assert.equal(
    JSON.stringify(workPlan.quarantineDiagnostics).includes(fixture.sessionsDirectory),
    false,
  );
});

void test('recall marker replay deduplicates repeated delivery and breaks equal sequence ties by marker ID', async (t) => {
  const fixture = await createCoordinatorFixture(t);
  const duplicate = fixture.createMarker('runtime-a', 1, {
    kind: RecallWorkMarkerTrigger.ACTIVITY,
  });
  await fixture.publishMarker(duplicate);
  await fixture.publishMarker(duplicate);
  const firstTie = fixture.createMarker('runtime-a', 2, {
    kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
    logicalSessionId: 'logical-session-1',
    oldLeafEntryId: 'old-a',
    newLeafEntryId: 'new-a',
  });
  const secondTieTrigger: RecallWorkMarkerTriggerPayload = {
    kind: RecallWorkMarkerTrigger.BRANCH_EXIT,
    logicalSessionId: 'logical-session-1',
    oldLeafEntryId: 'old-b',
    newLeafEntryId: 'new-b',
  };
  const secondTieIdentity = {
    version: firstTie.version,
    physicalSessionId: firstTie.physicalSessionId,
    physicalSessionPath: firstTie.physicalSessionPath,
    runtimeInstanceId: firstTie.runtimeInstanceId,
    runtimeSequence: firstTie.runtimeSequence,
    createdAtEpochMilliseconds: firstTie.createdAtEpochMilliseconds + 1,
    trigger: secondTieTrigger,
  };
  const secondTie: RecallWorkMarker = {
    ...secondTieIdentity,
    markerId: createRecallWorkMarkerId(secondTieIdentity),
  };
  await fixture.publishMarker(firstTie);
  await fixture.publishMarker(secondTie);

  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    markerQuarantineDirectory: fixture.markerQuarantineDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [fixture.sessionsDirectory],
  });

  assert.equal(workPlan.discoveredMarkerCount, 3);
  assert.deepEqual(
    workPlan.workItems
      .filter(({ marker }) => marker.runtimeSequence === 2)
      .map(({ marker }) => marker.markerId),
    [firstTie.markerId, secondTie.markerId].toSorted(),
  );
});

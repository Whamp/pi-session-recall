import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { coordinateRecallMarkerReplay } from './coordinate-recall-marker-replay.js';
import { RecallWorkMarkerTrigger } from './enums.js';
import { acknowledgeCoveredRecallMarkers } from './recall-marker-spool.js';
import {
  createRecallWorkMarkerId,
  encodeRecallWorkMarker,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
} from './recall-work-marker.js';

void test('recall marker acknowledgement requires exact target-generation checkpoint coverage', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-marker-spool-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'recall', 'markers', 'pending');
  const physicalSessionPath = join(sessionsDirectory, 'session.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await mkdir(markerSpoolDirectory, { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');

  const markers: RecallWorkMarker[] = [1, 2].map((runtimeSequence) => {
    const identity = {
      version: RECALL_WORK_MARKER_VERSION,
      physicalSessionId: 'physical-session-1',
      physicalSessionPath,
      runtimeInstanceId: 'runtime-1',
      runtimeSequence,
      createdAtEpochMilliseconds: 1_753_315_200_000 + runtimeSequence,
      trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
    } as const;
    return { ...identity, markerId: createRecallWorkMarkerId(identity) };
  });
  for (const marker of markers) {
    await writeFile(
      join(markerSpoolDirectory, `${marker.markerId}.json`),
      await encodeRecallWorkMarker(marker, { trustedSessionRoots: [sessionsDirectory] }),
    );
  }
  const retainedMarkerDirectory = join(directory, 'recall', 'markers', 'rollback-retained');
  const workPlan = await coordinateRecallMarkerReplay({
    markerSpoolDirectory,
    markerQuarantineDirectory: join(directory, 'recall', 'markers', 'quarantine'),
    retainedMarkerDirectory,
    targetGenerationId: 'generation-1',
    trustedSessionRoots: [sessionsDirectory],
  });

  assert.equal((await readdir(markerSpoolDirectory)).length, 2);
  assert.equal(
    await acknowledgeCoveredRecallMarkers(workPlan, {
      generationId: 'generation-other',
      coveredMarkerIds: markers.map(({ markerId }) => markerId),
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 99 }],
    }),
    0,
  );
  assert.equal((await readdir(markerSpoolDirectory)).length, 2);

  assert.equal(
    await acknowledgeCoveredRecallMarkers(workPlan, {
      generationId: 'generation-1',
      coveredMarkerIds: [markers[0]?.markerId ?? 'missing'],
      runtimeSequences: [{ runtimeInstanceId: 'runtime-1', sequence: 99 }],
    }),
    1,
  );
  assert.deepEqual(await readdir(markerSpoolDirectory), [`${markers[1]?.markerId}.json`]);
  const firstMarker = markers[0];
  assert.ok(firstMarker);
  assert.equal(
    await readFile(join(retainedMarkerDirectory, `${firstMarker.markerId}.json`), 'utf8'),
    await encodeRecallWorkMarker(firstMarker, { trustedSessionRoots: [sessionsDirectory] }),
  );

  const outsidePath = join(directory, 'outside.json');
  await writeFile(outsidePath, 'must remain');
  await assert.rejects(
    () =>
      acknowledgeCoveredRecallMarkers(
        { ...workPlan, sourceMarkerIds: ['../../outside'] },
        {
          generationId: 'generation-1',
          coveredMarkerIds: ['../../outside'],
          runtimeSequences: [],
        },
      ),
    /marker ID invalid/iu,
  );
  assert.equal(await readFile(outsidePath, 'utf8'), 'must remain');
});

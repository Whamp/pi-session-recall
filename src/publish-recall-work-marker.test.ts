import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, open, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallWorkMarkerTrigger } from './enums.js';
import {
  publishRecallWorkMarker,
  type RecallMarkerPublicationFile,
  type RecallMarkerPublicationFilesystem,
} from './publish-recall-work-marker.js';
import {
  createRecallWorkMarkerId,
  decodeRecallWorkMarker,
  RECALL_WORK_MARKER_VERSION,
  type RecallWorkMarker,
} from './recall-work-marker.js';

async function createPublicationFixture(t: test.TestContext): Promise<{
  marker: RecallWorkMarker;
  markerSpoolDirectory: string;
  sessionsDirectory: string;
}> {
  const directory = await mkdtemp(join(tmpdir(), 'publish-recall-work-marker-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const sessionsDirectory = join(directory, 'sessions');
  const markerSpoolDirectory = join(directory, 'recall', 'markers', 'pending');
  const physicalSessionPath = join(sessionsDirectory, 'session.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(physicalSessionPath, '{}\n');
  const identity = {
    version: RECALL_WORK_MARKER_VERSION,
    physicalSessionId: 'physical-session-1',
    physicalSessionPath,
    runtimeInstanceId: 'runtime-instance-1',
    runtimeSequence: 1,
    createdAtEpochMilliseconds: 1_753_315_200_000,
    trigger: { kind: RecallWorkMarkerTrigger.ACTIVITY },
  } as const;
  return {
    sessionsDirectory,
    markerSpoolDirectory,
    marker: { ...identity, markerId: createRecallWorkMarkerId(identity) },
  };
}

function createTestFilesystem(
  wrapFile: (file: RecallMarkerPublicationFile) => RecallMarkerPublicationFile = (file) => file,
  renameFile: RecallMarkerPublicationFilesystem['renameFile'] = rename,
  wrapDirectory: (file: RecallMarkerPublicationFile) => RecallMarkerPublicationFile = (file) =>
    file,
): RecallMarkerPublicationFilesystem {
  return {
    async createDirectory(path) {
      await mkdir(path, { recursive: true });
    },
    async openExclusiveFile(path) {
      return wrapFile(await open(path, 'wx', 0o600));
    },
    renameFile,
    async openDirectory(path) {
      return wrapDirectory(await open(path, 'r'));
    },
    async removeFile(path) {
      await rm(path, { force: true });
    },
  };
}

function replaceFileSync(
  file: RecallMarkerPublicationFile,
  sync: RecallMarkerPublicationFile['sync'],
): RecallMarkerPublicationFile {
  return {
    writeFile: file.writeFile.bind(file),
    sync,
    close: file.close.bind(file),
  };
}

void test('recall work marker publication exposes only a complete strict marker before signaling', async (t) => {
  const fixture = await createPublicationFixture(t);
  const markerPath = join(fixture.markerSpoolDirectory, `${fixture.marker.markerId}.json`);
  let signalCount = 0;

  await publishRecallWorkMarker(fixture.marker, {
    markerSpoolDirectory: fixture.markerSpoolDirectory,
    trustedSessionRoots: [fixture.sessionsDirectory],
    workerSignal: {
      signalDetachedWorker() {
        signalCount += 1;
        assert.equal(existsSync(markerPath), true);
      },
    },
  });

  assert.equal(signalCount, 1);
  const spoolNames = await readdir(fixture.markerSpoolDirectory);
  assert.deepEqual(spoolNames, [`${fixture.marker.markerId}.json`]);
  const markerSource = await import('node:fs/promises').then((filesystem) =>
    filesystem.readFile(markerPath, 'utf8'),
  );
  assert.deepEqual(
    await decodeRecallWorkMarker(markerSource, {
      trustedSessionRoots: [fixture.sessionsDirectory],
    }),
    fixture.marker,
  );
});

void test('recall work marker publication keeps a durable marker when worker signaling fails', async (t) => {
  const fixture = await createPublicationFixture(t);

  await assert.rejects(
    () =>
      publishRecallWorkMarker(fixture.marker, {
        markerSpoolDirectory: fixture.markerSpoolDirectory,
        trustedSessionRoots: [fixture.sessionsDirectory],
        workerSignal: {
          signalDetachedWorker() {
            throw new Error('injected worker signal failure');
          },
        },
      }),
    /Recall marker publication worker signal failed/iu,
  );

  assert.deepEqual(await readdir(fixture.markerSpoolDirectory), [
    `${fixture.marker.markerId}.json`,
  ]);
});

for (const fault of ['file_sync', 'rename', 'directory_sync'] as const) {
  void test(`recall work marker publication rejects ${fault} without exposing a partial marker`, async (t) => {
    const fixture = await createPublicationFixture(t);
    const filesystem =
      fault === 'file_sync'
        ? createTestFilesystem((file) =>
            replaceFileSync(file, async () => {
              throw new Error('injected file sync failure');
            }),
          )
        : fault === 'rename'
          ? createTestFilesystem(
              (file) => file,
              async () => {
                throw new Error('injected rename failure');
              },
            )
          : createTestFilesystem(
              (file) => file,
              rename,
              (directory) =>
                replaceFileSync(directory, async () => {
                  throw new Error('injected directory sync failure');
                }),
            );
    let signalCount = 0;

    await assert.rejects(
      () =>
        publishRecallWorkMarker(fixture.marker, {
          markerSpoolDirectory: fixture.markerSpoolDirectory,
          trustedSessionRoots: [fixture.sessionsDirectory],
          filesystem,
          workerSignal: {
            signalDetachedWorker() {
              signalCount += 1;
            },
          },
        }),
      new RegExp(`Recall marker publication ${fault.replace('_', ' ')}`, 'u'),
    );

    assert.equal(signalCount, 0);
    const spoolNames = await readdir(fixture.markerSpoolDirectory);
    const discoverableNames = spoolNames.filter((name) => name.endsWith('.json'));
    if (fault === 'directory_sync') {
      assert.deepEqual(discoverableNames, [`${fixture.marker.markerId}.json`]);
      const markerPath = join(fixture.markerSpoolDirectory, discoverableNames[0] ?? 'missing');
      const source = await import('node:fs/promises').then((module) =>
        module.readFile(markerPath, 'utf8'),
      );
      await decodeRecallWorkMarker(source, {
        trustedSessionRoots: [fixture.sessionsDirectory],
      });
    } else {
      assert.deepEqual(discoverableNames, []);
    }
  });
}

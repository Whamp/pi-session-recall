import assert from 'node:assert/strict';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { setTimeout as sleep } from 'node:timers/promises';

import { RecallBackgroundIndexProcessState } from './enums.js';
import {
  readRecallBackgroundIndexStatusRecord,
  recallBackgroundIndexWorkerRequestPath,
  removeRecallBackgroundIndexWorkerRequest,
  updateRecallBackgroundIndexStatusRecord,
} from './recall-background-index-build.js';
import { tryAcquireRecallRebuildOwnershipLock } from './recall-rebuild-ownership-lock.js';

void test('missing background index status means no build has started', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-background-status-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  assert.equal(await readRecallBackgroundIndexStatusRecord(join(root, 'missing.json')), null);
});

void test('background index status mutation waits for flock ownership', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-background-flock-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statusPath = join(root, 'background-index-status.json');
  const buildId = '11111111-1111-4111-8111-111111111111';
  const status = {
    version: 1,
    buildId,
    generationId: null,
    embeddingProfileId: 'embedding-profile-v1',
    processId: process.pid,
    processState: RecallBackgroundIndexProcessState.RUNNING,
    startedAt: '2026-08-02T10:00:00.000Z',
    updatedAt: '2026-08-02T10:00:00.000Z',
    completedAt: null,
    progress: null,
    latestCheckpoint: null,
    latestActionableError: null,
  } as const;
  await writeFile(statusPath, `${JSON.stringify(status)}\n`, 'utf8');
  const owner = await tryAcquireRecallRebuildOwnershipLock(`${statusPath}.control-lock`);
  assert.ok(owner);
  t.after(() => owner.release());

  const update = updateRecallBackgroundIndexStatusRecord(
    statusPath,
    buildId,
    process.pid,
    (current) => ({ ...current, updatedAt: '2026-08-02T10:01:00.000Z' }),
  );
  const settledBeforeRelease = await Promise.race([
    update.then(
      () => true,
      () => true,
    ),
    sleep(50).then(() => false),
  ]);
  assert.equal(settledBeforeRelease, false);

  await owner.release();
  await update;
  assert.equal(
    (await readRecallBackgroundIndexStatusRecord(statusPath))?.updatedAt,
    '2026-08-02T10:01:00.000Z',
  );
});

void test('background worker request paths are scoped per build id', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-background-request-path-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const baseRequestPath = join(root, 'background-index-request.json');
  const firstBuildId = '11111111-1111-4111-8111-111111111111';
  const secondBuildId = '22222222-2222-4222-8222-222222222222';
  const firstRequestPath = recallBackgroundIndexWorkerRequestPath(baseRequestPath, firstBuildId);
  const secondRequestPath = recallBackgroundIndexWorkerRequestPath(baseRequestPath, secondBuildId);

  assert.notEqual(firstRequestPath, secondRequestPath);
  assert.equal(firstRequestPath, `${baseRequestPath}.${firstBuildId}`);
  assert.equal(secondRequestPath, `${baseRequestPath}.${secondBuildId}`);

  await writeFile(firstRequestPath, `${JSON.stringify({ buildId: firstBuildId })}\n`, 'utf8');
  await writeFile(secondRequestPath, `${JSON.stringify({ buildId: secondBuildId })}\n`, 'utf8');
  await removeRecallBackgroundIndexWorkerRequest(firstRequestPath);

  await assert.rejects(() => access(firstRequestPath), { code: 'ENOENT' });
  const remainingRequest: unknown = JSON.parse(await readFile(secondRequestPath, 'utf8'));
  assert.ok(remainingRequest !== null && typeof remainingRequest === 'object');
  assert.equal(Reflect.get(remainingRequest, 'buildId'), secondBuildId);
});

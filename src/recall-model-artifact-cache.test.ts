import assert from 'node:assert/strict';
import { access, copyFile, mkdir, mkdtemp, readFile, readdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  createRecallModelArtifactCache,
  type RecallModelArtifactTransport,
} from './recall-model-artifact-cache.js';
import {
  createRecallModelArtifactFixtureGguf as createFixtureGguf,
  createRecallModelArtifactFixtureProfile as createFixtureProfile,
} from './recall-model-artifact.test-utils.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

void test('model artifact cache reports missing and refuses download without explicit approval', async (t) => {
  const cacheDirectory = await mkdtemp(join(tmpdir(), 'recall-model-cache-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(cacheDirectory, { recursive: true, force: true });
  });
  let downloadCount = 0;
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact() {
      downloadCount += 1;
    },
  };
  const cache = createRecallModelArtifactCache({
    cacheDirectory,
    profile: createRecommendedEmbeddingGemmaModelProfile(),
    transport,
  });

  const status = await cache.verifyArtifact();

  assert.equal(status.state, 'missing');
  assert.match(status.repair, /approve the pinned model download/u);
  await assert.rejects(
    () => cache.downloadArtifact({ approved: false }),
    /Recall model download approval required/u,
  );
  assert.equal(downloadCount, 0);
  assert.deepEqual(await readdir(cacheDirectory), []);
});

void test('approved model download validates and atomically activates the pinned GGUF once', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-download-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  });
  const fixtureArtifact = createFixtureGguf();
  const sourcePath = join(root, 'fixture.gguf');
  await writeFile(sourcePath, fixtureArtifact);
  const requests: Array<{ sourceUrl: string; destinationPath: string }> = [];
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact(sourceUrl, destinationPath) {
      requests.push({ sourceUrl, destinationPath });
      await copyFile(sourcePath, destinationPath);
    },
  };
  const cache = createRecallModelArtifactCache({
    cacheDirectory: join(root, 'cache'),
    profile: createFixtureProfile(fixtureArtifact),
    transport,
  });

  const downloaded = await cache.downloadArtifact({ approved: true });
  const reused = await cache.downloadArtifact({ approved: true });

  assert.equal(downloaded.state, 'valid');
  assert.deepEqual(reused, downloaded);
  assert.equal(requests.length, 1);
  assert.equal(requests[0]?.sourceUrl, 'https://models.invalid/pinned/fixture.gguf');
  assert.match(requests[0]?.destinationPath ?? '', /\.partial-[0-9a-f-]+$/u);
  assert.deepEqual(await readFile(downloaded.artifactPath), fixtureArtifact);
  assert.equal(downloaded.partialPaths.length, 0);
});

void test('model status and doctor distinguish partial corrupt incompatible valid and removed artifacts', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-doctor-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  });
  const fixtureArtifact = createFixtureGguf();
  const profile = createFixtureProfile(fixtureArtifact);
  const transport: RecallModelArtifactTransport = {
    async downloadArtifact() {
      throw new Error('transport must not run when repairing compatible pinned bytes');
    },
  };
  const cache = createRecallModelArtifactCache({
    cacheDirectory: join(root, 'cache'),
    profile,
    transport,
  });
  const missing = await cache.verifyArtifact();
  await mkdir(join(missing.artifactPath, '..'), { recursive: true });
  await writeFile(`${missing.artifactPath}.partial-interrupted`, 'unfinished');

  const partial = await cache.verifyArtifact();
  assert.equal(partial.state, 'partial');
  assert.match(partial.repair, /fresh pinned download/u);

  await writeFile(missing.artifactPath, 'not a model');
  const corrupt = await cache.verifyArtifact();
  assert.equal(corrupt.state, 'corrupt');
  assert.match(corrupt.issue ?? '', /size mismatch/u);
  assert.match(corrupt.repair, /replace the corrupt artifact/u);

  await writeFile(missing.artifactPath, fixtureArtifact);
  const incompatible = await cache.verifyArtifact();
  assert.equal(incompatible.state, 'incompatible');
  assert.match(incompatible.issue ?? '', /no activation receipt/u);

  const repaired = await cache.repairArtifact({ approved: true });
  assert.equal(repaired.state, 'valid');
  const inspection = await cache.inspectArtifact();
  assert.equal(inspection.profile, profile);
  assert.equal(inspection.status.state, 'valid');
  assert.deepEqual(await cache.diagnoseArtifact(), {
    healthy: true,
    status: repaired,
    action: 'No repair required.',
  });

  await assert.rejects(
    () => cache.removeArtifact({ approved: false }),
    /Recall model removal approval required/u,
  );
  assert.equal((await cache.verifyArtifact()).state, 'valid');
  const removed = await cache.removeArtifact({ approved: true });
  assert.equal(removed.state, 'missing');
});

void test('download rejects checksum mismatch and checksum-valid non-GGUF bytes', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-invalid-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  });
  const validArtifact = createFixtureGguf();
  const checksumMismatch = Buffer.from(validArtifact);
  checksumMismatch[checksumMismatch.length - 1] = 1;
  const invalidGguf = Buffer.from(validArtifact);
  invalidGguf.write('NOPE', 0, 'ascii');
  const cases = [
    {
      name: 'checksum mismatch',
      artifact: checksumMismatch,
      profile: createFixtureProfile(validArtifact),
      expectedError: /GGUF artifact SHA-256 mismatch/u,
    },
    {
      name: 'invalid GGUF',
      artifact: invalidGguf,
      profile: createFixtureProfile(invalidGguf),
      expectedError: /GGUF magic invalid/u,
    },
  ];

  for (const invalidCase of cases) {
    await t.test(invalidCase.name, async () => {
      const cache = createRecallModelArtifactCache({
        cacheDirectory: join(root, invalidCase.name),
        profile: invalidCase.profile,
        transport: {
          async downloadArtifact(sourceUrl, destinationPath) {
            void sourceUrl;
            await writeFile(destinationPath, invalidCase.artifact);
          },
        },
      });
      await assert.rejects(
        () => cache.downloadArtifact({ approved: true }),
        invalidCase.expectedError,
      );
      assert.equal((await cache.verifyArtifact()).state, 'partial');
    });
  }
});

void test('rejected model download remains partial and never becomes available', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-rejected-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  });
  const fixtureArtifact = createFixtureGguf();
  const cache = createRecallModelArtifactCache({
    cacheDirectory: join(root, 'cache'),
    profile: createFixtureProfile(fixtureArtifact),
    transport: {
      async downloadArtifact(sourceUrl, destinationPath) {
        void sourceUrl;
        await writeFile(destinationPath, Buffer.from('GGUF-corrupt-download'));
      },
    },
  });

  await assert.rejects(
    () => cache.downloadArtifact({ approved: true }),
    /Recall downloaded model artifact rejected: GGUF artifact size mismatch/u,
  );
  const status = await cache.verifyArtifact();
  assert.equal(status.state, 'partial');
  assert.equal(status.partialPaths.length, 1);
  await assert.rejects(() => access(status.artifactPath), { code: 'ENOENT' });
});

void test('model artifact cache rejects path traversal segments before filesystem access', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-model-containment-'));
  t.after(async () => {
    const { rm } = await import('node:fs/promises');
    await rm(root, { recursive: true, force: true });
  });
  const cacheDirectory = join(root, 'cache');
  const fixtureArtifact = createFixtureGguf();
  const profile = createFixtureProfile(fixtureArtifact);
  const invalidProfiles = [
    { ...profile, profileId: '../victim' },
    { ...profile, profileId: '/absolute' },
    { ...profile, profileId: '.' },
    { ...profile, source: { ...profile.source, revision: '..' } },
    { ...profile, source: { ...profile.source, revision: '/absolute' } },
    { ...profile, source: { ...profile.source, artifact: '..' } },
    { ...profile, source: { ...profile.source, artifact: 'nested/model.gguf' } },
    { ...profile, source: { ...profile.source, artifact: String.raw`nested\model.gguf` } },
  ];

  for (const invalidProfile of invalidProfiles) {
    assert.throws(
      () => createRecallModelArtifactCache({ cacheDirectory, profile: invalidProfile }),
      /Recall model artifact cache path invalid/u,
    );
  }
  await assert.rejects(() => access(cacheDirectory), { code: 'ENOENT' });
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { LocalOctenModelStatusKind } from './enums.js';
import {
  createLocalOctenModelManager,
  type LocalOctenArtifactIdentity,
} from './local-octen-model-manager.js';

function sha256(content: Uint8Array): string {
  return createHash('sha256').update(content).digest('hex');
}

const MODEL = new TextEncoder().encode('verified-model');
const TOKENIZER = new TextEncoder().encode('verified-tokenizer');
const TEST_ARTIFACT: LocalOctenArtifactIdentity = {
  artifactId: 'test-local-octen-v1',
  releaseUrl: 'https://example.invalid/releases/test-local-octen-v1',
  nativeDimensions: 4,
  files: [
    {
      fileName: 'model.onnx',
      url: 'https://example.invalid/model.onnx',
      bytes: MODEL.byteLength,
      sha256: sha256(MODEL),
    },
    {
      fileName: 'tokenizer.json',
      url: 'https://example.invalid/tokenizer.json',
      bytes: TOKENIZER.byteLength,
      sha256: sha256(TOKENIZER),
    },
  ],
};

function createDownloadSource(
  files: ReadonlyMap<string, Uint8Array>,
): (url: string, signal?: AbortSignal) => AsyncIterable<Uint8Array> {
  return async function* downloadSource(url, signal) {
    signal?.throwIfAborted();
    const content = files.get(url);
    if (!content) {
      throw new Error(`Unexpected test download URL: ${url}`);
    }
    const midpoint = Math.ceil(content.byteLength / 2);
    yield content.slice(0, midpoint);
    signal?.throwIfAborted();
    yield content.slice(midpoint);
  };
}

void test('local model download publishes only a complete checksum-verified artifact', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'local-octen-model-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const manager = createLocalOctenModelManager({
    modelRootDirectory: root,
    artifact: TEST_ARTIFACT,
    downloadSource: createDownloadSource(
      new Map([
        [TEST_ARTIFACT.files[0]!.url, MODEL],
        [TEST_ARTIFACT.files[1]!.url, TOKENIZER],
      ]),
    ),
  });

  assert.deepEqual(await manager.status(), {
    kind: LocalOctenModelStatusKind.MISSING,
    artifactId: TEST_ARTIFACT.artifactId,
    modelDirectory: join(root, TEST_ARTIFACT.artifactId),
    totalBytes: MODEL.byteLength + TOKENIZER.byteLength,
    partialDownloads: 0,
    detail: 'Local Octen model is not downloaded',
  });

  const progress: string[] = [];
  const result = await manager.download({
    approved: true,
    onProgress(event) {
      progress.push(event.kind);
    },
  });

  assert.equal(result.downloaded, true);
  assert.equal(result.modelDirectory, join(root, TEST_ARTIFACT.artifactId));
  assert.deepEqual(await readFile(join(result.modelDirectory, 'model.onnx')), Buffer.from(MODEL));
  assert.deepEqual(
    await readFile(join(result.modelDirectory, 'tokenizer.json')),
    Buffer.from(TOKENIZER),
  );
  assert.deepEqual(progress, [
    'preparing',
    'downloading-file',
    'file-verified',
    'downloading-file',
    'file-verified',
    'activated',
  ]);
  assert.equal((await manager.status()).kind, LocalOctenModelStatusKind.READY);
  assert.deepEqual(await readdir(root), [TEST_ARTIFACT.artifactId]);

  const second = await manager.download({ approved: true });
  assert.equal(second.downloaded, false);
});

void test('local model download requires approval before reading the network', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'local-octen-approval-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  let requested = false;
  const manager = createLocalOctenModelManager({
    modelRootDirectory: root,
    artifact: TEST_ARTIFACT,
    downloadSource: async function* () {
      requested = true;
      yield MODEL;
    },
  });

  await assert.rejects(
    manager.download({ approved: false }),
    /requires explicit approval.*\d+ bytes/u,
  );
  assert.equal(requested, false);
  assert.deepEqual(await readdir(root), []);
});

void test('checksum failure removes the partial artifact and preserves a valid model', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'local-octen-corrupt-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const goodFiles = new Map([
    [TEST_ARTIFACT.files[0]!.url, MODEL],
    [TEST_ARTIFACT.files[1]!.url, TOKENIZER],
  ]);
  const goodManager = createLocalOctenModelManager({
    modelRootDirectory: root,
    artifact: TEST_ARTIFACT,
    downloadSource: createDownloadSource(goodFiles),
  });
  await goodManager.download({ approved: true });
  const originalReceipt = await readFile(
    join(root, TEST_ARTIFACT.artifactId, 'model-receipt.json'),
    'utf8',
  );

  await writeFile(join(root, TEST_ARTIFACT.artifactId, 'model.onnx'), 'corrupt', 'utf8');
  const corruptFiles = new Map(goodFiles);
  corruptFiles.set(TEST_ARTIFACT.files[1]!.url, new TextEncoder().encode('tampered-tokenizer'));
  const repairManager = createLocalOctenModelManager({
    modelRootDirectory: root,
    artifact: TEST_ARTIFACT,
    downloadSource: createDownloadSource(corruptFiles),
  });

  await assert.rejects(
    repairManager.download({ approved: true }),
    /tokenizer\.json checksum mismatch/u,
  );
  assert.equal(
    await readFile(join(root, TEST_ARTIFACT.artifactId, 'model-receipt.json'), 'utf8'),
    originalReceipt,
  );
  assert.equal(
    await readFile(join(root, TEST_ARTIFACT.artifactId, 'model.onnx'), 'utf8'),
    'corrupt',
  );
  assert.deepEqual(await readdir(root), [TEST_ARTIFACT.artifactId]);
});

void test('model diagnosis verifies every checksum before the runtime smoke probe', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'local-octen-doctor-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  let probedDirectory: string | undefined;
  const manager = createLocalOctenModelManager({
    modelRootDirectory: root,
    artifact: TEST_ARTIFACT,
    downloadSource: createDownloadSource(
      new Map([
        [TEST_ARTIFACT.files[0]!.url, MODEL],
        [TEST_ARTIFACT.files[1]!.url, TOKENIZER],
      ]),
    ),
    async probeRuntime(modelDirectory) {
      probedDirectory = modelDirectory;
      return { dimensions: 4, norm: 1 };
    },
  });
  await manager.download({ approved: true });

  const healthy = await manager.doctor();
  assert.equal(healthy.healthy, true);
  assert.equal(healthy.runtime?.dimensions, 4);
  assert.equal(probedDirectory, join(root, TEST_ARTIFACT.artifactId));

  await writeFile(join(root, TEST_ARTIFACT.artifactId, 'model.onnx'), 'tampered-model', 'utf8');
  probedDirectory = undefined;
  const corrupt = await manager.doctor();
  assert.equal(corrupt.healthy, false);
  assert.match(corrupt.detail, /model\.onnx checksum mismatch/u);
  assert.equal(probedDirectory, undefined);
});

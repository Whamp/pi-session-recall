import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  assertRecallIndexManifestCompatible,
  createRecallIndexManifest,
  readRecallIndexManifest,
  writeRecallIndexManifest,
  type RecallEmbeddingModelIdentity,
} from './recall-index-manifest.js';
import { normalizeRecallProjectLineages } from './resolve-project-identity.js';

const OCTEN_IDENTITY: RecallEmbeddingModelIdentity = {
  requestModel: 'octen-embed',
  servedModelId: 'Octen/Octen-Embedding-4B',
  nativeDimensions: 2_560,
  storedDimensions: 1_024,
  transformation: 'vendor-prefix-then-l2-v1',
};

function createManifest(overrides: Partial<RecallEmbeddingModelIdentity> = {}) {
  return createRecallIndexManifest({
    embeddingIdentity: { ...OCTEN_IDENTITY, ...overrides },
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  });
}

void test('index manifest binds Octen native and stored widths, tokenizer, chunking, and inner product', () => {
  const manifest = createManifest();

  assert.deepEqual(manifest.embedding, OCTEN_IDENTITY);
  assert.deepEqual(manifest.chunkPolicy, {
    version: 3,
    maxTokens: 512,
    overlapTokens: 64,
    boundaryAlgorithm: 'markdown-structure-v1',
  });
  assert.equal(manifest.tokenizer.model, 'Octen/Octen-Embedding-4B');
  assert.equal(manifest.zvec.metric, 'inner-product');
  assert.equal('embeddingCacheVersion' in manifest, false);
  assert.equal('canaryVector' in manifest.embedding, false);
});

void test('index manifest round-trips atomically', async () => {
  const root = await mkdtemp(join(tmpdir(), 'recall-manifest-'));
  const path = join(root, 'index-manifest.json');
  try {
    const expected = createManifest();
    await writeRecallIndexManifest(path, expected);

    assert.deepEqual(await readRecallIndexManifest(path), expected);
    assert.equal((await readFile(path, 'utf8')).endsWith('\n'), true);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

void test('manifest compatibility requires rebuild when model or stored width changes', () => {
  const actual = createManifest();

  assert.throws(
    () =>
      assertRecallIndexManifestCompatible(
        actual,
        createManifest({ storedDimensions: 768 }),
        '/recall/index-manifest.json',
      ),
    /embedding\.storedDimensions: expected 768, received 1024[\s\S]*psr index --rebuild/,
  );
  assert.throws(
    () =>
      assertRecallIndexManifestCompatible(
        actual,
        createManifest({ servedModelId: 'other/model' }),
        '/recall/index-manifest.json',
      ),
    /embedding\.servedModelId: expected "other\/model"/,
  );
});

void test('manifest compatibility binds chunk and project-lineage policy', () => {
  const actual = createRecallIndexManifest({
    embeddingIdentity: OCTEN_IDENTITY,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
    projectLineages: normalizeRecallProjectLineages({}),
  });
  const changedChunking = createRecallIndexManifest({
    embeddingIdentity: OCTEN_IDENTITY,
    chunkPolicy: { maxTokens: 256, overlapTokens: 32 },
    projectLineages: normalizeRecallProjectLineages({}),
  });
  const changedLineage = createRecallIndexManifest({
    embeddingIdentity: OCTEN_IDENTITY,
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
    projectLineages: normalizeRecallProjectLineages({
      'git-origin:github.com/Whamp/pi-session-recall': ['/historical/recall'],
    }),
  });

  assert.throws(
    () => assertRecallIndexManifestCompatible(actual, changedChunking, '/manifest.json'),
    /chunkPolicy\.maxTokens/,
  );
  assert.throws(
    () => assertRecallIndexManifestCompatible(actual, changedLineage, '/manifest.json'),
    /projectIdentity\.lineageDigest/,
  );
});

void test('manifest creation rejects stored widths larger than the native Octen vector', () => {
  assert.throws(
    () => createManifest({ nativeDimensions: 512, storedDimensions: 1_024 }),
    /stored dimensions 1024 exceed native dimensions 512/,
  );
});

void test('manifest reader rejects old generation formats without compatibility adoption', async () => {
  const root = await mkdtemp(join(tmpdir(), 'recall-old-manifest-'));
  const path = join(root, 'index-manifest.json');
  try {
    await writeFile(path, '{"manifestVersion":5}\n', 'utf8');

    await assert.rejects(
      readRecallIndexManifest(path),
      /Recall index manifest invalid[\s\S]*psr index --rebuild/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

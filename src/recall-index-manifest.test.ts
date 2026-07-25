import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  assertRecallIndexManifestCompatible,
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  readRecallIndexManifest,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';

const embeddingIdentity = {
  requestModel: 'octen-embed',
  servedModelId: 'octen-embed',
  artifact: 'Octen-Embedding-4B.Q8_0.gguf',
  dimensions: 3,
  quantization: 'Q8_0',
  pooling: 'last',
};

void test('index manifest round-trips the complete reproducibility identity atomically', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const manifestPath = join(directory, 'index-manifest.json');
  const canaryFingerprint = createRecallEmbeddingCanaryFingerprint([0.25, -0.5, 1], 3);
  const manifest = createRecallIndexManifest({ embeddingIdentity, canaryFingerprint });

  await writeRecallIndexManifest(manifestPath, manifest);

  assert.deepEqual(await readRecallIndexManifest(manifestPath), manifest);
  assert.deepEqual(await readdir(directory), ['index-manifest.json']);
  assert.equal(manifest.embedding.canaryFingerprint.length, 64);
  assert.equal(manifest.tokenizer.assets[0]?.fileName, 'tokenizer.json');
  assert.equal(manifest.tokenizer.assets[1]?.fileName, 'tokenizer_config.json');
  assert.deepEqual(manifest.chunkPolicy, {
    version: 2,
    maxTokens: 1_024,
    overlapTokens: 128,
    boundaryAlgorithm: 'markdown-structure-v1',
    normalization: 'unicode-nfc-v1',
  });
});

void test('index manifest incompatibility reports every mismatch with the rebuild command', () => {
  const expected = createRecallIndexManifest({
    embeddingIdentity,
    canaryFingerprint: createRecallEmbeddingCanaryFingerprint([0.25, -0.5, 1], 3),
  });
  const actual = structuredClone(expected);
  actual.embedding.dimensions = 2;
  actual.embedding.pooling = 'mean';
  actual.tokenizer.revision = 'mutable-main';
  actual.chunkPolicy.maxTokens = 512;
  actual.conversationSchemaVersion = 1;
  actual.zvec.ftsConfigurationVersion = 99;

  assert.throws(
    () => assertRecallIndexManifestCompatible(actual, expected, '/data/index-manifest.json'),
    (error) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /embedding\.dimensions/);
      assert.match(error.message, /embedding\.pooling/);
      assert.match(error.message, /tokenizer\.revision/);
      assert.match(error.message, /chunkPolicy\.maxTokens/);
      assert.match(error.message, /conversationSchemaVersion/);
      assert.match(error.message, /zvec\.ftsConfigurationVersion/);
      assert.match(error.message, /\/pi-session-recall-index --rebuild/);
      return true;
    },
  );
  assert.throws(
    () => assertRecallIndexManifestCompatible(null, expected, '/data/index-manifest.json'),
    /Recall index manifest missing.*\/pi-session-recall-index --rebuild/,
  );
});

void test('index manifest reader rejects malformed or unversioned data actionably', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invalid-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const malformedPath = join(directory, 'malformed.json');
  const unversionedPath = join(directory, 'unversioned.json');
  await writeFile(malformedPath, '{');
  await writeFile(unversionedPath, '{}');

  await assert.rejects(
    () => readRecallIndexManifest(malformedPath),
    /Recall index manifest invalid.*\/pi-session-recall-index --rebuild/,
  );
  await assert.rejects(
    () => readRecallIndexManifest(unversionedPath),
    /Recall index manifest invalid.*\/pi-session-recall-index --rebuild/,
  );
  assert.equal(await readRecallIndexManifest(join(directory, 'missing.json')), null);
});

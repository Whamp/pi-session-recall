import assert from 'node:assert/strict';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createEmbeddingVectorCacheIdentity } from './embedding-vector-cache.js';
import {
  assertRecallIndexManifestCompatible,
  createRecallEmbeddingCanaryFingerprint,
  createRecallIndexManifest,
  readRecallIndexManifest,
  recoverRecallEmbeddingCanaryFromManifest,
  writeRecallIndexManifest,
} from './recall-index-manifest.js';
import {
  normalizeRecallProjectLineages,
  PROJECT_IDENTITY_METADATA_SCHEMA_VERSION,
} from './resolve-project-identity.js';

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
  const canaryEmbedding = [0.25, -0.5, 1];
  const manifest = createRecallIndexManifest({ embeddingIdentity, canaryEmbedding });

  await writeRecallIndexManifest(manifestPath, manifest);

  assert.deepEqual(await readRecallIndexManifest(manifestPath), manifest);
  assert.deepEqual(await readdir(directory), ['index-manifest.json']);
  assert.equal(manifest.manifestVersion, 5);
  assert.deepEqual(manifest.importPolicy, { version: 1 });
  assert.equal(Object.hasOwn(createEmbeddingVectorCacheIdentity(manifest), 'importPolicy'), false);
  assert.equal(
    manifest.embedding.canaryFingerprint,
    createRecallEmbeddingCanaryFingerprint(canaryEmbedding, 3),
  );
  assert.deepEqual(manifest.embedding.canaryVector, canaryEmbedding);
  assert.equal(manifest.embedding.canaryMinimumCosineSimilarity, 0.9995);
  assert.equal(manifest.tokenizer.assets[0]?.fileName, 'tokenizer.json');
  assert.equal(manifest.tokenizer.assets[1]?.fileName, 'tokenizer_config.json');
  assert.deepEqual(manifest.chunkPolicy, {
    version: 2,
    maxTokens: 1_024,
    overlapTokens: 128,
    boundaryAlgorithm: 'markdown-structure-v1',
    normalization: 'unicode-nfc-v1',
  });
  assert.equal(manifest.conversationSchemaVersion, 8);
  assert.equal(manifest.provenanceSchemaVersion, 8);
  assert.equal(PROJECT_IDENTITY_METADATA_SCHEMA_VERSION, 3);
  assert.deepEqual(manifest.projectIdentity, {
    policyVersion: 4,
    metadataSchemaVersion: 3,
    lineagePolicyVersion: 1,
    lineageDigest: '44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a',
  });
  assert.equal(manifest.zvec.schemaVersion, 7);
  assert.equal(manifest.zvec.ftsConfigurationVersion, 2);
});

void test('index manifest canonically digests project lineage and rejects changed lineage policy', () => {
  const target = 'git-origin:github.com/Whamp/successor';
  const actual = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
    projectLineages: normalizeRecallProjectLineages({
      [target]: ['/historical/zeta', '/historical/alpha'],
    }),
  });
  const equivalent = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
    projectLineages: normalizeRecallProjectLineages({
      [target]: ['/historical/alpha', '/historical/zeta'],
    }),
  });
  const changed = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
    projectLineages: normalizeRecallProjectLineages({
      [target]: ['/historical/alpha', '/historical/replacement'],
    }),
  });

  assert.equal(actual.projectIdentity.lineagePolicyVersion, 1);
  assert.match(actual.projectIdentity.lineageDigest, /^[a-f0-9]{64}$/u);
  assert.equal(equivalent.projectIdentity.lineageDigest, actual.projectIdentity.lineageDigest);
  assert.notEqual(changed.projectIdentity.lineageDigest, actual.projectIdentity.lineageDigest);
  assert.throws(
    () => assertRecallIndexManifestCompatible(actual, changed, '/data/index-manifest.json'),
    /projectIdentity\.lineageDigest.*\/pi-session-recall-index --rebuild/s,
  );
});

void test('index manifest records an explicitly bounded chunk policy', () => {
  const manifest = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
    chunkPolicy: { maxTokens: 512, overlapTokens: 64 },
  });

  assert.equal(manifest.chunkPolicy.maxTokens, 512);
  assert.equal(manifest.chunkPolicy.overlapTokens, 64);
});

void test('index manifest rejects invalid chunk geometry before indexing', () => {
  assert.throws(
    () =>
      createRecallIndexManifest({
        embeddingIdentity,
        canaryEmbedding: [0.25, -0.5, 1],
        chunkPolicy: { maxTokens: 512, overlapTokens: 512 },
      }),
    /Recall chunk policy invalid/,
  );
});

void test('index manifest tolerates same-model canary jitter and rejects material drift', () => {
  const actual = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [1, 0, 0],
  });
  const slotJitter = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [1, 0.027, 0],
  });
  const materialDrift = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0, 1, 0],
  });

  assert.doesNotThrow(() =>
    assertRecallIndexManifestCompatible(actual, slotJitter, '/data/index-manifest.json'),
  );
  assert.throws(
    () => assertRecallIndexManifestCompatible(actual, materialDrift, '/data/index-manifest.json'),
    /embedding\.canaryCosineSimilarity.*\/pi-session-recall-index --rebuild/s,
  );
});

void test('index manifest incompatibility reports every mismatch with the rebuild command', () => {
  const expected = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
  });
  const actual = structuredClone(expected);
  actual.embedding.dimensions = 2;
  actual.embedding.pooling = 'mean';
  actual.tokenizer.revision = 'mutable-main';
  actual.chunkPolicy.maxTokens = 512;
  actual.conversationSchemaVersion = 1;
  actual.projectIdentity.policyVersion = 1;
  actual.projectIdentity.metadataSchemaVersion = 1;
  actual.projectIdentity.lineagePolicyVersion = 99;
  actual.projectIdentity.lineageDigest = '0'.repeat(64);
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
      assert.match(error.message, /projectIdentity\.policyVersion/);
      assert.match(error.message, /projectIdentity\.metadataSchemaVersion/);
      assert.match(error.message, /projectIdentity\.lineagePolicyVersion/);
      assert.match(error.message, /projectIdentity\.lineageDigest/);
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

void test('index manifest recovers only an intact canary from an incompatible generation', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-manifest-canary-recovery-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const manifestPath = join(directory, 'index-manifest.json');
  const manifest = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
  });

  await writeFile(
    manifestPath,
    JSON.stringify({ ...manifest, manifestVersion: manifest.manifestVersion - 1 }),
  );
  const recovered = await recoverRecallEmbeddingCanaryFromManifest(
    manifestPath,
    embeddingIdentity.dimensions,
  );
  assert.deepEqual(recovered?.canaryVector, manifest.embedding.canaryVector);
  assert.equal(
    recovered?.canaryMinimumCosineSimilarity,
    manifest.embedding.canaryMinimumCosineSimilarity,
  );

  await writeFile(
    manifestPath,
    JSON.stringify({
      ...manifest,
      manifestVersion: manifest.manifestVersion - 1,
      embedding: { ...manifest.embedding, canaryVector: [1, 0, 0] },
    }),
  );
  assert.equal(
    await recoverRecallEmbeddingCanaryFromManifest(manifestPath, embeddingIdentity.dimensions),
    null,
  );

  await writeFile(manifestPath, '{broken');
  assert.equal(
    await recoverRecallEmbeddingCanaryFromManifest(manifestPath, embeddingIdentity.dimensions),
    null,
  );
});

void test('index manifest reader rejects malformed or unversioned data actionably', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-invalid-manifest-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const malformedPath = join(directory, 'malformed.json');
  const unversionedPath = join(directory, 'unversioned.json');
  const preScopePath = join(directory, 'pre-scope.json');
  const currentManifest = createRecallIndexManifest({
    embeddingIdentity,
    canaryEmbedding: [0.25, -0.5, 1],
  });
  const { projectIdentity, ...preScopeManifest } = currentManifest;
  void projectIdentity;
  await writeFile(malformedPath, '{');
  await writeFile(unversionedPath, '{}');
  await writeFile(preScopePath, JSON.stringify(preScopeManifest));

  await assert.rejects(
    () => readRecallIndexManifest(malformedPath),
    /Recall index manifest invalid.*\/pi-session-recall-index --rebuild/,
  );
  await assert.rejects(
    () => readRecallIndexManifest(unversionedPath),
    /Recall index manifest invalid.*\/pi-session-recall-index --rebuild/,
  );
  await assert.rejects(
    () => readRecallIndexManifest(preScopePath),
    /Recall index manifest invalid.*\/pi-session-recall-index --rebuild/s,
  );
  assert.equal(await readRecallIndexManifest(join(directory, 'missing.json')), null);
});

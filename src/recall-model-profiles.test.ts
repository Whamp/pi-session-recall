import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createOctenEmbeddingModelProfile,
  createRecallEmbeddingProfileIdentity,
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
  resolveRecallStoredDimensionSelection,
  type RecallEmbeddingModelProfile,
} from './recall-model-profiles.js';

void test('recommended model profiles retain distinct immutable semantic identities', () => {
  assert.deepEqual(
    [
      createRecommendedEmbeddingGemmaModelProfile().profileId,
      createRecommendedQwenRerankingModelProfile().profileId,
      createRecommendedQmdQueryPlanningModelProfile().profileId,
    ],
    [
      'embeddinggemma-300m-q8-0-v1',
      'qwen3-reranker-0.6b-q8-0-v1',
      'qmd-query-expansion-1.7b-q4-k-m-v1',
    ],
  );
});

void test('stored width changes semantic profile identity without changing model artifact identity', () => {
  const native = createRecommendedEmbeddingGemmaModelProfile(768);
  const reduced = createRecommendedEmbeddingGemmaModelProfile(512);

  assert.notEqual(
    createRecallEmbeddingProfileIdentity(native),
    createRecallEmbeddingProfileIdentity(reduced),
  );
  assert.deepEqual(native.identity, reduced.identity);
  assert.deepEqual(native.source, reduced.source);
});

void test('other embedding profiles default native and accept only declared explicit widths', () => {
  const genericIdentity = {
    requestModel: 'other-model',
    servedModelId: 'other/model',
    artifact: 'other-model.fp32',
    dimensions: 6,
    quantization: 'fp32',
    pooling: 'mean',
  };
  const nativeProfile: RecallEmbeddingModelProfile = {
    identity: genericIdentity,
    queryInputPrefix: 'query: ',
    documentInputPrefix: 'document: ',
  };
  assert.deepEqual(resolveRecallStoredDimensionSelection(nativeProfile), {
    nativeDimensions: 6,
    storedDimensions: 6,
    evidenceStatus: 'native-width',
    evidenceSources: [],
  });

  const explicitProfile: RecallEmbeddingModelProfile = {
    ...nativeProfile,
    storedDimensions: 4,
    storedDimensionRange: {
      minimum: 2,
      maximum: 6,
      evidenceStatus: 'unverified-override',
    },
    storedDimensionEvidenceSources: ['https://example.test/other-model-widths'],
  };
  assert.deepEqual(resolveRecallStoredDimensionSelection(explicitProfile), {
    nativeDimensions: 6,
    storedDimensions: 4,
    evidenceStatus: 'unverified-override',
    evidenceSources: ['https://example.test/other-model-widths'],
  });
  assert.throws(
    () =>
      resolveRecallStoredDimensionSelection({
        ...explicitProfile,
        storedDimensions: 1,
      }),
    /Recall stored dimensions unsupported: expected 2 through 6, received 1/u,
  );
});

void test('embedding profiles declare their default stored width and evidence status', () => {
  const embeddingGemma = createRecommendedEmbeddingGemmaModelProfile();
  assert.deepEqual(embeddingGemma.storedDimensionChoices, [
    { dimensions: 768, evidenceStatus: 'verified-mrl' },
    { dimensions: 512, evidenceStatus: 'verified-mrl' },
    { dimensions: 256, evidenceStatus: 'verified-mrl' },
    { dimensions: 128, evidenceStatus: 'verified-mrl' },
  ]);
  assert.equal(embeddingGemma.storedDimensions, 768);

  const octen = createOctenEmbeddingModelProfile({
    requestModel: 'octen-embedding-4b',
    servedModelId: 'Octen/Octen-Embedding-4B',
    artifact: 'Octen-Embedding-4B.Q8_0.gguf',
    dimensions: 2_560,
    quantization: 'Q8_0',
    pooling: 'last',
    normalization: 'l2',
  });
  assert.equal(octen.storedDimensions, 1_024);
  assert.deepEqual(octen.storedDimensionRange, {
    minimum: 1,
    maximum: 2_560,
    evidenceStatus: 'vendor-supported-prefix',
  });
});

import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

void test('recommended EmbeddingGemma profile pins artifact and embedding semantics', () => {
  const profile = createRecommendedEmbeddingGemmaModelProfile();

  assert.deepEqual(profile, {
    profileId: 'embeddinggemma-300m-q8-0-v1',
    purpose: 'Embed recall queries and conversation documents for local semantic retrieval.',
    identity: {
      requestModel: 'embeddinggemma-300M-Q8_0',
      servedModelId: 'google/embeddinggemma-300M',
      artifact: 'embeddinggemma-300M-Q8_0.gguf',
      artifactRepository: 'ggml-org/embeddinggemma-300M-GGUF',
      artifactRevision: '0f741b5a6585bd53aeb15cd1372c56f2a0f65e12',
      artifactSha256: 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
      dimensions: 768,
      quantization: 'Q8_0',
      pooling: 'mean',
      normalization: 'l2',
    },
    source: {
      repository: 'ggml-org/embeddinggemma-300M-GGUF',
      revision: '0f741b5a6585bd53aeb15cd1372c56f2a0f65e12',
      artifact: 'embeddinggemma-300M-Q8_0.gguf',
      byteSize: 333_590_944,
      sha256: 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
      downloadUrl:
        'https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12/embeddinggemma-300M-Q8_0.gguf',
    },
    license: {
      id: 'gemma',
      name: 'Gemma Terms of Use',
      url: 'https://ai.google.dev/gemma/terms',
      distributionStatus: 'review-required',
    },
    nativeDimensions: 768,
    storedDimensions: 768,
    storedDimensionChoices: [
      { dimensions: 768, evidenceStatus: 'verified-mrl' },
      { dimensions: 512, evidenceStatus: 'verified-mrl' },
      { dimensions: 256, evidenceStatus: 'verified-mrl' },
      { dimensions: 128, evidenceStatus: 'verified-mrl' },
    ],
    storedDimensionEvidenceSources: ['https://huggingface.co/google/embeddinggemma-300m'],
    queryInputPrefix: 'task: search result | query: ',
    documentInputPrefix: 'title: none | text: ',
    normalization: 'l2',
    tokenizer: {
      kind: 'gguf-metadata',
      model: 'google/embeddinggemma-300M',
      artifactSha256: 'b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
      identity:
        'embeddinggemma-300M-Q8_0.gguf@b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63',
    },
    canary: {
      policy: 'repeat-cosine-v1',
      operation: 'query',
      query: 'Which session evidence explains the retained implementation decision?',
      expectedDimensions: 768,
      expectedNormalization: 'l2',
      minimumRepeatCosineSimilarity: 0.9995,
    },
  });
  assert.ok(Object.isFrozen(profile));
  assert.ok(Object.isFrozen(profile.source));
});

import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecommendedQwenRerankingModelProfile } from './recall-model-profiles.js';

void test('recommended Qwen reranker profile pins artifact and score semantics', () => {
  const profile = createRecommendedQwenRerankingModelProfile();

  assert.deepEqual(profile, {
    profileId: 'qwen3-reranker-0.6b-q8-0-v1',
    model: 'qwen3-reranker-0.6b-q8_0',
    purpose: 'Score recall evidence against a submitted query for deep reranking.',
    scoreMeaning: 'higher-is-more-relevant',
    scoreRange: { minimum: 0, maximum: 1 },
    scorePolicy: 'llama-cpp-qwen3-rank-probability-v1',
    source: {
      repository: 'ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF',
      revision: 'a02f48bb4f057028298c21fa033da2b30d7742d5',
      artifact: 'qwen3-reranker-0.6b-q8_0.gguf',
      byteSize: 639_153_184,
      sha256: '22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48',
      downloadUrl:
        'https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf',
    },
    license: {
      id: 'apache-2.0',
      name: 'Apache License 2.0',
      url: 'https://www.apache.org/licenses/LICENSE-2.0',
      distributionStatus: 'review-required',
    },
  });
  assert.ok(Object.isFrozen(profile));
  assert.ok(Object.isFrozen(profile.source));
  assert.ok(Object.isFrozen(profile.scoreRange));
});

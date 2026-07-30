import assert from 'node:assert/strict';
import test from 'node:test';

import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createConfiguredOctenEmbeddingModelProfile,
  createRecommendedOctenHttpCandidateId,
  createRecommendedOctenHttpInferenceCandidate,
  readRecommendedOctenHttpStoredDimensions,
} from './recommended-octen-inference-candidate.js';

void test('configured Octen candidates reconstruct every vendor-supported stored width', async () => {
  const config = await loadRecallConversationConfig({
    homeDirectory: '/home/fixture',
    environment: {
      PI_RECALL_DATA_DIRECTORY: '/recall/data',
      PI_RECALL_EMBEDDING_DIMENSIONS: '2560',
    },
  });

  for (const storedDimensions of [1, 1_024, 2_560]) {
    const candidateId = createRecommendedOctenHttpCandidateId(storedDimensions);
    const candidate = createRecommendedOctenHttpInferenceCandidate(config, storedDimensions);
    const profile = createConfiguredOctenEmbeddingModelProfile(config, storedDimensions);

    assert.equal(candidate.candidateId, candidateId);
    assert.equal(candidate.profileId, 'octen-embedding-4b');
    assert.equal(candidate.artifact, null);
    assert.equal(profile.identity.dimensions, 2_560);
    assert.equal(profile.storedDimensions, storedDimensions);
    assert.equal(
      readRecommendedOctenHttpStoredDimensions(candidateId, config.embeddingDimensions),
      storedDimensions,
    );
  }

  assert.equal(readRecommendedOctenHttpStoredDimensions('another-candidate', 2_560), null);
  assert.equal(readRecommendedOctenHttpStoredDimensions('recommended-octen-http-0', 2_560), null);
  assert.equal(
    readRecommendedOctenHttpStoredDimensions('recommended-octen-http-2561', 2_560),
    null,
  );
  assert.throws(
    () => createRecommendedOctenHttpCandidateId(0),
    /Recall Octen setup stored dimensions invalid/u,
  );
  assert.throws(
    () => createRecommendedOctenHttpCandidateId(2_561),
    /Recall Octen setup stored dimensions invalid/u,
  );
});

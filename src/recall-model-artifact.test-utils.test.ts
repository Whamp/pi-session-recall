import assert from 'node:assert/strict';
import test from 'node:test';

import {
  createRecallModelArtifactFixtureGguf,
  createRecallModelArtifactFixtureProfile,
} from './recall-model-artifact.test-utils.js';

void test('model artifact test utility produces a profile matching its GGUF bytes', () => {
  const artifact = createRecallModelArtifactFixtureGguf();
  const profile = createRecallModelArtifactFixtureProfile(artifact);

  assert.equal(artifact.subarray(0, 4).toString('ascii'), 'GGUF');
  assert.equal(profile.source.byteSize, artifact.length);
});

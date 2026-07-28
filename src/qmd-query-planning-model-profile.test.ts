import assert from 'node:assert/strict';
import test from 'node:test';

import { createRecommendedQmdQueryPlanningModelProfile } from './recall-model-profiles.js';

void test('recommended QMD query planner profile pins artifact and planning semantics', () => {
  const profile = createRecommendedQmdQueryPlanningModelProfile();

  assert.equal(profile.profileId, 'qmd-query-expansion-1.7b-q4-k-m-v1');
  assert.equal(profile.model, 'qmd-query-expansion-1.7B-q4_k_m');
  assert.equal(profile.promptPolicy, 'qmd-query-expansion-no-think-v1');
  assert.equal(profile.grammarVersion, 'qmd-bounded-query-plan-v2');
  assert.equal(
    profile.grammar,
    [
      'root ::= lex lex? lex? vec vec? vec? hyde?',
      'lex ::= "lex: " content "\\n"',
      'vec ::= "vec: " content "\\n"',
      'hyde ::= "hyde: " content "\\n"',
      'content ::= [^\\n]{1,512}',
    ].join('\n'),
  );
  assert.deepEqual(profile.planBounds, {
    minimumLexQueries: 1,
    maximumLexQueries: 3,
    minimumVecQueries: 1,
    maximumVecQueries: 3,
    maximumHydeQueries: 1,
  });
  assert.deepEqual(profile.generationPolicy, {
    contextSize: 2_048,
    maximumOutputTokens: 600,
    temperature: 0.7,
    topK: 20,
    topP: 0.8,
    repeatPenaltyLastTokens: 64,
    presencePenalty: 0.5,
  });
  assert.equal(profile.source.repository, 'tobil/qmd-query-expansion-1.7B-gguf');
  assert.equal(profile.source.revision, '7816de0b72572c6c860ca1eddf97ba9e7fb8cc65');
  assert.equal(profile.source.artifact, 'qmd-query-expansion-1.7B-q4_k_m.gguf');
  assert.equal(profile.source.byteSize, 1_282_438_912);
  assert.equal(
    profile.source.sha256,
    '000dfb1c06efa6a049e9f64ba921c3740e2454f62abab6fa10e77bd30bb2bcc0',
  );
  assert.equal(
    profile.source.downloadUrl,
    'https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/resolve/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/qmd-query-expansion-1.7B-q4_k_m.gguf',
  );
  assert.deepEqual(profile.license, {
    id: 'mit',
    name: 'MIT License',
    url: 'https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/blob/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/README.md',
    distributionStatus: 'review-required',
  });
  assert.deepEqual(profile.conformanceCanary, {
    query: 'Copper Finch',
    recallIntent: 'Find Pi conversation evidence about the exact Copper Finch recovery entity.',
    protectedTerms: ['Copper', 'Finch'],
  });
});

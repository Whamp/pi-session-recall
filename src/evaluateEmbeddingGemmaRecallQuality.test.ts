import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { EmbeddedInferenceDevicePolicy } from './enums.js';
import { evaluateEmbeddingGemmaRecallQuality } from './evaluateEmbeddingGemmaRecallQuality.js';

void test('EmbeddingGemma evaluation requires the committed quality corpus before model loading', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-embeddinggemma-evaluation-'));
  t.after(() => rm(root, { recursive: true, force: true }));

  await assert.rejects(
    () => evaluateEmbeddingGemmaRecallQuality(EmbeddedInferenceDevicePolicy.CPU, root),
    /ENOENT.*recall-quality-cases\.json/u,
  );
});

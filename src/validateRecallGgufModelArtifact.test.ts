import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { createRecallModelArtifactFixtureGguf } from './recall-model-artifact.test-utils.js';
import { validateRecallGgufModelArtifact } from './validateRecallGgufModelArtifact.js';

void test('GGUF validation accepts exact size, checksum, and structure', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-gguf-validation-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const artifact = createRecallModelArtifactFixtureGguf();
  const path = join(root, 'fixture.gguf');
  await writeFile(path, artifact);

  await validateRecallGgufModelArtifact(path, {
    byteSize: artifact.length,
    sha256: createHash('sha256').update(artifact).digest('hex'),
  });
});

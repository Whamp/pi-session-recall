import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  inspectRecallConversationCorpus,
  selectRecallConversationCorpusSample,
} from './recall-conversation-corpus.js';

void test('conversation corpus inspection and sampling are metadata-only and deterministic', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-corpus-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'nested'));
  await writeFile(join(root, 'small.jsonl'), 'a');
  await writeFile(join(root, 'nested', 'large.jsonl'), '12345');
  await writeFile(join(root, 'ignored.txt'), 'ignored');

  const corpus = await inspectRecallConversationCorpus(root);

  assert.deepEqual(corpus.inspection, { sessionCount: 2, sourceByteSize: 6 });
  assert.equal(selectRecallConversationCorpusSample(corpus.files, 1)[0]?.sourceByteSize, 1);
});

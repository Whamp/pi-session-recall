import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  isPathInsideRecallEvaluationArea,
  writeAtomicRecallEvaluationFile,
} from './recall-evaluation-file-system.js';

void test('recall evaluation paths stay bounded and publishable files replace atomically', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-evaluation-file-system-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const evidencePath = join(directory, 'nested', 'evidence.json');

  assert.equal(isPathInsideRecallEvaluationArea(directory, directory), true);
  assert.equal(isPathInsideRecallEvaluationArea(directory, evidencePath), true);
  assert.equal(isPathInsideRecallEvaluationArea(directory, `${directory}-nearby`), false);

  await writeAtomicRecallEvaluationFile(evidencePath, 'first');
  await writeAtomicRecallEvaluationFile(evidencePath, 'second');
  assert.equal(await readFile(evidencePath, 'utf8'), 'second');
});

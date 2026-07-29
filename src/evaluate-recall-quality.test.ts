import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import test from 'node:test';

import { DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS } from './create-deterministic-recall-quality-dependencies.js';
import { createDeterministicRecallQualityConfig } from './evaluate-recall-quality.js';

const execFileAsync = promisify(execFile);

void test('recall quality CLI help states bounded work and report outputs without model calls', async () => {
  const { stdout } = await execFileAsync(
    process.execPath,
    ['--import', 'tsx', 'src/evaluate-recall-quality.ts', '--help'],
    { cwd: process.cwd(), encoding: 'utf8' },
  );

  assert.match(stdout, /checksum-fixed evaluation corpus/);
  assert.match(stdout, /never scans the production session corpus/);
  assert.match(stdout, /docs\/evaluation\/recall-quality-report\.md/);
  assert.match(stdout, /docs\/evaluation\/recall-quality-results\.json/);
});

void test('committed recall quality config pins the deterministic embedding identity', () => {
  const config = createDeterministicRecallQualityConfig(process.cwd());

  assert.equal(config.embeddingBaseUrl, 'in-process://deterministic-fixture-v1');
  assert.equal(config.embeddingModel, 'deterministic-fixture-v1');
  assert.equal(config.embeddingDimensions, DETERMINISTIC_RECALL_QUALITY_EMBEDDING_DIMENSIONS);
});

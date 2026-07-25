import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import test from 'node:test';

const execFileAsync = promisify(execFile);

void test('recall quality CLI help states bounded work and report outputs without model calls', async () => {
  const { stdout } = await execFileAsync(
    process.execPath,
    ['--import', 'tsx', 'src/evaluate-recall-quality.ts', '--help'],
    { cwd: process.cwd(), encoding: 'utf8' },
  );

  assert.match(stdout, /fixed 8-file evaluation corpus/);
  assert.match(stdout, /never scans the production session corpus/);
  assert.match(stdout, /docs\/evaluation\/recall-quality-report\.md/);
  assert.match(stdout, /docs\/evaluation\/recall-quality-results\.json/);
});

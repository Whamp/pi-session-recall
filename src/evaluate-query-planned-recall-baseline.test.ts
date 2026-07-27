import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const EXEC_FILE_ASYNC = promisify(execFile);

void test('query-planned recall baseline CLI documents private isolation and publishable outputs', async () => {
  const sourceDirectory = dirname(fileURLToPath(import.meta.url));
  const projectDirectory = dirname(sourceDirectory);
  const { stdout, stderr } = await EXEC_FILE_ASYNC(
    process.execPath,
    [
      '--import',
      'tsx',
      join(sourceDirectory, 'evaluate-query-planned-recall-baseline.ts'),
      '--help',
    ],
    { cwd: projectDirectory },
  );

  assert.equal(stderr, '');
  assert.match(stdout, /\.recall-data\/query-planned-recall\/manifest\.json/u);
  assert.match(stdout, /evaluation\/query-planned-recall-controls\.json/u);
  assert.match(stdout, /docs\/evaluation\/query-planned-hybrid-baseline\.json/u);
  assert.match(stdout, /never scans or writes the production recall corpus/u);
});

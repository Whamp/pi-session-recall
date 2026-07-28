import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const EXEC_FILE_ASYNC = promisify(execFile);

void test('live query-planned profile acceptance CLI documents the complete measured matrix', async () => {
  const projectDirectory = dirname(dirname(fileURLToPath(import.meta.url)));
  const { stdout, stderr } = await EXEC_FILE_ASYNC(
    process.execPath,
    [
      '--import',
      'tsx',
      join(projectDirectory, 'src', 'evaluate-query-planned-profile-acceptance.ts'),
      '--help',
    ],
    { cwd: projectDirectory },
  );

  assert.equal(stderr, '');
  assert.match(stdout, /--accelerated-device <metal\|cuda\|vulkan>/u);
  assert.match(stdout, /--http-planner-url/u);
  assert.match(stdout, /--http-reranker-url/u);
  assert.match(stdout, /embedded CPU, embedded accelerator, and HTTP/u);
  assert.match(stdout, /embeddinggemma-quality-cpu\.json/u);
  assert.match(stdout, /rerun resumes/u);
  assert.match(
    stdout,
    /exact commit, corpus, profile, backend, device, and adapter configuration/u,
  );
  assert.match(stdout, /reports profile and case progress/u);
  assert.match(stdout, /never publishes private query, plan, or source text/u);
});

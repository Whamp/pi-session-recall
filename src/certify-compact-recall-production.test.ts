import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import test from 'node:test';

void test('production certification rejects missing arguments before reading production state', () => {
  const result = spawnSync(
    process.execPath,
    [
      '--import',
      'tsx',
      new URL('./certify-compact-recall-production.ts', import.meta.url).pathname,
    ],
    { encoding: 'utf8' },
  );

  assert.notEqual(result.status, 0);
  assert.match(
    result.stderr,
    /Compact recall production certification usage: --candidate <database-target> --block-device <name> --changed-session <path> --output <path>/u,
  );
  assert.doesNotMatch(result.stderr, /\.pi\/agent\/recall|\.pi\/agent\/sessions/u);
});

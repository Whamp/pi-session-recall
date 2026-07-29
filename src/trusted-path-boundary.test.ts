import assert from 'node:assert/strict';
import { mkdir, mkdtemp, rm, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  isCanonicalPathWithinBoundary,
  resolveCanonicalPathBoundary,
} from './trusted-path-boundary.js';

void test('trusted path boundary resolves symlinks and missing descendants canonically', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'trusted-path-boundary-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const root = join(directory, 'root');
  const target = join(directory, 'target');
  await mkdir(root);
  await mkdir(target);
  await symlink(target, join(root, 'link'));

  assert.equal(
    await resolveCanonicalPathBoundary(join(root, 'missing', 'session.jsonl')),
    join(root, 'missing', 'session.jsonl'),
  );
  assert.equal(
    await resolveCanonicalPathBoundary(join(root, 'link', 'session.jsonl')),
    join(target, 'session.jsonl'),
  );
  assert.equal(isCanonicalPathWithinBoundary(join(root, 'child'), root), true);
  assert.equal(isCanonicalPathWithinBoundary(join(directory, 'outside'), root), false);
});

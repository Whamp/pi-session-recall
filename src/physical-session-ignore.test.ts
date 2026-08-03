import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  addIgnoredPhysicalSessionPath,
  listIgnoredPhysicalSessionPaths,
  normalizePhysicalSessionPath,
  removeIgnoredPhysicalSessionPath,
} from './physical-session-ignore.js';

const EXEC_FILE_ASYNC = promisify(execFile);
const PSR_EXECUTABLE_PATH = fileURLToPath(new URL('../bin/psr', import.meta.url));
const PSR_PROJECT_DIRECTORY = fileURLToPath(new URL('..', import.meta.url));

void test('physical session ignore paths resolve lexically without filesystem rules', () => {
  assert.equal(
    normalizePhysicalSessionPath('/working/project', '../sessions/*.jsonl'),
    '/working/sessions/*.jsonl',
  );
});

void test('physical session ignore state persists sorted idempotent additions and removals', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'physical-session-ignore-state-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'physical-session-ignore.json');
  const firstPath = resolve(root, 'sessions', 'a.jsonl');
  const secondPath = resolve(root, 'sessions', 'b.jsonl');

  assert.equal(await addIgnoredPhysicalSessionPath(statePath, secondPath), true);
  assert.equal(await addIgnoredPhysicalSessionPath(statePath, firstPath), true);
  assert.equal(await addIgnoredPhysicalSessionPath(statePath, firstPath), false);
  assert.deepEqual(await listIgnoredPhysicalSessionPaths(statePath), [firstPath, secondPath]);
  assert.equal(
    await readFile(statePath, 'utf8'),
    `${JSON.stringify({ version: 1, ignoredPhysicalSessionPaths: [firstPath, secondPath] })}\n`,
  );

  assert.equal(await removeIgnoredPhysicalSessionPath(statePath, firstPath), true);
  assert.equal(await removeIgnoredPhysicalSessionPath(statePath, firstPath), false);
  assert.deepEqual(await listIgnoredPhysicalSessionPaths(statePath), [secondPath]);
  assert.deepEqual(await readdir(root), ['physical-session-ignore.json']);
});

void test('physical session ignore state rejects malformed and noncanonical persistence', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'physical-session-ignore-invalid-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const statePath = join(root, 'physical-session-ignore.json');

  for (const invalidState of [
    '{',
    '{"version":2,"ignoredPhysicalSessionPaths":[]}',
    '{"version":1,"ignoredPhysicalSessionPaths":["relative.jsonl"]}',
    '{"version":1,"ignoredPhysicalSessionPaths":["/z.jsonl","/a.jsonl"]}',
  ]) {
    await writeFile(statePath, `${invalidState}\n`, 'utf8');
    await assert.rejects(
      listIgnoredPhysicalSessionPaths(statePath),
      new RegExp(`Physical session ignore state invalid at ${statePath}`, 'u'),
    );
  }
});

void test('public CLI resolves its runtime while normalizing paths from the caller directory', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'physical-session-ignore-cli-cwd-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'recall');
  const relativeIgnoredPath = join('sessions', 'session.jsonl');
  const normalizedIgnoredPath = resolve(root, relativeIgnoredPath);
  const environment = { ...process.env, PI_RECALL_DATA_DIRECTORY: dataDirectory };

  const addition = await EXEC_FILE_ASYNC(
    PSR_EXECUTABLE_PATH,
    ['ignore', 'add', relativeIgnoredPath],
    { cwd: root, env: environment },
  );

  assert.equal(addition.stderr, '');
  assert.equal(addition.stdout, `Ignored: ${normalizedIgnoredPath}\n`);
  assert.deepEqual(
    await listIgnoredPhysicalSessionPaths(join(dataDirectory, 'physical-session-ignore.json')),
    [normalizedIgnoredPath],
  );
});

void test('concurrent public CLI additions retain every acknowledged ignore path', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'physical-session-ignore-concurrent-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const dataDirectory = join(root, 'recall');
  const ignoredPaths = Array.from({ length: 24 }, (_, index) =>
    resolve(root, 'sessions', `session-${String(index).padStart(2, '0')}.jsonl`),
  );
  const environment = { ...process.env, PI_RECALL_DATA_DIRECTORY: dataDirectory };

  const additions = await Promise.all(
    ignoredPaths.map((ignoredPath) =>
      EXEC_FILE_ASYNC(PSR_EXECUTABLE_PATH, ['ignore', 'add', ignoredPath], {
        cwd: PSR_PROJECT_DIRECTORY,
        env: environment,
      }),
    ),
  );
  for (const [index, addition] of additions.entries()) {
    assert.equal(addition.stderr, '');
    assert.equal(addition.stdout, `Ignored: ${ignoredPaths[index]}\n`);
  }

  const listing = await EXEC_FILE_ASYNC(PSR_EXECUTABLE_PATH, ['ignore', 'list'], {
    cwd: PSR_PROJECT_DIRECTORY,
    env: environment,
  });
  assert.equal(listing.stderr, '');
  assert.deepEqual(listing.stdout.trimEnd().split('\n'), ignoredPaths);
});

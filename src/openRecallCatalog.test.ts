import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { DatabaseSync } from 'node:sqlite';

import { RecallProjectIdentitySource } from './enums.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import type { InvocationRecord } from './createSessionInvocationRecords.js';
import { openRecallCatalog } from './openRecallCatalog.js';
import { parseRepositoryIdentity } from './resolve-project-identity.js';

function createInvocationRecord(
  sessionPath: string,
  overrides: Partial<InvocationRecord> = {},
): InvocationRecord {
  return {
    kind: 'tool_call',
    toolName: 'read',
    toolCallId: 'call-1',
    sessionPath,
    sessionId: 'session-1',
    entryId: 'assistant-1',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockIndex: 0,
    timestamp: '2026-08-09T12:00:00Z',
    sessionOrigin: '/project',
    projectAttribution: null,
    isError: false,
    searchableText: 'tool="read"\npath="/project/src/queue.ts"',
    ...overrides,
  };
}

void test('recall catalog creates and replaces one physical session state', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-catalog-create-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const catalog = openRecallCatalog(join(root, 'recall-catalog.sqlite'));
  t.after(() => catalog.close());

  catalog.replacePhysicalSession({
    sessionPath: '/sessions/one.jsonl',
    size: 123,
    mtimeMs: 456,
    documentIds: ['dense-1', 'dense-2'],
    denseDocumentIds: ['dense-1', 'dense-2'],
    invocations: [],
  });

  assert.deepEqual(catalog.readPhysicalSessionState('/sessions/one.jsonl'), {
    size: 123,
    mtimeMs: 456,
    documentIds: ['dense-1', 'dense-2'],
    denseDocumentIds: ['dense-1', 'dense-2'],
  });
  assert.deepEqual(catalog.listPhysicalSessionPaths(), ['/sessions/one.jsonl']);
  assert.equal(catalog.requiresInvocationBackfill('/sessions/one.jsonl'), false);
});

void test('recall catalog rolls back one failed replacement without touching unrelated sessions', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-catalog-rollback-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const catalog = openRecallCatalog(join(root, 'recall-catalog.sqlite'));
  t.after(() => catalog.close());
  catalog.replacePhysicalSession({
    sessionPath: '/sessions/one.jsonl',
    size: 10,
    mtimeMs: 20,
    documentIds: ['one-before'],
    denseDocumentIds: ['one-before'],
    invocations: [createInvocationRecord('/sessions/one.jsonl')],
  });
  catalog.replacePhysicalSession({
    sessionPath: '/sessions/unrelated.jsonl',
    size: 30,
    mtimeMs: 40,
    documentIds: ['unrelated-before'],
    denseDocumentIds: ['unrelated-before'],
    invocations: [],
  });

  assert.throws(() =>
    catalog.replacePhysicalSession({
      sessionPath: '/sessions/one.jsonl',
      size: 11,
      mtimeMs: 21,
      documentIds: ['one-after'],
      denseDocumentIds: ['one-after'],
      invocations: [
        createInvocationRecord('/sessions/one.jsonl', {
          sourceLineStart: 0,
          sourceLineEnd: 0,
        }),
      ],
    }),
  );

  assert.deepEqual(catalog.readPhysicalSessionState('/sessions/one.jsonl'), {
    size: 10,
    mtimeMs: 20,
    documentIds: ['one-before'],
    denseDocumentIds: ['one-before'],
  });
  assert.deepEqual(catalog.readPhysicalSessionState('/sessions/unrelated.jsonl'), {
    size: 30,
    mtimeMs: 40,
    documentIds: ['unrelated-before'],
    denseDocumentIds: ['unrelated-before'],
  });
});

void test('recall catalog searches Invocation records globally and by exact project identity', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-catalog-search-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const catalog = openRecallCatalog(join(root, 'recall-catalog.sqlite'));
  t.after(() => catalog.close());
  const alphaProject = parseRepositoryIdentity('git-origin:github.com/acme/alpha');
  const betaProject = parseRepositoryIdentity('git-origin:github.com/acme/beta');
  for (const [index, projectIdentity] of [alphaProject, betaProject].entries()) {
    const sessionPath = `/sessions/${index}.jsonl`;
    catalog.replacePhysicalSession({
      sessionPath,
      size: 100 + index,
      mtimeMs: 200 + index,
      documentIds: [`dense-${index}`],
      denseDocumentIds: [`dense-${index}`],
      invocations: [
        createInvocationRecord(sessionPath, {
          toolCallId: `call-${index}`,
          entryId: `assistant-${index}`,
          sourceLineStart: 10 + index,
          sourceLineEnd: 10 + index,
          projectAttribution: {
            projectIdentity,
            identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
          },
          searchableText: `tool="read"\npath="/project/src/compact-catalog-${index}.ts"`,
        }),
      ],
    });
  }

  const globalMatches = catalog.searchInvocations('compact catalog', 10);
  assert.equal(globalMatches.length, 2);
  assert.deepEqual(
    globalMatches.map((match) => ({
      sessionPath: match.sessionPath,
      entryId: match.entryId,
      sourceLineStart: match.sourceLineStart,
    })),
    [
      { sessionPath: '/sessions/0.jsonl', entryId: 'assistant-0', sourceLineStart: 10 },
      { sessionPath: '/sessions/1.jsonl', entryId: 'assistant-1', sourceLineStart: 11 },
    ],
  );

  const projectMatches = catalog.searchInvocations('compact catalog', 10, betaProject);
  assert.equal(projectMatches.length, 1);
  assert.equal(projectMatches[0]?.sessionPath, '/sessions/1.jsonl');
  assert.equal(projectMatches[0]?.projectAttribution?.projectIdentity, betaProject);
});

void test('recall catalog migrates legacy indexed-session knowledge exactly once', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-catalog-migrate-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const legacyStatePath = join(root, 'index-state.json');
  await writeFile(
    legacyStatePath,
    `${JSON.stringify({
      version: 3,
      importPolicyVersion: SESSION_IMPORT_POLICY_VERSION,
      sessions: {
        '/sessions/legacy.jsonl': {
          size: 321,
          mtimeMs: 654,
          chunks: [{ id: 'legacy-dense-1' }, { id: 'legacy-dense-2' }],
        },
      },
    })}\n`,
  );

  const migrated = openRecallCatalog(catalogPath, { legacyStatePath });
  assert.deepEqual(migrated.readPhysicalSessionState('/sessions/legacy.jsonl'), {
    size: 321,
    mtimeMs: 654,
    documentIds: ['legacy-dense-1', 'legacy-dense-2'],
    denseDocumentIds: [],
  });
  assert.equal(migrated.requiresInvocationBackfill('/sessions/legacy.jsonl'), true);
  migrated.close();

  await writeFile(
    legacyStatePath,
    `${JSON.stringify({
      version: 3,
      importPolicyVersion: SESSION_IMPORT_POLICY_VERSION,
      sessions: {},
    })}\n`,
  );
  const reopened = openRecallCatalog(catalogPath, { legacyStatePath });
  t.after(() => reopened.close());
  assert.deepEqual(reopened.listPhysicalSessionPaths(), ['/sessions/legacy.jsonl']);
});

void test('recall catalog rejects an incompatible schema with a rebuild action', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-catalog-schema-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const catalogPath = join(root, 'recall-catalog.sqlite');
  const incompatible = new DatabaseSync(catalogPath);
  incompatible.exec('PRAGMA user_version = 99');
  incompatible.close();

  assert.throws(
    () => openRecallCatalog(catalogPath),
    /Recall catalog schema incompatible.*found version 99.*psr index --rebuild/u,
  );
});

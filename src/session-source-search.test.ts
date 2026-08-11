import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { RecallProjectIdentitySource, RecallSearchScope } from './enums.js';
import { searchSessionSourceFiles } from './session-source-search.js';
import {
  parseRepositoryIdentity,
  type ResolvedProjectIdentity,
} from './resolve-project-identity.js';

function sessionJsonl(entries: object[]): string {
  return `${entries.map((entry) => JSON.stringify(entry)).join('\n')}\n`;
}

function createProjectIdentity(value: string): ResolvedProjectIdentity {
  return {
    projectIdentity: parseRepositoryIdentity(`git-origin:github.com/Whamp/${value}`),
    identitySource: RecallProjectIdentitySource.GIT_ORIGIN,
  };
}

void test('source search finds complete result and bash evidence within scope and limit', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-source-search-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionsDirectory = join(root, 'sessions');
  const projectAPath = join(sessionsDirectory, 'a.jsonl');
  const projectBPath = join(sessionsDirectory, 'b.jsonl');
  await mkdir(sessionsDirectory, { recursive: true });
  await writeFile(
    projectAPath,
    sessionJsonl([
      {
        type: 'session',
        version: 3,
        id: 'session-a',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: '/projects/a',
      },
      {
        type: 'message',
        id: 'result-a',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-a',
          toolName: 'bash',
          isError: true,
          content: [
            {
              type: 'text',
              text: 'Disk CT1000P3PSSD8 failed for archive.tar.zst with --force-repair',
            },
          ],
        },
      },
      {
        type: 'message',
        id: 'bash-a',
        parentId: 'result-a',
        timestamp: '2026-08-10T10:02:00Z',
        message: {
          role: 'bashExecution',
          command: 'cat diagnostic.log',
          output: 'COMMAND_OUTPUT_ONLY alpha beta gamma',
          cancelled: false,
          truncated: false,
        },
      },
      {
        type: 'session',
        version: 3,
        id: 'malformed-reused-session',
        timestamp: '2026-08-10T10:03:00Z',
      },
      {
        type: 'message',
        id: 'must-not-inherit-project',
        parentId: null,
        timestamp: '2026-08-10T10:04:00Z',
        message: { role: 'toolResult', content: 'MALFORMED_HEADER_BOUNDARY' },
      },
    ]),
  );
  await writeFile(
    projectBPath,
    sessionJsonl([
      {
        type: 'session',
        version: 3,
        id: 'session-b',
        timestamp: '2026-08-10T11:00:00Z',
        cwd: '/projects/b',
      },
      {
        type: 'message',
        id: 'result-b',
        parentId: null,
        timestamp: '2026-08-10T11:01:00Z',
        message: {
          role: 'toolResult',
          toolCallId: 'call-b',
          toolName: 'read',
          isError: false,
          content: [{ type: 'text', text: 'COMMAND_OUTPUT_ONLY from project b' }],
        },
      },
    ]),
  );

  const projectA = createProjectIdentity('project-a');
  const projectB = createProjectIdentity('project-b');
  const resolveProjectIdentity = async (sessionOrigin: string) =>
    sessionOrigin === '/projects/a' ? projectA : projectB;

  for (const resultOnlyQuery of ['CT1000P3PSSD8', 'archive.tar.zst', 'failed', '--force-repair']) {
    const resultOnly = await searchSessionSourceFiles({
      sessionsDirectory,
      ignoredPhysicalSessionPaths: new Set(),
      query: resultOnlyQuery,
      limit: 5,
      scope: RecallSearchScope.PROJECT,
      invocationProjectIdentity: projectA.projectIdentity,
      resolveProjectIdentity,
    });
    assert.equal(resultOnly.results.length, 1);
    const result = resultOnly.results[0];
    assert.equal(result?.sessionPath, projectAPath);
    assert.equal(result?.sourceLineStart, 2);
    assert.equal(result?.sourceLineEnd, 2);
    assert.equal(result?.entryId, 'result-a');
    assert.equal(result?.sessionOrigin, '/projects/a');
    assert.match(result?.text ?? '', new RegExp(resultOnlyQuery, 'u'));
  }

  const projectBOnly = await searchSessionSourceFiles({
    sessionsDirectory,
    ignoredPhysicalSessionPaths: new Set(),
    query: 'COMMAND_OUTPUT_ONLY',
    limit: 5,
    scope: RecallSearchScope.PROJECT,
    invocationProjectIdentity: projectB.projectIdentity,
    resolveProjectIdentity,
  });
  assert.deepEqual(
    projectBOnly.results.map((result) => result.entryId),
    ['result-b'],
  );

  const malformedBoundary = await searchSessionSourceFiles({
    sessionsDirectory,
    ignoredPhysicalSessionPaths: new Set(),
    query: 'MALFORMED_HEADER_BOUNDARY',
    limit: 5,
    scope: RecallSearchScope.PROJECT,
    invocationProjectIdentity: projectA.projectIdentity,
    resolveProjectIdentity,
  });
  assert.deepEqual(malformedBoundary.results, []);

  const globalLimited = await searchSessionSourceFiles({
    sessionsDirectory,
    ignoredPhysicalSessionPaths: new Set(),
    query: 'COMMAND_OUTPUT_ONLY',
    limit: 1,
    scope: RecallSearchScope.GLOBAL,
    invocationProjectIdentity: null,
    resolveProjectIdentity: async () => {
      throw new Error('global source search must not resolve project identities');
    },
  });
  assert.deepEqual(
    globalLimited.results.map((result) => result.entryId),
    ['bash-a'],
  );
  assert.equal(globalLimited.filesScanned, 2);
});

void test('source search reports a missing file without hiding valid matches', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-source-missing-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const validPath = join(root, 'a-valid.jsonl');
  const missingPath = join(root, 'z-missing.jsonl');
  await writeFile(
    validPath,
    sessionJsonl([
      {
        type: 'session',
        version: 3,
        id: 'valid-session',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: '/projects/a',
      },
      {
        type: 'message',
        id: 'valid-result',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: { role: 'toolResult', content: 'MISSING_FILE_CONTINUATION' },
      },
    ]),
  );

  const search = await searchSessionSourceFiles(
    {
      sessionsDirectory: root,
      ignoredPhysicalSessionPaths: new Set(),
      query: 'MISSING_FILE_CONTINUATION',
      limit: 1,
      scope: RecallSearchScope.GLOBAL,
      invocationProjectIdentity: null,
      resolveProjectIdentity: async () => null,
    },
    { listSessionPaths: async () => [missingPath, validPath] },
  );

  assert.equal(search.results[0]?.entryId, 'valid-result');
  assert.deepEqual(search.failures, [
    {
      sessionPath: missingPath,
      error: `Source session missing at ${missingPath}`,
    },
  ]);
});

void test('source search honors cancellation while scanning session lines', async (t) => {
  const root = await mkdtemp(join(tmpdir(), 'recall-source-cancel-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sessionPath = join(root, 'cancel.jsonl');
  await writeFile(
    sessionPath,
    sessionJsonl([
      {
        type: 'session',
        version: 3,
        id: 'cancel-session',
        timestamp: '2026-08-10T10:00:00Z',
        cwd: '/projects/a',
      },
      {
        type: 'message',
        id: 'cancel-result',
        parentId: null,
        timestamp: '2026-08-10T10:01:00Z',
        message: { role: 'toolResult', content: 'CANCELLED_SOURCE_EVIDENCE' },
      },
    ]),
  );
  const abortController = new AbortController();
  const project = createProjectIdentity('project-a');

  await assert.rejects(
    searchSessionSourceFiles({
      sessionsDirectory: root,
      ignoredPhysicalSessionPaths: new Set(),
      query: 'CANCELLED_SOURCE_EVIDENCE',
      limit: 5,
      scope: RecallSearchScope.PROJECT,
      invocationProjectIdentity: project.projectIdentity,
      resolveProjectIdentity: async () => {
        abortController.abort(new Error('Source search cancelled by test'));
        return project;
      },
      signal: abortController.signal,
    }),
    (error: unknown) =>
      error instanceof Error &&
      (error.name === 'AbortError' || error.message === 'Source search cancelled by test'),
  );
});

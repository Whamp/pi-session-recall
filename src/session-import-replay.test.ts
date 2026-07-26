import assert from 'node:assert/strict';
import { cp, mkdtemp, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { SessionImportFormat, SessionImportReplayOutcome } from './enums.js';
import { replaySessionImportCorpus } from './session-import-replay.js';

void test('session import replay is deterministic, machine-readable, and source read-only', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-session-import-replay-'));
  const corpusRoot = join(directory, 'corpus');
  await mkdir(corpusRoot);
  const fixtureDirectory = join(import.meta.dirname, 'fixtures/session-import');
  await Promise.all(
    [
      'canonical-unicode-separators.jsonl',
      'pi-v1-linear.jsonl',
      'pi-session-reuse-history.jsonl',
    ].map((fixtureName) => cp(join(fixtureDirectory, fixtureName), join(corpusRoot, fixtureName))),
  );
  const corruptPath = join(corpusRoot, 'missing-parent.jsonl');
  await writeFile(
    corruptPath,
    [
      {
        type: 'session',
        version: 3,
        id: 'corrupt-session',
        timestamp: '2026-01-10T10:00:00Z',
        cwd: '/corrupt/project',
      },
      {
        type: 'message',
        id: 'orphan-entry',
        parentId: 'missing-entry',
        timestamp: '2026-01-10T10:00:01Z',
        message: { role: 'user', content: 'must not become searchable' },
      },
    ]
      .map((record) => JSON.stringify(record))
      .join('\n'),
  );
  const bytesBefore = await readFile(corruptPath);
  const metadataBefore = await stat(corruptPath);

  const first = await replaySessionImportCorpus(corpusRoot);
  const second = await replaySessionImportCorpus(corpusRoot);

  assert.deepEqual(second, first);
  assert.equal(first.physicalFiles, 4);
  assert.equal(first.acceptedPhysicalFiles, 3);
  assert.equal(first.rejectedPhysicalFiles, 1);
  assert.equal(first.logicalSessions, 4);
  assert.deepEqual(first.formats[SessionImportFormat.CANONICAL_JSONL], {
    physicalFiles: 1,
    logicalSessions: 1,
  });
  assert.deepEqual(first.formats[SessionImportFormat.PI_V1_LINEAR], {
    physicalFiles: 1,
    logicalSessions: 1,
  });
  assert.deepEqual(first.formats[SessionImportFormat.PI_SESSION_REUSE_HISTORY], {
    physicalFiles: 1,
    logicalSessions: 2,
  });
  const rejected = first.files.find((file) => file.outcome === SessionImportReplayOutcome.REJECTED);
  assert.equal(rejected?.sessionPath, 'missing-parent.jsonl');
  assert.match(rejected?.error ?? '', /entry orphan-entry has missing parent missing-entry/u);
  assert.equal(rejected?.documents, 0);
  assert.match(first.replayDigest, /^[a-f0-9]{64}$/u);
  assert.deepEqual(await readFile(corruptPath), bytesBefore);
  const metadataAfter = await stat(corruptPath);
  assert.deepEqual(
    {
      size: metadataAfter.size,
      mode: metadataAfter.mode,
      mtimeMs: metadataAfter.mtimeMs,
      ino: metadataAfter.ino,
    },
    {
      size: metadataBefore.size,
      mode: metadataBefore.mode,
      mtimeMs: metadataBefore.mtimeMs,
      ino: metadataBefore.ino,
    },
  );
});

void test('session import replay refuses the production recall data directory', async () => {
  await assert.rejects(
    () => replaySessionImportCorpus(join(process.env.HOME ?? '', '.pi/agent/recall')),
    /refuses the production recall index directory/u,
  );
});

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cp, mkdtemp, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { SessionImportFormat, SessionImportReplayOutcome } from './enums.js';
import {
  replaySessionImportCorpus,
  type SessionImportReplayResult,
} from './session-import-replay.js';

const historicalCorpusReplayExpectation = {
  schemaVersion: 1,
  physicalFiles: 121,
  acceptedPhysicalFiles: 119,
  rejectedPhysicalFiles: 2,
  logicalSessions: 170,
  documents: 34_029,
  formats: {
    [SessionImportFormat.CANONICAL_JSONL]: { physicalFiles: 9, logicalSessions: 9 },
    [SessionImportFormat.PI_V1_LINEAR]: { physicalFiles: 77, logicalSessions: 77 },
    [SessionImportFormat.PI_SESSION_REUSE_HISTORY]: {
      physicalFiles: 33,
      logicalSessions: 84,
    },
  },
  rejection: {
    sourceLine: 212,
    entryId: '8d2b86d9',
    missingParentId: '74da12a2',
    documentsPerFile: 0,
  },
  sourceSetDigest: '87ab99beec4a66bd236fb214f8ca4ceca4b6e00d054ea877e5baa56f50a640bd',
  outcomeDigest: 'a63931337880808394f2412a1b7109148d7e646fd54cd8a3e354c8b4291e38cd',
};

function sha256ReplayEvidence(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

function normalizeReplayError(error: string | null) {
  if (!error) {
    return null;
  }
  const match = error.match(/:(\d+): entry ([^ ]+) has missing parent ([^ ]+)$/u);
  if (!match) {
    return { unrecognized: error };
  }
  return {
    line: Number(match[1]),
    entryId: match[2],
    missingParentId: match[3],
  };
}

function createHistoricalReplayOutcomeDigest(result: SessionImportReplayResult): string {
  const files = result.files
    .map(({ format, outcome, logicalSessions, documents, importDigest, error }) => ({
      format,
      outcome,
      logicalSessions,
      documents,
      importDigest,
      error: normalizeReplayError(error),
    }))
    .sort((left, right) => JSON.stringify(left).localeCompare(JSON.stringify(right)));
  return sha256ReplayEvidence(
    JSON.stringify({
      physicalFiles: result.physicalFiles,
      acceptedPhysicalFiles: result.acceptedPhysicalFiles,
      rejectedPhysicalFiles: result.rejectedPhysicalFiles,
      logicalSessions: result.logicalSessions,
      formats: result.formats,
      files,
    }),
  );
}

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

void test('historical corpus replay expectation is frozen in the repository', async () => {
  const fixturePath = join(
    import.meta.dirname,
    'fixtures/session-import/historical-corpus-replay-expectation.json',
  );
  const fixture: unknown = JSON.parse(await readFile(fixturePath, 'utf8'));
  assert.deepEqual(fixture, historicalCorpusReplayExpectation);
});

const historicalCorpusRoot = process.env.PI_SESSION_IMPORT_CORPUS_ROOT;
void test(
  'historical corpus replay matches the privacy-safe frozen expectation',
  { skip: historicalCorpusRoot ? false : 'PI_SESSION_IMPORT_CORPUS_ROOT is not set' },
  async () => {
    if (!historicalCorpusRoot) {
      return;
    }
    const replay = await replaySessionImportCorpus(historicalCorpusRoot);
    const sourceHashes = await Promise.all(
      replay.files.map(async (file) =>
        sha256ReplayEvidence(await readFile(join(replay.corpusRoot, file.sessionPath))),
      ),
    );
    sourceHashes.sort();
    const rejected = replay.files.filter(
      (file) => file.outcome === SessionImportReplayOutcome.REJECTED,
    );

    assert.deepEqual(
      {
        physicalFiles: replay.physicalFiles,
        acceptedPhysicalFiles: replay.acceptedPhysicalFiles,
        rejectedPhysicalFiles: replay.rejectedPhysicalFiles,
        logicalSessions: replay.logicalSessions,
        documents: replay.files.reduce((total, file) => total + file.documents, 0),
        formats: replay.formats,
      },
      {
        physicalFiles: historicalCorpusReplayExpectation.physicalFiles,
        acceptedPhysicalFiles: historicalCorpusReplayExpectation.acceptedPhysicalFiles,
        rejectedPhysicalFiles: historicalCorpusReplayExpectation.rejectedPhysicalFiles,
        logicalSessions: historicalCorpusReplayExpectation.logicalSessions,
        documents: historicalCorpusReplayExpectation.documents,
        formats: historicalCorpusReplayExpectation.formats,
      },
    );
    assert.ok(
      rejected.every(
        (file) =>
          file.documents === historicalCorpusReplayExpectation.rejection.documentsPerFile &&
          file.error?.includes(
            `:${historicalCorpusReplayExpectation.rejection.sourceLine}: entry ${historicalCorpusReplayExpectation.rejection.entryId} has missing parent ${historicalCorpusReplayExpectation.rejection.missingParentId}`,
          ),
      ),
    );
    assert.equal(
      sha256ReplayEvidence(JSON.stringify(sourceHashes)),
      historicalCorpusReplayExpectation.sourceSetDigest,
    );
    assert.equal(
      createHistoricalReplayOutcomeDigest(replay),
      historicalCorpusReplayExpectation.outcomeDigest,
    );
  },
);

void test('session import replay refuses the production recall data directory', async () => {
  await assert.rejects(
    () => replaySessionImportCorpus(join(process.env.HOME ?? '', '.pi/agent/recall')),
    /refuses the production recall index directory/u,
  );
});

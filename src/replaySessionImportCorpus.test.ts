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
} from './replaySessionImportCorpus.js';

const HISTORICAL_CORPUS_REPLAY_EXPECTATION = {
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
  outcomeDigest: 'a2cd0ac2a66edf799ad333050f05900005cfc1e784b487af373d70d25cd6f635',
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
    .sort((left, right) => {
      const leftJson = JSON.stringify(left);
      const rightJson = JSON.stringify(right);
      return leftJson < rightJson ? -1 : leftJson > rightJson ? 1 : 0;
    });
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
  const relocatedCorpusRoot = join(directory, 'relocated-corpus');
  await cp(corpusRoot, relocatedCorpusRoot, { recursive: true, preserveTimestamps: true });
  const relocated = await replaySessionImportCorpus(relocatedCorpusRoot);

  assert.deepEqual(second, first);
  assert.deepEqual(
    relocated.files.map(({ sessionPath, importDigest }) => ({ sessionPath, importDigest })),
    first.files.map(({ sessionPath, importDigest }) => ({ sessionPath, importDigest })),
  );
  assert.equal(first.physicalFiles, 4);
  assert.equal(first.acceptedPhysicalFiles, 3);
  assert.equal(first.rejectedPhysicalFiles, 1);
  assert.equal(first.logicalSessions, 4);
  assert.deepEqual(
    first.files
      .filter((file) => file.outcome === SessionImportReplayOutcome.ACCEPTED)
      .map((file) => file.sessionPath),
    ['canonical-unicode-separators.jsonl', 'pi-session-reuse-history.jsonl', 'pi-v1-linear.jsonl'],
  );
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
  assert.deepEqual(fixture, HISTORICAL_CORPUS_REPLAY_EXPECTATION);
});

const HISTORICAL_CORPUS_ROOT = process.env.PI_SESSION_IMPORT_CORPUS_ROOT;
void test(
  'historical corpus replay matches the privacy-safe frozen expectation',
  { skip: HISTORICAL_CORPUS_ROOT ? false : 'PI_SESSION_IMPORT_CORPUS_ROOT is not set' },
  async () => {
    if (!HISTORICAL_CORPUS_ROOT) {
      return;
    }
    const replay = await replaySessionImportCorpus(HISTORICAL_CORPUS_ROOT);
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
        physicalFiles: HISTORICAL_CORPUS_REPLAY_EXPECTATION.physicalFiles,
        acceptedPhysicalFiles: HISTORICAL_CORPUS_REPLAY_EXPECTATION.acceptedPhysicalFiles,
        rejectedPhysicalFiles: HISTORICAL_CORPUS_REPLAY_EXPECTATION.rejectedPhysicalFiles,
        logicalSessions: HISTORICAL_CORPUS_REPLAY_EXPECTATION.logicalSessions,
        documents: HISTORICAL_CORPUS_REPLAY_EXPECTATION.documents,
        formats: HISTORICAL_CORPUS_REPLAY_EXPECTATION.formats,
      },
    );
    assert.ok(
      rejected.every(
        (file) =>
          file.documents === HISTORICAL_CORPUS_REPLAY_EXPECTATION.rejection.documentsPerFile &&
          file.error?.includes(
            `:${HISTORICAL_CORPUS_REPLAY_EXPECTATION.rejection.sourceLine}: entry ${HISTORICAL_CORPUS_REPLAY_EXPECTATION.rejection.entryId} has missing parent ${HISTORICAL_CORPUS_REPLAY_EXPECTATION.rejection.missingParentId}`,
          ),
      ),
    );
    assert.equal(
      sha256ReplayEvidence(JSON.stringify(sourceHashes)),
      HISTORICAL_CORPUS_REPLAY_EXPECTATION.sourceSetDigest,
    );
    assert.equal(
      createHistoricalReplayOutcomeDigest(replay),
      HISTORICAL_CORPUS_REPLAY_EXPECTATION.outcomeDigest,
    );
  },
);

void test('session import replay refuses the production recall data directory', async () => {
  await assert.rejects(
    () => replaySessionImportCorpus(join(process.env.HOME ?? '', '.pi/agent/recall')),
    /Recall test data root overlaps protected path/u,
  );
});

void test('session import replay refuses the environment-configured recall data directory', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'recall-replay-environment-guard-'));
  const dataDirectory = join(directory, 'production-recall');
  await mkdir(dataDirectory);
  const previousConfigPath = process.env.PI_RECALL_CONFIG;
  const previousDataDirectory = process.env.PI_RECALL_DATA_DIRECTORY;
  process.env.PI_RECALL_CONFIG = join(directory, 'missing-recall.json');
  process.env.PI_RECALL_DATA_DIRECTORY = dataDirectory;
  try {
    await assert.rejects(
      () => replaySessionImportCorpus(dataDirectory),
      /Recall test data root overlaps protected path/u,
    );
  } finally {
    if (previousConfigPath === undefined) {
      delete process.env.PI_RECALL_CONFIG;
    } else {
      process.env.PI_RECALL_CONFIG = previousConfigPath;
    }
    if (previousDataDirectory === undefined) {
      delete process.env.PI_RECALL_DATA_DIRECTORY;
    } else {
      process.env.PI_RECALL_DATA_DIRECTORY = previousDataDirectory;
    }
  }
});

void test('session import replay refuses the recall.json-configured data directory', async () => {
  const homeDirectory = await mkdtemp(join(tmpdir(), 'recall-replay-file-guard-'));
  const agentDirectory = join(homeDirectory, '.pi/agent');
  const dataDirectory = join(homeDirectory, 'production-recall');
  await mkdir(agentDirectory, { recursive: true });
  await mkdir(dataDirectory);
  await writeFile(join(agentDirectory, 'recall.json'), JSON.stringify({ dataDirectory }));
  const previousConfigPath = process.env.PI_RECALL_CONFIG;
  const previousDataDirectory = process.env.PI_RECALL_DATA_DIRECTORY;
  process.env.PI_RECALL_CONFIG = join(agentDirectory, 'recall.json');
  delete process.env.PI_RECALL_DATA_DIRECTORY;
  try {
    await assert.rejects(
      () => replaySessionImportCorpus(dataDirectory),
      /Recall test data root overlaps protected path/u,
    );
  } finally {
    if (previousConfigPath === undefined) {
      delete process.env.PI_RECALL_CONFIG;
    } else {
      process.env.PI_RECALL_CONFIG = previousConfigPath;
    }
    if (previousDataDirectory === undefined) {
      delete process.env.PI_RECALL_DATA_DIRECTORY;
    } else {
      process.env.PI_RECALL_DATA_DIRECTORY = previousDataDirectory;
    }
  }
});

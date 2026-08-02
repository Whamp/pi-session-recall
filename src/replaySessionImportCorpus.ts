import { createHash } from 'node:crypto';
import { lstat, readFile, readdir, realpath, stat } from 'node:fs/promises';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { SessionImportFormat, SessionImportReplayOutcome } from './enums.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
  type SessionConversationImport,
} from './session-conversation-index.js';

interface SessionImportSourceSnapshot {
  sha256: string;
  size: number;
  mode: number;
  mtimeMs: number;
  ino: number;
}

/** Aggregate physical-file and logical-session counts for one detected format. */
export interface SessionImportReplayFormatCount {
  physicalFiles: number;
  logicalSessions: number;
}

/** Deterministic replay result for one physical session source. */
export interface SessionImportReplayFileResult {
  sessionPath: string;
  format: SessionImportFormat | null;
  outcome: SessionImportReplayOutcome;
  logicalSessions: number;
  documents: number;
  importDigest: string | null;
  error: string | null;
}

/** Machine-readable result of one guarded, read-only session import corpus replay. */
export interface SessionImportReplayResult {
  corpusRoot: string;
  physicalFiles: number;
  acceptedPhysicalFiles: number;
  rejectedPhysicalFiles: number;
  logicalSessions: number;
  formats: Record<SessionImportFormat, SessionImportReplayFormatCount>;
  files: SessionImportReplayFileResult[];
  replayDigest: string;
}

const REPLAY_TOKENIZER: ConversationTextTokenizer = {
  encodeConversationText(text) {
    const tokenCount = text.match(/\S+/gu)?.length ?? 0;
    return { ids: [...Array(tokenCount).keys()] };
  },
};

function sha256(value: string | Buffer): string {
  return createHash('sha256').update(value).digest('hex');
}

async function listReplaySessionFiles(corpusRoot: string): Promise<string[]> {
  const files: string[] = [];
  async function visit(directory: string): Promise<void> {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
        continue;
      }
      if (entry.name.endsWith('.jsonl') && (entry.isFile() || entry.isSymbolicLink())) {
        const fileStats = await stat(path);
        if (fileStats.isFile()) {
          files.push(path);
        }
      }
    }
  }
  await visit(corpusRoot);
  return files.sort();
}

async function snapshotSessionImportSource(
  sessionPath: string,
): Promise<SessionImportSourceSnapshot> {
  const [bytes, metadata] = await Promise.all([readFile(sessionPath), lstat(sessionPath)]);
  return {
    sha256: sha256(bytes),
    size: metadata.size,
    mode: metadata.mode,
    mtimeMs: metadata.mtimeMs,
    ino: metadata.ino,
  };
}

function assertReplaySourcesUnchanged(
  before: Map<string, SessionImportSourceSnapshot>,
  after: Map<string, SessionImportSourceSnapshot>,
): void {
  for (const [sessionPath, expected] of before) {
    const actual = after.get(sessionPath);
    if (!actual || JSON.stringify(actual) !== JSON.stringify(expected)) {
      throw new Error(
        `Recall session import replay mutated source metadata or bytes: ${sessionPath}`,
      );
    }
  }
}

function overlapsPath(left: string, right: string): boolean {
  const leftToRight = relative(left, right);
  const rightToLeft = relative(right, left);
  const contains = (value: string): boolean =>
    value === '' || (!value.startsWith('..') && !isAbsolute(value));
  return contains(leftToRight) || contains(rightToLeft);
}

function assertReplayCorpusOutsideProduction(
  corpusRoot: string,
  productionRecallDirectory: string,
): void {
  if (overlapsPath(corpusRoot, productionRecallDirectory)) {
    throw new Error(
      `Recall session import replay refuses the production recall index directory: ${corpusRoot}`,
    );
  }
}

async function resolveReplayCorpusRoot(corpusRoot: string): Promise<string> {
  const requestedRoot = resolve(corpusRoot);
  const recallConfig = await loadRecallConversationConfig();
  const productionRecallDirectory = resolve(dirname(recallConfig.databasePath));
  assertReplayCorpusOutsideProduction(requestedRoot, productionRecallDirectory);
  const resolvedRoot = await realpath(requestedRoot);
  assertReplayCorpusOutsideProduction(resolvedRoot, productionRecallDirectory);
  const rootStats = await stat(resolvedRoot);
  if (!rootStats.isDirectory()) {
    throw new Error(`Recall session import replay corpus root is not a directory: ${resolvedRoot}`);
  }
  return resolvedRoot;
}

function createImportDigest(
  format: SessionImportFormat,
  logicalSessions: SessionConversationImport['logicalSessions'],
  chunks: SessionConversationChunk[],
  corpusRoot: string,
  sessionPath: string,
): string {
  return sha256(
    JSON.stringify({
      format,
      logicalSessions,
      documents: chunks.map((chunk) => {
        if (chunk.sessionPath !== sessionPath) {
          throw new Error(
            `Recall session import replay chunk path does not identify its physical session file: ${chunk.sessionPath}`,
          );
        }
        return {
          id: chunk.id,
          checksum: chunk.checksum,
          sessionId: chunk.sessionId.value,
          entryId: chunk.entryId.value,
          parentEntryId: chunk.parentEntryId?.value ?? null,
          contributingEntryIds: chunk.contributingEntryIds.map((entryId) => entryId.value),
          sessionPath: relative(corpusRoot, chunk.sessionPath),
          sourceLineStart: chunk.sourceLineStart,
          sourceLineEnd: chunk.sourceLineEnd,
        };
      }),
    }),
  );
}

function createEmptyFormatCounts(): Record<SessionImportFormat, SessionImportReplayFormatCount> {
  return {
    [SessionImportFormat.CANONICAL_JSONL]: { physicalFiles: 0, logicalSessions: 0 },
    [SessionImportFormat.PI_V1_LINEAR]: { physicalFiles: 0, logicalSessions: 0 },
    [SessionImportFormat.PI_SESSION_REUSE_HISTORY]: {
      physicalFiles: 0,
      logicalSessions: 0,
    },
  };
}

/** Replays session import beneath an explicit corpus root without opening or writing recall storage. */
export async function replaySessionImportCorpus(
  corpusRoot: string,
): Promise<SessionImportReplayResult> {
  const resolvedRoot = await resolveReplayCorpusRoot(corpusRoot);
  const sessionPaths = await listReplaySessionFiles(resolvedRoot);
  const beforeSnapshots = new Map<string, SessionImportSourceSnapshot>();
  for (const sessionPath of sessionPaths) {
    beforeSnapshots.set(sessionPath, await snapshotSessionImportSource(sessionPath));
  }

  const formats = createEmptyFormatCounts();
  const files: SessionImportReplayFileResult[] = [];
  for (const sessionPath of sessionPaths) {
    const relativeSessionPath = relative(resolvedRoot, sessionPath);
    let imported: SessionConversationImport;
    try {
      imported = await readSessionConversationImport(sessionPath, {
        tokenizer: REPLAY_TOKENIZER,
      });
    } catch (error) {
      files.push({
        sessionPath: relativeSessionPath,
        format: null,
        outcome: SessionImportReplayOutcome.REJECTED,
        logicalSessions: 0,
        documents: 0,
        importDigest: null,
        error: error instanceof Error ? error.message : String(error),
      });
      continue;
    }
    const formatCount = formats[imported.format];
    formatCount.physicalFiles += 1;
    formatCount.logicalSessions += imported.logicalSessions.length;
    files.push({
      sessionPath: relativeSessionPath,
      format: imported.format,
      outcome: SessionImportReplayOutcome.ACCEPTED,
      logicalSessions: imported.logicalSessions.length,
      documents: imported.chunks.length,
      importDigest: createImportDigest(
        imported.format,
        imported.logicalSessions,
        imported.chunks,
        resolvedRoot,
        sessionPath,
      ),
      error: null,
    });
  }

  const afterSnapshots = new Map<string, SessionImportSourceSnapshot>();
  for (const sessionPath of sessionPaths) {
    afterSnapshots.set(sessionPath, await snapshotSessionImportSource(sessionPath));
  }
  assertReplaySourcesUnchanged(beforeSnapshots, afterSnapshots);
  const acceptedPhysicalFiles = files.filter(
    (file) => file.outcome === SessionImportReplayOutcome.ACCEPTED,
  ).length;
  const logicalSessions = files.reduce((total, file) => total + file.logicalSessions, 0);
  const replayDigest = sha256(
    JSON.stringify({
      sources: Array.from(beforeSnapshots, ([sessionPath, snapshot]) => ({
        sessionPath: relative(resolvedRoot, sessionPath),
        ...snapshot,
      })),
      files,
    }),
  );
  return {
    corpusRoot: resolvedRoot,
    physicalFiles: sessionPaths.length,
    acceptedPhysicalFiles,
    rejectedPhysicalFiles: sessionPaths.length - acceptedPhysicalFiles,
    logicalSessions,
    formats,
    files,
    replayDigest,
  };
}

function readReplayCorpusRootArgument(argumentsToParse: string[]): string {
  if (argumentsToParse.length !== 2 || argumentsToParse[0] !== '--corpus-root') {
    throw new Error(
      'Recall session import replay usage: npm run replay:session-import -- --corpus-root <path>',
    );
  }
  const corpusRoot = argumentsToParse[1];
  if (!corpusRoot) {
    throw new Error('Recall session import replay requires an explicit corpus root');
  }
  return corpusRoot;
}

async function runSessionImportReplayCommand(): Promise<void> {
  const corpusRoot = readReplayCorpusRootArgument(process.argv.slice(2));
  const result = await replaySessionImportCorpus(corpusRoot);
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

const COMMAND_PATH = process.argv[1] ? resolve(process.argv[1]) : '';
if (COMMAND_PATH === fileURLToPath(import.meta.url)) {
  runSessionImportReplayCommand().catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}

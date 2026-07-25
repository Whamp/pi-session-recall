import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { RecallChunkPolicy } from './recall-index-manifest.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

const conversationIndexStateSchema = Type.Object({
  version: Type.Literal(1),
  sessions: Type.Record(
    Type.String(),
    Type.Object({
      size: Type.Number({ minimum: 0 }),
      mtimeMs: Type.Number({ minimum: 0 }),
      chunks: Type.Array(
        Type.Object({
          id: Type.String(),
        }),
      ),
    }),
  ),
});

interface IndexedSessionState {
  size: number;
  mtimeMs: number;
  chunks: Array<{ id: string }>;
}

interface ConversationIndexState {
  version: 1;
  sessions: Record<string, IndexedSessionState>;
}

/** Progress from scanning session files before indexing changed recall evidence. */
export interface ConversationIndexProgress {
  scannedSessions: number;
  totalSessions: number;
  sessionPath: string;
}

/** Counts and failures produced by one incremental conversation indexing pass. */
export interface ConversationIndexSummary {
  scannedSessions: number;
  indexedSessions: number;
  removedSessions: number;
  cacheHits: number;
  newlyEmbeddedChunks: number;
  embeddingRequestCount: number;
  deletedChunks: number;
  failedSessions: Array<{ sessionPath: string; error: string }>;
}

/** Dependencies and paths required for one incremental conversation indexing pass. */
export interface IncrementalSessionIndexerOptions {
  sessionsDirectory: string;
  statePath: string;
  store: ConversationChunkStore;
  embeddingCache: EmbeddingVectorCache;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy?: RecallChunkPolicy;
  signal?: AbortSignal;
  onProgress?: (progress: ConversationIndexProgress) => void;
}

async function listSessionFiles(directory: string): Promise<string[]> {
  const files: string[] = [];
  async function visit(current: string): Promise<void> {
    let entries;
    try {
      entries = await readdir(current, { withFileTypes: true });
    } catch (error) {
      if (readNodeErrorCode(error) === 'ENOENT') {
        return;
      }
      throw error;
    }
    for (const entry of entries) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        files.push(path);
      }
    }
  }
  await visit(directory);
  return files.sort();
}

async function readConversationIndexState(statePath: string): Promise<ConversationIndexState> {
  try {
    const parsed: unknown = JSON.parse(await readFile(statePath, 'utf8'));
    return Value.Parse(conversationIndexStateSchema, parsed);
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return { version: 1, sessions: {} };
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Recall index state invalid at ${statePath}: ${message}`, { cause: error });
  }
}

async function writeConversationIndexState(
  statePath: string,
  state: ConversationIndexState,
): Promise<void> {
  await mkdir(dirname(statePath), { recursive: true });
  const temporaryPath = `${statePath}.${process.pid}.tmp`;
  await writeFile(temporaryPath, `${JSON.stringify(state)}\n`, 'utf8');
  await rename(temporaryPath, statePath);
}

async function readChangedSessionChunks(
  sessionPath: string,
  previous: IndexedSessionState | undefined,
  tokenizer: ConversationTextTokenizer,
  chunkPolicy?: RecallChunkPolicy,
): Promise<
  | {
      size: number;
      mtimeMs: number;
      chunks: Awaited<ReturnType<typeof readSessionConversationChunks>>;
    }
  | undefined
> {
  const fileStats = await stat(sessionPath);
  if (previous && previous.size === fileStats.size && previous.mtimeMs === fileStats.mtimeMs) {
    return undefined;
  }
  return {
    size: fileStats.size,
    mtimeMs: fileStats.mtimeMs,
    chunks: await readSessionConversationChunks(sessionPath, {
      tokenizer,
      ...(chunkPolicy ? chunkPolicy : {}),
    }),
  };
}

function throwIfIndexingAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new Error('Recall conversation indexing cancelled', { cause: signal.reason });
  }
}

/** Incrementally indexes changed Pi sessions while embedding only dense-searchable evidence. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const state = await readConversationIndexState(options.statePath);
  const sessionFiles = await listSessionFiles(options.sessionsDirectory);
  const liveSessionPaths = new Set(sessionFiles);
  const summary: ConversationIndexSummary = {
    scannedSessions: 0,
    indexedSessions: 0,
    removedSessions: 0,
    cacheHits: 0,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
    deletedChunks: 0,
    failedSessions: [],
  };

  for (const stalePath of Object.keys(state.sessions).filter(
    (path) => !liveSessionPaths.has(path),
  )) {
    throwIfIndexingAborted(options.signal);
    const stale = state.sessions[stalePath];
    const staleIds = stale?.chunks.map((chunk) => chunk.id) ?? [];
    options.store.deleteChunks(staleIds);
    summary.deletedChunks += staleIds.length;
    summary.removedSessions += 1;
    delete state.sessions[stalePath];
  }
  if (summary.removedSessions > 0) {
    await writeConversationIndexState(options.statePath, state);
  }

  let sessionsSinceCheckpoint = 0;
  for (const sessionPath of sessionFiles) {
    throwIfIndexingAborted(options.signal);
    summary.scannedSessions += 1;
    options.onProgress?.({
      scannedSessions: summary.scannedSessions,
      totalSessions: sessionFiles.length,
      sessionPath,
    });

    const previous = state.sessions[sessionPath];
    let changedSession: Awaited<ReturnType<typeof readChangedSessionChunks>>;
    try {
      changedSession = await readChangedSessionChunks(
        sessionPath,
        previous,
        options.tokenizer,
        options.chunkPolicy,
      );
    } catch (error) {
      summary.failedSessions.push({
        sessionPath,
        error: error instanceof Error ? error.message : String(error),
      });
      continue;
    }
    if (!changedSession) {
      continue;
    }

    const { chunks } = changedSession;
    const currentIds = new Set(chunks.map((chunk) => chunk.id));
    const removedIds =
      previous?.chunks.filter((chunk) => !currentIds.has(chunk.id)).map((chunk) => chunk.id) ?? [];
    options.store.deleteChunks(removedIds);
    summary.deletedChunks += removedIds.length;

    for (let start = 0; start < chunks.length; start += 128) {
      const chunkBatch = chunks.slice(start, start + 128);
      const denseChunkBatch = chunkBatch.filter((chunk) => chunk.isDenseSearchable);
      const cacheResult = await options.embeddingCache.resolveEmbeddingVectors(
        denseChunkBatch.map((chunk) => chunk.content),
        options.signal,
      );
      let denseChunkIndex = 0;
      const indexedChunks: IndexedSessionConversationChunk[] = chunkBatch.map((chunk) => {
        if (!chunk.isDenseSearchable) {
          return { ...chunk, isDenseSearchable: false };
        }
        const embedding = cacheResult.vectors[denseChunkIndex];
        denseChunkIndex += 1;
        if (!embedding) {
          throw new Error(`Recall embedding missing for conversation chunk ${chunk.id}`);
        }
        return { ...chunk, isDenseSearchable: true, embedding };
      });
      options.store.upsertChunks(indexedChunks);
      summary.cacheHits += cacheResult.cacheHits;
      summary.newlyEmbeddedChunks += cacheResult.newlyEmbeddedChunks;
      summary.embeddingRequestCount += cacheResult.embeddingRequestCount;
    }

    state.sessions[sessionPath] = {
      size: changedSession.size,
      mtimeMs: changedSession.mtimeMs,
      chunks: chunks.map(({ id }) => ({ id })),
    };
    summary.indexedSessions += 1;
    sessionsSinceCheckpoint += 1;
    if (sessionsSinceCheckpoint >= 100) {
      await writeConversationIndexState(options.statePath, state);
      sessionsSinceCheckpoint = 0;
    }
  }

  if (sessionsSinceCheckpoint > 0) {
    await writeConversationIndexState(options.statePath, state);
  }
  return summary;
}

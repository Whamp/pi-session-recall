import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';
import { SESSION_IMPORT_POLICY_VERSION } from './session-jsonl-import.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

const conversationIndexStateSchema = Type.Object({
  version: Type.Literal(2),
  importPolicyVersion: Type.Literal(SESSION_IMPORT_POLICY_VERSION),
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
  version: 2;
  importPolicyVersion: typeof SESSION_IMPORT_POLICY_VERSION;
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
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
  onProgress?: (progress: ConversationIndexProgress) => void;
}

/** Dependencies for reconciling one active Pi session without scanning sibling files. */
export interface IncrementalConversationSessionIndexerOptions {
  sessionPath: string;
  statePath: string;
  store: ConversationChunkStore;
  embeddingCache: EmbeddingVectorCache;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy?: RecallChunkPolicy;
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
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
      return {
        version: 2,
        importPolicyVersion: SESSION_IMPORT_POLICY_VERSION,
        sessions: {},
      };
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

function createConversationIndexSummary(): ConversationIndexSummary {
  return {
    scannedSessions: 0,
    indexedSessions: 0,
    removedSessions: 0,
    cacheHits: 0,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
    deletedChunks: 0,
    failedSessions: [],
  };
}

function removeIndexedConversationSession(
  state: ConversationIndexState,
  sessionPath: string,
  store: ConversationChunkStore,
  summary: ConversationIndexSummary,
): boolean {
  const previous = state.sessions[sessionPath];
  if (!previous) {
    return false;
  }
  const staleIds = previous.chunks.map((chunk) => chunk.id);
  store.deleteChunks(staleIds);
  summary.deletedChunks += staleIds.length;
  summary.removedSessions += 1;
  delete state.sessions[sessionPath];
  return true;
}

async function indexChangedConversationSessionFile(
  options: Omit<IncrementalSessionIndexerOptions, 'sessionsDirectory' | 'onProgress'>,
  state: ConversationIndexState,
  sessionPath: string,
  summary: ConversationIndexSummary,
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): Promise<boolean> {
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
    if (!previous) {
      return false;
    }
    const staleIds = previous.chunks.map((chunk) => chunk.id);
    options.store.deleteChunks(staleIds);
    summary.deletedChunks += staleIds.length;
    delete state.sessions[sessionPath];
    return true;
  }
  if (!changedSession) {
    return false;
  }

  const { chunks } = changedSession;
  const sessionOrigins = Array.from(new Set(chunks.map((chunk) => chunk.cwd).filter(Boolean)));
  const projectIdentityEntries = await Promise.all(
    sessionOrigins.map(
      async (sessionOrigin): Promise<[string, ResolvedProjectIdentity | null]> => [
        sessionOrigin,
        await resolveSessionProjectIdentity(sessionOrigin),
      ],
    ),
  );
  const projectIdentityBySessionOrigin = new Map(projectIdentityEntries);
  const attributedChunks = chunks.map((chunk) => ({
    ...chunk,
    projectAttribution: projectIdentityBySessionOrigin.get(chunk.cwd) ?? null,
  }));
  const currentIds = new Set(attributedChunks.map((chunk) => chunk.id));
  const removedIds =
    previous?.chunks.filter((chunk) => !currentIds.has(chunk.id)).map((chunk) => chunk.id) ?? [];
  options.store.deleteChunks(removedIds);
  summary.deletedChunks += removedIds.length;

  for (let start = 0; start < attributedChunks.length; start += 128) {
    const chunkBatch = attributedChunks.slice(start, start + 128);
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
  return true;
}

function createSessionProjectIdentityResolver(
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null> {
  const projectIdentityBySessionOrigin = new Map<string, Promise<ResolvedProjectIdentity | null>>();
  return (sessionOrigin) => {
    const existing = projectIdentityBySessionOrigin.get(sessionOrigin);
    if (existing) {
      return existing;
    }
    const resolution = resolveProjectIdentity
      ? resolveProjectIdentity(sessionOrigin)
      : Promise.resolve(null);
    projectIdentityBySessionOrigin.set(sessionOrigin, resolution);
    return resolution;
  };
}

/** Reconciles one active Pi session while embedding only new dense-searchable evidence. */
export async function indexChangedConversationSession(
  options: IncrementalConversationSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const state = await readConversationIndexState(options.statePath);
  const summary = createConversationIndexSummary();
  throwIfIndexingAborted(options.signal);
  summary.scannedSessions = 1;
  try {
    await stat(options.sessionPath);
  } catch (error) {
    if (readNodeErrorCode(error) !== 'ENOENT') {
      throw error;
    }
    if (removeIndexedConversationSession(state, options.sessionPath, options.store, summary)) {
      await writeConversationIndexState(options.statePath, state);
    }
    return summary;
  }
  const changed = await indexChangedConversationSessionFile(
    options,
    state,
    options.sessionPath,
    summary,
    createSessionProjectIdentityResolver(options.resolveProjectIdentity),
  );
  if (changed) {
    await writeConversationIndexState(options.statePath, state);
  }
  return summary;
}

/** Incrementally indexes changed Pi sessions while embedding only dense-searchable evidence. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const state = await readConversationIndexState(options.statePath);
  const sessionFiles = await listSessionFiles(options.sessionsDirectory);
  const liveSessionPaths = new Set(sessionFiles);
  const summary = createConversationIndexSummary();

  for (const stalePath of Object.keys(state.sessions).filter(
    (path) => !liveSessionPaths.has(path),
  )) {
    throwIfIndexingAborted(options.signal);
    removeIndexedConversationSession(state, stalePath, options.store, summary);
  }
  if (summary.removedSessions > 0) {
    await writeConversationIndexState(options.statePath, state);
  }

  const resolveSessionProjectIdentity = createSessionProjectIdentityResolver(
    options.resolveProjectIdentity,
  );
  let sessionsSinceCheckpoint = 0;
  for (const sessionPath of sessionFiles) {
    throwIfIndexingAborted(options.signal);
    summary.scannedSessions += 1;
    options.onProgress?.({
      scannedSessions: summary.scannedSessions,
      totalSessions: sessionFiles.length,
      sessionPath,
    });
    const changed = await indexChangedConversationSessionFile(
      options,
      state,
      sessionPath,
      summary,
      resolveSessionProjectIdentity,
    );
    if (!changed) {
      continue;
    }
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

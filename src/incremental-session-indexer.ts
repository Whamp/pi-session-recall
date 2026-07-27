import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type {
  RecallDiagnosticsClock,
  RecallLiveSessionDiagnosticMetrics,
} from './recall-operation-diagnostics.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
} from './session-conversation-index.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
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
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics;
  diagnosticsClock?: RecallDiagnosticsClock;
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
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics;
  diagnosticsClock?: RecallDiagnosticsClock;
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
  chunkPolicy: RecallChunkPolicy | undefined,
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): Promise<
  | { changed: false; size: number }
  | {
      changed: true;
      size: number;
      mtimeMs: number;
      chunks: Awaited<ReturnType<typeof readSessionConversationChunks>>;
    }
> {
  const startedAtMilliseconds = diagnosticsClock?.monotonicMilliseconds();
  try {
    const fileStats = await stat(sessionPath);
    const changed =
      !previous || previous.size !== fileStats.size || previous.mtimeMs !== fileStats.mtimeMs;
    if (diagnosticMetrics) {
      diagnosticMetrics.sourceByteSize = fileStats.size;
      diagnosticMetrics.changed = changed;
      diagnosticMetrics.skipped = !changed;
    }
    if (!changed) {
      return { changed: false, size: fileStats.size };
    }
    return {
      changed: true,
      size: fileStats.size,
      mtimeMs: fileStats.mtimeMs,
      chunks: await readSessionConversationChunks(sessionPath, {
        tokenizer,
        ...(chunkPolicy ? chunkPolicy : {}),
      }),
    };
  } finally {
    if (diagnosticMetrics && diagnosticsClock && startedAtMilliseconds !== undefined) {
      diagnosticMetrics.physicalSessionPreparationMilliseconds += Math.max(
        diagnosticsClock.monotonicMilliseconds() - startedAtMilliseconds,
        0,
      );
    }
  }
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

function runTimedDatabaseWrite<T>(
  operation: () => T,
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): T {
  const startedAtMilliseconds = diagnosticsClock?.monotonicMilliseconds();
  try {
    return operation();
  } finally {
    if (diagnosticMetrics && diagnosticsClock && startedAtMilliseconds !== undefined) {
      diagnosticMetrics.databaseWriteMilliseconds += Math.max(
        diagnosticsClock.monotonicMilliseconds() - startedAtMilliseconds,
        0,
      );
    }
  }
}

async function writeConversationIndexStateWithDiagnostics(
  statePath: string,
  state: ConversationIndexState,
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): Promise<void> {
  const startedAtMilliseconds = diagnosticsClock?.monotonicMilliseconds();
  try {
    await writeConversationIndexState(statePath, state);
  } finally {
    if (diagnosticMetrics && diagnosticsClock && startedAtMilliseconds !== undefined) {
      diagnosticMetrics.indexStateCheckpointMilliseconds += Math.max(
        diagnosticsClock.monotonicMilliseconds() - startedAtMilliseconds,
        0,
      );
    }
  }
}

function removeIndexedConversationSession(
  state: ConversationIndexState,
  sessionPath: string,
  store: ConversationChunkStore,
  summary: ConversationIndexSummary,
  diagnosticMetrics?: RecallLiveSessionDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): boolean {
  const previous = state.sessions[sessionPath];
  if (!previous) {
    return false;
  }
  const staleIds = previous.chunks.map((chunk) => chunk.id);
  runTimedDatabaseWrite(() => store.deleteChunks(staleIds), diagnosticMetrics, diagnosticsClock);
  summary.deletedChunks += staleIds.length;
  if (diagnosticMetrics) {
    diagnosticMetrics.deletedDocumentCount += staleIds.length;
  }
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
      options.diagnosticMetrics,
      options.diagnosticsClock,
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
    runTimedDatabaseWrite(
      () => options.store.deleteChunks(staleIds),
      options.diagnosticMetrics,
      options.diagnosticsClock,
    );
    summary.deletedChunks += staleIds.length;
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.deletedDocumentCount += staleIds.length;
    }
    delete state.sessions[sessionPath];
    return true;
  }
  if (!changedSession.changed) {
    return false;
  }

  const { chunks } = changedSession;
  const sessionOrigins = Array.from(new Set(chunks.map((chunk) => chunk.cwd).filter(Boolean)));
  const projectIdentityStartedAtMilliseconds = options.diagnosticsClock?.monotonicMilliseconds();
  let projectIdentityEntries: Array<[string, ResolvedProjectIdentity | null]>;
  try {
    projectIdentityEntries = await Promise.all(
      sessionOrigins.map(
        async (sessionOrigin): Promise<[string, ResolvedProjectIdentity | null]> => [
          sessionOrigin,
          await resolveSessionProjectIdentity(sessionOrigin),
        ],
      ),
    );
  } finally {
    if (
      options.diagnosticMetrics &&
      options.diagnosticsClock &&
      projectIdentityStartedAtMilliseconds !== undefined
    ) {
      options.diagnosticMetrics.projectIdentityResolutionMilliseconds += Math.max(
        options.diagnosticsClock.monotonicMilliseconds() - projectIdentityStartedAtMilliseconds,
        0,
      );
    }
  }
  const projectIdentityBySessionOrigin = new Map(projectIdentityEntries);
  const attributedChunks = chunks.map((chunk) => ({
    ...chunk,
    projectAttribution: projectIdentityBySessionOrigin.get(chunk.cwd) ?? null,
  }));
  const currentIds = new Set(attributedChunks.map((chunk) => chunk.id));
  const removedIds =
    previous?.chunks.filter((chunk) => !currentIds.has(chunk.id)).map((chunk) => chunk.id) ?? [];
  runTimedDatabaseWrite(
    () => options.store.deleteChunks(removedIds),
    options.diagnosticMetrics,
    options.diagnosticsClock,
  );
  summary.deletedChunks += removedIds.length;
  if (options.diagnosticMetrics) {
    options.diagnosticMetrics.deletedDocumentCount += removedIds.length;
  }

  for (let start = 0; start < attributedChunks.length; start += 128) {
    const chunkBatch = attributedChunks.slice(start, start + 128);
    const denseChunkBatch = chunkBatch.filter((chunk) => chunk.isDenseSearchable);
    const cacheResolutionStartedAtMilliseconds = options.diagnosticsClock?.monotonicMilliseconds();
    const diagnosticMetrics = options.diagnosticMetrics;
    const serverRequestMillisecondsBefore =
      diagnosticMetrics?.embeddingServerRequestMilliseconds ?? 0;
    let cacheResult: Awaited<ReturnType<EmbeddingVectorCache['resolveEmbeddingVectors']>>;
    try {
      cacheResult = await options.embeddingCache.resolveEmbeddingVectors(
        denseChunkBatch.map((chunk) => chunk.content),
        options.signal,
        diagnosticMetrics
          ? {
              recordEmbeddingServerRequest(milliseconds) {
                diagnosticMetrics.embeddingServerRequestMilliseconds += milliseconds;
                diagnosticMetrics.embeddingRequestCount += 1;
              },
            }
          : undefined,
      );
    } catch (error) {
      if (
        options.diagnosticMetrics &&
        options.diagnosticsClock &&
        cacheResolutionStartedAtMilliseconds !== undefined
      ) {
        const totalResolutionMilliseconds = Math.max(
          options.diagnosticsClock.monotonicMilliseconds() - cacheResolutionStartedAtMilliseconds,
          0,
        );
        const serverRequestMilliseconds =
          options.diagnosticMetrics.embeddingServerRequestMilliseconds -
          serverRequestMillisecondsBefore;
        options.diagnosticMetrics.embeddingCacheResolutionMilliseconds += Math.max(
          totalResolutionMilliseconds - serverRequestMilliseconds,
          0,
        );
      }
      throw error;
    }
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.embeddingCacheResolutionMilliseconds +=
        cacheResult.embeddingCacheResolutionMilliseconds;
      options.diagnosticMetrics.cacheHitCount += cacheResult.cacheHits;
      options.diagnosticMetrics.newEmbeddingCount += cacheResult.newlyEmbeddedChunks;
    }
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
    runTimedDatabaseWrite(
      () => options.store.upsertChunks(indexedChunks),
      options.diagnosticMetrics,
      options.diagnosticsClock,
    );
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.upsertedDocumentCount += indexedChunks.length;
    }
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
    const previous = state.sessions[options.sessionPath];
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.sourceByteSize = 0;
      options.diagnosticMetrics.changed = previous !== undefined;
      options.diagnosticMetrics.skipped = previous === undefined;
    }
    if (
      removeIndexedConversationSession(
        state,
        options.sessionPath,
        options.store,
        summary,
        options.diagnosticMetrics,
        options.diagnosticsClock,
      )
    ) {
      await writeConversationIndexStateWithDiagnostics(
        options.statePath,
        state,
        options.diagnosticMetrics,
        options.diagnosticsClock,
      );
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
    await writeConversationIndexStateWithDiagnostics(
      options.statePath,
      state,
      options.diagnosticMetrics,
      options.diagnosticsClock,
    );
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

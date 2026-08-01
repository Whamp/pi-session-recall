import { mkdir, readdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { SESSION_IMPORT_POLICY_VERSION } from './import-session-jsonl.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationChunks,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';
import type {
  ConversationChunkStore,
  IndexedSessionConversationChunk,
} from './zvec-conversation-store.js';

const conversationIndexStateSchema = Type.Object({
  version: Type.Literal(3),
  importPolicyVersion: Type.Literal(SESSION_IMPORT_POLICY_VERSION),
  sessions: Type.Record(
    Type.String(),
    Type.Object({
      size: Type.Number({ minimum: 0 }),
      mtimeMs: Type.Number({ minimum: 0 }),
      chunks: Type.Array(Type.Object({ id: Type.String() })),
    }),
  ),
});

interface IndexedSessionState {
  size: number;
  mtimeMs: number;
  chunks: Array<{ id: string }>;
}

interface ConversationIndexState {
  version: 3;
  importPolicyVersion: typeof SESSION_IMPORT_POLICY_VERSION;
  sessions: Record<string, IndexedSessionState>;
}

/** Progress from scanning physical session files during explicit `psr` maintenance. */
export interface ConversationIndexProgress {
  scannedSessions: number;
  totalSessions: number;
  sessionPath: string;
}

/** Counts and source failures from one explicit incremental indexing pass. */
export interface ConversationIndexSummary {
  scannedSessions: number;
  indexedSessions: number;
  removedSessions: number;
  reusedVectors: number;
  newlyEmbeddedChunks: number;
  embeddingRequestCount: number;
  deletedChunks: number;
  failedSessions: Array<{ sessionPath: string; error: string }>;
}

/** Dependencies required to update the one zvec collection from Pi session JSONL files. */
export interface IncrementalSessionIndexerOptions {
  sessionsDirectory: string;
  statePath: string;
  store: ConversationChunkStore;
  embeddingProvider: RecallEmbeddingProvider;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy: RecallChunkPolicy;
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
  onProgress?: (progress: ConversationIndexProgress) => void;
}

async function listRecallSessionFiles(directory: string): Promise<string[]> {
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
        version: 3,
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
    reusedVectors: 0,
    newlyEmbeddedChunks: 0,
    embeddingRequestCount: 0,
    deletedChunks: 0,
    failedSessions: [],
  };
}

function removeIndexedSession(
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

function createSessionProjectIdentityResolver(
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null> {
  const resolutions = new Map<string, Promise<ResolvedProjectIdentity | null>>();
  return (sessionOrigin) => {
    const existing = resolutions.get(sessionOrigin);
    if (existing) {
      return existing;
    }
    const resolution = resolveProjectIdentity
      ? resolveProjectIdentity(sessionOrigin)
      : Promise.resolve(null);
    resolutions.set(sessionOrigin, resolution);
    return resolution;
  };
}

async function attributeRecallChunksToProjects(
  chunks: readonly SessionConversationChunk[],
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): Promise<SessionConversationChunk[]> {
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
  return chunks.map((chunk) => ({
    ...chunk,
    projectAttribution: projectIdentityBySessionOrigin.get(chunk.cwd) ?? null,
  }));
}

async function prepareChangedRecallRows(
  chunks: readonly SessionConversationChunk[],
  store: ConversationChunkStore,
  embeddingProvider: RecallEmbeddingProvider,
  summary: ConversationIndexSummary,
  signal?: AbortSignal,
): Promise<IndexedSessionConversationChunk[]> {
  const changedRows: IndexedSessionConversationChunk[] = [];
  for (let start = 0; start < chunks.length; start += 128) {
    throwIfIndexingAborted(signal);
    const batch = chunks.slice(start, start + 128);
    const ids = batch.map((chunk) => chunk.id);
    const existingChunks = store.fetchConversationChunks(ids);
    const existingVectors = store.fetchVectors(
      batch.filter((chunk) => chunk.isDenseSearchable).map((chunk) => chunk.id),
    );
    const rowsNeedingWrite = batch.filter((chunk) => {
      const existing = existingChunks.get(chunk.id);
      return (
        !existing ||
        existing.checksum !== chunk.checksum ||
        (chunk.isDenseSearchable && !existingVectors.has(chunk.id))
      );
    });
    summary.reusedVectors += batch.filter(
      (chunk) =>
        chunk.isDenseSearchable &&
        existingChunks.get(chunk.id)?.checksum === chunk.checksum &&
        existingVectors.has(chunk.id),
    ).length;
    const denseRows = rowsNeedingWrite.filter((chunk) => chunk.isDenseSearchable);
    const embeddings =
      denseRows.length === 0
        ? []
        : await embeddingProvider.embedDocuments(
            denseRows.map((chunk) => chunk.content),
            signal,
          );
    if (denseRows.length > 0) {
      summary.embeddingRequestCount += 1;
      summary.newlyEmbeddedChunks += denseRows.length;
    }
    let denseIndex = 0;
    for (const chunk of rowsNeedingWrite) {
      if (!chunk.isDenseSearchable) {
        changedRows.push({ ...chunk, isDenseSearchable: false });
        continue;
      }
      const embedding = embeddings[denseIndex];
      denseIndex += 1;
      if (!embedding) {
        throw new Error(`Recall embedding missing for conversation chunk ${chunk.id}`);
      }
      changedRows.push({ ...chunk, isDenseSearchable: true, embedding });
    }
  }
  return changedRows;
}

async function indexChangedRecallSessionFile(
  options: IncrementalSessionIndexerOptions,
  state: ConversationIndexState,
  sessionPath: string,
  summary: ConversationIndexSummary,
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): Promise<boolean> {
  const previous = state.sessions[sessionPath];
  const fileStats = await stat(sessionPath);
  if (previous && previous.size === fileStats.size && previous.mtimeMs === fileStats.mtimeMs) {
    return false;
  }

  let chunks: SessionConversationChunk[];
  try {
    chunks = await readSessionConversationChunks(sessionPath, {
      tokenizer: options.tokenizer,
      ...options.chunkPolicy,
    });
  } catch (error) {
    summary.failedSessions.push({
      sessionPath,
      error: error instanceof Error ? error.message : String(error),
    });
    if (previous) {
      removeIndexedSession(state, sessionPath, options.store, summary);
      return true;
    }
    return false;
  }

  const attributedChunks = await attributeRecallChunksToProjects(
    chunks,
    resolveSessionProjectIdentity,
  );
  const currentIds = new Set(attributedChunks.map((chunk) => chunk.id));
  const removedIds =
    previous?.chunks.filter((chunk) => !currentIds.has(chunk.id)).map((chunk) => chunk.id) ?? [];
  const changedRows = await prepareChangedRecallRows(
    attributedChunks,
    options.store,
    options.embeddingProvider,
    summary,
    options.signal,
  );

  options.store.deleteChunks(removedIds);
  for (let start = 0; start < changedRows.length; start += 128) {
    options.store.upsertChunks(changedRows.slice(start, start + 128));
  }
  summary.deletedChunks += removedIds.length;
  state.sessions[sessionPath] = {
    size: fileStats.size,
    mtimeMs: fileStats.mtimeMs,
    chunks: attributedChunks.map(({ id }) => ({ id })),
  };
  summary.indexedSessions += 1;
  return true;
}

/** Incrementally updates one zvec collection from changed, new, and removed session files. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const state = await readConversationIndexState(options.statePath);
  const sessionFiles = await listRecallSessionFiles(options.sessionsDirectory);
  const liveSessionPaths = new Set(sessionFiles);
  const summary = createConversationIndexSummary();

  for (const stalePath of Object.keys(state.sessions).filter(
    (path) => !liveSessionPaths.has(path),
  )) {
    throwIfIndexingAborted(options.signal);
    removeIndexedSession(state, stalePath, options.store, summary);
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
    const stateChanged = await indexChangedRecallSessionFile(
      options,
      state,
      sessionPath,
      summary,
      resolveSessionProjectIdentity,
    );
    if (stateChanged) {
      sessionsSinceCheckpoint += 1;
    }
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

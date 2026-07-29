import { mkdir, readFile, rename, stat, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import type { EmbeddingVectorCache } from './embedding-vector-cache.js';
import { RecallDiagnosticErrorCategory, RecallDiagnosticStatus } from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { listRecallConversationSessionFiles } from './recall-conversation-corpus.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import {
  accumulateRecallIndexMetrics,
  createRecallIndexMetrics,
  type RecallDiagnosticsClock,
  type RecallIndexDiagnosticMetrics,
  type RecallPhysicalSessionDiagnostic,
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

/** Validated session and document identity counts persisted by one index generation. */
export interface RecallConversationIndexStateSummary {
  sessionCount: number;
  documentIds: string[];
}

/** Reads validated session state for pre-activation generation conformance checks. */
export async function readRecallConversationIndexStateSummary(
  statePath: string,
): Promise<RecallConversationIndexStateSummary> {
  const state = await readConversationIndexState(statePath);
  const documentIds = Object.values(state.sessions).flatMap((session) =>
    session.chunks.map((chunk) => chunk.id),
  );
  if (new Set(documentIds).size !== documentIds.length) {
    throw new Error(`Recall index state contains duplicate document IDs at ${statePath}`);
  }
  return {
    sessionCount: Object.keys(state.sessions).length,
    documentIds,
  };
}

/** Progress from scanning session files before indexing changed recall evidence. */
export interface ConversationIndexProgress {
  scannedSessions: number;
  totalSessions: number;
  sessionPath: string;
}

/** Latest physical session durably represented by one index-generation checkpoint. */
export interface ConversationIndexCheckpoint {
  checkpointedSessions: number;
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
  onCheckpoint?: (checkpoint: ConversationIndexCheckpoint) => void;
  diagnosticMetrics?: RecallIndexDiagnosticMetrics;
  diagnosticsClock?: RecallDiagnosticsClock;
  onPhysicalSessionCheck?: (completion: RecallPhysicalSessionDiagnostic) => void;
  eligibleContributorEntryIdsBySessionPath?: ReadonlyMap<
    string,
    ReadonlyMap<string, ReadonlySet<string>>
  >;
  /**
   * When false, skips immediate removal of stale-path sessions and suppresses
   * error-path chunk deletion for previously indexed sessions, deferring to confirmed
   * deletion via the incremental worker. Defaults to true (rebuild/legacy behavior).
   */
  retireMissingSourcesImmediately?: boolean;
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
  diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
  eligibleContributorEntryIdsByLogicalSessionId?: ReadonlyMap<string, ReadonlySet<string>>,
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
        ...(eligibleContributorEntryIdsByLogicalSessionId
          ? { eligibleContributorEntryIdsByLogicalSessionId }
          : {}),
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

interface PhysicalSessionIndexOutcome {
  stateChanged: boolean;
  failed: boolean;
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

async function runTimedDatabaseWrite<T>(
  operation: () => Promise<T>,
  diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): Promise<T> {
  const startedAtMilliseconds = diagnosticsClock?.monotonicMilliseconds();
  try {
    return await operation();
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
  diagnosticMetrics?: RecallIndexDiagnosticMetrics,
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

async function removeIndexedConversationSession(
  state: ConversationIndexState,
  sessionPath: string,
  store: ConversationChunkStore,
  summary: ConversationIndexSummary,
  diagnosticMetrics?: RecallIndexDiagnosticMetrics,
  diagnosticsClock?: RecallDiagnosticsClock,
): Promise<boolean> {
  const previous = state.sessions[sessionPath];
  if (!previous) {
    return false;
  }
  const staleIds = previous.chunks.map((chunk) => chunk.id);
  await runTimedDatabaseWrite(
    () => store.deleteChunks(staleIds),
    diagnosticMetrics,
    diagnosticsClock,
  );
  summary.deletedChunks += staleIds.length;
  if (diagnosticMetrics) {
    diagnosticMetrics.deletedDocumentCount += staleIds.length;
  }
  summary.removedSessions += 1;
  delete state.sessions[sessionPath];
  return true;
}

async function indexChangedConversationSessionFile(
  options: Omit<
    IncrementalSessionIndexerOptions,
    'sessionsDirectory' | 'onProgress' | 'onPhysicalSessionCheck'
  >,
  state: ConversationIndexState,
  sessionPath: string,
  summary: ConversationIndexSummary,
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
): Promise<PhysicalSessionIndexOutcome> {
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
      options.eligibleContributorEntryIdsBySessionPath?.get(sessionPath),
    );
  } catch (error) {
    summary.failedSessions.push({
      sessionPath,
      error: error instanceof Error ? error.message : String(error),
    });
    if (!previous) {
      return { stateChanged: false, failed: true };
    }
    if (options.retireMissingSourcesImmediately === false) {
      return { stateChanged: false, failed: true };
    }
    const staleIds = previous.chunks.map((chunk) => chunk.id);
    await runTimedDatabaseWrite(
      () => options.store.deleteChunks(staleIds),
      options.diagnosticMetrics,
      options.diagnosticsClock,
    );
    summary.deletedChunks += staleIds.length;
    if (options.diagnosticMetrics) {
      options.diagnosticMetrics.deletedDocumentCount += staleIds.length;
    }
    delete state.sessions[sessionPath];
    return { stateChanged: true, failed: true };
  }
  if (!changedSession.changed) {
    return { stateChanged: false, failed: false };
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
  await runTimedDatabaseWrite(
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
    await runTimedDatabaseWrite(
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
  return { stateChanged: true, failed: false };
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

interface PhysicalSessionDiagnosticOutcome {
  indexedSessionCount: number;
  removedSessionCount: number;
  failedSessionCount: number;
}

async function runPhysicalSessionCheck<T extends PhysicalSessionDiagnosticOutcome>(options: {
  indexerOptions: IncrementalSessionIndexerOptions;
  sessionPath: string;
  performPhysicalSessionCheck: (diagnosticMetrics?: RecallIndexDiagnosticMetrics) => T | Promise<T>;
}): Promise<T> {
  const { indexerOptions } = options;
  const physicalSessionMetrics =
    indexerOptions.diagnosticMetrics || indexerOptions.onPhysicalSessionCheck
      ? createRecallIndexMetrics()
      : undefined;
  const startedAtMilliseconds = physicalSessionMetrics
    ? indexerOptions.diagnosticsClock?.monotonicMilliseconds()
    : undefined;

  function completePhysicalSessionCheck(
    outcome: PhysicalSessionDiagnosticOutcome,
    status: RecallDiagnosticStatus,
    errorCategory?: RecallDiagnosticErrorCategory,
  ): void {
    if (!physicalSessionMetrics) {
      return;
    }
    physicalSessionMetrics.indexedSessionCount = outcome.indexedSessionCount;
    physicalSessionMetrics.removedSessionCount = outcome.removedSessionCount;
    physicalSessionMetrics.failedSessionCount = outcome.failedSessionCount;
    if (indexerOptions.diagnosticMetrics) {
      accumulateRecallIndexMetrics(indexerOptions.diagnosticMetrics, physicalSessionMetrics);
    }
    const elapsedMilliseconds =
      indexerOptions.diagnosticsClock && startedAtMilliseconds !== undefined
        ? Math.max(
            indexerOptions.diagnosticsClock.monotonicMilliseconds() - startedAtMilliseconds,
            0,
          )
        : 0;
    indexerOptions.onPhysicalSessionCheck?.({
      sessionPath: options.sessionPath,
      status,
      metrics: physicalSessionMetrics,
      elapsedMilliseconds,
      indexedSessionCount: outcome.indexedSessionCount,
      removedSessionCount: outcome.removedSessionCount,
      failedSessionCount: outcome.failedSessionCount,
      ...(errorCategory ? { errorCategory } : {}),
    });
  }

  let outcome: T;
  try {
    outcome = await options.performPhysicalSessionCheck(physicalSessionMetrics);
  } catch (error) {
    const cancelled = indexerOptions.signal?.aborted === true;
    const status = cancelled ? RecallDiagnosticStatus.CANCELLED : RecallDiagnosticStatus.FAILED;
    const errorCategory = cancelled
      ? RecallDiagnosticErrorCategory.OPERATION_CANCELLED
      : RecallDiagnosticErrorCategory.OPERATION_FAILED;
    completePhysicalSessionCheck(
      {
        indexedSessionCount: 0,
        removedSessionCount: 0,
        failedSessionCount: cancelled ? 0 : 1,
      },
      status,
      errorCategory,
    );
    throw error;
  }
  const failed = outcome.failedSessionCount > 0;
  completePhysicalSessionCheck(
    outcome,
    failed ? RecallDiagnosticStatus.FAILED : RecallDiagnosticStatus.SUCCEEDED,
    failed ? RecallDiagnosticErrorCategory.OPERATION_FAILED : undefined,
  );
  return outcome;
}

async function scanPhysicalSessionFiles(options: IncrementalSessionIndexerOptions): Promise<{
  state: ConversationIndexState;
  sessionFiles: string[];
}> {
  const scanStartedAtMilliseconds = options.diagnosticsClock?.monotonicMilliseconds();
  try {
    const liveSessionFiles = await listRecallConversationSessionFiles(options.sessionsDirectory);
    return {
      state: await readConversationIndexState(options.statePath),
      sessionFiles:
        options.eligibleContributorEntryIdsBySessionPath === undefined
          ? liveSessionFiles
          : liveSessionFiles.filter((sessionPath) =>
              options.eligibleContributorEntryIdsBySessionPath?.has(sessionPath),
            ),
    };
  } finally {
    if (
      options.diagnosticMetrics &&
      options.diagnosticsClock &&
      scanStartedAtMilliseconds !== undefined
    ) {
      options.diagnosticMetrics.physicalSessionScanMilliseconds += Math.max(
        options.diagnosticsClock.monotonicMilliseconds() - scanStartedAtMilliseconds,
        0,
      );
    }
  }
}

/** Incrementally indexes changed Pi sessions while embedding only dense-searchable evidence. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const { state, sessionFiles } = await scanPhysicalSessionFiles(options);
  const liveSessionPaths = new Set(sessionFiles);
  const summary = createConversationIndexSummary();

  if (options.retireMissingSourcesImmediately !== false) {
    for (const stalePath of Object.keys(state.sessions).filter(
      (path) => !liveSessionPaths.has(path),
    )) {
      throwIfIndexingAborted(options.signal);
      await runPhysicalSessionCheck({
        indexerOptions: options,
        sessionPath: stalePath,
        async performPhysicalSessionCheck(physicalSessionMetrics) {
          if (physicalSessionMetrics) {
            physicalSessionMetrics.sourceByteSize = 0;
            physicalSessionMetrics.changed = true;
            physicalSessionMetrics.skipped = false;
          }
          const removed = await removeIndexedConversationSession(
            state,
            stalePath,
            options.store,
            summary,
            physicalSessionMetrics,
            options.diagnosticsClock,
          );
          return {
            indexedSessionCount: 0,
            removedSessionCount: removed ? 1 : 0,
            failedSessionCount: 0,
          };
        },
      });
    }
    if (summary.removedSessions > 0) {
      await writeConversationIndexStateWithDiagnostics(
        options.statePath,
        state,
        options.diagnosticMetrics,
        options.diagnosticsClock,
      );
    }
  }

  const resolveSessionProjectIdentity = createSessionProjectIdentityResolver(
    options.resolveProjectIdentity,
  );
  for (const sessionPath of sessionFiles) {
    throwIfIndexingAborted(options.signal);
    summary.scannedSessions += 1;
    options.onProgress?.({
      scannedSessions: summary.scannedSessions,
      totalSessions: sessionFiles.length,
      sessionPath,
    });
    throwIfIndexingAborted(options.signal);
    const outcome = await runPhysicalSessionCheck({
      indexerOptions: options,
      sessionPath,
      async performPhysicalSessionCheck(physicalSessionMetrics) {
        if (physicalSessionMetrics) {
          physicalSessionMetrics.scannedSessionCount = 1;
        }
        const indexOutcome = await indexChangedConversationSessionFile(
          {
            ...options,
            ...(physicalSessionMetrics ? { diagnosticMetrics: physicalSessionMetrics } : {}),
          },
          state,
          sessionPath,
          summary,
          resolveSessionProjectIdentity,
        );
        return {
          ...indexOutcome,
          indexedSessionCount: indexOutcome.stateChanged && !indexOutcome.failed ? 1 : 0,
          removedSessionCount: 0,
          failedSessionCount: indexOutcome.failed ? 1 : 0,
        };
      },
    });
    if (outcome.stateChanged) {
      await writeConversationIndexStateWithDiagnostics(
        options.statePath,
        state,
        options.diagnosticMetrics,
        options.diagnosticsClock,
      );
    }
    options.onCheckpoint?.({
      checkpointedSessions: summary.scannedSessions,
      totalSessions: sessionFiles.length,
      sessionPath,
    });
  }

  await writeConversationIndexStateWithDiagnostics(
    options.statePath,
    state,
    options.diagnosticMetrics,
    options.diagnosticsClock,
  );
  return summary;
}

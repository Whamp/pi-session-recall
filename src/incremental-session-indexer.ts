import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { listRecallSessionFiles } from './listRecallSessionFiles.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';
import { openSqliteRecallDatabase, type SqliteRecallDatabase } from './sqlite-recall-database.js';

/** Read-only active-database vectors whose embedding profile the caller already verified. */
export interface RecallVectorReuseReader {
  fetchDocuments(ids: string[]): Map<string, SessionConversationChunk>;
  fetchVectors(ids: string[]): Map<string, number[]>;
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

/** Dependencies required to update recall storage from Pi session JSONL files. */
export interface IncrementalSessionIndexerOptions {
  sessionsDirectory: string;
  /** Explicit physical session files for bounded maintenance without pruning unselected sessions. */
  selectedPhysicalSessionPaths?: readonly string[];
  /** Already-open unified database owned by the service for this maintenance pass. */
  database?: SqliteRecallDatabase;
  /** Path used only when the indexer itself owns the database handle. */
  databasePath?: string;
  /** Read-only active database used only during staged embedding reuse. */
  vectorReuseReader?: RecallVectorReuseReader;
  embeddingProvider: RecallEmbeddingProvider;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy: RecallChunkPolicy;
  ignoredPhysicalSessionPaths: ReadonlySet<string>;
  resolveProjectIdentity?: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
  rebuild?: boolean;
  onProgress?: (event: RecallIndexProgressEvent) => void;
}

interface PlannedPhysicalSessionFile {
  sessionPath: string;
  size: number;
  mtimeMs: number;
  change: 'new' | 'changed';
}

interface MaintenanceWorksetPlan {
  discoveredFiles: number;
  filesToIndex: PlannedPhysicalSessionFile[];
  missingFiles: string[];
  ignoredIndexedFiles: string[];
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

function emitMaintenanceWorksetProgress(
  options: IncrementalSessionIndexerOptions,
  summary: ConversationIndexSummary,
  completedFiles: number,
  totalFiles: number,
  sessionPath: string,
): void {
  options.onProgress?.({
    kind: 'indexing-maintenance-workset',
    completedFiles,
    totalFiles,
    sessionPath,
    indexedSessions: summary.indexedSessions,
    newlyEmbeddedDocuments: summary.newlyEmbeddedChunks,
    reusedVectors: summary.reusedVectors,
    deletedDocuments: summary.deletedChunks,
    failedSessions: summary.failedSessions.length,
  });
}

function removeIndexedSession(
  database: SqliteRecallDatabase,
  sessionPath: string,
  summary: ConversationIndexSummary,
): boolean {
  const previous = database.readPhysicalSessionState(sessionPath);
  if (!previous || !database.deletePhysicalSession(sessionPath)) {
    return false;
  }
  summary.deletedChunks += previous.documentIds.length;
  summary.removedSessions += 1;
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

async function prepareChangedRecallEmbeddingMap(
  chunks: readonly (SessionConversationChunk & { isDenseSearchable: true })[],
  database: SqliteRecallDatabase,
  embeddingProvider: RecallEmbeddingProvider,
  vectorReuseReader: RecallVectorReuseReader | undefined,
  summary: ConversationIndexSummary,
  onBatchPrepared: () => void,
  signal?: AbortSignal,
): Promise<Map<string, readonly number[]>> {
  const embeddingByDocumentId = new Map<string, readonly number[]>();
  for (let start = 0; start < chunks.length; start += 128) {
    throwIfIndexingAborted(signal);
    const batch = chunks.slice(start, start + 128);
    const ids = batch.map((chunk) => chunk.id);
    const currentDocuments = database.fetchDenseDocuments(ids);
    const currentVectors = database.fetchDenseVectors(ids);
    const chunksMissingCurrentVector = batch.filter((chunk) => {
      const currentDocument = currentDocuments.get(chunk.id);
      const currentVector = currentVectors.get(chunk.id);
      if (currentDocument?.checksum !== chunk.checksum || !currentVector) {
        return true;
      }
      embeddingByDocumentId.set(chunk.id, currentVector);
      summary.reusedVectors += 1;
      return false;
    });

    const legacyIds = chunksMissingCurrentVector.map((chunk) => chunk.id);
    const legacyDocuments = vectorReuseReader?.fetchDocuments(legacyIds);
    const legacyVectors = vectorReuseReader?.fetchVectors(legacyIds);
    const chunksNeedingEmbedding = chunksMissingCurrentVector.filter((chunk) => {
      const legacyVector = legacyVectors?.get(chunk.id);
      if (legacyDocuments?.get(chunk.id)?.checksum !== chunk.checksum || !legacyVector) {
        return true;
      }
      embeddingByDocumentId.set(chunk.id, legacyVector);
      summary.reusedVectors += 1;
      return false;
    });

    const embeddings =
      chunksNeedingEmbedding.length === 0
        ? []
        : await embeddingProvider.embedDocuments(
            chunksNeedingEmbedding.map((chunk) => chunk.content),
            signal,
          );
    if (chunksNeedingEmbedding.length > 0) {
      summary.embeddingRequestCount += 1;
      summary.newlyEmbeddedChunks += chunksNeedingEmbedding.length;
    }
    for (const [index, chunk] of chunksNeedingEmbedding.entries()) {
      const embedding = embeddings[index];
      if (!embedding) {
        throw new Error(`Recall embedding missing for conversation chunk ${chunk.id}`);
      }
      embeddingByDocumentId.set(chunk.id, embedding);
    }
    onBatchPrepared();
  }
  return embeddingByDocumentId;
}

async function indexChangedRecallSessionFile(
  options: IncrementalSessionIndexerOptions,
  database: SqliteRecallDatabase,
  plannedFile: PlannedPhysicalSessionFile,
  summary: ConversationIndexSummary,
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
  onBatchPrepared: () => void,
): Promise<void> {
  const { sessionPath } = plannedFile;
  const previous = database.readPhysicalSessionState(sessionPath);

  let imported: Awaited<ReturnType<typeof readSessionConversationImport>>;
  try {
    imported = await readSessionConversationImport(sessionPath, {
      tokenizer: options.tokenizer,
      ...options.chunkPolicy,
      resolveProjectIdentity: resolveSessionProjectIdentity,
    });
  } catch (error) {
    summary.failedSessions.push({
      sessionPath,
      error: error instanceof Error ? error.message : String(error),
    });
    options.onProgress?.({ kind: 'physical-session-file-failed', sessionPath });
    if (previous) {
      removeIndexedSession(database, sessionPath, summary);
    }
    return;
  }

  const attributedChunks = await attributeRecallChunksToProjects(
    imported.chunks,
    resolveSessionProjectIdentity,
  );
  const denseChunks = attributedChunks.filter(
    (chunk): chunk is SessionConversationChunk & { isDenseSearchable: true } =>
      chunk.isDenseSearchable,
  );
  const currentIds = new Set(denseChunks.map((chunk) => chunk.id));
  const removedIds =
    previous?.denseDocumentIds.filter((documentId) => !currentIds.has(documentId)) ?? [];
  const denseEmbeddings = await prepareChangedRecallEmbeddingMap(
    denseChunks,
    database,
    options.embeddingProvider,
    options.vectorReuseReader,
    summary,
    onBatchPrepared,
    options.signal,
  );

  throwIfIndexingAborted(options.signal);
  database.replacePhysicalSession({
    sessionPath,
    size: plannedFile.size,
    mtimeMs: plannedFile.mtimeMs,
    documentIds: denseChunks.map((chunk) => chunk.id),
    denseDocuments: denseChunks,
    denseEmbeddings,
    invocations: imported.invocations,
  });
  summary.deletedChunks += removedIds.length;
  summary.indexedSessions += 1;
}

async function planMaintenanceWorkset(
  sessionFiles: readonly string[],
  database: SqliteRecallDatabase,
  ignoredPhysicalSessionPaths: ReadonlySet<string>,
  pruneUnselectedSessions: boolean,
): Promise<MaintenanceWorksetPlan> {
  const filesToIndex: PlannedPhysicalSessionFile[] = [];
  for (const sessionPath of sessionFiles) {
    if (ignoredPhysicalSessionPaths.has(sessionPath)) {
      continue;
    }
    const fileStats = await stat(sessionPath);
    const previous = database.readPhysicalSessionState(sessionPath);
    if (!previous) {
      filesToIndex.push({
        sessionPath,
        size: fileStats.size,
        mtimeMs: fileStats.mtimeMs,
        change: 'new',
      });
    } else if (
      previous.size !== fileStats.size ||
      previous.mtimeMs !== fileStats.mtimeMs ||
      database.requiresInvocationBackfill(sessionPath)
    ) {
      filesToIndex.push({
        sessionPath,
        size: fileStats.size,
        mtimeMs: fileStats.mtimeMs,
        change: 'changed',
      });
    }
  }
  const liveSessionPaths = new Set(sessionFiles);
  const indexedSessionPaths = database.listPhysicalSessionPaths();
  return {
    discoveredFiles: sessionFiles.length,
    filesToIndex,
    missingFiles: pruneUnselectedSessions
      ? indexedSessionPaths
          .filter(
            (sessionPath) =>
              !ignoredPhysicalSessionPaths.has(sessionPath) && !liveSessionPaths.has(sessionPath),
          )
          .sort()
      : [],
    ignoredIndexedFiles: indexedSessionPaths
      .filter(
        (sessionPath) =>
          ignoredPhysicalSessionPaths.has(sessionPath) &&
          (pruneUnselectedSessions || liveSessionPaths.has(sessionPath)),
      )
      .sort(),
  };
}

/** Indexes eligible new or changed physical session files and removes missing or ignored ones. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const { databasePath } = options;
  if (!options.database && !databasePath) {
    throw new Error('Recall indexing database path missing');
  }
  const database = options.database ?? openSqliteRecallDatabase(databasePath ?? '');
  const ownsDatabase = options.database === undefined;
  try {
    options.onProgress?.({ kind: 'discovering-physical-session-files' });
    const sessionsDirectory = resolve(options.sessionsDirectory);
    const selectedPhysicalSessionPaths = options.selectedPhysicalSessionPaths?.map((sessionPath) =>
      resolve(sessionPath),
    );
    const sessionFiles = selectedPhysicalSessionPaths
      ? [...new Set(selectedPhysicalSessionPaths)].sort()
      : await listRecallSessionFiles(sessionsDirectory);
    options.onProgress?.({ kind: 'planning-maintenance-workset' });
    const workset = await planMaintenanceWorkset(
      sessionFiles,
      database,
      options.ignoredPhysicalSessionPaths,
      selectedPhysicalSessionPaths === undefined,
    );
    const summary = createConversationIndexSummary();
    summary.scannedSessions = workset.discoveredFiles;
    options.onProgress?.({
      kind: 'maintenance-workset-planned',
      discoveredFiles: workset.discoveredFiles,
      newFiles: workset.filesToIndex.filter((file) => file.change === 'new').length,
      changedFiles: workset.filesToIndex.filter((file) => file.change === 'changed').length,
      missingFiles: workset.missingFiles.length,
      ignoredRemovals: workset.ignoredIndexedFiles.length,
      rebuild: options.rebuild ?? false,
    });

    const indexedPathsToRemove = [...workset.missingFiles, ...workset.ignoredIndexedFiles];
    const totalWorksetFiles = indexedPathsToRemove.length + workset.filesToIndex.length;
    let completedWorksetFiles = 0;
    for (const sessionPath of indexedPathsToRemove) {
      throwIfIndexingAborted(options.signal);
      removeIndexedSession(database, sessionPath, summary);
      completedWorksetFiles += 1;
      emitMaintenanceWorksetProgress(
        options,
        summary,
        completedWorksetFiles,
        totalWorksetFiles,
        sessionPath,
      );
    }

    const resolveSessionProjectIdentity = createSessionProjectIdentityResolver(
      options.resolveProjectIdentity,
    );
    if (workset.filesToIndex.length > 0) {
      options.onProgress?.({ kind: 'indexing-changed-physical-session-files' });
    }
    for (const plannedFile of workset.filesToIndex) {
      throwIfIndexingAborted(options.signal);
      await indexChangedRecallSessionFile(
        options,
        database,
        plannedFile,
        summary,
        resolveSessionProjectIdentity,
        () => {
          emitMaintenanceWorksetProgress(
            options,
            summary,
            completedWorksetFiles,
            totalWorksetFiles,
            plannedFile.sessionPath,
          );
        },
      );
      completedWorksetFiles += 1;
      emitMaintenanceWorksetProgress(
        options,
        summary,
        completedWorksetFiles,
        totalWorksetFiles,
        plannedFile.sessionPath,
      );
    }
    return summary;
  } finally {
    if (ownsDatabase) {
      database.close();
    }
  }
}

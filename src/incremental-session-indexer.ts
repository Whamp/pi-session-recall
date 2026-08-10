import { stat } from 'node:fs/promises';
import { resolve } from 'node:path';

import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import { openRecallCatalog, type RecallCatalog } from './recall-catalog.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { listRecallSessionFiles } from './recall-session-files.js';
import type { ResolvedProjectIdentity } from './resolve-project-identity.js';
import {
  readSessionConversationImport,
  type ConversationTextTokenizer,
  type SessionConversationChunk,
} from './session-conversation-index.js';
import type { DenseRecallDocument } from './dense-recall-conversation-store.js';

/** Dense-document persistence required by incremental compact-layout indexing. */
export interface DenseRecallIndexStore {
  upsertDocuments(documents: DenseRecallDocument[]): void;
  deleteDocuments(ids: string[]): void;
  fetchDocuments(this: void, ids: string[]): Map<string, SessionConversationChunk>;
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
  catalogPath: string;
  store: DenseRecallIndexStore;
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
  catalog: RecallCatalog,
  sessionPath: string,
  store: DenseRecallIndexStore,
  summary: ConversationIndexSummary,
): boolean {
  const previous = catalog.readPhysicalSessionState(sessionPath);
  if (!previous) {
    return false;
  }
  store.deleteDocuments(previous.denseDocumentIds);
  catalog.deletePhysicalSession(sessionPath);
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

async function prepareChangedRecallRows(
  chunks: readonly SessionConversationChunk[],
  store: DenseRecallIndexStore,
  embeddingProvider: RecallEmbeddingProvider,
  summary: ConversationIndexSummary,
  onBatchPrepared: () => void,
  signal?: AbortSignal,
): Promise<DenseRecallDocument[]> {
  const changedRows: DenseRecallDocument[] = [];
  for (let start = 0; start < chunks.length; start += 128) {
    throwIfIndexingAborted(signal);
    const batch = chunks.slice(start, start + 128);
    const ids = batch.map((chunk) => chunk.id);
    const existingChunks = store.fetchDocuments(ids);
    const existingVectors = store.fetchVectors(ids);
    const rowsNeedingWrite = batch.filter((chunk) => {
      const existing = existingChunks.get(chunk.id);
      return !existing || existing.checksum !== chunk.checksum || !existingVectors.has(chunk.id);
    });
    summary.reusedVectors += batch.length - rowsNeedingWrite.length;
    const rowsNeedingEmbedding = rowsNeedingWrite;
    const embeddings =
      rowsNeedingEmbedding.length === 0
        ? []
        : await embeddingProvider.embedDocuments(
            rowsNeedingEmbedding.map((chunk) => chunk.content),
            signal,
          );
    if (rowsNeedingEmbedding.length > 0) {
      summary.embeddingRequestCount += 1;
      summary.newlyEmbeddedChunks += rowsNeedingEmbedding.length;
    }
    for (const [index, chunk] of rowsNeedingEmbedding.entries()) {
      const embedding = embeddings[index];
      if (!embedding) {
        throw new Error(`Recall embedding missing for conversation chunk ${chunk.id}`);
      }
      changedRows.push({ ...chunk, embedding });
    }
    onBatchPrepared();
  }
  return changedRows;
}

async function indexChangedRecallSessionFile(
  options: IncrementalSessionIndexerOptions,
  catalog: RecallCatalog,
  plannedFile: PlannedPhysicalSessionFile,
  summary: ConversationIndexSummary,
  resolveSessionProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>,
  onBatchPrepared: () => void,
): Promise<void> {
  const { sessionPath } = plannedFile;
  const previous = catalog.readPhysicalSessionState(sessionPath);

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
      removeIndexedSession(catalog, sessionPath, options.store, summary);
    }
    return;
  }

  const attributedChunks = await attributeRecallChunksToProjects(
    imported.chunks,
    resolveSessionProjectIdentity,
  );
  const denseChunks = attributedChunks;
  const currentIds = new Set(denseChunks.map((chunk) => chunk.id));
  const removedIds =
    previous?.denseDocumentIds.filter((documentId) => !currentIds.has(documentId)) ?? [];
  const changedRows = await prepareChangedRecallRows(
    denseChunks,
    options.store,
    options.embeddingProvider,
    summary,
    onBatchPrepared,
    options.signal,
  );

  options.store.deleteDocuments(removedIds);
  for (let start = 0; start < changedRows.length; start += 128) {
    options.store.upsertDocuments(changedRows.slice(start, start + 128));
  }
  catalog.replacePhysicalSession({
    sessionPath,
    size: plannedFile.size,
    mtimeMs: plannedFile.mtimeMs,
    documentIds: denseChunks.map((chunk) => chunk.id),
    denseDocumentIds: denseChunks.map((chunk) => chunk.id),
    invocations: imported.invocations,
  });
  summary.deletedChunks += removedIds.length;
  summary.indexedSessions += 1;
}

async function planMaintenanceWorkset(
  sessionFiles: readonly string[],
  catalog: RecallCatalog,
  ignoredPhysicalSessionPaths: ReadonlySet<string>,
): Promise<MaintenanceWorksetPlan> {
  const filesToIndex: PlannedPhysicalSessionFile[] = [];
  for (const sessionPath of sessionFiles) {
    if (ignoredPhysicalSessionPaths.has(sessionPath)) {
      continue;
    }
    const fileStats = await stat(sessionPath);
    const previous = catalog.readPhysicalSessionState(sessionPath);
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
      catalog.requiresInvocationBackfill(sessionPath)
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
  const indexedSessionPaths = catalog.listPhysicalSessionPaths();
  return {
    discoveredFiles: sessionFiles.length,
    filesToIndex,
    missingFiles: indexedSessionPaths
      .filter(
        (sessionPath) =>
          !ignoredPhysicalSessionPaths.has(sessionPath) && !liveSessionPaths.has(sessionPath),
      )
      .sort(),
    ignoredIndexedFiles: indexedSessionPaths
      .filter((sessionPath) => ignoredPhysicalSessionPaths.has(sessionPath))
      .sort(),
  };
}

/** Indexes eligible new or changed physical session files and removes missing or ignored ones. */
export async function indexChangedConversationSessions(
  options: IncrementalSessionIndexerOptions,
): Promise<ConversationIndexSummary> {
  const catalog = openRecallCatalog(options.catalogPath);
  try {
    options.onProgress?.({ kind: 'discovering-physical-session-files' });
    const sessionsDirectory = resolve(options.sessionsDirectory);
    const sessionFiles = await listRecallSessionFiles(sessionsDirectory);
    options.onProgress?.({ kind: 'planning-maintenance-workset' });
    const workset = await planMaintenanceWorkset(
      sessionFiles,
      catalog,
      options.ignoredPhysicalSessionPaths,
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
      removeIndexedSession(catalog, sessionPath, options.store, summary);
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
        catalog,
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
    catalog.close();
  }
}

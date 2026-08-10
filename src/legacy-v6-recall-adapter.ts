import { fuseRecallSearchCandidates } from './fuse-recall-search-candidates.js';
import {
  indexChangedLegacyV6ConversationSessions,
  type LegacyV6ConversationIndexSummary,
} from './legacy-v6-incremental-session-indexer.js';
import {
  assertLegacyV6RecallIndexManifestCompatible,
  readLegacyV6RecallIndexManifest,
  type LegacyV6RecallIndexManifest,
} from './legacy-v6-recall-index-manifest.js';
import type { RecallChunkPolicy } from './recall-chunk-policy.js';
import type { RecallIndexProgressEvent } from './recall-index-progress.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  rankFusedRecallSearchResults,
  type RankedRecallSearchResult,
} from './rank-recall-search-results.js';
import type { ProjectIdentity, ResolvedProjectIdentity } from './resolve-project-identity.js';
import type {
  ConversationTextTokenizer,
  SessionConversationChunk,
} from './session-conversation-index.js';
import {
  openZvecConversationStore,
  type ZvecConversationStore,
} from './zvec-conversation-store.js';

/** Paths owned by the temporary legacy-v6 rollback adapter. */
export interface LegacyV6RecallDatabasePaths {
  databasePath: string;
  statePath: string;
  manifestPath: string;
}

/** Exact search output preserved from the production manifest-v6 service. */
export interface LegacyV6RecallSearch {
  results: RankedRecallSearchResult[];
  totalChunks: number;
}

/** Result of one update-only legacy-v6 hourly indexing pass. */
export interface LegacyV6RecallIndexResult {
  indexSummary: LegacyV6ConversationIndexSummary;
  totalChunks: number;
}

/** Verifies the complete actual-v6 identity before any legacy database is opened. */
export async function assertLegacyV6RecallDatabaseCompatible(
  paths: LegacyV6RecallDatabasePaths,
  expectedManifest: LegacyV6RecallIndexManifest,
): Promise<LegacyV6RecallIndexManifest> {
  const actualManifest = await readLegacyV6RecallIndexManifest(paths.manifestPath);
  assertLegacyV6RecallIndexManifestCompatible(actualManifest, expectedManifest, paths.manifestPath);
  return actualManifest;
}

function openLegacyV6ZvecStore(
  paths: LegacyV6RecallDatabasePaths,
  dimensions: number,
  readOnly: boolean,
): ZvecConversationStore {
  return openZvecConversationStore({
    databasePath: paths.databasePath,
    dimensions,
    createIfMissing: false,
    readOnly,
  });
}

/** Searches actual-v6 dense, lexical, and identifier channels with original fusion and neighbors. */
export function searchLegacyV6RecallDatabase(options: {
  paths: LegacyV6RecallDatabasePaths;
  dimensions: number;
  query: string;
  queryEmbedding: number[];
  resultLimit: number;
  candidateLimit: number;
  projectIdentity?: ProjectIdentity;
}): LegacyV6RecallSearch {
  const store = openLegacyV6ZvecStore(options.paths, options.dimensions, true);
  try {
    const candidates = fuseRecallSearchCandidates(
      {
        denseCandidates: store.searchDenseCandidates(
          options.queryEmbedding,
          options.candidateLimit,
          options.projectIdentity,
        ),
        lexicalCandidates: store.searchLexicalCandidates(
          options.query,
          options.candidateLimit,
          options.projectIdentity,
        ),
        identifierCandidates: store.searchIdentifierCandidates(
          options.query,
          options.candidateLimit,
          options.projectIdentity,
        ),
      },
      options.candidateLimit * 3,
    );
    return {
      results: rankFusedRecallSearchResults(
        candidates,
        options.resultLimit,
        store.fetchConversationChunks,
      ),
      totalChunks: store.count(),
    };
  } finally {
    store.close();
  }
}

/** Runs update-only actual-v6 indexing; it never optimizes the Zvec collection. */
export async function indexLegacyV6RecallDatabase(options: {
  paths: LegacyV6RecallDatabasePaths;
  dimensions: number;
  sessionsDirectory: string;
  embeddingProvider: RecallEmbeddingProvider;
  tokenizer: ConversationTextTokenizer;
  chunkPolicy: RecallChunkPolicy;
  ignoredPhysicalSessionPaths: ReadonlySet<string>;
  resolveProjectIdentity: (sessionOrigin: string) => Promise<ResolvedProjectIdentity | null>;
  signal?: AbortSignal;
  onProgress?: (event: RecallIndexProgressEvent) => void;
}): Promise<LegacyV6RecallIndexResult> {
  const store = openLegacyV6ZvecStore(options.paths, options.dimensions, false);
  try {
    const indexSummary = await indexChangedLegacyV6ConversationSessions({
      sessionsDirectory: options.sessionsDirectory,
      statePath: options.paths.statePath,
      store,
      embeddingProvider: options.embeddingProvider,
      tokenizer: options.tokenizer,
      chunkPolicy: options.chunkPolicy,
      ignoredPhysicalSessionPaths: options.ignoredPhysicalSessionPaths,
      resolveProjectIdentity: options.resolveProjectIdentity,
      ...(options.signal ? { signal: options.signal } : {}),
      ...(options.onProgress ? { onProgress: options.onProgress } : {}),
    });
    return { indexSummary, totalChunks: store.count() };
  } finally {
    store.close();
  }
}

/** Opens checksum-addressable actual-v6 vectors only after the caller validates its manifest. */
export function openLegacyV6VectorReuseReader(
  paths: LegacyV6RecallDatabasePaths,
  dimensions: number,
): {
  fetchDocuments(ids: string[]): Map<string, SessionConversationChunk>;
  fetchVectors(ids: string[]): Map<string, number[]>;
  close(): void;
} {
  const store = openLegacyV6ZvecStore(paths, dimensions, true);
  return {
    fetchDocuments: (ids) => store.fetchConversationChunks(ids),
    fetchVectors: (ids) => store.fetchVectors(ids),
    close: () => store.close(),
  };
}

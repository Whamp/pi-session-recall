import { createHash } from 'node:crypto';

import { ZVecIndexType, ZVecOpen, type ZVecCollection } from '@zvec/zvec';

import type { RecallCoherentGenerationConfig } from './recall-coherent-generation.js';
import { openValidatedRecallGeneration } from './recall-coherent-generation.js';
import type { RecallDenseCandidate, RecallFullTextCandidate } from './fuse-recall-ranked-lists.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';
import { readActiveTargetRecallManifestFingerprint } from './read-active-target-recall-generation.js';
import { parseRecallGenerationSearchDocument } from './recall-physical-source-generation.js';
import type { ProjectIdentity } from './resolve-project-identity.js';
import {
  convertNormalizedRecallInnerProductToCosineDistance,
  createStoredRecallEmbedding,
} from './recall-stored-embedding.js';
import { visitExactZvecDocuments } from './visit-exact-zvec-documents.js';
import type { SessionConversationChunk } from './session-conversation-index.js';
import {
  createRecallZvecFullTextQuery,
  type RecallConversationSearchStore,
} from './zvec-conversation-store.js';

const TARGET_RECALL_DENSE_SEARCH_EF = 300;

function assertTargetRecallCandidateLimit(limit: number, channelName: string): void {
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    throw new Error(
      `Recall target generation candidate limit invalid (${channelName}): expected an integer from 1 to 200`,
    );
  }
}

function createTargetRecallProjectFilter(projectIdentity?: ProjectIdentity): string | null {
  if (projectIdentity === undefined) {
    return null;
  }
  const digest = createHash('sha256').update(projectIdentity).digest('hex');
  return `projectIdentityDigest = '${digest}'`;
}

function createTargetRecallEvidenceFilter(projectIdentity?: ProjectIdentity): string {
  const projectFilter = createTargetRecallProjectFilter(projectIdentity);
  return projectFilter === null
    ? "recordKind = 'evidence'"
    : `recordKind = 'evidence' AND ${projectFilter}`;
}

function parseTargetRecallSearchDocument(
  documentId: string,
  fields: Record<string, unknown>,
): SessionConversationChunk {
  return parseRecallGenerationSearchDocument(documentId, fields.recordJson);
}

function fetchTargetRecallSearchDocuments(
  lexicalSource: ZVecCollection,
  ids: readonly string[],
): Map<string, SessionConversationChunk> {
  if (ids.length === 0) {
    return new Map();
  }
  const documents = lexicalSource.fetchSync({
    ids: [...ids],
    outputFields: ['recordJson'],
    includeVector: false,
  });
  const result = new Map<string, SessionConversationChunk>();
  for (const id of ids) {
    const document = documents[id];
    if (document === undefined) {
      throw new Error(
        `Recall target generation incoherent occurrence missing from lexical/source store: ${id}`,
      );
    }
    result.set(id, parseTargetRecallSearchDocument(id, document.fields));
  }
  return result;
}

async function searchTargetRecallFullTextCandidates(
  lexicalSource: ZVecCollection,
  fieldName: 'content' | 'identifierContent',
  query: string,
  limit: number,
  channelName: 'lexical' | 'identifier',
  defaultOperator: 'AND' | 'OR',
  projectIdentity?: ProjectIdentity,
): Promise<RecallFullTextCandidate[]> {
  assertTargetRecallCandidateLimit(limit, channelName);
  const documents = await lexicalSource.query({
    fieldName,
    fts: createRecallZvecFullTextQuery(query),
    filter: createTargetRecallEvidenceFilter(projectIdentity),
    topk: limit,
    outputFields: ['recordJson'],
    includeVector: false,
    params: { indexType: ZVecIndexType.FTS, defaultOperator },
  });
  return documents.map((document) => ({
    ...parseTargetRecallSearchDocument(document.id, document.fields),
    fullTextScore: document.score,
  }));
}

/** Opens the pointer-selected validated target generation behind the existing search-store shape. */
export async function openActiveRecallGenerationSearchStore(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
): Promise<RecallConversationSearchStore> {
  const registryManifestFingerprint = await readActiveTargetRecallManifestFingerprint(
    config,
    generationId,
  );
  const opened = await openValidatedRecallGeneration(config, generationId);
  if (registryManifestFingerprint !== opened.manifestFingerprint) {
    throw new Error(
      `Recall target generation registry manifest fingerprint mismatch: expected ${opened.manifestFingerprint}, received ${registryManifestFingerprint}`,
    );
  }
  const paths = createRecallGenerationComponentPaths(opened.generationDirectory);
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  let dense: ZVecCollection;
  try {
    dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  } catch (error) {
    lexicalSource.closeSync();
    throw error;
  }
  let evidenceCount = 0;
  try {
    visitExactZvecDocuments(
      lexicalSource,
      {
        filter: "recordKind = 'evidence'",
        uniquePartitionField: 'evidenceOccurrenceId',
        outputFields: [],
      },
      () => {
        evidenceCount += 1;
      },
    );
  } catch (error) {
    lexicalSource.closeSync();
    dense.closeSync();
    throw error;
  }

  return {
    async searchDenseCandidates(embedding, limit, projectIdentity) {
      assertTargetRecallCandidateLimit(limit, 'dense');
      if (dense.stats.docCount === 0) {
        return [];
      }
      const storedEmbedding = createStoredRecallEmbedding(embedding, {
        nativeDimensions: config.embeddingProfile.identity.dimensions,
        storedDimensions:
          config.embeddingProfile.storedDimensions ?? config.embeddingProfile.identity.dimensions,
        source: `active target generation ${generationId} query`,
      });
      const projectFilter = createTargetRecallProjectFilter(projectIdentity);
      const documents = await dense.query({
        fieldName: 'embedding',
        vector: storedEmbedding,
        ...(projectFilter === null ? {} : { filter: projectFilter }),
        topk: limit,
        outputFields: ['evidenceOccurrenceId'],
        includeVector: false,
        params: { indexType: ZVecIndexType.HNSW, ef: TARGET_RECALL_DENSE_SEARCH_EF },
      });
      const searchDocuments = fetchTargetRecallSearchDocuments(
        lexicalSource,
        documents.map(({ id }) => id),
      );
      return documents.map((document): RecallDenseCandidate => {
        const searchDocument = searchDocuments.get(document.id);
        if (searchDocument === undefined) {
          throw new Error(
            `Recall target generation dense occurrence join failed for ${document.id}`,
          );
        }
        return {
          ...searchDocument,
          cosineDistance: convertNormalizedRecallInnerProductToCosineDistance(document.score),
        };
      });
    },
    searchLexicalCandidates(query, limit, projectIdentity) {
      return searchTargetRecallFullTextCandidates(
        lexicalSource,
        'content',
        query,
        limit,
        'lexical',
        'OR',
        projectIdentity,
      );
    },
    searchIdentifierCandidates(query, limit, projectIdentity) {
      return searchTargetRecallFullTextCandidates(
        lexicalSource,
        'identifierContent',
        query,
        limit,
        'identifier',
        'AND',
        projectIdentity,
      );
    },
    fetchConversationChunks(ids) {
      return fetchTargetRecallSearchDocuments(lexicalSource, ids);
    },
    close() {
      lexicalSource.closeSync();
      dense.closeSync();
    },
    count() {
      return evidenceCount;
    },
  };
}

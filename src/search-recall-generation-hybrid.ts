import { ZVecIndexType, ZVecOpen } from '@zvec/zvec';

import {
  openValidatedRecallGeneration,
  type RecallCoherentGenerationConfig,
} from './recall-coherent-generation.js';
import {
  parseRecallGenerationLexicalEvidence,
  type RecallGenerationLexicalEvidence,
} from './recall-physical-source-generation.js';
import { RECALL_RRF_RANK_CONSTANT } from './fuse-recall-ranked-lists.js';
import { createRecallGenerationComponentPaths } from './recall-generation-stores.js';

/** One cross-store target result joined by exact evidence occurrence ID. */
export interface RecallGenerationHybridSearchResult {
  evidence: RecallGenerationLexicalEvidence;
  denseRank: number | null;
  denseScore: number | null;
  lexicalRank: number | null;
  lexicalScore: number | null;
  fusedScore: number;
}

interface RecallGenerationRankedMatch {
  evidence: RecallGenerationLexicalEvidence;
  denseRank: number | null;
  denseScore: number | null;
  lexicalRank: number | null;
  lexicalScore: number | null;
}

function reciprocalRank(rank: number | null): number {
  return rank === null ? 0 : 1 / (RECALL_RRF_RANK_CONSTANT + rank);
}

/** Searches independent dense and lexical/source stores and joins them by occurrence ID. */
export async function searchRecallGenerationHybrid(
  config: Readonly<RecallCoherentGenerationConfig>,
  generationId: string,
  query: string,
  storedQueryEmbedding: readonly number[],
  limit: number,
): Promise<RecallGenerationHybridSearchResult[]> {
  if (!query.trim()) {
    throw new Error('Recall generation hybrid query must not be blank');
  }
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) {
    throw new Error('Recall generation hybrid limit must be from 1 to 200');
  }
  const opened = await openValidatedRecallGeneration(config, generationId);
  const paths = createRecallGenerationComponentPaths(opened.generationDirectory);
  const lexicalSource = ZVecOpen(paths.lexicalSourceStorePath, { readOnly: true });
  const dense = ZVecOpen(paths.denseStorePath, { readOnly: true });
  try {
    const [lexicalDocuments, denseDocuments] = await Promise.all([
      lexicalSource.query({
        fieldName: 'content',
        fts: { matchString: query },
        filter: "recordKind = 'evidence'",
        topk: limit,
        outputFields: ['recordJson'],
        includeVector: false,
        params: { indexType: ZVecIndexType.FTS, defaultOperator: 'OR' },
      }),
      dense.stats.docCount === 0
        ? Promise.resolve([])
        : dense.query({
            fieldName: 'embedding',
            vector: [...storedQueryEmbedding],
            topk: limit,
            outputFields: ['evidenceOccurrenceId'],
            includeVector: false,
            params: { indexType: ZVecIndexType.HNSW, ef: 300 },
          }),
    ]);
    const occurrenceIds = [
      ...new Set([...lexicalDocuments.map(({ id }) => id), ...denseDocuments.map(({ id }) => id)]),
    ];
    const fetched = lexicalSource.fetchSync({
      ids: occurrenceIds,
      outputFields: ['recordJson'],
      includeVector: false,
    });
    const matches = new Map<string, RecallGenerationRankedMatch>();
    for (const occurrenceId of occurrenceIds) {
      const lexicalRecord = fetched[occurrenceId];
      if (lexicalRecord === undefined) {
        throw new Error(
          `Recall generation hybrid incoherent dense occurrence missing from lexical/source store: ${occurrenceId}`,
        );
      }
      matches.set(occurrenceId, {
        evidence: parseRecallGenerationLexicalEvidence(lexicalRecord.fields.recordJson),
        denseRank: null,
        denseScore: null,
        lexicalRank: null,
        lexicalScore: null,
      });
    }
    for (const [index, document] of denseDocuments.entries()) {
      const match = matches.get(document.id);
      if (match !== undefined) {
        match.denseRank = index + 1;
        match.denseScore = document.score;
      }
    }
    for (const [index, document] of lexicalDocuments.entries()) {
      const match = matches.get(document.id);
      if (match !== undefined) {
        match.lexicalRank = index + 1;
        match.lexicalScore = document.score;
      }
    }
    return [...matches.values()]
      .map((match) => ({
        ...match,
        fusedScore: reciprocalRank(match.denseRank) + reciprocalRank(match.lexicalRank),
      }))
      .toSorted(
        (left, right) =>
          right.fusedScore - left.fusedScore ||
          left.evidence.evidenceOccurrenceId.localeCompare(
            right.evidence.evidenceOccurrenceId,
            'en-US',
          ),
      )
      .slice(0, limit);
  } finally {
    lexicalSource.closeSync();
    dense.closeSync();
  }
}

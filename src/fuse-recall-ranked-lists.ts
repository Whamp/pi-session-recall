import { compareRecallDocumentIds } from './compare-recall-document-ids.js';
import { RecallRankedListSource } from './enums.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

/** Version of the deterministic application-side hybrid rank-fusion policy. */
export const RECALL_RANK_FUSION_VERSION = 2;

/** Fixed reciprocal rank fusion (RRF) constant applied to each retrieval channel. */
export const RECALL_RRF_RANK_CONSTANT = 60;

/** One bounded dense candidate; cosine distance is lower-is-better. */
export interface RecallDenseCandidate extends SessionConversationChunk {
  cosineDistance: number;
}

/** One bounded full text search (FTS) candidate; zvec score is higher-is-better. */
export interface RecallFullTextCandidate extends SessionConversationChunk {
  fullTextScore: number;
}

/** One retained dense component; cosine distance is lower-is-better. */
export interface RecallDenseScore {
  rank: number;
  cosineDistance: number;
}

/** One retained full text search (FTS) component; score is higher-is-better. */
export interface RecallFullTextScore {
  rank: number;
  fullTextScore: number;
}

/** One document and its provider-native score within a recall ranked list. */
export interface RecallRankedListCandidate {
  document: SessionConversationChunk;
  nativeScore: number;
}

/** One independently bounded retrieval result list submitted to rank fusion. */
export interface RecallRankedList {
  source: RecallRankedListSource;
  query: string;
  weight: number;
  candidateLimit: number;
  higherNativeScoresRankFirst: boolean;
  candidates: readonly RecallRankedListCandidate[];
}

/** Retained source, query, rank, native score, and weight from one ranked list. */
export interface RecallRankedListEvidence {
  source: RecallRankedListSource;
  query: string;
  rank: number;
  nativeScore: number;
  weight: number;
}

/** One source-backed fused result with generic list evidence and legacy component details. */
export interface RecallSearchResult extends SessionConversationChunk {
  dense: RecallDenseScore | null;
  lexical: RecallFullTextScore | null;
  identifier: RecallFullTextScore | null;
  rankedListEvidence: RecallRankedListEvidence[];
  fusedScore: number;
}

function assertFiniteRecallCandidateScore(
  score: number,
  channelName: string,
  documentId: string,
): void {
  if (!Number.isFinite(score)) {
    throw new Error(
      `Recall candidate score invalid (${channelName}) for document ${documentId}: expected a finite number`,
    );
  }
}

function createRecallSearchResult(document: SessionConversationChunk): RecallSearchResult {
  return {
    ...document,
    dense: null,
    lexical: null,
    identifier: null,
    rankedListEvidence: [],
    fusedScore: 0,
  };
}

function reciprocalRankContribution(rank: number, weight: number): number {
  return weight / (RECALL_RRF_RANK_CONSTANT + rank);
}

function assertRecallRankedList(list: RecallRankedList): void {
  if (
    !Number.isInteger(list.candidateLimit) ||
    list.candidateLimit < 1 ||
    list.candidateLimit > 200
  ) {
    throw new Error(
      'Recall ranked-list candidate limit invalid: expected an integer from 1 to 200',
    );
  }
  if (!Number.isFinite(list.weight) || list.weight <= 0) {
    throw new Error('Recall ranked-list weight invalid: expected a positive finite number');
  }
  for (const candidate of list.candidates) {
    assertFiniteRecallCandidateScore(candidate.nativeScore, list.source, candidate.document.id);
  }
}

function retainLegacyRecallComponent(
  result: RecallSearchResult,
  evidence: RecallRankedListEvidence,
): void {
  if (evidence.source === RecallRankedListSource.DENSE && result.dense === null) {
    result.dense = { rank: evidence.rank, cosineDistance: evidence.nativeScore };
  }
  if (evidence.source === RecallRankedListSource.LEXICAL && result.lexical === null) {
    result.lexical = { rank: evidence.rank, fullTextScore: evidence.nativeScore };
  }
  if (evidence.source === RecallRankedListSource.IDENTIFIER && result.identifier === null) {
    result.identifier = { rank: evidence.rank, fullTextScore: evidence.nativeScore };
  }
}

/** Fuses independently bounded weighted ranked lists and caps the deterministic RRF pool before grouping. */
export function fuseRecallRankedLists(
  lists: readonly RecallRankedList[],
  fusedPoolLimit: number,
): RecallSearchResult[] {
  if (!Number.isInteger(fusedPoolLimit) || fusedPoolLimit < 1 || fusedPoolLimit > 600) {
    throw new Error('Recall fused pool limit invalid: expected an integer from 1 to 600');
  }

  const resultsById = new Map<string, RecallSearchResult>();
  for (const list of lists) {
    assertRecallRankedList(list);
    const admittedCandidates = list.candidates
      .toSorted((left, right) => {
        const nativeScoreOrder = list.higherNativeScoresRankFirst
          ? right.nativeScore - left.nativeScore
          : left.nativeScore - right.nativeScore;
        return nativeScoreOrder || compareRecallDocumentIds(left.document.id, right.document.id);
      })
      .slice(0, list.candidateLimit);
    const admittedDocumentIds = new Set<string>();
    for (const [index, candidate] of admittedCandidates.entries()) {
      if (admittedDocumentIds.has(candidate.document.id)) {
        continue;
      }
      admittedDocumentIds.add(candidate.document.id);
      const result =
        resultsById.get(candidate.document.id) ?? createRecallSearchResult(candidate.document);
      const evidence: RecallRankedListEvidence = {
        source: list.source,
        query: list.query,
        rank: index + 1,
        nativeScore: candidate.nativeScore,
        weight: list.weight,
      };
      result.rankedListEvidence.push(evidence);
      result.fusedScore += reciprocalRankContribution(evidence.rank, evidence.weight);
      retainLegacyRecallComponent(result, evidence);
      resultsById.set(result.id, result);
    }
  }

  return Array.from(resultsById.values())
    .toSorted(
      (left, right) =>
        right.fusedScore - left.fusedScore || compareRecallDocumentIds(left.id, right.id),
    )
    .slice(0, fusedPoolLimit);
}

import type { SessionConversationChunk } from './session-conversation-index.js';

/** Version of the deterministic application-side hybrid rank-fusion policy. */
export const RECALL_RANK_FUSION_VERSION = 1;

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

/** Bounded dense, ordinary FTS, and case-preserving identifier candidate channels. */
export interface RecallSearchCandidateChannels {
  denseCandidates: RecallDenseCandidate[];
  lexicalCandidates: RecallFullTextCandidate[];
  identifierCandidates: RecallFullTextCandidate[];
}

/** One source-backed hybrid result with every component rank and score retained. */
export interface RecallSearchResult extends SessionConversationChunk {
  dense: RecallDenseScore | null;
  lexical: RecallFullTextScore | null;
  identifier: RecallFullTextScore | null;
  fusedScore: number;
}

function compareRecallDocumentIds(leftId: string, rightId: string): number {
  if (leftId < rightId) {
    return -1;
  }
  if (leftId > rightId) {
    return 1;
  }
  return 0;
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
    fusedScore: 0,
  };
}

function reciprocalRankContribution(rank: number | null): number {
  return rank === null ? 0 : 1 / (RECALL_RRF_RANK_CONSTANT + rank);
}

/** Applies deterministic reciprocal rank fusion (RRF); scores must be finite and limit must be an integer from 1 to 600. */
export function fuseRecallSearchCandidates(
  channels: RecallSearchCandidateChannels,
  resultLimit: number,
): RecallSearchResult[] {
  if (!Number.isInteger(resultLimit) || resultLimit < 1 || resultLimit > 600) {
    throw new Error('Recall result limit invalid: expected an integer from 1 to 600');
  }

  const resultsById = new Map<string, RecallSearchResult>();
  const getResult = (document: SessionConversationChunk): RecallSearchResult => {
    const existing = resultsById.get(document.id);
    if (existing) {
      return existing;
    }
    const result = createRecallSearchResult(document);
    resultsById.set(document.id, result);
    return result;
  };

  for (const candidate of channels.denseCandidates) {
    assertFiniteRecallCandidateScore(candidate.cosineDistance, 'dense', candidate.id);
  }
  const denseCandidates = channels.denseCandidates.toSorted(
    (left, right) =>
      left.cosineDistance - right.cosineDistance || compareRecallDocumentIds(left.id, right.id),
  );
  for (const [index, candidate] of denseCandidates.entries()) {
    const { cosineDistance, ...document } = candidate;
    const result = getResult(document);
    if (result.dense === null) {
      result.dense = { rank: index + 1, cosineDistance };
    }
  }

  const addFullTextCandidates = (
    candidates: RecallFullTextCandidate[],
    channelName: 'lexical' | 'identifier',
  ): void => {
    for (const candidate of candidates) {
      assertFiniteRecallCandidateScore(candidate.fullTextScore, channelName, candidate.id);
    }
    const sortedCandidates = candidates.toSorted(
      (left, right) =>
        right.fullTextScore - left.fullTextScore || compareRecallDocumentIds(left.id, right.id),
    );
    for (const [index, candidate] of sortedCandidates.entries()) {
      const { fullTextScore, ...document } = candidate;
      const result = getResult(document);
      if (channelName === 'lexical' && result.lexical === null) {
        result.lexical = { rank: index + 1, fullTextScore };
      }
      if (channelName === 'identifier' && result.identifier === null) {
        result.identifier = { rank: index + 1, fullTextScore };
      }
    }
  };

  addFullTextCandidates(channels.lexicalCandidates, 'lexical');
  addFullTextCandidates(channels.identifierCandidates, 'identifier');

  return Array.from(resultsById.values())
    .map((result) => ({
      ...result,
      fusedScore:
        reciprocalRankContribution(result.dense?.rank ?? null) +
        reciprocalRankContribution(result.lexical?.rank ?? null) +
        reciprocalRankContribution(result.identifier?.rank ?? null),
    }))
    .toSorted(
      (left, right) =>
        right.fusedScore - left.fusedScore || compareRecallDocumentIds(left.id, right.id),
    )
    .slice(0, resultLimit);
}

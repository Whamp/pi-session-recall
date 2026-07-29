import { compareRecallDocumentIds } from './compare-recall-document-ids.js';
import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type { RecallRerankingProvider } from './recall-inference-capabilities.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

/** Version of Qwen reranking, duplicate suppression, and active-branch scoring policy. */
export const RECALL_RERANK_POLICY_VERSION = 1;

/** Small additive prior used to favor active-branch evidence without filtering other branches. */
export const RECALL_ACTIVE_BRANCH_PRIOR = 0.01;

/** Maximum cosine distance admitted for a dense-only default hybrid result. */
export const RECALL_MAX_DENSE_ONLY_COSINE_DISTANCE = 0.5;

/** Readable context reconstructed from exact same-run atomic chunk provenance. */
export interface RecallNeighborContext {
  content: string;
  chunks: SessionConversationChunk[];
}

/** One hybrid recall candidate after deterministic fusion or optional local Qwen scoring. */
export interface RankedRecallSearchResult extends RecallSearchResult {
  rerankerScore: number | null;
  activeBranchPrior: number;
  rankingScore: number;
  duplicateOccurrences: RecallSearchResult[];
  neighborContext: RecallNeighborContext | null;
}

interface RecallCandidateGroup {
  representative: RecallSearchResult;
  duplicateOccurrences: RecallSearchResult[];
}

/** Inputs for reranking one bounded, already-fused recall candidate pool. */
export interface RerankRecallSearchResultsOptions {
  query: string;
  candidates: readonly RecallSearchResult[];
  resultLimit: number;
  reranker: RecallRerankingProvider;
  fetchConversationChunks: (ids: string[]) => Map<string, SessionConversationChunk>;
  signal?: AbortSignal;
}

function compareRecallCandidatePreference(
  left: RecallSearchResult,
  right: RecallSearchResult,
): number {
  return right.fusedScore - left.fusedScore || compareRecallDocumentIds(left.id, right.id);
}

function areSameRecallTextRun(
  left: SessionConversationChunk,
  right: SessionConversationChunk,
): boolean {
  return (
    left.schemaVersion === right.schemaVersion &&
    left.sessionId.value === right.sessionId.value &&
    left.sessionPath === right.sessionPath &&
    left.entryId.value === right.entryId.value &&
    haveMatchingEntryIds(left.contributingEntryIds, right.contributingEntryIds) &&
    left.documentKind === right.documentKind &&
    left.summaryKind === right.summaryKind &&
    left.evidenceKind === right.evidenceKind &&
    left.evidencePart === right.evidencePart &&
    left.isDenseSearchable === right.isDenseSearchable &&
    left.role === right.role &&
    left.textRunId === right.textRunId &&
    left.textRunIndex === right.textRunIndex &&
    left.chunkCount === right.chunkCount &&
    left.sourceLineStart === right.sourceLineStart &&
    left.sourceLineEnd === right.sourceLineEnd &&
    left.sourceBlockStart === right.sourceBlockStart &&
    left.sourceBlockEnd === right.sourceBlockEnd
  );
}

function getRecallSiblingOverlapCharacters(
  earlier: SessionConversationChunk,
  later: SessionConversationChunk,
): number | null {
  if (
    !areSameRecallTextRun(earlier, later) ||
    earlier.nextSiblingId !== later.id ||
    later.previousSiblingId !== earlier.id ||
    !earlier.siblingIds.includes(later.id) ||
    !later.siblingIds.includes(earlier.id) ||
    later.chunkIndex !== earlier.chunkIndex + 1 ||
    later.tokenStart !== earlier.tokenEnd - later.overlapTokenCount ||
    earlier.characterStart >= later.characterStart ||
    earlier.characterEnd > later.characterEnd
  ) {
    return null;
  }
  const overlapCharacters = earlier.characterEnd - later.characterStart;
  if (overlapCharacters < 0) {
    return null;
  }
  if (overlapCharacters === 0) {
    return later.overlapTokenCount === 0 ? 0 : null;
  }
  return overlapCharacters <= earlier.content.length &&
    overlapCharacters <= later.content.length &&
    later.overlapTokenCount > 0 &&
    earlier.content.slice(-overlapCharacters) === later.content.slice(0, overlapCharacters)
    ? overlapCharacters
    : null;
}

function areOverlappingRecallSiblings(
  left: RecallSearchResult,
  right: RecallSearchResult,
): boolean {
  const earlier = left.chunkIndex < right.chunkIndex ? left : right;
  const later = earlier === left ? right : left;
  return (getRecallSiblingOverlapCharacters(earlier, later) ?? 0) > 0;
}

function getRecallCandidateGroupMembers(group: RecallCandidateGroup): RecallSearchResult[] {
  return [group.representative, ...group.duplicateOccurrences];
}

function mergeRecallCandidateGroups(
  groups: RecallCandidateGroup[],
  incomingMembers: readonly RecallSearchResult[],
  matchingIndexes: readonly number[],
  missingRepresentativeError: string,
): void {
  const matchingCandidates = matchingIndexes.flatMap((index) => {
    const group = groups[index];
    return group ? getRecallCandidateGroupMembers(group) : [];
  });
  const combined = [...incomingMembers, ...matchingCandidates].toSorted(
    compareRecallCandidatePreference,
  );
  const representative = combined[0];
  const insertionIndex = matchingIndexes[0];
  if (!representative || insertionIndex === undefined) {
    throw new Error(missingRepresentativeError);
  }
  for (const index of matchingIndexes.toReversed()) {
    groups.splice(index, 1);
  }
  groups.splice(insertionIndex, 0, {
    representative,
    duplicateOccurrences: combined.slice(1),
  });
}

function createSiblingOverlapCandidateGroups(
  candidates: readonly RecallSearchResult[],
): RecallCandidateGroup[] {
  const groups: RecallCandidateGroup[] = [];
  for (const candidate of candidates) {
    const matchingIndexes = groups
      .map((group, index) => ({ group, index }))
      .filter(({ group }) =>
        getRecallCandidateGroupMembers(group).some((existing) =>
          areOverlappingRecallSiblings(candidate, existing),
        ),
      )
      .map(({ index }) => index);
    if (matchingIndexes.length === 0) {
      groups.push({ representative: candidate, duplicateOccurrences: [] });
      continue;
    }
    mergeRecallCandidateGroups(
      groups,
      [candidate],
      matchingIndexes,
      'Recall sibling overlap grouping lost its representative',
    );
  }
  return groups;
}

function areExactCrossSessionCopies(left: RecallSearchResult, right: RecallSearchResult): boolean {
  return (
    left.sessionPath !== right.sessionPath &&
    left.checksum === right.checksum &&
    left.content === right.content &&
    left.documentKind === right.documentKind &&
    left.summaryKind === right.summaryKind &&
    left.evidenceKind === right.evidenceKind &&
    left.evidencePart === right.evidencePart &&
    left.role === right.role
  );
}

function createCrossSessionCopyCandidateGroups(
  siblingGroups: readonly RecallCandidateGroup[],
): RecallCandidateGroup[] {
  const copyGroups: RecallCandidateGroup[] = [];
  for (const siblingGroup of siblingGroups) {
    const siblingMembers = getRecallCandidateGroupMembers(siblingGroup);
    const matchingIndexes = copyGroups
      .map((group, index) => ({ group, index }))
      .filter(({ group }) =>
        getRecallCandidateGroupMembers(group).some((existing) =>
          siblingMembers.some((candidate) => areExactCrossSessionCopies(candidate, existing)),
        ),
      )
      .map(({ index }) => index);
    if (matchingIndexes.length === 0) {
      copyGroups.push(siblingGroup);
      continue;
    }
    mergeRecallCandidateGroups(
      copyGroups,
      siblingMembers,
      matchingIndexes,
      'Recall exact-copy grouping lost its representative',
    );
  }
  return copyGroups;
}

function haveMatchingEntryIds(
  left: SessionConversationChunk['contributingEntryIds'],
  right: SessionConversationChunk['contributingEntryIds'],
): boolean {
  return (
    left.length === right.length &&
    left.every((entryId, index) => entryId.value === right[index]?.value)
  );
}

function isAtomicConversationChunk(chunk: SessionConversationChunk): boolean {
  return (
    chunk.documentKind === 'conversation' &&
    chunk.summaryKind === null &&
    chunk.evidenceKind === 'conversation' &&
    chunk.evidencePart === 'content' &&
    (chunk.role === 'user' || chunk.role === 'assistant' || chunk.role === 'custom') &&
    chunk.toolCallId === null &&
    chunk.toolName === null &&
    chunk.toolCallEntryId === null &&
    chunk.toolResultEntryId === null &&
    chunk.toolError === null
  );
}

function areContiguousAtomicSiblings(
  left: SessionConversationChunk,
  right: SessionConversationChunk,
): boolean {
  return (
    isAtomicConversationChunk(left) &&
    isAtomicConversationChunk(right) &&
    getRecallSiblingOverlapCharacters(left, right) !== null
  );
}

function stitchAtomicNeighborChunks(chunks: readonly SessionConversationChunk[]): string {
  const first = chunks[0];
  if (!first) {
    throw new Error('Recall neighbor expansion cannot stitch an empty chunk list');
  }
  let content = first.content;
  for (let index = 1; index < chunks.length; index += 1) {
    const previous = chunks[index - 1];
    const next = chunks[index];
    if (!previous || !next) {
      throw new Error(`Recall neighbor expansion missing chunk at index ${index}`);
    }
    const overlapCharacters = Math.max(0, previous.characterEnd - next.characterStart);
    if (overlapCharacters > 0) {
      content += next.content.slice(overlapCharacters);
    } else if (previous.characterEnd === next.characterStart) {
      content += next.content;
    } else {
      throw new Error(
        `Recall neighbor expansion source gap between documents ${previous.id} and ${next.id}`,
      );
    }
  }
  return content;
}

function expandRankedRecallNeighbors(
  results: readonly RankedRecallSearchResult[],
  fetchConversationChunks: (ids: string[]) => Map<string, SessionConversationChunk>,
): RankedRecallSearchResult[] {
  const neighborIds = Array.from(
    new Set(
      results.flatMap((result) =>
        isAtomicConversationChunk(result)
          ? [result.previousSiblingId, result.nextSiblingId].filter(
              (id): id is string => id !== null,
            )
          : [],
      ),
    ),
  );
  const chunksById = fetchConversationChunks(neighborIds);
  return results.map((result) => {
    if (!isAtomicConversationChunk(result)) {
      return { ...result, neighborContext: null };
    }
    const previous = result.previousSiblingId
      ? chunksById.get(result.previousSiblingId)
      : undefined;
    const next = result.nextSiblingId ? chunksById.get(result.nextSiblingId) : undefined;
    const chunks: SessionConversationChunk[] = [];
    if (previous && areContiguousAtomicSiblings(previous, result)) {
      chunks.push(previous);
    }
    chunks.push(result);
    if (next && areContiguousAtomicSiblings(result, next)) {
      chunks.push(next);
    }
    return {
      ...result,
      neighborContext:
        chunks.length === 1
          ? null
          : {
              content: stitchAtomicNeighborChunks(chunks),
              chunks,
            },
    };
  });
}

function assertRecallRankingResultLimit(resultLimit: number, errorMessage: string): void {
  if (!Number.isInteger(resultLimit) || resultLimit < 1 || resultLimit > 200) {
    throw new Error(errorMessage);
  }
}

function isEligibleHybridRecallCandidate(candidate: RecallSearchResult): boolean {
  return (
    candidate.lexical !== null ||
    candidate.identifier !== null ||
    candidate.dense === null ||
    candidate.dense.cosineDistance <= RECALL_MAX_DENSE_ONLY_COSINE_DISTANCE
  );
}

function createRecallCandidateGroups(
  candidates: readonly RecallSearchResult[],
): RecallCandidateGroup[] {
  return createCrossSessionCopyCandidateGroups(createSiblingOverlapCandidateGroups(candidates));
}

function getRecallCandidateGroupActivePrior(group: RecallCandidateGroup): number {
  return getRecallCandidateGroupMembers(group).some((candidate) => candidate.isOnActiveBranch)
    ? RECALL_ACTIVE_BRANCH_PRIOR
    : 0;
}

function createRankedRecallSearchResult(
  group: RecallCandidateGroup,
  rerankerScore: number | null,
): RankedRecallSearchResult {
  const activeBranchPrior = getRecallCandidateGroupActivePrior(group);
  return {
    ...group.representative,
    rerankerScore,
    activeBranchPrior,
    rankingScore: (rerankerScore ?? group.representative.fusedScore) + activeBranchPrior,
    duplicateOccurrences: group.duplicateOccurrences,
    neighborContext: null,
  };
}

/** Ranks fused recall candidates without Qwen while preserving duplicate and neighbor provenance. */
export function rankFusedRecallSearchResults(
  candidates: readonly RecallSearchResult[],
  resultLimit: number,
  fetchConversationChunks: (ids: string[]) => Map<string, SessionConversationChunk>,
): RankedRecallSearchResult[] {
  assertRecallRankingResultLimit(
    resultLimit,
    'Recall fused result limit invalid: expected an integer from 1 to 200',
  );
  const eligibleCandidates = candidates.filter(isEligibleHybridRecallCandidate);
  const rankedResults = createRecallCandidateGroups(eligibleCandidates)
    .map((group) => createRankedRecallSearchResult(group, null))
    .toSorted(
      (left, right) =>
        right.rankingScore - left.rankingScore ||
        right.fusedScore - left.fusedScore ||
        compareRecallDocumentIds(left.id, right.id),
    )
    .slice(0, resultLimit);
  return expandRankedRecallNeighbors(rankedResults, fetchConversationChunks);
}

/** Reranks original recall candidate text while retaining every hybrid component score. */
export async function rerankRecallSearchResults(
  options: RerankRecallSearchResultsOptions,
): Promise<RankedRecallSearchResult[]> {
  assertRecallRankingResultLimit(
    options.resultLimit,
    'Recall reranked result limit invalid: expected an integer from 1 to 200',
  );
  if (options.candidates.length === 0) {
    return [];
  }
  const candidateGroups = createRecallCandidateGroups(options.candidates);
  const scores = await options.reranker.rerankDocuments(
    options.query,
    candidateGroups.map((group) => group.representative.content),
    options.signal,
  );
  if (scores.length !== candidateGroups.length) {
    throw new Error(
      `Recall reranker score count mismatch: expected ${candidateGroups.length}, received ${scores.length}`,
    );
  }
  const rerankedResults = candidateGroups
    .map((group, index) => {
      const rerankerScore = scores[index];
      if (rerankerScore === undefined || !Number.isFinite(rerankerScore)) {
        throw new Error(
          `Recall reranker score invalid for candidate index ${index}: expected finite number`,
        );
      }
      return createRankedRecallSearchResult(group, rerankerScore);
    })
    .toSorted(
      (left, right) =>
        right.rankingScore - left.rankingScore ||
        (right.rerankerScore ?? 0) - (left.rerankerScore ?? 0) ||
        right.fusedScore - left.fusedScore ||
        compareRecallDocumentIds(left.id, right.id),
    )
    .slice(0, options.resultLimit);
  return expandRankedRecallNeighbors(rerankedResults, options.fetchConversationChunks);
}

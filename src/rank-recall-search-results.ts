import { compareRecallDocumentIds } from './compare-recall-document-ids.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

/** Small additive prior used to favor active-branch evidence without filtering other branches. */
export const RECALL_ACTIVE_BRANCH_PRIOR = 0.01;

/** Maximum cosine distance admitted for a dense conversation result. */
export const RECALL_MAX_DENSE_COSINE_DISTANCE = 0.5;

/** One bounded dense candidate; cosine distance is lower-is-better. */
export interface RecallDenseCandidate extends SessionConversationChunk {
  cosineDistance: number;
}

/** One dense candidate with a stable rank score for branch-aware result ordering. */
export interface RecallDenseSearchResult extends RecallDenseCandidate {
  denseRank: number;
  denseReciprocalRankScore: number;
}

/** Readable context reconstructed from exact same-run atomic chunk provenance. */
export interface RecallNeighborContext {
  content: string;
  chunks: SessionConversationChunk[];
}

/** One dense recall result after duplicate suppression and neighbor expansion. */
export interface RankedRecallSearchResult extends RecallDenseSearchResult {
  activeBranchPrior: number;
  rankingScore: number;
  duplicateOccurrences: RecallDenseSearchResult[];
  neighborContext: RecallNeighborContext | null;
}

interface RecallCandidateGroup {
  representative: RecallDenseSearchResult;
  duplicateOccurrences: RecallDenseSearchResult[];
}

function compareRecallCandidatePreference(
  left: RecallDenseSearchResult,
  right: RecallDenseSearchResult,
): number {
  return (
    right.denseReciprocalRankScore - left.denseReciprocalRankScore ||
    compareRecallDocumentIds(left.id, right.id)
  );
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
  left: RecallDenseSearchResult,
  right: RecallDenseSearchResult,
): boolean {
  const earlier = left.chunkIndex < right.chunkIndex ? left : right;
  const later = earlier === left ? right : left;
  return (getRecallSiblingOverlapCharacters(earlier, later) ?? 0) > 0;
}

function getRecallCandidateGroupMembers(group: RecallCandidateGroup): RecallDenseSearchResult[] {
  return [group.representative, ...group.duplicateOccurrences];
}

function mergeRecallCandidateGroups(
  groups: RecallCandidateGroup[],
  incomingMembers: readonly RecallDenseSearchResult[],
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
  candidates: readonly RecallDenseSearchResult[],
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

function areExactCrossSessionCopies(
  left: RecallDenseSearchResult,
  right: RecallDenseSearchResult,
): boolean {
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

function isAtomicConversationChunk(chunk: SessionConversationChunk): boolean {
  return (
    chunk.documentKind === 'conversation' &&
    chunk.summaryKind === null &&
    chunk.evidenceKind === 'conversation' &&
    chunk.evidencePart === 'content' &&
    (chunk.role === 'user' || chunk.role === 'assistant' || chunk.role === 'custom')
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

function assertRecallRankingResultLimit(resultLimit: number): void {
  if (!Number.isInteger(resultLimit) || resultLimit < 1 || resultLimit > 200) {
    throw new Error('Recall dense result limit invalid: expected an integer from 1 to 200');
  }
}

function createRecallCandidateGroups(
  candidates: readonly RecallDenseSearchResult[],
): RecallCandidateGroup[] {
  return createCrossSessionCopyCandidateGroups(createSiblingOverlapCandidateGroups(candidates));
}

function getRecallCandidateGroupActivePrior(group: RecallCandidateGroup): number {
  return getRecallCandidateGroupMembers(group).some((candidate) => candidate.isOnActiveBranch)
    ? RECALL_ACTIVE_BRANCH_PRIOR
    : 0;
}

function createRankedRecallSearchResult(group: RecallCandidateGroup): RankedRecallSearchResult {
  const activeBranchPrior = getRecallCandidateGroupActivePrior(group);
  return {
    ...group.representative,
    activeBranchPrior,
    rankingScore: group.representative.denseReciprocalRankScore + activeBranchPrior,
    duplicateOccurrences: group.duplicateOccurrences,
    neighborContext: null,
  };
}

/** Ranks dense recall candidates while preserving duplicate and neighbor provenance. */
export function rankDenseRecallSearchResults(
  candidates: readonly RecallDenseCandidate[],
  resultLimit: number,
  fetchConversationChunks: (ids: string[]) => Map<string, SessionConversationChunk>,
): RankedRecallSearchResult[] {
  assertRecallRankingResultLimit(resultLimit);
  for (const candidate of candidates) {
    if (!Number.isFinite(candidate.cosineDistance)) {
      throw new Error(
        `Dense recall candidate score invalid for document ${candidate.id}: expected a finite cosine distance`,
      );
    }
  }
  const denseResults: RecallDenseSearchResult[] = candidates
    .toSorted(
      (left, right) =>
        left.cosineDistance - right.cosineDistance || compareRecallDocumentIds(left.id, right.id),
    )
    .map((candidate, index) => ({
      ...candidate,
      denseRank: index + 1,
      denseReciprocalRankScore: 1 / (61 + index),
    }))
    .filter((candidate) => candidate.cosineDistance <= RECALL_MAX_DENSE_COSINE_DISTANCE);
  const rankedResults = createRecallCandidateGroups(denseResults)
    .map(createRankedRecallSearchResult)
    .toSorted(
      (left, right) =>
        right.rankingScore - left.rankingScore ||
        right.denseReciprocalRankScore - left.denseReciprocalRankScore ||
        compareRecallDocumentIds(left.id, right.id),
    )
    .slice(0, resultLimit);
  return expandRankedRecallNeighbors(rankedResults, fetchConversationChunks);
}

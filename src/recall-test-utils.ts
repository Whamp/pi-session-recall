import { RecallEvidenceRelation } from './enums.js';
import type { RecallDenseSearchResult } from './rank-recall-search-results.js';
import type { RecallConversationSearchResult } from './recall-conversation-service.js';
import type { SessionConversationChunk } from './session-conversation-index.js';

/** Test-only overrides for one complete recall evidence document fixture. */
export interface TestSessionConversationChunkOptions extends Partial<SessionConversationChunk> {
  id: string;
}

/** Builds one complete recall evidence document with deterministic provenance defaults. */
export function createTestSessionConversationChunk(
  options: TestSessionConversationChunkOptions,
): SessionConversationChunk {
  const content = options.content ?? `content ${options.id}`;
  return {
    schemaVersion: 8,
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    checksum: `checksum-${options.id}`,
    sessionId: { value: `session-${options.id}` },
    sessionPath: `/sessions/${options.id}.jsonl`,
    parentSessionPath: null,
    cwd: '/project',
    projectPath: '/project',
    projectAttribution: null,
    sessionName: `Session ${options.id}`,
    entryId: { value: `entry-${options.id}` },
    parentEntryId: null,
    childEntryIds: [],
    contributingEntryIds: [{ value: `entry-${options.id}` }],
    currentLeafId: { value: `entry-${options.id}` },
    branchPathLeafIds: [{ value: `entry-${options.id}` }],
    isOnActiveBranch: true,
    isVisibleInActiveContext: true,
    compactedByEntryIds: [],
    compactionFirstKeptEntryId: null,
    branchSummaryFromEntryId: null,
    role: 'assistant',
    timestamp: '2026-07-24T10:00:00Z',
    sourceLineStart: 2,
    sourceLineEnd: 2,
    sourceBlockStart: 0,
    sourceBlockEnd: 0,
    characterStart: 0,
    tokenStart: 0,
    tokenEnd: 4,
    tokenCount: 4,
    overlapTokenCount: 0,
    textRunId: `run-${options.id}`,
    textRunIndex: 0,
    chunkIndex: 0,
    chunkCount: 1,
    siblingIds: [],
    previousSiblingId: null,
    nextSiblingId: null,
    ...options,
    id: options.id,
    content,
    characterEnd: options.characterEnd ?? content.length,
  };
}

/** Builds one dense recall-result fixture on top of complete source provenance. */
export function createTestRecallDenseSearchResult(
  options: TestSessionConversationChunkOptions & Partial<RecallDenseSearchResult>,
): RecallDenseSearchResult {
  return {
    ...createTestSessionConversationChunk(options),
    cosineDistance: 0.1,
    denseRank: 1,
    denseReciprocalRankScore: 1 / 61,
    ...options,
  };
}

/** Builds one ranked recall-result fixture with no duplicates or neighbor expansion. */
export function createTestRankedRecallDenseSearchResult(
  options: TestSessionConversationChunkOptions & Partial<RecallConversationSearchResult>,
): RecallConversationSearchResult {
  return {
    ...createTestRecallDenseSearchResult(options),
    activeBranchPrior: 0,
    rankingScore: 0.02,
    duplicateOccurrences: [],
    neighborContext: null,
    resultKind: 'conversation',
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    ...options,
  };
}

import { RecallEvidenceRelation } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
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
    schemaVersion: 5,
    documentKind: 'conversation',
    summaryKind: null,
    evidenceKind: 'conversation',
    evidencePart: 'content',
    isDenseSearchable: true,
    checksum: `checksum-${options.id}`,
    sessionId: { value: `session-${options.id}` },
    sessionPath: `/sessions/${options.id}.jsonl`,
    parentSessionPath: null,
    cwd: '/project',
    projectPath: '/project',
    projectIdentity: null,
    projectIdentitySource: null,
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
    toolCallId: null,
    toolName: null,
    toolCallEntryId: null,
    toolResultEntryId: null,
    toolError: null,
    ...options,
    id: options.id,
    content,
    characterEnd: options.characterEnd ?? content.length,
  };
}

/** Builds one fused recall-result fixture on top of complete source provenance. */
export function createTestRecallSearchResult(
  options: TestSessionConversationChunkOptions & Partial<RecallSearchResult>,
): RecallSearchResult {
  return {
    ...createTestSessionConversationChunk(options),
    dense: { rank: 1, cosineDistance: 0.1 },
    lexical: null,
    identifier: null,
    fusedScore: 0.02,
    ...options,
  };
}

/** Builds one reranked recall-result fixture with no duplicates or neighbor expansion. */
export function createTestRankedRecallSearchResult(
  options: TestSessionConversationChunkOptions & Partial<RecallConversationSearchResult>,
): RecallConversationSearchResult {
  return {
    ...createTestRecallSearchResult(options),
    rerankerScore: 0.9,
    activeBranchPrior: 0,
    rankingScore: 0.9,
    duplicateOccurrences: [],
    neighborContext: null,
    evidenceRelation: RecallEvidenceRelation.UNRESTRICTED_GLOBAL,
    ...options,
  };
}

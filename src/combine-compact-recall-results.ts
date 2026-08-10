import type { RecallEvidenceRelation } from './enums.js';
import type { RecallCatalogInvocationSearchResult } from './openRecallCatalog.js';
import type { RankedRecallSearchResult } from './rank-recall-search-results.js';

/** One dense conversation result labeled for the compact normal-recall result union. */
export interface CompactRecallConversationResult extends RankedRecallSearchResult {
  resultKind: 'conversation';
  evidenceRelation: RecallEvidenceRelation;
}

/** One compact Invocation result with exact source provenance and SQLite rank. */
export interface CompactRecallInvocationResult extends RecallCatalogInvocationSearchResult {
  resultKind: 'invocation';
  content: string;
  evidenceRelation: RecallEvidenceRelation;
}

/** One normal recall result from either fast compact-layout store. */
export type CompactRecallSearchResult =
  | CompactRecallConversationResult
  | CompactRecallInvocationResult;

/** Version of the deterministic mixed conversation and Invocation result policy. */
export const COMPACT_RECALL_MIXED_RESULT_POLICY_VERSION = 1;

function assertCompactRecallResultLimit(limit: number): void {
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    throw new Error('Compact recall result limit invalid: expected an integer from 1 to 200');
  }
}

/**
 * Combines both fast-store result lists before the final limit, reserving at least one slot for
 * each kind when the limit permits and filling unused conversation capacity with Invocations.
 */
export function combineCompactRecallResults(
  conversations: readonly CompactRecallConversationResult[],
  invocations: readonly CompactRecallInvocationResult[],
  limit: number,
): CompactRecallSearchResult[] {
  assertCompactRecallResultLimit(limit);
  if (conversations.length === 0) {
    return invocations.slice(0, limit);
  }
  if (invocations.length === 0 || limit === 1) {
    return conversations.slice(0, limit);
  }

  const reservedInvocationCount = Math.max(1, Math.floor(limit / 3));
  const conversationCount = Math.min(conversations.length, limit - reservedInvocationCount);
  const invocationCount = Math.min(invocations.length, limit - conversationCount);
  const remainingCapacity = limit - conversationCount - invocationCount;
  const selectedConversations = conversations.slice(0, conversationCount);
  const selectedInvocations = invocations.slice(0, invocationCount + remainingCapacity);
  const combined: CompactRecallSearchResult[] = [];
  const pairCount = Math.min(selectedConversations.length, selectedInvocations.length);
  for (let index = 0; index < pairCount; index += 1) {
    const conversation = selectedConversations[index];
    const invocation = selectedInvocations[index];
    if (conversation && invocation) {
      combined.push(conversation, invocation);
    }
  }
  combined.push(...selectedConversations.slice(pairCount), ...selectedInvocations.slice(pairCount));
  return combined.slice(0, limit);
}

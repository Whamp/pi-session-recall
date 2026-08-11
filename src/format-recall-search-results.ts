import { RecallSearchScope } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type { RecallConversationSearch } from './recall-conversation-service.js';
import type { RankedRecallSearchResult } from './rank-recall-search-results.js';

function truncateRecallExcerpt(content: string, maxCharacters: number): string {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxCharacters) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxCharacters - 1)).trimEnd()}…`;
}

function formatRecallDocumentMetadata(result: RankedRecallSearchResult): string {
  if (result.documentKind === 'tool') {
    const parts = [
      `${result.evidenceKind}/${result.evidencePart}`,
      result.toolName ?? 'unknown tool',
    ];
    if (result.toolCallId) {
      parts.push(`call ${result.toolCallId}`);
    }
    if (result.toolError === true) {
      parts.push('error');
    }
    return parts.join(' · ');
  }
  if (result.documentKind === 'conversation') {
    return 'atomic conversation';
  }
  if (result.documentKind === 'turn_context') {
    return 'turn context';
  }
  return `${result.summaryKind ?? 'unknown'} summary`;
}

function formatRecallBranchLabel(result: Pick<RecallSearchResult, 'isOnActiveBranch'>): string {
  return result.isOnActiveBranch ? 'active branch' : 'abandoned branch';
}

function formatRecallEvidenceRelation(result: RecallConversationSearch['results'][number]): string {
  return result.evidenceRelation.replaceAll('_', ' ');
}

function formatRecallDuplicateOccurrence(occurrence: RecallSearchResult): string {
  return [
    `Duplicate occurrence: ${formatRecallBranchLabel(occurrence)}`,
    `${occurrence.sessionPath}#${occurrence.entryId.value}`,
    `document ${occurrence.id}`,
    `characters ${occurrence.characterStart}-${occurrence.characterEnd}`,
    `fused RRF ${occurrence.fusedScore.toFixed(4)}`,
  ].join(' · ');
}

function formatRecallScoreComponents(result: RankedRecallSearchResult): string {
  const components = [
    `ranking ${result.rankingScore.toFixed(4)}`,
    `active prior +${result.activeBranchPrior.toFixed(4)}`,
    `fused RRF ${result.fusedScore.toFixed(4)}`,
  ];
  if (result.dense) {
    components.push(
      `dense #${result.dense.rank} cosine distance ${result.dense.cosineDistance.toFixed(4)}`,
    );
  }
  if (result.lexical) {
    components.push(
      `lexical #${result.lexical.rank} FTS ${result.lexical.fullTextScore.toFixed(4)}`,
    );
  }
  if (result.identifier) {
    components.push(
      `identifier #${result.identifier.rank} FTS ${result.identifier.fullTextScore.toFixed(4)}`,
    );
  }
  return components.join(' · ');
}

/** Formats combined dense conversation and compact Invocation results with source locators. */
export function formatRecallSearchResults(
  search: RecallConversationSearch,
  maxExcerptCharacters = 2_000,
): string {
  const rankingDescription = `compact mixed retrieval v${search.searchPolicy.mixedResultPolicyVersion}`;
  const scopeDescription =
    search.searchPolicy.scope === RecallSearchScope.PROJECT
      ? `project scope for ${search.searchPolicy.invocationProjectIdentity ?? 'an unresolved project'}`
      : 'global scope';
  const countDescription = `${search.documentCounts.dense} dense documents and ${search.documentCounts.invocations} compact Invocations`;
  const lines = [
    `Recall searched ${scopeDescription} across ${countDescription} with ${rankingDescription} (active prior +${search.searchPolicy.activeBranchPrior.toFixed(4)}).`,
  ];
  if (search.results.length === 0) {
    lines.push('No matching past conversations found.');
    if (search.searchPolicy.scope === RecallSearchScope.PROJECT) {
      lines.push(
        'Retry with scope "global" to search every indexed session; project scope was not broadened automatically.',
      );
    }
    return lines.join('\n');
  }

  for (const [index, result] of search.results.entries()) {
    if (result.resultKind === 'invocation') {
      const errorLabel = result.isError === true ? ' · error' : '';
      lines.push(
        '',
        `${index + 1}. ${result.toolName} Invocation (FTS rank ${result.rank.toFixed(4)})`,
        `${result.timestamp || 'unknown time'} · ${result.kind.replaceAll('_', ' ')}${errorLabel} · session origin ${result.sessionOrigin || 'unknown'} · ${formatRecallEvidenceRelation(result)}`,
        truncateRecallExcerpt(result.searchableText, maxExcerptCharacters),
        `Source: ${result.sessionPath}:${result.sourceLineStart}-${result.sourceLineEnd}#${result.entryId}`,
      );
      continue;
    }
    const title = result.sessionName || result.sessionId.value;
    lines.push(
      '',
      `${index + 1}. ${title} (${formatRecallScoreComponents(result)})`,
      `${result.timestamp || 'unknown time'} · ${result.role} · ${formatRecallDocumentMetadata(result)} · ${formatRecallBranchLabel(result)} · session origin ${result.cwd || 'unknown'} · ${formatRecallEvidenceRelation(result)}`,
      truncateRecallExcerpt(
        result.neighborContext?.content ?? result.content,
        maxExcerptCharacters,
      ),
    );
    if (result.contributingEntryIds.length > 1) {
      lines.push(
        `Contributing entries: ${result.contributingEntryIds.map((id) => id.value).join(' → ')}`,
      );
    }
    if (result.neighborContext) {
      lines.push(
        `Expanded chunks: ${result.neighborContext.chunks
          .map((chunk) => `${chunk.id} [characters ${chunk.characterStart}-${chunk.characterEnd}]`)
          .join(' → ')}`,
      );
    }
    for (const duplicateOccurrence of result.duplicateOccurrences) {
      lines.push(formatRecallDuplicateOccurrence(duplicateOccurrence));
    }
    lines.push(
      `Source: ${result.sessionPath}:${result.sourceLineStart}-${result.sourceLineEnd}#${result.entryId.value}`,
    );
  }
  return lines.join('\n');
}

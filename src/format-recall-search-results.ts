import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type { RecallConversationSearch } from './recall-conversation-service.js';

function truncateRecallExcerpt(content: string, maxCharacters: number): string {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxCharacters) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxCharacters - 1)).trimEnd()}…`;
}

function formatRecallToolEvidenceMetadata(result: RecallSearchResult): string | null {
  if (result.documentKind !== 'tool') {
    return null;
  }
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

function formatRecallScoreComponents(result: RecallSearchResult): string {
  const components = [`fused RRF ${result.fusedScore.toFixed(4)}`];
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

/** Formats hybrid recall evidence with component scores and exact source provenance. */
export function formatRecallSearchResults(
  search: RecallConversationSearch,
  maxExcerptCharacters = 2_000,
): string {
  const lines = [
    `Recall searched ${search.totalChunks} indexed evidence documents with fusion v${search.searchPolicy.rankFusionVersion} (RRF k=${search.searchPolicy.reciprocalRankConstant}).`,
  ];
  if (search.results.length === 0) {
    lines.push('No matching past conversations found.');
    return lines.join('\n');
  }

  for (const [index, result] of search.results.entries()) {
    const title = result.sessionName || result.sessionId.value;
    const toolMetadata = formatRecallToolEvidenceMetadata(result);
    lines.push(
      '',
      `${index + 1}. ${title} (${formatRecallScoreComponents(result)})`,
      `${result.timestamp || 'unknown time'} · ${result.role}${toolMetadata ? ` · ${toolMetadata}` : ''} · ${result.cwd || 'unknown project'}`,
      truncateRecallExcerpt(result.content, maxExcerptCharacters),
    );
    if (result.toolCallEntryId || result.toolResultEntryId) {
      lines.push(
        `Call source: ${result.toolCallEntryId?.value ?? 'unknown'} · Result source: ${result.toolResultEntryId?.value ?? 'unknown'}`,
      );
    }
    lines.push(`Source: ${result.sessionPath}#${result.entryId.value}`);
  }
  return lines.join('\n');
}

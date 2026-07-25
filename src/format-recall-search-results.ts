import type { RecallSearchResult } from './fuse-recall-search-candidates.js';
import type { RecallConversationSearch } from './recall-conversation-service.js';

function truncateRecallExcerpt(content: string, maxCharacters: number): string {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxCharacters) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxCharacters - 1)).trimEnd()}…`;
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

/** Formats hybrid conversation matches with component scores and exact source provenance. */
export function formatRecallSearchResults(
  search: RecallConversationSearch,
  maxExcerptCharacters = 2_000,
): string {
  const lines = [
    `Recall searched ${search.totalChunks} indexed conversation chunks with fusion v${search.searchPolicy.rankFusionVersion} (RRF k=${search.searchPolicy.reciprocalRankConstant}).`,
  ];
  if (search.results.length === 0) {
    lines.push('No matching past conversations found.');
    return lines.join('\n');
  }

  for (const [index, result] of search.results.entries()) {
    const title = result.sessionName || result.sessionId.value;
    lines.push(
      '',
      `${index + 1}. ${title} (${formatRecallScoreComponents(result)})`,
      `${result.timestamp || 'unknown time'} · ${result.role} · ${result.cwd || 'unknown project'}`,
      truncateRecallExcerpt(result.content, maxExcerptCharacters),
      `Source: ${result.sessionPath}#${result.entryId.value}`,
    );
  }
  return lines.join('\n');
}

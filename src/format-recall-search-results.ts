import type { RecallConversationSearch } from './recall-conversation-service.js';

function truncateRecallExcerpt(content: string, maxCharacters: number): string {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxCharacters) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxCharacters - 1)).trimEnd()}…`;
}

/** Formats semantic conversation matches with the provenance needed for exact source recovery. */
export function formatRecallSearchResults(
  search: RecallConversationSearch,
  maxExcerptCharacters = 2_000,
): string {
  const lines = [
    `Recall searched ${search.totalChunks} indexed conversation chunks.`,
    `Incremental index: ${search.indexSummary.embeddedChunks} embedded, ${search.indexSummary.deletedChunks} removed.`,
  ];

  if (search.indexSummary.failedSessions.length > 0) {
    const count = search.indexSummary.failedSessions.length;
    lines.push(`Warning: ${count} session${count === 1 ? '' : 's'} failed to index.`);
  }
  if (search.results.length === 0) {
    lines.push('No matching past conversations found.');
    return lines.join('\n');
  }

  for (const [index, result] of search.results.entries()) {
    const title = result.sessionName || result.sessionId.value;
    lines.push(
      '',
      `${index + 1}. ${title} (score ${result.score.toFixed(4)})`,
      `${result.timestamp || 'unknown time'} · ${result.role} · ${result.cwd || 'unknown project'}`,
      truncateRecallExcerpt(result.content, maxExcerptCharacters),
      `Source: ${result.sessionPath}#${result.entryId.value}`,
    );
  }
  return lines.join('\n');
}

import { RecallSearchScope } from './enums.js';
import type { RecallSearchResult } from './fuse-recall-ranked-lists.js';
import type { RecallConversationSearch } from './recall-conversation-service.js';
import type { RankedRecallSearchResult } from './rank-recall-search-results.js';

function truncateRecallExcerpt(content: string, maxCharacters: number): string {
  const normalized = content.replace(/\s+/g, ' ').trim();
  if (normalized.length <= maxCharacters) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxCharacters - 1)).trimEnd()}…`;
}

function formatRecallToolEvidenceMetadata(result: RankedRecallSearchResult): string | null {
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

function formatRecallDocumentMetadata(result: RankedRecallSearchResult): string {
  const toolMetadata = formatRecallToolEvidenceMetadata(result);
  if (toolMetadata) {
    return toolMetadata;
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
    ...(result.rerankerScore === null ? [] : [`Qwen reranker ${result.rerankerScore.toFixed(4)}`]),
    `active prior +${result.activeBranchPrior.toFixed(4)}`,
    `fused RRF ${result.fusedScore.toFixed(4)}`,
    ...(result.topRankBonus !== undefined && result.topRankBonus > 0
      ? [`top-rank bonus +${result.topRankBonus.toFixed(4)}`]
      : []),
    ...(result.retrievalPositionRank === undefined ||
    result.retrievalPositionScore === undefined ||
    result.retrievalScoreWeight === undefined ||
    result.rerankerScoreWeight === undefined
      ? []
      : [
          `fused rank #${result.retrievalPositionRank} position ${result.retrievalPositionScore.toFixed(4)}`,
          `blend ${Math.round(result.retrievalScoreWeight * 100)}% retrieval / ${Math.round(result.rerankerScoreWeight * 100)}% reranker`,
        ]),
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

/** Formats hybrid or deeply reranked recall evidence with every source occurrence. */
export function formatRecallSearchResults(
  search: RecallConversationSearch,
  maxExcerptCharacters = 2_000,
): string {
  const rankingDescription =
    search.searchPolicy.rankingMode === 'hybrid'
      ? `deterministic fusion v${search.searchPolicy.rankFusionVersion} (RRF k=${search.searchPolicy.reciprocalRankConstant}) without Qwen reranking`
      : search.searchPolicy.rankingMode === 'query-planned'
        ? `agent query-planned fusion v${search.searchPolicy.rankFusionVersion} (RRF k=${search.searchPolicy.reciprocalRankConstant}) and position-aware Qwen ${search.searchPolicy.rerankerModel} policy v${search.searchPolicy.rerankPolicyVersion}`
        : `fusion v${search.searchPolicy.rankFusionVersion} (RRF k=${search.searchPolicy.reciprocalRankConstant}) and Qwen ${search.searchPolicy.rerankerModel} policy v${search.searchPolicy.rerankPolicyVersion}`;
  const scopeDescription =
    search.searchPolicy.scope === RecallSearchScope.PROJECT
      ? `project scope for ${search.searchPolicy.invocationProjectIdentity ?? 'an unresolved project'}`
      : 'global scope';
  const lines = [
    `Recall searched ${scopeDescription} across ${search.totalChunks} indexed evidence documents with ${rankingDescription} (active prior +${search.searchPolicy.activeBranchPrior.toFixed(4)}).`,
  ];
  const queryPlan = search.searchPolicy.queryPlan;
  if (queryPlan) {
    lines.push(
      `Agent query plan (${queryPlan.source}): ${queryPlan.plannedQueries.map((plannedQuery) => `${plannedQuery.type}: ${plannedQuery.query}`).join(' · ')}`,
      `Plan intent: ${queryPlan.intent ?? 'none'}`,
      `Fusion policy: submitted weight ${queryPlan.fusionPolicy.submittedQueryListWeight}, planned weight ${queryPlan.fusionPolicy.plannedQueryListWeight}, rank bonuses +${queryPlan.fusionPolicy.rankOneBonus.toFixed(4)} / +${queryPlan.fusionPolicy.rankTwoOrThreeBonus.toFixed(4)}.`,
      `Ranked lists: ${queryPlan.rankedLists.map((list) => `${list.source} ${list.admittedCandidateCount}/${list.candidateLimit} for ${list.query}`).join(' · ')}`,
    );
  }
  if (search.searchPolicy.rerankerIdentity) {
    lines.push(
      `Reranker identity: reranker profile ${search.searchPolicy.rerankerIdentity.profileId} · adapter ${search.searchPolicy.rerankerIdentity.adapterId} · cache ${search.searchPolicy.rerankerIdentity.cacheIdentity}.`,
    );
  }
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
    if (result.toolCallEntryId || result.toolResultEntryId) {
      lines.push(
        `Call source: ${result.toolCallEntryId?.value ?? 'unknown'} · Result source: ${result.toolResultEntryId?.value ?? 'unknown'}`,
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
    lines.push(`Source: ${result.sessionPath}#${result.entryId.value}`);
  }
  return lines.join('\n');
}

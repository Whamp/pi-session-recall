import type {
  RecallSourceNeighborhood,
  RecallSourceNeighborhoodEvidence,
} from './expand-recall-source-neighborhood.js';

function formatNullableSourceValue(value: string | number | null): string {
  return value === null ? 'none' : String(value);
}

function formatSourceNeighborhoodEvidence(evidence: RecallSourceNeighborhoodEvidence): string[] {
  const lines = [
    `### ${evidence.evidenceKind} · ${evidence.evidencePart} · ${evidence.role}`,
    '',
    evidence.content,
    '',
    `Contributing entries: ${evidence.contributingEntryIds.join(', ') || 'none'}`,
    `Current leaf entry ID: ${formatNullableSourceValue(evidence.currentLeafEntryId)}`,
    `Compacted by entries: ${evidence.compactedByEntryIds.join(', ') || 'none'}`,
  ];
  if (evidence.toolName !== null || evidence.toolCallId !== null) {
    lines.push(
      `Tool: ${formatNullableSourceValue(evidence.toolName)} · call ${formatNullableSourceValue(evidence.toolCallId)} · call entry ${formatNullableSourceValue(evidence.toolCallEntryId)} · result entry ${formatNullableSourceValue(evidence.toolResultEntryId)}`,
    );
  }
  if (evidence.compactionFirstKeptEntryId !== null) {
    lines.push(`Compaction first kept entry: ${evidence.compactionFirstKeptEntryId}`);
  }
  if (evidence.branchSummaryFromEntryId !== null) {
    lines.push(`Branch summary from entry: ${evidence.branchSummaryFromEntryId}`);
  }
  lines.push('Evidence occurrences:');
  for (const occurrence of evidence.occurrences) {
    lines.push(
      `- ${occurrence.evidenceOccurrenceId} · lines ${occurrence.sourceLineStart}-${occurrence.sourceLineEnd} · blocks ${formatNullableSourceValue(occurrence.sourceBlockStart)}-${formatNullableSourceValue(occurrence.sourceBlockEnd)} · chars ${occurrence.characterStart}-${occurrence.characterEnd} · tokens ${occurrence.tokenStart}-${occurrence.tokenEnd} · chunk ${occurrence.chunkIndex + 1}/${occurrence.chunkCount}`,
    );
  }
  return lines;
}

/** Formats one exact source neighborhood with source text and complete occurrence locators. */
export function formatRecallSourceNeighborhood(expansion: RecallSourceNeighborhood): string {
  const lines = [
    '# Recall source neighborhood',
    '',
    `Anchor evidence occurrence ID: ${expansion.anchorEvidenceOccurrenceId}`,
    `Physical session path: ${expansion.physicalSessionPath}`,
    `Physical source identity: ${expansion.physicalSourceIdentity}`,
    `Logical session occurrence ID: ${expansion.logicalSessionOccurrenceId}`,
    `Raw session ID: ${expansion.rawSessionId}`,
    `Requested entries: previous ${expansion.requestedEntryCounts.previous}, next ${expansion.requestedEntryCounts.next}`,
    `Returned entries: previous ${expansion.returnedEntryCounts.previous}, next ${expansion.returnedEntryCounts.next}`,
    `Selected branch-path leaf entry ID: ${formatNullableSourceValue(expansion.branchPathLeafEntryId)}`,
  ];
  for (const entry of expansion.entries) {
    lines.push(
      '',
      `## Entry ${entry.pathOrder}: ${entry.entryId}`,
      `Parent entry ID: ${formatNullableSourceValue(entry.parentEntryId)}`,
      `Entry type: ${entry.entryType}`,
      `Timestamp: ${entry.timestamp}`,
      `Source order: ${entry.sourceOrder}`,
    );
    if (entry.placeholder) {
      lines.push(
        '[Structural placeholder: this counted entry has no returnable indexed evidence.]',
      );
      continue;
    }
    for (const evidence of entry.evidence) {
      lines.push('', ...formatSourceNeighborhoodEvidence(evidence));
    }
  }
  return lines.join('\n');
}

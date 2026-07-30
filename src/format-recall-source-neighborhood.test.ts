import assert from 'node:assert/strict';
import test from 'node:test';

import type { RecallSourceNeighborhood } from './expand-recall-source-neighborhood.js';
import { formatRecallSourceNeighborhood } from './format-recall-source-neighborhood.js';

void test('source-neighborhood formatter retains source text and exact occurrence locators', () => {
  const expansion: RecallSourceNeighborhood = {
    anchorEvidenceOccurrenceId: 'occurrence-1',
    physicalSourceIdentity: 'physical-source',
    physicalSessionPath: '/disposable/sessions/source.jsonl',
    sessionsRootRelativePath: 'source.jsonl',
    logicalSessionOccurrenceId: 'logical-occurrence',
    rawSessionId: 'raw-session',
    requestedEntryCounts: { previous: 0, next: 0 },
    returnedEntryCounts: { previous: 0, next: 0 },
    branchPathLeafEntryId: null,
    entries: [
      {
        entryAnchorId: 'anchor-1',
        entryId: 'entry-1',
        parentEntryId: null,
        entryType: 'message',
        timestamp: '2026-08-15T00:00:00.000Z',
        sourceOrder: 1,
        pathOrder: 0,
        placeholder: false,
        evidence: [
          {
            documentKind: 'conversation',
            summaryKind: null,
            evidenceKind: 'conversation',
            evidencePart: 'content',
            role: 'assistant',
            content: 'source neighborhood ownership evidence',
            contributingEntryIds: ['entry-1'],
            branchPathLeafEntryIds: ['entry-1'],
            currentLeafEntryId: 'entry-1',
            compactedByEntryIds: [],
            isOnActiveBranch: true,
            isVisibleInActiveContext: true,
            toolCallId: null,
            toolName: null,
            toolCallEntryId: null,
            toolResultEntryId: null,
            toolError: null,
            compactionFirstKeptEntryId: null,
            branchSummaryFromEntryId: null,
            occurrences: [
              {
                evidenceOccurrenceId: 'occurrence-1',
                sourceLineStart: 2,
                sourceLineEnd: 2,
                sourceBlockStart: 0,
                sourceBlockEnd: 0,
                characterStart: 0,
                characterEnd: 38,
                tokenStart: 0,
                tokenEnd: 4,
                textRunIndex: 0,
                chunkIndex: 0,
                chunkCount: 1,
              },
            ],
          },
        ],
      },
    ],
  };
  const formatted = formatRecallSourceNeighborhood(expansion);
  assert.match(formatted, /source neighborhood ownership evidence/u);
  assert.match(formatted, /occurrence-1 · lines 2-2 · blocks 0-0/u);
});

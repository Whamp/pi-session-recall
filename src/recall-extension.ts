import { Type } from 'typebox';

import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
} from '@earendil-works/pi-coding-agent';

import { formatRecallSearchResults } from './format-recall-search-results.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';

/** Registers hybrid recall of past Pi conversations. Pi requires extension factories to be default exports. */
export default async function recallExtension(
  pi: Pick<ExtensionAPI, 'registerTool' | 'registerCommand'>,
): Promise<void> {
  const config = await loadRecallConversationConfig();
  const service = createRecallConversationService(config);

  pi.registerTool({
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search a prebuilt compatible index of past Pi conversations with dense, lexical, and case-preserving identifier retrieval backed by local embeddings and zvec FTS. Excludes hidden reasoning, keeps raw tool evidence lexical-only, and returns atomic or turn-context excerpts with exact contributing-entry provenance. Run /pi-session-recall-index explicitly to update the index. Output is truncated to 2000 lines or 50KB.',
    promptSnippet:
      'Search past Pi conversations by meaning or exact text and recover remembered details',
    promptGuidelines: [
      'Use pi-session-recall when a task depends on a conversation or detail from a past session and the current context does not contain reliable source evidence.',
      'Treat pi-session-recall results as search leads; cite their session path and every listed contributing entry when relying on turn-context evidence.',
    ],
    parameters: Type.Object({
      query: Type.String({
        minLength: 1,
        description:
          'Natural-language description, exact identifier, filename, command, hash, or quoted text to recover',
      }),
      limit: Type.Optional(
        Type.Integer({
          minimum: 1,
          maximum: 10,
          description: 'Maximum matches to return (default 5)',
        }),
      ),
    }),

    async execute(toolCallId, parameters, signal) {
      void toolCallId;
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = await service.search(query, parameters.limit ?? 5, signal);
      const formatted = formatRecallSearchResults(search);
      const truncation = truncateHead(formatted, {
        maxLines: DEFAULT_MAX_LINES,
        maxBytes: DEFAULT_MAX_BYTES,
      });
      const text = truncation.truncated
        ? `${truncation.content}\n\n[Recall output truncated to ${formatSize(DEFAULT_MAX_BYTES)}.]`
        : truncation.content;
      return {
        content: [{ type: 'text', text }],
        details: {
          totalChunks: search.totalChunks,
          searchPolicy: search.searchPolicy,
          sources: search.results.map((result) => ({
            documentKind: result.documentKind,
            sessionPath: result.sessionPath,
            entryId: result.entryId.value,
            contributingEntryIds: result.contributingEntryIds.map((id) => id.value),
            fusedScore: result.fusedScore,
            dense: result.dense,
            lexical: result.lexical,
            identifier: result.identifier,
          })),
        },
      };
    },
  });

  pi.registerCommand('pi-session-recall-index', {
    description:
      'Incrementally index all Pi conversations for dense and full-text search, then optimize zvec',
    async handler(argumentsText, context) {
      void argumentsText;
      context.ui.setStatus('pi-session-recall', 'indexing conversations…');
      try {
        const result = await service.index(
          undefined,
          (progress) => {
            context.ui.setStatus(
              'pi-session-recall',
              `indexing ${progress.scannedSessions}/${progress.totalSessions}`,
            );
          },
          true,
        );
        const failures = result.indexSummary.failedSessions.length;
        const message = [
          `Recall index ready: ${result.totalChunks} chunks`,
          `${result.indexSummary.cacheHits} cache hits`,
          `${result.indexSummary.newlyEmbeddedChunks} newly embedded`,
          `${result.indexSummary.embeddingRequestCount} embedding requests`,
          `${result.indexSummary.deletedChunks} removed`,
          failures > 0 ? `${failures} failed sessions` : undefined,
        ]
          .filter((part) => part !== undefined)
          .join(' · ');
        context.ui.notify(message, failures > 0 ? 'warning' : 'info');
      } finally {
        context.ui.setStatus('pi-session-recall', undefined);
      }
    },
  });
}

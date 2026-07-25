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
import { runRecallIndexCommand } from './recall-index-command.js';
import {
  MAX_RECALL_FINAL_RESULT_COUNT,
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

/** Registers hybrid recall of past Pi conversations. Pi requires extension factories to be default exports. */
export default async function recallExtension(
  pi: Pick<ExtensionAPI, 'registerTool' | 'registerCommand'>,
): Promise<void> {
  const qualityGateDecision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);
  const configured = await loadRecallConversationConfig();
  const selectedPolicy = qualityGateDecision.selectedPolicy;
  const config = selectedPolicy
    ? {
        ...configured,
        chunkPolicy: {
          maxTokens: selectedPolicy.chunkPolicy.maxTokens,
          overlapTokens: selectedPolicy.chunkPolicy.overlapTokens,
        },
        searchCandidateLimits: {
          dense: selectedPolicy.candidateCount,
          lexical: selectedPolicy.candidateCount,
          identifier: selectedPolicy.candidateCount,
        },
      }
    : configured;
  const defaultResultLimit = selectedPolicy?.finalCount ?? 5;
  const service = createRecallConversationService(config);

  pi.registerTool({
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search a prebuilt compatible index of past Pi conversations with dense, lexical, and case-preserving identifier retrieval. Search defaults to deterministic hybrid ranking; choose deep-rerank only when ambiguous evidence warrants slower local Qwen scoring. Excludes hidden reasoning, keeps raw tool evidence lexical-only, labels active and abandoned branches, and expands only valid same-run atomic neighbors with exact provenance. After the committed quality gate passes, run /pi-session-recall-index explicitly to update the index. Output is truncated to 2000 lines or 50KB.',
    promptSnippet:
      'Search past Pi conversations by meaning or exact text and recover remembered details',
    promptGuidelines: [
      'Use pi-session-recall when a task depends on a conversation or detail from a past session and the current context does not contain reliable source evidence.',
      'Treat pi-session-recall results as search leads; cite the primary source, every listed contributing entry, and any duplicate or expanded-chunk provenance used as evidence.',
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
          maximum: MAX_RECALL_FINAL_RESULT_COUNT,
          description: `Maximum matches to return (default ${defaultResultLimit})`,
        }),
      ),
      mode: Type.Optional(
        Type.Union([Type.Literal('hybrid'), Type.Literal('deep-rerank')], {
          description:
            'Ranking depth: hybrid is the fast default; deep-rerank adds slower local Qwen scoring',
        }),
      ),
    }),

    async execute(toolCallId, parameters, signal) {
      void toolCallId;
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = await service.search(query, parameters.limit ?? defaultResultLimit, {
        mode: parameters.mode ?? 'hybrid',
        ...(signal ? { signal } : {}),
      });
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
            summaryKind: result.summaryKind,
            evidenceKind: result.evidenceKind,
            sessionPath: result.sessionPath,
            entryId: result.entryId.value,
            contributingEntryIds: result.contributingEntryIds.map((id) => id.value),
            isOnActiveBranch: result.isOnActiveBranch,
            rankingScore: result.rankingScore,
            rerankerScore: result.rerankerScore,
            activeBranchPrior: result.activeBranchPrior,
            fusedScore: result.fusedScore,
            dense: result.dense,
            lexical: result.lexical,
            identifier: result.identifier,
            duplicateOccurrences: result.duplicateOccurrences.map((occurrence) => ({
              documentId: occurrence.id,
              documentKind: occurrence.documentKind,
              summaryKind: occurrence.summaryKind,
              evidenceKind: occurrence.evidenceKind,
              sessionPath: occurrence.sessionPath,
              entryId: occurrence.entryId.value,
              contributingEntryIds: occurrence.contributingEntryIds.map((id) => id.value),
              isOnActiveBranch: occurrence.isOnActiveBranch,
              characterStart: occurrence.characterStart,
              characterEnd: occurrence.characterEnd,
              fusedScore: occurrence.fusedScore,
              dense: occurrence.dense,
              lexical: occurrence.lexical,
              identifier: occurrence.identifier,
            })),
            expandedChunks:
              result.neighborContext?.chunks.map((chunk) => ({
                documentId: chunk.id,
                sessionPath: chunk.sessionPath,
                entryId: chunk.entryId.value,
                role: chunk.role,
                textRunId: chunk.textRunId,
                chunkIndex: chunk.chunkIndex,
                characterStart: chunk.characterStart,
                characterEnd: chunk.characterEnd,
              })) ?? [],
          })),
        },
      };
    },
  });

  pi.registerCommand('pi-session-recall-index', {
    description:
      'Index production sessions only after the committed quality gate passes; use --rebuild to replace an incompatible generation',
    async handler(argumentsText, context) {
      await runRecallIndexCommand({
        argumentsText,
        qualityGateDecision,
        service,
        ui: {
          setStatus(status) {
            context.ui.setStatus('pi-session-recall', status);
          },
          notify(message, level) {
            context.ui.notify(message, level);
          },
        },
      });
    },
  });
}

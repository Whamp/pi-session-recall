import { Type } from 'typebox';

import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
} from '@earendil-works/pi-coding-agent';

import { RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  MAX_AGENT_RECALL_PLANNED_QUERY_COUNT,
  type RecallConversationSearch,
  type RecallConversationService,
  type RecallPlannedRetrievalQuery,
  type RecallSearchMode,
} from './recall-conversation-service.js';
import { runRecallIndexCommand } from './recall-index-command.js';
import {
  MAX_RECALL_FINAL_RESULT_COUNT,
  readRecallQualityGateDecision,
  RECALL_QUALITY_RESULTS_PATH,
} from './recall-quality-gate.js';

/** Model-visible parameters for recall search; invocation directory is intentionally absent. */
export interface PiRecallParameters {
  query: string;
  limit?: number;
  mode?: RecallSearchMode;
  scope?: 'project' | 'global';
  plan?: readonly RecallPlannedRetrievalQuery[];
  intent?: string;
}

interface PiRecallInvocationContext {
  cwd: ExtensionContext['cwd'];
}

/** Applies trusted Pi tool context and project-default scope to one recall service search. */
export async function searchPiRecall(
  service: RecallConversationService,
  parameters: PiRecallParameters,
  context: PiRecallInvocationContext,
  defaultResultLimit: number,
  signal?: AbortSignal,
): Promise<RecallConversationSearch> {
  return service.search(parameters.query.trim(), parameters.limit ?? defaultResultLimit, {
    mode: parameters.mode ?? 'hybrid',
    scope: parameters.scope === 'global' ? RecallSearchScope.GLOBAL : RecallSearchScope.PROJECT,
    invocationDirectory: context.cwd,
    ...(parameters.plan ? { plan: parameters.plan } : {}),
    ...(parameters.intent !== undefined ? { intent: parameters.intent } : {}),
    ...(signal ? { signal } : {}),
  });
}

/** Builds the structured Pi tool details for one source-backed recall search. */
export function createPiRecallToolDetails(search: RecallConversationSearch) {
  return {
    totalChunks: search.totalChunks,
    searchPolicy: search.searchPolicy,
    sources: search.results.map((result) => ({
      documentKind: result.documentKind,
      summaryKind: result.summaryKind,
      evidenceKind: result.evidenceKind,
      evidenceRelation: result.evidenceRelation,
      sessionOrigin: result.cwd,
      projectIdentity: result.projectAttribution?.projectIdentity ?? null,
      projectIdentitySource: result.projectAttribution?.identitySource ?? null,
      sessionPath: result.sessionPath,
      entryId: result.entryId.value,
      contributingEntryIds: result.contributingEntryIds.map((id) => id.value),
      isOnActiveBranch: result.isOnActiveBranch,
      rankingScore: result.rankingScore,
      rerankerScore: result.rerankerScore,
      activeBranchPrior: result.activeBranchPrior,
      fusedScore: result.fusedScore,
      ...(result.topRankBonus === undefined
        ? {}
        : {
            topRankBonus: result.topRankBonus,
            retrievalPositionRank: result.retrievalPositionRank ?? null,
            retrievalPositionScore: result.retrievalPositionScore ?? null,
            retrievalScoreWeight: result.retrievalScoreWeight ?? null,
            rerankerScoreWeight: result.rerankerScoreWeight ?? null,
          }),
      rankedListEvidence: result.rankedListEvidence,
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
        ...(occurrence.topRankBonus === undefined ? {} : { topRankBonus: occurrence.topRankBonus }),
        rankedListEvidence: occurrence.rankedListEvidence,
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
  };
}

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
  let recallWarningHandler: ((message: string) => void) | undefined;
  const service = createRecallConversationService(config, {
    notifyWarning(message) {
      recallWarningHandler?.(message);
    },
  });
  pi.registerTool({
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search past Pi conversations with dense, lexical, and case-preserving identifier retrieval. It defaults to project scope; choose global explicitly for cross-project evidence. Search defaults to deterministic hybrid ranking; choose deep-rerank only when ambiguous evidence warrants slower local Qwen scoring, or query-planned to route an agent-supplied lex/vec/hyde plan through bounded QMD fusion and reranking. Excludes hidden reasoning and derived recall output, keeps other raw tool evidence lexical-only, labels active and abandoned branches, and expands only valid same-run atomic neighbors with exact provenance. Use /pi-session-recall-index for explicit catch-up or repair; interactive Pi lifecycle and search operations never perform whole-session maintenance. Output is truncated to 2000 lines or 50KB.',
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
        Type.Union(
          [Type.Literal('hybrid'), Type.Literal('deep-rerank'), Type.Literal('query-planned')],
          {
            description:
              'Ranking depth: hybrid is the fast default; deep-rerank adds local Qwen scoring; query-planned requires an agent-supplied plan and applies QMD fusion plus position-aware reranking',
          },
        ),
      ),
      plan: Type.Optional(
        Type.Array(
          Type.Object(
            {
              type: Type.Union([Type.Literal('lex'), Type.Literal('vec'), Type.Literal('hyde')]),
              query: Type.String({
                minLength: 1,
                description:
                  'Single-line retrieval text: keywords for lex, a semantic reformulation for vec, or a hypothetical answer passage for hyde',
              }),
            },
            { additionalProperties: false },
          ),
          {
            minItems: 1,
            maxItems: MAX_AGENT_RECALL_PLANNED_QUERY_COUNT,
            description:
              'Agent-supplied retrieval plan required by query-planned mode; lex routes only to ordinary lexical retrieval, while vec and hyde route only to dense retrieval',
          },
        ),
      ),
      intent: Type.Optional(
        Type.String({
          minLength: 1,
          description:
            'Optional context that guides deep-rerank or query-planned reranking without creating a retrieval list; hybrid mode rejects intent',
        }),
      ),
      scope: Type.Optional(
        StringEnum(['project', 'global'] as const, {
          description:
            'Corpus boundary: project is the default from Pi trusted cwd; global searches every indexed session',
        }),
      ),
    }),

    async execute(toolCallId, parameters, signal, onUpdate, context) {
      void onUpdate;
      void toolCallId;
      recallWarningHandler = (message) => context.ui.notify(message, 'warning');
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = await searchPiRecall(
        service,
        { ...parameters, query },
        context,
        defaultResultLimit,
        signal,
      );
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
        details: createPiRecallToolDetails(search),
      };
    },
  });

  pi.registerCommand('pi-session-recall-index', {
    description:
      'Index production sessions only after the committed quality gate passes; use --rebuild to replace an incompatible generation',
    async handler(argumentsText, context) {
      recallWarningHandler = (message) => context.ui.notify(message, 'warning');
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

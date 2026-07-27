import { Type } from 'typebox';

import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
} from '@earendil-works/pi-coding-agent';

import { createRecallLiveSessionIngestion } from './create-recall-live-session-ingestion.js';
import { RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationSearch,
  type RecallConversationService,
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
}

interface PiRecallInvocationContext {
  cwd: ExtensionContext['cwd'];
  sessionManager: Pick<ExtensionContext['sessionManager'], 'getSessionFile'>;
}

/** Applies trusted Pi tool context and project-default scope to one recall service search. */
export async function searchPiRecall(
  service: RecallConversationService,
  parameters: PiRecallParameters,
  context: PiRecallInvocationContext,
  defaultResultLimit: number,
  signal?: AbortSignal,
): Promise<RecallConversationSearch> {
  const activeSessionPath = context.sessionManager.getSessionFile();
  return service.search(parameters.query.trim(), parameters.limit ?? defaultResultLimit, {
    mode: parameters.mode ?? 'hybrid',
    scope: parameters.scope === 'global' ? RecallSearchScope.GLOBAL : RecallSearchScope.PROJECT,
    invocationDirectory: context.cwd,
    ...(activeSessionPath ? { activeSessionPath } : {}),
    ...(signal ? { signal } : {}),
  });
}

/** Prevents full corpus catch-up from blocking any Pi runtime during session startup. */
export function shouldRunRecallStartupCatchUp(mode: ExtensionContext['mode']): boolean {
  void mode;
  return false;
}

/** Returns whether a persistent Pi runtime should ingest sessions from lifecycle events. */
export function shouldRunRecallLifecycleIngestion(mode: ExtensionContext['mode']): boolean {
  return mode === 'tui' || mode === 'rpc';
}

/** Registers hybrid recall of past Pi conversations. Pi requires extension factories to be default exports. */
export default async function recallExtension(
  pi: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'>,
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
  const liveSessionIngestion = createRecallLiveSessionIngestion(service, (message) => {
    recallWarningHandler?.(message);
  });
  if (qualityGateDecision.automatedGatePassed && selectedPolicy) {
    pi.on('session_start', (event, context) => {
      void event;
      recallWarningHandler = (message) => context.ui.notify(message, 'warning');
      if (shouldRunRecallStartupCatchUp(context.mode)) {
        void liveSessionIngestion.catchUpSessions();
      }
    });
    pi.on('agent_settled', (event, context) => {
      void event;
      if (shouldRunRecallLifecycleIngestion(context.mode)) {
        recallWarningHandler = (message) => context.ui.notify(message, 'warning');
        void liveSessionIngestion.reconcileActiveSession(context.sessionManager.getSessionFile());
      }
    });
  }

  pi.registerTool({
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search past Pi conversations with dense, lexical, and case-preserving identifier retrieval. Recall refreshes the trusted active session before every search and keeps other sessions current from Pi lifecycle events. It defaults to project scope; choose global explicitly for cross-project evidence. Search defaults to deterministic hybrid ranking; choose deep-rerank only when ambiguous evidence warrants slower local Qwen scoring. Excludes hidden reasoning and derived recall output, keeps other raw tool evidence lexical-only, labels active and abandoned branches, and expands only valid same-run atomic neighbors with exact provenance. Use /pi-session-recall-index for manual full catch-up or repair. Output is truncated to 2000 lines or 50KB.',
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
        details: {
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

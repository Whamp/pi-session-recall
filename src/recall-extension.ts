import { Type } from 'typebox';

import { StringEnum } from '@earendil-works/pi-ai';
import type {
  ExtensionAPI,
  ExtensionContext,
  Theme,
  ToolDefinition,
  TruncationResult,
} from '@earendil-works/pi-coding-agent';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  keyHint,
  truncateHead,
} from '@earendil-works/pi-coding-agent';
import { Text, truncateToWidth } from '@earendil-works/pi-tui';

import { RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationSearch,
  type RecallConversationService,
  type RecallConversationToolService,
} from './recall-conversation-service.js';
import type { RecallIndexMaintenanceStatus } from './recall-index-maintenance-status.js';
import {
  formatSessionSourceSearchResults,
  type SessionSourceSearch,
} from './session-source-search.js';

const DEFAULT_RECALL_FINAL_RESULT_COUNT = 5;
const MAX_RECALL_FINAL_RESULT_COUNT = 10;

const PI_RECALL_PARAMETERS = Type.Object({
  query: Type.String({
    minLength: 1,
    description:
      'Natural-language description, exact identifier, filename, command, hash, or quoted text to recover',
  }),
  limit: Type.Optional(
    Type.Integer({
      minimum: 1,
      maximum: MAX_RECALL_FINAL_RESULT_COUNT,
      description: `Maximum matches to return (default ${DEFAULT_RECALL_FINAL_RESULT_COUNT})`,
    }),
  ),
  scope: Type.Optional(
    StringEnum(['project', 'global'] as const, {
      description:
        'Corpus boundary: project is the default from Pi trusted cwd; global searches every eligible session',
    }),
  ),
  source: Type.Optional(
    Type.Boolean({
      description:
        'Slowly scan original session JSONL for complete raw tool results and bash output; never runs unless true',
    }),
  ),
});

/** Model-visible parameters for read-only recall search. */
export interface PiRecallParameters {
  query: string;
  limit?: number;
  scope?: 'project' | 'global';
  source?: boolean;
}

interface PiRecallInvocationContext {
  cwd: ExtensionContext['cwd'];
}

/** Applies trusted Pi invocation context and project-default scope to one indexed search. */
export async function searchPiRecall(
  service: RecallConversationService,
  parameters: PiRecallParameters,
  context: PiRecallInvocationContext,
  signal?: AbortSignal,
): Promise<RecallConversationSearch> {
  return service.search(
    parameters.query.trim(),
    parameters.limit ?? DEFAULT_RECALL_FINAL_RESULT_COUNT,
    {
      scope: parameters.scope === 'global' ? RecallSearchScope.GLOBAL : RecallSearchScope.PROJECT,
      invocationDirectory: context.cwd,
      ...(signal ? { signal } : {}),
    },
  );
}

/** Creates complete source geometry details for every displayed indexed recall result. */
export function createPiRecallToolDetails(search: RecallConversationSearch) {
  return {
    totalChunks: search.totalChunks,
    documentCounts: search.documentCounts,
    searchPolicy: search.searchPolicy,
    sources: search.results.map((result) =>
      result.resultKind === 'invocation'
        ? {
            resultKind: result.resultKind,
            evidenceKind: 'invocation' as const,
            evidenceRelation: result.evidenceRelation,
            toolName: result.toolName,
            toolCallId: result.toolCallId,
            sessionOrigin: result.sessionOrigin,
            projectIdentity: result.projectAttribution?.projectIdentity ?? null,
            projectIdentitySource: result.projectAttribution?.identitySource ?? null,
            sessionPath: result.sessionPath,
            entryId: result.entryId,
            sourceLineStart: result.sourceLineStart,
            sourceLineEnd: result.sourceLineEnd,
            sourceBlockIndex: result.sourceBlockIndex,
            invocationRank: result.rank,
          }
        : {
            documentKind: result.documentKind,
            summaryKind: result.summaryKind,
            evidenceKind: result.evidenceKind,
            evidencePart: result.evidencePart,
            evidenceRelation: result.evidenceRelation,
            sessionOrigin: result.cwd,
            projectIdentity: result.projectAttribution?.projectIdentity ?? null,
            projectIdentitySource: result.projectAttribution?.identitySource ?? null,
            sessionPath: result.sessionPath,
            entryId: result.entryId.value,
            contributingEntryIds: result.contributingEntryIds.map((id) => id.value),
            sourceLineStart: result.sourceLineStart,
            sourceLineEnd: result.sourceLineEnd,
            sourceBlockStart: result.sourceBlockStart,
            sourceBlockEnd: result.sourceBlockEnd,
            characterStart: result.characterStart,
            characterEnd: result.characterEnd,
            isOnActiveBranch: result.isOnActiveBranch,
            rankingScore: result.rankingScore,
            activeBranchPrior: result.activeBranchPrior,
            fusedScore: result.fusedScore,
            dense: result.dense,
            lexical: result.lexical,
            identifier: result.identifier,
            duplicateOccurrences: result.duplicateOccurrences.map((occurrence) => ({
              documentId: occurrence.id,
              sessionPath: occurrence.sessionPath,
              entryId: occurrence.entryId.value,
              contributingEntryIds: occurrence.contributingEntryIds.map((id) => id.value),
              sourceLineStart: occurrence.sourceLineStart,
              sourceLineEnd: occurrence.sourceLineEnd,
              sourceBlockStart: occurrence.sourceBlockStart,
              sourceBlockEnd: occurrence.sourceBlockEnd,
              characterStart: occurrence.characterStart,
              characterEnd: occurrence.characterEnd,
            })),
            expandedChunks:
              result.neighborContext?.chunks.map((chunk) => ({
                documentId: chunk.id,
                sessionPath: chunk.sessionPath,
                entryId: chunk.entryId.value,
                sourceLineStart: chunk.sourceLineStart,
                sourceLineEnd: chunk.sourceLineEnd,
                sourceBlockStart: chunk.sourceBlockStart,
                sourceBlockEnd: chunk.sourceBlockEnd,
                characterStart: chunk.characterStart,
                characterEnd: chunk.characterEnd,
              })) ?? [],
          },
    ),
  };
}

/** Creates physical source locators and read failures for one explicit source scan. */
export function createPiRecallSourceToolDetails(search: SessionSourceSearch) {
  return {
    filesScanned: search.filesScanned,
    searchPolicy: {
      scope: search.scope,
      invocationProjectIdentity: search.invocationProjectIdentity,
      rankingMode: 'source' as const,
    },
    sources: search.results.map((result) => ({
      sessionPath: result.sessionPath,
      entryId: result.entryId,
      sourceLineStart: result.sourceLineStart,
      sourceLineEnd: result.sourceLineEnd,
      sessionOrigin: result.sessionOrigin,
    })),
    sourceFailures: search.failures,
  };
}

interface PiRecallIndexMaintenanceDetails extends Omit<RecallIndexMaintenanceStatus, 'version'> {
  ageMinutesAtExecution: number;
}

/** Search policy and source geometry retained with one indexed Pi recall tool result. */
export interface PiRecallToolDetails {
  totalChunks: number;
  documentCounts: { dense: number; invocations: number };
  searchPolicy: RecallConversationSearch['searchPolicy'];
  sources: ReturnType<typeof createPiRecallToolDetails>['sources'];
  /** Index maintenance freshness fixed at tool execution time. */
  indexMaintenanceStatus?: PiRecallIndexMaintenanceDetails;
  /** UTF-8 bytes in the bounded evidence body returned for nonzero results. */
  returnedBytes?: number;
  /** Lines in the bounded evidence body returned for nonzero results. */
  returnedLines?: number;
  truncation?: TruncationResult;
}

/** Source-scan policy, physical locators, and per-file failures retained with tool output. */
export interface PiRecallSourceToolDetails extends ReturnType<
  typeof createPiRecallSourceToolDetails
> {
  /** UTF-8 bytes in the bounded source evidence body returned for nonzero results. */
  returnedBytes?: number;
  /** Lines in the bounded source evidence body returned for nonzero results. */
  returnedLines?: number;
  truncation?: TruncationResult;
}

interface PiRecallRenderContext {
  isError: boolean;
  lastComponent?: unknown;
}

type PiRecallCallRenderTheme = Pick<Theme, 'bold' | 'fg'>;
type PiRecallRenderTheme = Pick<Theme, 'fg'>;

function createPiRecallIndexMaintenanceDetails(
  status: RecallIndexMaintenanceStatus,
  currentTime: Date,
): PiRecallIndexMaintenanceDetails {
  return {
    completedAt: status.completedAt,
    scannedSessions: status.scannedSessions,
    failedSessions: status.failedSessions,
    ageMinutesAtExecution: Math.floor(
      Math.max(0, currentTime.valueOf() - new Date(status.completedAt).valueOf()) / 60_000,
    ),
  };
}

function formatPiRecallIndexMaintenanceAge(ageMinutes: number): string {
  if (ageMinutes < 60) {
    return `${ageMinutes}m`;
  }
  if (ageMinutes < 24 * 60) {
    return `${Math.floor(ageMinutes / 60)}h`;
  }
  return `${Math.floor(ageMinutes / (24 * 60))}d`;
}

/** Creates the complete Pi recall tool definition around one conversation service. */
export function createPiRecallToolDefinition(
  service: RecallConversationToolService,
  getCurrentTime: () => Date = () => new Date(),
) {
  return {
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search the explicitly maintained compact recall database across dense conversations and compact tool-call or command Invocations. It defaults to project scope; choose global explicitly for cross-project evidence. Set source true only for a slower scan of original session JSONL when complete raw tool results, bash output, or omitted payloads are required. Source scanning never runs automatically. Every result cites the physical JSONL path and line range. Search never updates the index or writes cache data; only standalone `psr index` maintenance does. Output is truncated to 2000 lines or 50KB.',
    promptSnippet:
      'Search past Pi conversations, or explicitly scan complete raw output when indexed evidence is insufficient',
    promptGuidelines: [
      'Use pi-session-recall when a task depends on a past conversation or detail absent from the current context.',
      'Use source true only for complete raw tool results or bash output because it slowly scans original session files.',
      'Treat results as search leads and read the cited JSONL lines when surrounding source context matters.',
    ],
    parameters: PI_RECALL_PARAMETERS,

    renderCall(parameters, theme: PiRecallCallRenderTheme, context: PiRecallRenderContext) {
      const text =
        context.lastComponent instanceof Text ? context.lastComponent : new Text('', 0, 0);
      const title = theme.fg('toolTitle', theme.bold('pi-session-recall'));
      if (context.isError) {
        text.setText(title);
        return text;
      }

      const displayQuery = truncateToWidth(parameters.query.trim().replace(/\s+/gu, ' '), 60, '…');
      const sourceLabel = parameters.source ? 'source ' : '';
      text.setText(`${title} ${theme.fg('muted', `${sourceLabel}“${displayQuery}”`)}`);
      return text;
    },

    async execute(toolCallId, parameters, signal, onUpdate, context: PiRecallInvocationContext) {
      void onUpdate;
      void toolCallId;
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = parameters.source
        ? await service.searchSource(query, parameters.limit ?? DEFAULT_RECALL_FINAL_RESULT_COUNT, {
            scope:
              parameters.scope === 'global' ? RecallSearchScope.GLOBAL : RecallSearchScope.PROJECT,
            invocationDirectory: context.cwd,
            ...(signal ? { signal } : {}),
          })
        : await searchPiRecall(service, { ...parameters, query }, context, signal);
      const sourceSearch = 'filesScanned' in search;
      const formatted = sourceSearch
        ? formatSessionSourceSearchResults(search)
        : formatRecallSearchResults(search);
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
          ...(sourceSearch
            ? createPiRecallSourceToolDetails(search)
            : createPiRecallToolDetails(search)),
          ...(!sourceSearch && search.indexMaintenanceStatus
            ? {
                indexMaintenanceStatus: createPiRecallIndexMaintenanceDetails(
                  search.indexMaintenanceStatus,
                  getCurrentTime(),
                ),
              }
            : {}),
          ...(search.results.length > 0
            ? {
                returnedBytes: truncation.outputBytes,
                returnedLines: truncation.outputLines,
              }
            : {}),
          ...(truncation.truncated ? { truncation } : {}),
        },
      };
    },

    renderResult(result, options, theme: PiRecallRenderTheme, context: PiRecallRenderContext) {
      const text =
        context.lastComponent instanceof Text ? context.lastComponent : new Text('', 0, 0);
      const content = result.content[0];
      const contentText = content?.type === 'text' ? content.text : '';
      const renderedContentText = contentText
        .split('\n')
        .map((line) => theme.fg('toolOutput', line))
        .join('\n');
      if (context.isError) {
        text.setText(renderedContentText);
        return text;
      }

      const details = result.details;
      if (!details) {
        text.setText(renderedContentText);
        return text;
      }
      const resultCount = details.sources.length;
      const scope =
        details.searchPolicy.scope === RecallSearchScope.PROJECT ? 'project scope' : 'global scope';
      const sourceSearch = 'sourceFailures' in details;
      const indexMaintenanceStatus = sourceSearch ? undefined : details.indexMaintenanceStatus;
      const freshnessParts = indexMaintenanceStatus
        ? [
            `index checked ${formatPiRecallIndexMaintenanceAge(indexMaintenanceStatus.ageMinutesAtExecution)} ago`,
            ...(indexMaintenanceStatus.failedSessions > 0
              ? [
                  `${indexMaintenanceStatus.failedSessions.toLocaleString('en-US')} failed ${indexMaintenanceStatus.failedSessions === 1 ? 'session' : 'sessions'}`,
                ]
              : []),
          ]
        : [];
      if (resultCount === 0) {
        const zeroMatchSummary = [
          sourceSearch
            ? 'No matching source-backed evidence found'
            : 'No matching past conversations found',
          scope,
          ...freshnessParts,
        ].join(' · ');
        text.setText(theme.fg('toolOutput', zeroMatchSummary));
        return text;
      }
      if (options.expanded) {
        text.setText(renderedContentText);
        return text;
      }

      const resultLabel = sourceSearch
        ? resultCount === 1
          ? 'source result'
          : 'source results'
        : resultCount === 1
          ? 'recall result'
          : 'recall results';
      const summaryParts = [`${resultCount} ${resultLabel}`, scope];
      if (details.returnedBytes !== undefined && details.returnedLines !== undefined) {
        const lineLabel = details.returnedLines === 1 ? 'line' : 'lines';
        summaryParts.push(
          `${formatSize(details.returnedBytes)} / ${details.returnedLines.toLocaleString('en-US')} ${lineLabel}`,
        );
      }
      if (details.truncation?.truncated) {
        summaryParts.push('output truncated');
      }
      summaryParts.push(...freshnessParts);
      const summary = summaryParts.join(' · ');
      const rendered = `${theme.fg('toolOutput', summary)} ${theme.fg('muted', '(')}${keyHint('app.tools.expand', 'to expand')}${theme.fg('muted', ')')}`;
      text.setText(rendered);
      return text;
    },
  } satisfies ToolDefinition<
    typeof PI_RECALL_PARAMETERS,
    PiRecallToolDetails | PiRecallSourceToolDetails | undefined
  >;
}

/** Registers read-only hybrid recall; all index writes belong to the standalone `psr` CLI. */
export default async function recallExtension(
  pi: Pick<ExtensionAPI, 'registerTool'>,
): Promise<void> {
  const service = createRecallConversationService(await loadRecallConversationConfig());
  pi.registerTool(createPiRecallToolDefinition(service));
}

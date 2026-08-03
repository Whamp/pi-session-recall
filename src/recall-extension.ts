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
import { Text } from '@earendil-works/pi-tui';

import { RecallSearchScope } from './enums.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationSearch,
  type RecallConversationService,
} from './recall-conversation-service.js';

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
        'Corpus boundary: project is the default from Pi trusted cwd; global searches every indexed session',
    }),
  ),
});

/** Model-visible parameters for read-only recall search. */
export interface PiRecallParameters {
  query: string;
  limit?: number;
  scope?: 'project' | 'global';
}

interface PiRecallInvocationContext {
  cwd: ExtensionContext['cwd'];
}

/** Applies trusted Pi invocation context and project-default scope to one search. */
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

/** Creates complete source geometry details for every displayed recall result. */
export function createPiRecallToolDetails(search: RecallConversationSearch) {
  return {
    totalChunks: search.totalChunks,
    searchPolicy: search.searchPolicy,
    sources: search.results.map((result) => ({
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
    })),
  };
}

/** Search policy and source geometry retained with one Pi recall tool result. */
export interface PiRecallToolDetails {
  totalChunks: number;
  searchPolicy: RecallConversationSearch['searchPolicy'];
  sources: ReturnType<typeof createPiRecallToolDetails>['sources'];
  truncation?: TruncationResult;
}

interface PiRecallRenderContext {
  isError: boolean;
  lastComponent?: unknown;
}

type PiRecallRenderTheme = Pick<Theme, 'fg'>;

/** Creates the complete Pi recall tool definition around one conversation service. */
export function createPiRecallToolDefinition(service: RecallConversationService) {
  return {
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search the explicitly maintained Pi session index with dense, lexical, and case-preserving identifier retrieval. It defaults to project scope; choose global explicitly for cross-project evidence. Results cite the source JSONL path and line range so the surrounding records can be read directly. Search never updates the index; only standalone `psr index` maintenance does. Output is truncated to 2000 lines or 50KB.',
    promptSnippet:
      'Search past Pi conversations by meaning or exact text and recover source-backed details',
    promptGuidelines: [
      'Use pi-session-recall when a task depends on a past conversation or detail absent from the current context.',
      'Treat results as search leads and read the cited JSONL lines when surrounding source context matters.',
    ],
    parameters: PI_RECALL_PARAMETERS,

    async execute(toolCallId, parameters, signal, onUpdate, context: PiRecallInvocationContext) {
      void onUpdate;
      void toolCallId;
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = await searchPiRecall(service, { ...parameters, query }, context, signal);
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
          ...createPiRecallToolDetails(search),
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
      if (resultCount === 0) {
        text.setText(theme.fg('toolOutput', `No matching past conversations found · ${scope}`));
        return text;
      }
      if (options.expanded) {
        text.setText(renderedContentText);
        return text;
      }

      const resultLabel = resultCount === 1 ? 'recall result' : 'recall results';
      const summaryParts = [
        `${resultCount} ${resultLabel}`,
        scope,
        `${details.totalChunks.toLocaleString('en-US')} indexed documents`,
      ];
      if (details.truncation?.truncated) {
        summaryParts.push('output truncated');
      }
      const summary = summaryParts.join(' · ');
      const rendered = `${theme.fg('toolOutput', summary)} ${theme.fg('muted', '(')}${keyHint('app.tools.expand', 'to expand')}${theme.fg('muted', ')')}`;
      text.setText(rendered);
      return text;
    },
  } satisfies ToolDefinition<typeof PI_RECALL_PARAMETERS, PiRecallToolDetails | undefined>;
}

/** Registers read-only hybrid recall; all index writes belong to the standalone `psr` CLI. */
export default async function recallExtension(
  pi: Pick<ExtensionAPI, 'registerTool'>,
): Promise<void> {
  const service = createRecallConversationService(await loadRecallConversationConfig());
  pi.registerTool(createPiRecallToolDefinition(service));
}

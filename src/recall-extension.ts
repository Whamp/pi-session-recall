import { randomUUID } from 'node:crypto';

import { Type } from 'typebox';

import { StringEnum } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import {
  DEFAULT_MAX_BYTES,
  DEFAULT_MAX_LINES,
  formatSize,
  truncateHead,
} from '@earendil-works/pi-coding-agent';

import { applyRecallQualityPolicyToConversationConfig } from './applyRecallQualityPolicyToConversationConfig.js';
import {
  createConfiguredRecallInferenceRuntime,
  resolveRecallInferenceConfigurationPath,
} from './configured-recall-inference-runtime.js';
import { RecallSearchScope } from './enums.js';
import type { RecallDetachedWorkerSignal } from './create-recall-detached-worker-signal.js';
import { formatRecallSearchResults } from './format-recall-search-results.js';
import { isUnknownRecord } from './is-unknown-record.js';
import { loadRecallConversationConfig } from './recall-conversation-config.js';
import {
  readRecallFirstIndexSetupState,
  resolveRecallFirstIndexSetupStatePath,
} from './recall-first-index-setup-command.js';
import {
  createRecallConversationService,
  type RecallConversationConfig,
  MAX_AGENT_RECALL_PLANNED_QUERY_COUNT,
  type RecallConversationSearch,
  type RecallConversationService,
  type RecallPlannedRetrievalQuery,
  type RecallSearchMode,
} from './recall-conversation-service.js';
import { publishRecallWorkMarker } from './publish-recall-work-marker.js';
import {
  registerRecallLifecycleMarkers,
  type RecallLifecycleRuntimeFactory,
} from './register-recall-lifecycle-markers.js';
import { runRecallIndexCommand } from './recall-index-command.js';
import {
  readRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
} from './recall-inference-configuration.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import { createRecommendedEmbeddingGemmaConversationRuntime } from './recommended-embeddinggemma-conversation-service.js';
import {
  assertRecallInstallationConfigured,
  resolveRecallInstallationMode,
} from './resolveRecallInstallationMode.js';
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

/** One cached recall service whose inference resources are always owned by the extension. */
export interface RecallExtensionServiceRuntime {
  service: RecallConversationService;
  dispose(): Promise<void>;
}

interface OwnedRecallExtensionServiceRuntime {
  runtime: RecallExtensionServiceRuntime;
  inferenceConfigurationKey: string;
  activeOperationCount: number;
  idleResolvers: Array<() => void>;
}

interface RecallExtensionServiceRuntimeLease {
  service: RecallConversationService;
  release(): void;
}

function canonicalizeRecallInferenceCacheKeyValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(canonicalizeRecallInferenceCacheKeyValue);
  }
  if (!isUnknownRecord(value)) {
    return value;
  }
  const canonicalValue: Record<string, unknown> = {};
  for (const key of Object.keys(value).toSorted()) {
    canonicalValue[key] = canonicalizeRecallInferenceCacheKeyValue(value[key]);
  }
  return canonicalValue;
}

function createRecallInferenceRuntimeCacheKey(
  inferenceConfiguration: RecallInferenceConfiguration,
  selectedEmbeddingProfileId?: string,
): string {
  return JSON.stringify(
    canonicalizeRecallInferenceCacheKeyValue({
      inferenceConfiguration,
      selectedEmbeddingProfile: selectedEmbeddingProfileId ?? null,
    }),
  );
}

/** Explicit startup seams used by tests to avoid production recall paths and worker processes. */
export interface RecallExtensionStartupOptions {
  config?: RecallConversationConfig;
  workerSignal?: RecallDetachedWorkerSignal;
  lifecycleRuntimeFactory?: RecallLifecycleRuntimeFactory;
  createServiceRuntime?: (
    inferenceConfiguration: RecallInferenceConfiguration,
    selectedEmbeddingProfileId?: string,
  ) => RecallExtensionServiceRuntime | Promise<RecallExtensionServiceRuntime>;
  registerServiceRuntimeShutdown?: (disposeRuntime: () => Promise<void>) => void;
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
export function createPiRecallToolDetails(
  search: Pick<RecallConversationSearch, 'results' | 'searchPolicy' | 'totalChunks'>,
) {
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
  pi: Pick<ExtensionAPI, 'on' | 'registerTool' | 'registerCommand'>,
  startupOptions: RecallExtensionStartupOptions = {},
): Promise<void> {
  const qualityGateDecision = await readRecallQualityGateDecision(RECALL_QUALITY_RESULTS_PATH);
  const configured = startupOptions.config ?? (await loadRecallConversationConfig());
  const config = applyRecallQualityPolicyToConversationConfig(configured, qualityGateDecision);
  const defaultResultLimit = qualityGateDecision.selectedPolicy?.finalCount ?? 5;
  let recallWarningHandler: ((message: string) => void) | undefined;

  const recommendedEmbeddingProfile = createRecommendedEmbeddingGemmaModelProfile();

  // Lazy service resolution — recreated only when the effective inference configuration changes.
  let cachedRuntimeOwnership: OwnedRecallExtensionServiceRuntime | undefined;
  let serviceRuntimeOperationQueue = Promise.resolve();

  async function runSerializedServiceRuntimeOperation<Result>(
    operation: () => Promise<Result>,
  ): Promise<Result> {
    const previousOperation = serviceRuntimeOperationQueue;
    let completeOperation = (): void => {};
    serviceRuntimeOperationQueue = new Promise<void>((resolve) => {
      completeOperation = resolve;
    });
    await previousOperation;
    try {
      return await operation();
    } finally {
      completeOperation();
    }
  }

  async function resolveInstallationMode() {
    return resolveRecallInstallationMode(config);
  }

  async function createServiceRuntime(
    inferenceConfiguration: RecallInferenceConfiguration,
    selectedEmbeddingProfileId?: string,
  ): Promise<RecallExtensionServiceRuntime> {
    if (startupOptions.createServiceRuntime) {
      return startupOptions.createServiceRuntime(
        inferenceConfiguration,
        selectedEmbeddingProfileId,
      );
    }
    if (inferenceConfiguration.embedding) {
      return createConfiguredRecallInferenceRuntime(config, {
        inferenceConfiguration,
        onWarning(message) {
          recallWarningHandler?.(message);
        },
      });
    }
    if (selectedEmbeddingProfileId) {
      return createRecommendedEmbeddingGemmaConversationRuntime(config, {
        onWarning(message) {
          recallWarningHandler?.(message);
        },
      });
    }
    return {
      service: createRecallConversationService(config, {
        notifyWarning(message) {
          recallWarningHandler?.(message);
        },
        ...(startupOptions.workerSignal === undefined
          ? {}
          : { workerSignal: startupOptions.workerSignal }),
      }),
      async dispose() {},
    };
  }

  async function waitForServiceRuntimeIdle(
    ownership: OwnedRecallExtensionServiceRuntime,
  ): Promise<void> {
    if (ownership.activeOperationCount === 0) {
      return;
    }
    await new Promise<void>((resolve) => {
      ownership.idleResolvers.push(resolve);
    });
  }

  function createServiceRuntimeLease(
    ownership: OwnedRecallExtensionServiceRuntime,
  ): RecallExtensionServiceRuntimeLease {
    ownership.activeOperationCount += 1;
    let released = false;
    return {
      service: ownership.runtime.service,
      release() {
        if (released) {
          return;
        }
        released = true;
        ownership.activeOperationCount -= 1;
        if (ownership.activeOperationCount === 0) {
          const idleResolvers = ownership.idleResolvers.splice(0);
          for (const resolve of idleResolvers) {
            resolve();
          }
        }
      },
    };
  }

  async function acquireServiceRuntimeLeaseWithoutConcurrentReplacement(): Promise<RecallExtensionServiceRuntimeLease> {
    const [inferenceConfiguration, firstIndexSetupState] = await Promise.all([
      readRecallInferenceConfiguration(resolveRecallInferenceConfigurationPath(config)),
      readRecallFirstIndexSetupState(resolveRecallFirstIndexSetupStatePath(config)),
    ]);

    const selectedEmbeddingProfile = firstIndexSetupState.embedding?.profileId;

    if (
      !inferenceConfiguration.embedding &&
      selectedEmbeddingProfile &&
      selectedEmbeddingProfile !== recommendedEmbeddingProfile.profileId
    ) {
      throw new Error(
        `Recall configured embedding profile unsupported: ${selectedEmbeddingProfile}; run setup:recall status instead of silently selecting another profile`,
      );
    }

    const inferenceConfigurationKey = createRecallInferenceRuntimeCacheKey(
      inferenceConfiguration,
      selectedEmbeddingProfile,
    );

    if (cachedRuntimeOwnership?.inferenceConfigurationKey === inferenceConfigurationKey) {
      return createServiceRuntimeLease(cachedRuntimeOwnership);
    }

    const previousOwnership = cachedRuntimeOwnership;
    if (previousOwnership !== undefined) {
      await waitForServiceRuntimeIdle(previousOwnership);
      await previousOwnership.runtime.dispose();
      cachedRuntimeOwnership = undefined;
    }

    const newRuntime = await createServiceRuntime(inferenceConfiguration, selectedEmbeddingProfile);
    const newOwnership: OwnedRecallExtensionServiceRuntime = {
      runtime: newRuntime,
      inferenceConfigurationKey,
      activeOperationCount: 0,
      idleResolvers: [],
    };
    cachedRuntimeOwnership = newOwnership;
    return createServiceRuntimeLease(newOwnership);
  }

  async function useServiceRuntime<Result>(
    operation: (service: RecallConversationService) => Promise<Result>,
  ): Promise<Result> {
    const lease = await runSerializedServiceRuntimeOperation(
      acquireServiceRuntimeLeaseWithoutConcurrentReplacement,
    );
    try {
      return await operation(lease.service);
    } finally {
      lease.release();
    }
  }

  async function disposeCachedServiceRuntime(): Promise<void> {
    await runSerializedServiceRuntimeOperation(async () => {
      const ownership = cachedRuntimeOwnership;
      if (ownership === undefined) {
        return;
      }
      await waitForServiceRuntimeIdle(ownership);
      await ownership.runtime.dispose();
      cachedRuntimeOwnership = undefined;
    });
  }

  registerRecallLifecycleMarkers(
    pi,
    {
      async publishRecallWorkMarker(marker) {
        await publishRecallWorkMarker(marker, {
          markerSpoolDirectory: config.markerSpoolDirectory,
          workerOwnershipLockPath: config.workerOwnershipLockPath,
          trustedSessionRoots: [config.sessionsDirectory],
          ...(startupOptions.workerSignal === undefined
            ? {}
            : { workerSignal: startupOptions.workerSignal }),
        });
      },
    },
    startupOptions.lifecycleRuntimeFactory ?? {
      createRuntimeInstanceId: randomUUID,
      nowEpochMilliseconds: Date.now,
    },
  );
  if (startupOptions.registerServiceRuntimeShutdown !== undefined) {
    startupOptions.registerServiceRuntimeShutdown(disposeCachedServiceRuntime);
  } else {
    pi.on('session_shutdown', async () => {
      await disposeCachedServiceRuntime();
    });
  }
  pi.registerTool({
    name: 'pi-session-recall',
    label: 'Pi Session Recall',
    description:
      'Search past Pi conversations with dense, lexical, and case-preserving identifier retrieval. It defaults to project scope; choose global explicitly for cross-project evidence. Search defaults to deterministic hybrid ranking; choose deep-rerank only when ambiguous evidence warrants slower local Qwen scoring, or query-planned to route an agent-supplied plan or invoke the configured query planner before bounded QMD fusion and reranking. Excludes hidden reasoning and derived recall output, keeps other raw tool evidence lexical-only, labels active and abandoned branches, and expands only valid same-run atomic neighbors with exact provenance. Use /pi-session-recall-index for explicit catch-up or repair; interactive Pi lifecycle and search operations never perform whole-session maintenance. Output is truncated to 2000 lines or 50KB.',
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
              'Ranking depth: hybrid is the fast default; deep-rerank adds local Qwen scoring; query-planned uses an agent plan or the configured query planner, then applies QMD fusion plus position-aware reranking',
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
              'Optional agent-supplied retrieval plan for query-planned mode; omit it to invoke the configured query planner. lex routes only to ordinary lexical retrieval, while vec and hyde route only to dense retrieval',
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
      assertRecallInstallationConfigured(await resolveInstallationMode());
      const query = parameters.query.trim();
      if (!query) {
        throw new Error('Recall query must not be blank');
      }
      const search = await useServiceRuntime((service) =>
        searchPiRecall(service, { ...parameters, query }, context, defaultResultLimit, signal),
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
      'Index production sessions after the quality gate; use --rebuild for detached replacement work and --status, --stop, --resume, or --discard to control it',
    async handler(argumentsText, context) {
      recallWarningHandler = (message) => context.ui.notify(message, 'warning');
      assertRecallInstallationConfigured(await resolveInstallationMode());
      await useServiceRuntime((service) =>
        runRecallIndexCommand({
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
        }),
      );
    },
  });
}

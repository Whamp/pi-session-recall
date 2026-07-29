import { dirname, join } from 'node:path';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
} from './embedded-embeddinggemma-provider.js';
import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import { createQmdHttpQueryPlanningProvider } from './createQmdHttpQueryPlanningProvider.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import { EmbeddedInferenceDevicePolicy, RecallInferenceBackend } from './enums.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationDependencies,
  type RecallConversationService,
} from './recall-conversation-service.js';
import {
  readRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
  type RecallInferenceConfigurationCandidate,
} from './recall-inference-configuration.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const EMBEDDED_EMBEDDING_CANDIDATE_ID = 'recommended-embeddinggemma-embedded';
const HTTP_EMBEDDING_CANDIDATE_ID = 'recommended-embeddinggemma-http';
const EMBEDDED_RERANKING_CANDIDATE_ID = 'recommended-qwen-reranker-embedded';
const HTTP_RERANKING_CANDIDATE_ID = 'recommended-qwen-reranker-http';
const EMBEDDED_QUERY_PLANNING_CANDIDATE_ID = 'recommended-qmd-query-planner-embedded';
const HTTP_QUERY_PLANNING_CANDIDATE_ID = 'recommended-qmd-query-planner-http';

/** Disposable conversation service reconstructed from exact verified inference selections. */
export interface ConfiguredRecallInferenceRuntime {
  service: RecallConversationService;
  embeddingProvider: RecallEmbeddingProvider;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  embeddingDimensions: number;
  dispose(): Promise<void>;
}

/** Registered custom setup candidates plus the runtime factory that reconstructs them. */
export interface RecallInferenceAdapterRegistry {
  candidates: readonly RecallInferenceConfigurationCandidate[];
  createConfiguredRuntime(
    config: RecallConversationConfig,
    configuration: RecallInferenceConfiguration,
  ): ConfiguredRecallInferenceRuntime | Promise<ConfiguredRecallInferenceRuntime>;
}

/** Optional warning sink, state path, and registered custom adapter reconstruction boundaries. */
export interface ConfiguredRecallInferenceRuntimeOptions {
  onWarning?: (warning: string) => void;
  inferenceConfigurationPath?: string;
  adapterRegistries?: readonly RecallInferenceAdapterRegistry[];
  /**
   * When true, reconstructs embedding from `pendingEmbeddingReplacement.selection` if present.
   * Staging/background workers must prefer the pending profile during an approved replacement.
   */
  preferPendingEmbeddingReplacement?: boolean;
}

function readEmbeddedDevicePolicy(
  capability: string,
  policy?: string,
): EmbeddedInferenceDevicePolicy {
  if (policy === EmbeddedInferenceDevicePolicy.AUTO) {
    return EmbeddedInferenceDevicePolicy.AUTO;
  }
  if (policy === EmbeddedInferenceDevicePolicy.CPU) {
    return EmbeddedInferenceDevicePolicy.CPU;
  }
  if (policy === EmbeddedInferenceDevicePolicy.METAL) {
    return EmbeddedInferenceDevicePolicy.METAL;
  }
  if (policy === EmbeddedInferenceDevicePolicy.CUDA) {
    return EmbeddedInferenceDevicePolicy.CUDA;
  }
  if (policy === EmbeddedInferenceDevicePolicy.VULKAN) {
    return EmbeddedInferenceDevicePolicy.VULKAN;
  }
  throw new Error(
    `Recall configured ${capability} device policy unsupported: ${policy ?? 'missing'}; no device was substituted`,
  );
}

function readRequiredHttpEndpoint(capability: string, endpoint: string | null): string {
  if (!endpoint) {
    throw new Error(
      `Recall configured ${capability} HTTP endpoint missing; no backend was substituted`,
    );
  }
  return endpoint;
}

function assertExactConfiguredAdapter(
  capability: string,
  actualProfileId: string,
  expectedProfileId: string,
  actualAdapterId: string,
  expectedAdapterId: string,
): void {
  if (actualProfileId !== expectedProfileId || actualAdapterId !== expectedAdapterId) {
    throw new Error(
      `Recall configured ${capability} identity unsupported: expected ${expectedProfileId}/${expectedAdapterId}, received ${actualProfileId}/${actualAdapterId}; no model or adapter was substituted`,
    );
  }
}

function candidateMatchesSelection(
  candidate: RecallInferenceConfigurationCandidate,
  selection: NonNullable<RecallInferenceConfiguration['embedding']>,
): boolean {
  return (
    candidate.capability === selection.capability &&
    candidate.candidateId === selection.candidateId &&
    candidate.profileId === selection.profileId &&
    candidate.backend === selection.backend &&
    candidate.adapterId === selection.adapterId
  );
}

function selectionUsesBuiltInAdapter(
  selection: NonNullable<RecallInferenceConfiguration['embedding']>,
): boolean {
  return (
    selection.candidateId === EMBEDDED_EMBEDDING_CANDIDATE_ID ||
    selection.candidateId === HTTP_EMBEDDING_CANDIDATE_ID ||
    selection.candidateId === EMBEDDED_RERANKING_CANDIDATE_ID ||
    selection.candidateId === HTTP_RERANKING_CANDIDATE_ID ||
    selection.candidateId === EMBEDDED_QUERY_PLANNING_CANDIDATE_ID ||
    selection.candidateId === HTTP_QUERY_PLANNING_CANDIDATE_ID
  );
}

async function createRegisteredRecallInferenceRuntime(
  config: RecallConversationConfig,
  configuration: RecallInferenceConfiguration,
  registries: readonly RecallInferenceAdapterRegistry[],
): Promise<ConfiguredRecallInferenceRuntime | undefined> {
  const selections = [
    configuration.embedding,
    configuration.reranking,
    configuration.queryPlanning,
  ].filter((selection) => selection !== null);
  const customSelections = selections.filter(
    (selection) => !selectionUsesBuiltInAdapter(selection),
  );
  if (customSelections.length === 0) {
    return undefined;
  }
  const registry = registries.find((candidateRegistry) =>
    customSelections.every((selection) =>
      candidateRegistry.candidates.some((candidate) =>
        candidateMatchesSelection(candidate, selection),
      ),
    ),
  );
  return registry?.createConfiguredRuntime(config, configuration);
}

/** Resolves the authoritative mixed inference configuration beside index-generation data. */
export function resolveRecallInferenceConfigurationPath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'inference-configuration.json');
}

function resolveConfiguredRuntimeConfiguration(
  configuration: RecallInferenceConfiguration,
  preferPendingEmbeddingReplacement: boolean,
): RecallInferenceConfiguration {
  if (!preferPendingEmbeddingReplacement || !configuration.pendingEmbeddingReplacement) {
    return configuration;
  }
  return {
    ...configuration,
    embedding: configuration.pendingEmbeddingReplacement.selection,
  };
}

/** Creates a service from exact embedded/HTTP selections and keeps model profiles backend-neutral. */
export async function createConfiguredRecallInferenceRuntime(
  config: RecallConversationConfig,
  options: ConfiguredRecallInferenceRuntimeOptions = {},
): Promise<ConfiguredRecallInferenceRuntime> {
  const configurationPath =
    options.inferenceConfigurationPath ?? resolveRecallInferenceConfigurationPath(config);
  const persistedConfiguration = await readRecallInferenceConfiguration(configurationPath, {
    generationRegistryPath: config.generationRegistryPath,
  });
  const configuration = resolveConfiguredRuntimeConfiguration(
    persistedConfiguration,
    options.preferPendingEmbeddingReplacement === true,
  );
  const embeddingSelection = configuration.embedding;
  if (!embeddingSelection) {
    throw new Error(
      'Recall configured inference runtime requires a verified embedding capability; run setup before recall',
    );
  }
  const registeredRuntime = await createRegisteredRecallInferenceRuntime(
    config,
    configuration,
    options.adapterRegistries ?? [],
  );
  if (registeredRuntime) {
    return registeredRuntime;
  }
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const disposableProviders: Array<{ dispose(): Promise<void> }> = [];
  const tokenizerProvider = createEmbeddedEmbeddingGemmaProvider(embeddingProfile, {
    modelCacheDirectory,
    device:
      embeddingSelection.backend === RecallInferenceBackend.EMBEDDED
        ? readEmbeddedDevicePolicy('embedding', embeddingSelection.device?.policy)
        : EmbeddedInferenceDevicePolicy.CPU,
    ...(options.onWarning ? { onWarning: options.onWarning } : {}),
  });
  disposableProviders.push(tokenizerProvider);

  let embeddingProvider;
  if (embeddingSelection.candidateId === EMBEDDED_EMBEDDING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'embedding',
      embeddingSelection.profileId,
      embeddingProfile.profileId,
      embeddingSelection.adapterId,
      'node-llama-cpp-embedded-v2',
    );
    if (embeddingSelection.backend !== RecallInferenceBackend.EMBEDDED) {
      throw new Error(
        `Recall configured embedding backend mismatch for ${embeddingSelection.candidateId}: ${embeddingSelection.backend}`,
      );
    }
    embeddingProvider = tokenizerProvider;
  } else if (embeddingSelection.candidateId === HTTP_EMBEDDING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'embedding',
      embeddingSelection.profileId,
      embeddingProfile.profileId,
      embeddingSelection.adapterId,
      'llama-cpp-http-embedding-v1',
    );
    if (embeddingSelection.backend !== RecallInferenceBackend.LLAMA_CPP_HTTP) {
      throw new Error(
        `Recall configured embedding backend mismatch for ${embeddingSelection.candidateId}: ${embeddingSelection.backend}`,
      );
    }
    embeddingProvider = createLlamaCppHttpEmbeddingProvider(embeddingProfile, {
      baseUrl: readRequiredHttpEndpoint('embedding', embeddingSelection.endpoint),
      batchSize: config.embeddingBatchSize,
    });
  } else {
    throw new Error(
      `Recall configured embedding adapter unavailable: ${embeddingSelection.candidateId}; no adapter was substituted`,
    );
  }

  const dependencies: RecallConversationDependencies = {
    embeddingProfile,
    embeddingProvider,
    tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(embeddingProfile),
    loadTokenizer: () => tokenizerProvider.loadConversationTokenizer(),
    backgroundIndexServiceFactory: {
      moduleUrl: import.meta.url,
      exportName: 'createConfiguredRecallBackgroundService',
    },
  };
  configureRerankingCapability(
    configuration,
    modelCacheDirectory,
    dependencies,
    disposableProviders,
    options.onWarning ? { onWarning: options.onWarning } : {},
  );
  configureQueryPlanningCapability(
    configuration,
    modelCacheDirectory,
    dependencies,
    disposableProviders,
    options.onWarning ? { onWarning: options.onWarning } : {},
  );
  const service = createRecallConversationService(config, dependencies);
  return {
    service,
    embeddingProvider,
    loadTokenizer: () => tokenizerProvider.loadConversationTokenizer(),
    embeddingDimensions: embeddingProfile.identity.dimensions,
    async dispose() {
      for (const provider of disposableProviders.reverse()) {
        await provider.dispose();
      }
    },
  };
}

function configureRerankingCapability(
  configuration: RecallInferenceConfiguration,
  modelCacheDirectory: string,
  dependencies: RecallConversationDependencies,
  disposableProviders: Array<{ dispose(): Promise<void> }>,
  options: { onWarning?: (warning: string) => void },
): void {
  const selection = configuration.reranking;
  if (!selection) {
    dependencies.rerankingProfile = null;
    dependencies.reranker = null;
    return;
  }
  const profile = createRecommendedQwenRerankingModelProfile();
  dependencies.rerankingProfile = profile;
  if (selection.candidateId === EMBEDDED_RERANKING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'reranking',
      selection.profileId,
      profile.profileId,
      selection.adapterId,
      'node-llama-cpp-qwen-reranking-logit-recovery-v1',
    );
    if (selection.backend !== RecallInferenceBackend.EMBEDDED) {
      throw new Error(
        `Recall configured reranking backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    const provider = createEmbeddedQwenRerankingProvider(profile, {
      modelCacheDirectory,
      device: readEmbeddedDevicePolicy('reranking', selection.device?.policy),
      ...(options.onWarning ? { onWarning: options.onWarning } : {}),
    });
    dependencies.reranker = provider;
    disposableProviders.push(provider);
    return;
  }
  if (selection.candidateId === HTTP_RERANKING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'reranking',
      selection.profileId,
      profile.profileId,
      selection.adapterId,
      'llama-cpp-http-reranking-v1',
    );
    if (selection.backend !== RecallInferenceBackend.LLAMA_CPP_HTTP) {
      throw new Error(
        `Recall configured reranking backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    dependencies.reranker = createQwenHttpRerankingProvider(profile, {
      baseUrl: readRequiredHttpEndpoint('reranking', selection.endpoint),
    });
    return;
  }
  throw new Error(
    `Recall configured reranking adapter unavailable: ${selection.candidateId}; no adapter was substituted`,
  );
}

function configureQueryPlanningCapability(
  configuration: RecallInferenceConfiguration,
  modelCacheDirectory: string,
  dependencies: RecallConversationDependencies,
  disposableProviders: Array<{ dispose(): Promise<void> }>,
  options: { onWarning?: (warning: string) => void },
): void {
  const selection = configuration.queryPlanning;
  if (!selection) {
    return;
  }
  const profile = createRecommendedQmdQueryPlanningModelProfile();
  dependencies.queryPlanningProfile = profile;
  if (selection.candidateId === EMBEDDED_QUERY_PLANNING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'query planning',
      selection.profileId,
      profile.profileId,
      selection.adapterId,
      'node-llama-cpp-qmd-query-planning-v1',
    );
    if (selection.backend !== RecallInferenceBackend.EMBEDDED) {
      throw new Error(
        `Recall configured query planning backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
      modelCacheDirectory,
      device: readEmbeddedDevicePolicy('query planning', selection.device?.policy),
      ...(options.onWarning ? { onWarning: options.onWarning } : {}),
    });
    dependencies.queryPlanner = provider;
    disposableProviders.push(provider);
    return;
  }
  if (selection.candidateId === HTTP_QUERY_PLANNING_CANDIDATE_ID) {
    assertExactConfiguredAdapter(
      'query planning',
      selection.profileId,
      profile.profileId,
      selection.adapterId,
      'llama-cpp-http-query-planning-v1',
    );
    if (selection.backend !== RecallInferenceBackend.LLAMA_CPP_HTTP) {
      throw new Error(
        `Recall configured query planning backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    dependencies.queryPlanner = createQmdHttpQueryPlanningProvider(profile, {
      baseUrl: readRequiredHttpEndpoint('query planning', selection.endpoint),
    });
    return;
  }
  throw new Error(
    `Recall configured query planning adapter unavailable: ${selection.candidateId}; no adapter was substituted`,
  );
}

/** Reconstructs the exact configured embedding runtime inside a detached staging worker. */
export async function createConfiguredRecallBackgroundService(
  config: RecallConversationConfig,
): Promise<RecallConversationService> {
  return (
    await createConfiguredRecallInferenceRuntime(config, {
      preferPendingEmbeddingReplacement: true,
    })
  ).service;
}

import { dirname, join } from 'node:path';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
  type EmbeddedInferenceDevicePolicy,
} from './embedded-embeddinggemma-provider.js';
import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { createLlamaCppHttpEmbeddingProvider } from './llama-cpp-http-embedding-provider.js';
import { createQmdHttpQueryPlanningProvider } from './qmd-http-query-planning-provider.js';
import { createQwenHttpRerankingProvider } from './qwen-http-reranking-provider.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationDependencies,
  type RecallConversationService,
} from './recall-conversation-service.js';
import {
  readRecallInferenceConfiguration,
  type RecallInferenceConfiguration,
} from './recall-inference-configuration.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

const EMBEDDED_EMBEDDING_CANDIDATE_ID = 'recommended-embeddinggemma-embedded';
const HTTP_EMBEDDING_CANDIDATE_ID = 'recommended-embeddinggemma-http';
const EMBEDDED_RERANKING_CANDIDATE_ID = 'recommended-qwen-reranker-embedded';
const HTTP_RERANKING_CANDIDATE_ID = 'recommended-qwen-reranker-http';
const EMBEDDED_QUERY_PLANNING_CANDIDATE_ID = 'recommended-qmd-query-planner-embedded';
const HTTP_QUERY_PLANNING_CANDIDATE_ID = 'recommended-qmd-query-planner-http';

/** Disposable conversation service reconstructed from exact verified inference selections. */
export interface ConfiguredRecallInferenceRuntime {
  service: RecallConversationService;
  dispose(): Promise<void>;
}

/** Optional warning sink for automatic same-profile CPU fallback during embedded execution. */
export interface ConfiguredRecallInferenceRuntimeOptions {
  onWarning?: (warning: string) => void;
  inferenceConfigurationPath?: string;
}

function readEmbeddedDevicePolicy(
  capability: string,
  policy: string | undefined,
): EmbeddedInferenceDevicePolicy {
  if (
    policy === 'auto' ||
    policy === 'cpu' ||
    policy === 'metal' ||
    policy === 'cuda' ||
    policy === 'vulkan'
  ) {
    return policy;
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

/** Resolves the authoritative mixed inference configuration beside index-generation data. */
export function resolveRecallInferenceConfigurationPath(config: RecallConversationConfig): string {
  return join(dirname(config.manifestPath), 'inference-configuration.json');
}

/** Creates a service from exact embedded/HTTP selections and keeps model profiles backend-neutral. */
export async function createConfiguredRecallInferenceRuntime(
  config: RecallConversationConfig,
  options: ConfiguredRecallInferenceRuntimeOptions = {},
): Promise<ConfiguredRecallInferenceRuntime> {
  const configurationPath =
    options.inferenceConfigurationPath ?? resolveRecallInferenceConfigurationPath(config);
  const configuration = await readRecallInferenceConfiguration(configurationPath);
  const embeddingSelection = configuration.embedding;
  if (!embeddingSelection) {
    throw new Error(
      'Recall configured inference runtime requires a verified embedding capability; run setup before recall',
    );
  }
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const disposableProviders: Array<{ dispose(): Promise<void> }> = [];
  const tokenizerProvider = createEmbeddedEmbeddingGemmaProvider(embeddingProfile, {
    modelCacheDirectory,
    device:
      embeddingSelection.backend === 'embedded'
        ? readEmbeddedDevicePolicy('embedding', embeddingSelection.device?.policy)
        : 'cpu',
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
    if (embeddingSelection.backend !== 'embedded') {
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
    if (embeddingSelection.backend !== 'llama-cpp-http') {
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
    options.onWarning,
  );
  configureQueryPlanningCapability(
    configuration,
    modelCacheDirectory,
    dependencies,
    disposableProviders,
    options.onWarning,
  );
  const service = createRecallConversationService(config, dependencies);
  return {
    service,
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
  onWarning: ((warning: string) => void) | undefined,
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
    if (selection.backend !== 'embedded') {
      throw new Error(
        `Recall configured reranking backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    const provider = createEmbeddedQwenRerankingProvider(profile, {
      modelCacheDirectory,
      device: readEmbeddedDevicePolicy('reranking', selection.device?.policy),
      ...(onWarning ? { onWarning } : {}),
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
    if (selection.backend !== 'llama-cpp-http') {
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
  onWarning: ((warning: string) => void) | undefined,
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
    if (selection.backend !== 'embedded') {
      throw new Error(
        `Recall configured query planning backend mismatch for ${selection.candidateId}: ${selection.backend}`,
      );
    }
    const provider = createEmbeddedQmdQueryPlanningProvider(profile, {
      modelCacheDirectory,
      device: readEmbeddedDevicePolicy('query planning', selection.device?.policy),
      ...(onWarning ? { onWarning } : {}),
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
    if (selection.backend !== 'llama-cpp-http') {
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
  return (await createConfiguredRecallInferenceRuntime(config)).service;
}

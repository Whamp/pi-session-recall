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
import {
  createRecommendedOptionalInferenceCandidates,
  type RecommendedOptionalInferenceCandidateOptions,
} from './createRecommendedOptionalInferenceCandidates.js';
import {
  createRecommendedEmbeddingGemmaHttpInferenceCandidate,
  createRecommendedEmbeddingGemmaInferenceCandidate,
} from './recommended-embeddinggemma-inference-candidate.js';
import {
  EmbeddedInferenceDevicePolicy,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import type { RecallBackgroundIndexServiceFactory } from './recall-background-index-build.js';
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
import type {
  RecallEmbeddingProvider,
  RecallIdentifiedQueryPlanningProvider,
  RecallIdentifiedRerankingProvider,
} from './recall-inference-capabilities.js';
import type { RecallTokenizerManifestIdentity } from './recall-index-manifest.js';
import {
  createRecommendedEmbeddingGemmaModelProfile,
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
  type RecallEmbeddingModelProfile,
  type RecallQueryPlanningModelProfile,
  type RecallRerankingModelProfile,
} from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const EMBEDDED_DEVICE_POLICY_BY_NAME: Readonly<
  Record<string, EmbeddedInferenceDevicePolicy | undefined>
> = Object.freeze({
  [EmbeddedInferenceDevicePolicy.AUTO]: EmbeddedInferenceDevicePolicy.AUTO,
  [EmbeddedInferenceDevicePolicy.CPU]: EmbeddedInferenceDevicePolicy.CPU,
  [EmbeddedInferenceDevicePolicy.METAL]: EmbeddedInferenceDevicePolicy.METAL,
  [EmbeddedInferenceDevicePolicy.CUDA]: EmbeddedInferenceDevicePolicy.CUDA,
  [EmbeddedInferenceDevicePolicy.VULKAN]: EmbeddedInferenceDevicePolicy.VULKAN,
});

/** Disposable conversation service reconstructed from exact verified inference selections. */
export interface ConfiguredRecallInferenceRuntime {
  service: RecallConversationService;
  embeddingProvider: RecallEmbeddingProvider;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  embeddingDimensions: number;
  dispose(): Promise<void>;
}

/** Reconstructed embedding capability with exact profile, provider, and tokenizer semantics. */
export interface ConfiguredRecallEmbeddingCapability {
  capability: RecallInferenceCapability.EMBEDDING;
  profile: RecallEmbeddingModelProfile;
  provider: RecallEmbeddingProvider;
  tokenizerIdentity: RecallTokenizerManifestIdentity;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  embeddingDimensions: number;
  backgroundIndexServiceFactory?: RecallBackgroundIndexServiceFactory;
  dispose(): Promise<void>;
}

/** Reconstructed reranking capability with its exact profile and identified provider. */
export interface ConfiguredRecallRerankingCapability {
  capability: RecallInferenceCapability.RERANKING;
  profile: RecallRerankingModelProfile;
  provider: RecallIdentifiedRerankingProvider;
  dispose(): Promise<void>;
}

/** Reconstructed query-planning capability with its exact profile and identified provider. */
export interface ConfiguredRecallQueryPlanningCapability {
  capability: RecallInferenceCapability.QUERY_PLANNING;
  profile: RecallQueryPlanningModelProfile;
  provider: RecallIdentifiedQueryPlanningProvider;
  dispose(): Promise<void>;
}

/** One capability runtime reconstructed from an exact persisted inference selection. */
export type ConfiguredRecallInferenceCapability =
  | ConfiguredRecallEmbeddingCapability
  | ConfiguredRecallRerankingCapability
  | ConfiguredRecallQueryPlanningCapability;

/** Optional runtime signals passed while reconstructing exact inference adapters. */
export interface RecallInferenceAdapterFactoryOptions {
  onWarning?: (warning: string) => void;
}

/** Configuration, persisted selection, and signals available to one adapter factory. */
export interface RecallInferenceAdapterFactoryContext extends RecallInferenceAdapterFactoryOptions {
  config: RecallConversationConfig;
  selection: NonNullable<RecallInferenceConfiguration['embedding']>;
}

/** One setup candidate paired with its capability-specific runtime factory. */
export interface RecallInferenceAdapterRegistration {
  candidate: RecallInferenceConfigurationCandidate;
  createConfiguredCapability(
    context: RecallInferenceAdapterFactoryContext,
  ): ConfiguredRecallInferenceCapability | Promise<ConfiguredRecallInferenceCapability>;
}

/** Registered setup candidates and capability factories resolved by exact persisted identity. */
export interface RecallInferenceAdapterRegistry {
  registrations: readonly RecallInferenceAdapterRegistration[];
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
  const configuredPolicy = policy ? EMBEDDED_DEVICE_POLICY_BY_NAME[policy] : undefined;
  if (configuredPolicy) {
    return configuredPolicy;
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

function formatConfiguredCapability(capability: RecallInferenceCapability): string {
  return capability === RecallInferenceCapability.QUERY_PLANNING ? 'query planning' : capability;
}

function resolveExactInferenceAdapterRegistration(
  registrations: readonly RecallInferenceAdapterRegistration[],
  selection: NonNullable<RecallInferenceConfiguration['embedding']>,
): RecallInferenceAdapterRegistration {
  const exactMatches = registrations.filter(({ candidate }) =>
    candidateMatchesSelection(candidate, selection),
  );
  const [exactMatch] = exactMatches;
  const capability = formatConfiguredCapability(selection.capability);
  if (exactMatches.length > 1) {
    throw new Error(
      `Recall configured ${capability} adapter ambiguous: ${selection.candidateId}/${selection.profileId}/${selection.backend}/${selection.adapterId}`,
    );
  }
  if (exactMatch) {
    return exactMatch;
  }

  const candidateIdMatches = registrations.filter(
    ({ candidate }) =>
      candidate.capability === selection.capability &&
      candidate.candidateId === selection.candidateId,
  );
  const [expectedRegistration] = candidateIdMatches;
  if (!expectedRegistration) {
    throw new Error(
      `Recall configured ${capability} adapter unavailable: ${selection.candidateId}; no adapter was substituted`,
    );
  }
  const identityMatch = candidateIdMatches.find(
    ({ candidate }) =>
      candidate.profileId === selection.profileId && candidate.adapterId === selection.adapterId,
  );
  if (!identityMatch) {
    throw new Error(
      `Recall configured ${capability} identity unsupported: expected ${expectedRegistration.candidate.profileId}/${expectedRegistration.candidate.adapterId}, received ${selection.profileId}/${selection.adapterId}; no model or adapter was substituted`,
    );
  }
  throw new Error(
    `Recall configured ${capability} backend mismatch for ${selection.candidateId}: ${selection.backend}`,
  );
}

function findBuiltInInferenceCandidate(
  candidates: readonly RecallInferenceConfigurationCandidate[],
  capability: RecallInferenceCapability,
  backend: RecallInferenceBackend,
): RecallInferenceConfigurationCandidate {
  const candidate = candidates.find(
    (item) => item.capability === capability && item.backend === backend,
  );
  if (!candidate) {
    throw new Error(
      `Recall built-in ${capability} adapter registration missing for backend ${backend}`,
    );
  }
  return candidate;
}

/** Creates the six built-in setup candidates and their exact capability runtime factories. */
export function createRecommendedRecallInferenceAdapterRegistry(
  config: RecallConversationConfig,
  options: RecommendedOptionalInferenceCandidateOptions = {},
): RecallInferenceAdapterRegistry {
  const candidates = createRecommendedOptionalInferenceCandidates(config, options);
  const embeddingProfile = createRecommendedEmbeddingGemmaModelProfile();
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  return {
    registrations: [
      {
        candidate: createRecommendedEmbeddingGemmaInferenceCandidate(config),
        createConfiguredCapability({ selection, onWarning }) {
          const provider = createEmbeddedEmbeddingGemmaProvider(embeddingProfile, {
            modelCacheDirectory,
            device: readEmbeddedDevicePolicy('embedding', selection.device?.policy),
            ...(onWarning ? { onWarning } : {}),
          });
          return {
            capability: RecallInferenceCapability.EMBEDDING,
            profile: embeddingProfile,
            provider,
            tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(embeddingProfile),
            loadTokenizer: () => provider.loadConversationTokenizer(),
            embeddingDimensions: embeddingProfile.identity.dimensions,
            dispose: () => provider.dispose(),
          };
        },
      },
      {
        candidate: createRecommendedEmbeddingGemmaHttpInferenceCandidate(config),
        createConfiguredCapability({ selection, onWarning }) {
          const tokenizerProvider = createEmbeddedEmbeddingGemmaProvider(embeddingProfile, {
            modelCacheDirectory,
            device: EmbeddedInferenceDevicePolicy.CPU,
            ...(onWarning ? { onWarning } : {}),
          });
          return {
            capability: RecallInferenceCapability.EMBEDDING,
            profile: embeddingProfile,
            provider: createLlamaCppHttpEmbeddingProvider(embeddingProfile, {
              baseUrl: readRequiredHttpEndpoint('embedding', selection.endpoint),
              batchSize: config.embeddingBatchSize,
            }),
            tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(embeddingProfile),
            loadTokenizer: () => tokenizerProvider.loadConversationTokenizer(),
            embeddingDimensions: embeddingProfile.identity.dimensions,
            dispose: () => tokenizerProvider.dispose(),
          };
        },
      },
      {
        candidate: findBuiltInInferenceCandidate(
          candidates,
          RecallInferenceCapability.RERANKING,
          RecallInferenceBackend.EMBEDDED,
        ),
        createConfiguredCapability({ selection, onWarning }) {
          const provider = createEmbeddedQwenRerankingProvider(rerankingProfile, {
            modelCacheDirectory,
            device: readEmbeddedDevicePolicy('reranking', selection.device?.policy),
            ...(onWarning ? { onWarning } : {}),
          });
          return {
            capability: RecallInferenceCapability.RERANKING,
            profile: rerankingProfile,
            provider,
            dispose: () => provider.dispose(),
          };
        },
      },
      {
        candidate: findBuiltInInferenceCandidate(
          candidates,
          RecallInferenceCapability.RERANKING,
          RecallInferenceBackend.LLAMA_CPP_HTTP,
        ),
        createConfiguredCapability({ selection }) {
          return {
            capability: RecallInferenceCapability.RERANKING,
            profile: rerankingProfile,
            provider: createQwenHttpRerankingProvider(rerankingProfile, {
              baseUrl: readRequiredHttpEndpoint('reranking', selection.endpoint),
            }),
            async dispose() {},
          };
        },
      },
      {
        candidate: findBuiltInInferenceCandidate(
          candidates,
          RecallInferenceCapability.QUERY_PLANNING,
          RecallInferenceBackend.EMBEDDED,
        ),
        createConfiguredCapability({ selection, onWarning }) {
          const provider = createEmbeddedQmdQueryPlanningProvider(queryPlanningProfile, {
            modelCacheDirectory,
            device: readEmbeddedDevicePolicy('query planning', selection.device?.policy),
            ...(onWarning ? { onWarning } : {}),
          });
          return {
            capability: RecallInferenceCapability.QUERY_PLANNING,
            profile: queryPlanningProfile,
            provider,
            dispose: () => provider.dispose(),
          };
        },
      },
      {
        candidate: findBuiltInInferenceCandidate(
          candidates,
          RecallInferenceCapability.QUERY_PLANNING,
          RecallInferenceBackend.LLAMA_CPP_HTTP,
        ),
        createConfiguredCapability({ selection }) {
          return {
            capability: RecallInferenceCapability.QUERY_PLANNING,
            profile: queryPlanningProfile,
            provider: createQmdHttpQueryPlanningProvider(queryPlanningProfile, {
              baseUrl: readRequiredHttpEndpoint('query planning', selection.endpoint),
            }),
            async dispose() {},
          };
        },
      },
    ],
  };
}

function readConfiguredInferenceSelections(
  configuration: RecallInferenceConfiguration,
): Array<NonNullable<RecallInferenceConfiguration['embedding']>> {
  const slots = [
    {
      capability: RecallInferenceCapability.EMBEDDING,
      selection: configuration.embedding,
    },
    {
      capability: RecallInferenceCapability.RERANKING,
      selection: configuration.reranking,
    },
    {
      capability: RecallInferenceCapability.QUERY_PLANNING,
      selection: configuration.queryPlanning,
    },
  ];
  const selections: Array<NonNullable<RecallInferenceConfiguration['embedding']>> = [];
  for (const { capability, selection } of slots) {
    if (!selection) {
      continue;
    }
    if (selection.capability !== capability) {
      throw new Error(
        `Recall configured ${formatConfiguredCapability(capability)} capability mismatch: received ${formatConfiguredCapability(selection.capability)}`,
      );
    }
    selections.push(selection);
  }
  return selections;
}

async function createRegisteredRecallInferenceRuntime(
  config: RecallConversationConfig,
  configuration: RecallInferenceConfiguration,
  registries: readonly RecallInferenceAdapterRegistry[],
  options: RecallInferenceAdapterFactoryOptions,
): Promise<ConfiguredRecallInferenceRuntime> {
  const selections = readConfiguredInferenceSelections(configuration);
  const registrations = registries.flatMap((registry) => registry.registrations);
  const selectedAdapters = selections.map((selection) => ({
    selection,
    registration: resolveExactInferenceAdapterRegistration(registrations, selection),
  }));

  const capabilities: ConfiguredRecallInferenceCapability[] = [];
  for (const { selection, registration } of selectedAdapters) {
    const capability = await registration.createConfiguredCapability({
      config,
      selection,
      ...options,
    });
    if (capability.capability !== selection.capability) {
      throw new Error(
        `Recall configured ${formatConfiguredCapability(selection.capability)} adapter ${selection.candidateId} returned ${formatConfiguredCapability(capability.capability)} capability`,
      );
    }
    capabilities.push(capability);
  }
  const embedding = capabilities.find(
    (capability) => capability.capability === RecallInferenceCapability.EMBEDDING,
  );
  if (!embedding) {
    throw new Error('Recall configured inference registry returned no embedding capability');
  }
  const reranking = capabilities.find(
    (capability) => capability.capability === RecallInferenceCapability.RERANKING,
  );
  const queryPlanning = capabilities.find(
    (capability) => capability.capability === RecallInferenceCapability.QUERY_PLANNING,
  );
  const dependencies: RecallConversationDependencies = {
    embeddingProfile: embedding.profile,
    embeddingProvider: embedding.provider,
    tokenizerIdentity: embedding.tokenizerIdentity,
    loadTokenizer: () => embedding.loadTokenizer(),
    rerankingProfile: reranking?.profile ?? null,
    reranker: reranking?.provider ?? null,
    ...(queryPlanning
      ? {
          queryPlanningProfile: queryPlanning.profile,
          queryPlanner: queryPlanning.provider,
        }
      : {}),
    backgroundIndexServiceFactory: embedding.backgroundIndexServiceFactory ?? {
      moduleUrl: import.meta.url,
      exportName: 'createConfiguredRecallBackgroundService',
    },
  };
  const service = createRecallConversationService(config, dependencies);
  return {
    service,
    embeddingProvider: embedding.provider,
    loadTokenizer: () => embedding.loadTokenizer(),
    embeddingDimensions: embedding.embeddingDimensions,
    async dispose() {
      for (const capability of capabilities.reverse()) {
        await capability.dispose();
      }
    },
  };
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
  return createRegisteredRecallInferenceRuntime(
    config,
    configuration,
    [createRecommendedRecallInferenceAdapterRegistry(config), ...(options.adapterRegistries ?? [])],
    options.onWarning ? { onWarning: options.onWarning } : {},
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

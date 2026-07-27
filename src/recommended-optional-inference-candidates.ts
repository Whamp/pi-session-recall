import { dirname, join } from 'node:path';

import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import type { EmbeddedInferenceDevicePolicy } from './embedded-embeddinggemma-provider.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { createQmdHttpQueryPlanningProvider } from './qmd-http-query-planning-provider.js';
import { createQwenHttpRerankingProvider } from './qwen-http-reranking-provider.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import type { RecallPlannedRetrievalQuery } from './recall-inference-capabilities.js';
import type {
  RecallInferenceArtifactState,
  RecallInferenceConfigurationCandidate,
  RecallInferenceDeviceStatus,
} from './recall-inference-configuration.js';
import {
  createRecallModelArtifactCache,
  type RecallDownloadableModelProfile,
} from './recall-model-artifact-cache.js';
import {
  createRecommendedQmdQueryPlanningModelProfile,
  createRecommendedQwenRerankingModelProfile,
} from './recall-model-profiles.js';

/** Fixed independent scores required before a reranking candidate can be accepted. */
export interface RecallRerankingSetupConformanceEvidence {
  query: string;
  documents: readonly string[];
  expectedScores: readonly number[];
  maximumAbsoluteDifference?: number;
}

/** Fixed independent typed plan required before a query planner candidate can be accepted. */
export interface RecallQueryPlanningSetupConformanceEvidence {
  expectedPlan: readonly Readonly<RecallPlannedRetrievalQuery>[];
}

/** Optional built-in evidence and backend settings supplied by release acceptance. */
export interface RecommendedOptionalInferenceCandidateOptions {
  rerankingConformance?: RecallRerankingSetupConformanceEvidence;
  queryPlanningConformance?: RecallQueryPlanningSetupConformanceEvidence;
  queryPlanningBaseUrl?: string;
  device?: EmbeddedInferenceDevicePolicy;
}

function mapOptionalModelArtifactState(state: string): RecallInferenceArtifactState {
  if (
    state === 'valid' ||
    state === 'missing' ||
    state === 'partial' ||
    state === 'corrupt' ||
    state === 'incompatible'
  ) {
    return state;
  }
  throw new Error(`Recall optional inference artifact state unsupported: ${state}`);
}

function createOptionalArtifactBoundaries(
  config: RecallConversationConfig,
  profile: RecallDownloadableModelProfile,
): Pick<
  RecallInferenceConfigurationCandidate,
  'artifact' | 'inspectHealth' | 'prepareArtifact' | 'repairArtifact'
> {
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const cache = createRecallModelArtifactCache({ cacheDirectory: modelCacheDirectory, profile });
  return {
    artifact: {
      path: join(modelCacheDirectory, profile.source.artifact),
      repository: profile.source.repository,
      revision: profile.source.revision,
      sha256: profile.source.sha256,
      byteSize: profile.source.byteSize,
    },
    async inspectHealth() {
      const inspection = await cache.inspectArtifact();
      return {
        artifactState: mapOptionalModelArtifactState(inspection.status.state),
        requiredRepair: inspection.status.state === 'valid' ? null : inspection.status.repair,
      };
    },
    async prepareArtifact(approved) {
      await cache.downloadArtifact({ approved });
    },
    async repairArtifact(approved) {
      await cache.repairArtifact({ approved });
    },
  };
}

function createHttpHealthBoundaries(): Pick<
  RecallInferenceConfigurationCandidate,
  'artifact' | 'inspectHealth'
> {
  return {
    artifact: null,
    async inspectHealth() {
      return { artifactState: 'not-required', requiredRepair: null };
    },
  };
}

function createPendingDeviceStatus(
  policy: EmbeddedInferenceDevicePolicy,
): RecallInferenceDeviceStatus {
  return { policy, computeBackend: 'pending', names: [] };
}

/** Creates embedded and llama.cpp HTTP candidates for optional reranking and query planning. */
export function createRecommendedOptionalInferenceCandidates(
  config: RecallConversationConfig,
  options: RecommendedOptionalInferenceCandidateOptions = {},
): readonly RecallInferenceConfigurationCandidate[] {
  const rerankingProfile = createRecommendedQwenRerankingModelProfile();
  const queryPlanningProfile = createRecommendedQmdQueryPlanningModelProfile();
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const devicePolicy = options.device ?? 'auto';
  const rerankingDevice = createPendingDeviceStatus(devicePolicy);
  const queryPlanningDevice = createPendingDeviceStatus(devicePolicy);
  const queryPlanningBaseUrl =
    options.queryPlanningBaseUrl ?? config.queryPlannerBaseUrl ?? 'http://192.168.0.67:8092/v1';

  const embeddedReranking: RecallInferenceConfigurationCandidate = {
    capability: 'reranking',
    candidateId: 'recommended-qwen-reranker-embedded',
    profileId: rerankingProfile.profileId,
    backend: 'embedded',
    adapterId: 'node-llama-cpp-qwen-reranking-logit-recovery-v1',
    endpoint: null,
    device: rerankingDevice,
    ...createOptionalArtifactBoundaries(config, rerankingProfile),
    async verifyCapabilityConformance() {
      const evidence = options.rerankingConformance;
      if (!evidence) {
        throw new Error(
          'Recall recommended reranking conformance evidence unavailable: supply independently accepted fixed scores; model output was not guessed',
        );
      }
      const provider = createEmbeddedQwenRerankingProvider(rerankingProfile, {
        modelCacheDirectory,
        device: devicePolicy,
      });
      try {
        const service = createRecallConversationService(config, {
          rerankingProfile,
          reranker: provider,
        });
        const verification = await service.verifyRerankingCapability(evidence);
        rerankingDevice.computeBackend = provider.executionIdentity.computeBackend;
        rerankingDevice.names = [...provider.executionIdentity.deviceNames];
        return {
          profileId: verification.profileId,
          adapterId: verification.executionIdentity.adapterId,
          backend: verification.executionIdentity.backend,
          cacheIdentity: verification.executionIdentity.cacheIdentity,
          measurement: { ...verification.measurement },
        };
      } finally {
        await provider.dispose();
      }
    },
  };

  const httpRerankingProvider = () =>
    createQwenHttpRerankingProvider(rerankingProfile, { baseUrl: config.rerankerBaseUrl });
  const httpReranking: RecallInferenceConfigurationCandidate = {
    capability: 'reranking',
    candidateId: 'recommended-qwen-reranker-http',
    profileId: rerankingProfile.profileId,
    backend: 'llama-cpp-http',
    adapterId: 'llama-cpp-http-reranking-v1',
    endpoint: config.rerankerBaseUrl,
    device: null,
    ...createHttpHealthBoundaries(),
    async verifyCapabilityConformance() {
      const evidence = options.rerankingConformance;
      if (!evidence) {
        throw new Error(
          'Recall recommended reranking conformance evidence unavailable: supply independently accepted fixed scores; model output was not guessed',
        );
      }
      const provider = httpRerankingProvider();
      const verification = await createRecallConversationService(config, {
        rerankingProfile,
        reranker: provider,
      }).verifyRerankingCapability(evidence);
      return {
        profileId: verification.profileId,
        adapterId: verification.executionIdentity.adapterId,
        backend: verification.executionIdentity.backend,
        cacheIdentity: verification.executionIdentity.cacheIdentity,
        measurement: { ...verification.measurement },
      };
    },
  };

  const embeddedQueryPlanning: RecallInferenceConfigurationCandidate = {
    capability: 'query-planning',
    candidateId: 'recommended-qmd-query-planner-embedded',
    profileId: queryPlanningProfile.profileId,
    backend: 'embedded',
    adapterId: 'node-llama-cpp-qmd-query-planning-v1',
    endpoint: null,
    device: queryPlanningDevice,
    ...createOptionalArtifactBoundaries(config, queryPlanningProfile),
    async verifyCapabilityConformance() {
      const evidence = options.queryPlanningConformance;
      if (!evidence) {
        throw new Error(
          'Recall recommended query planning conformance evidence unavailable: supply an independently accepted fixed plan; model output was not guessed',
        );
      }
      const provider = createEmbeddedQmdQueryPlanningProvider(queryPlanningProfile, {
        modelCacheDirectory,
        device: devicePolicy,
      });
      try {
        const verification = await createRecallConversationService(config, {
          queryPlanningProfile,
          queryPlanner: provider,
        }).verifyQueryPlanningCapability({ expectedPlan: evidence.expectedPlan });
        queryPlanningDevice.computeBackend = provider.executionIdentity.computeBackend;
        queryPlanningDevice.names = [...provider.executionIdentity.deviceNames];
        return {
          profileId: verification.profileId,
          adapterId: verification.executionIdentity.adapterId,
          backend: verification.executionIdentity.backend,
          cacheIdentity: verification.executionIdentity.cacheIdentity,
          measurement: { ...verification.measurement },
        };
      } finally {
        await provider.dispose();
      }
    },
  };

  const httpQueryPlanning: RecallInferenceConfigurationCandidate = {
    capability: 'query-planning',
    candidateId: 'recommended-qmd-query-planner-http',
    profileId: queryPlanningProfile.profileId,
    backend: 'llama-cpp-http',
    adapterId: 'llama-cpp-http-query-planning-v1',
    endpoint: queryPlanningBaseUrl,
    device: null,
    ...createHttpHealthBoundaries(),
    async verifyCapabilityConformance() {
      const evidence = options.queryPlanningConformance;
      if (!evidence) {
        throw new Error(
          'Recall recommended query planning conformance evidence unavailable: supply an independently accepted fixed plan; model output was not guessed',
        );
      }
      const provider = createQmdHttpQueryPlanningProvider(queryPlanningProfile, {
        baseUrl: queryPlanningBaseUrl,
      });
      const verification = await createRecallConversationService(config, {
        queryPlanningProfile,
        queryPlanner: provider,
      }).verifyQueryPlanningCapability({ expectedPlan: evidence.expectedPlan });
      return {
        profileId: verification.profileId,
        adapterId: verification.executionIdentity.adapterId,
        backend: verification.executionIdentity.backend,
        cacheIdentity: verification.executionIdentity.cacheIdentity,
        measurement: { ...verification.measurement },
      };
    },
  };

  return [embeddedReranking, httpReranking, embeddedQueryPlanning, httpQueryPlanning];
}

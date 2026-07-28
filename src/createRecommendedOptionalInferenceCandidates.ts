import { readFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { createEmbeddedQmdQueryPlanningProvider } from './embedded-qmd-query-planning-provider.js';
import {
  EmbeddedInferenceDevicePolicy,
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import { createEmbeddedQwenRerankingProvider } from './embedded-qwen-reranking-provider.js';
import { createQmdHttpQueryPlanningProvider } from './createQmdHttpQueryPlanningProvider.js';
import { createQwenHttpRerankingProvider } from './createQwenHttpRerankingProvider.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import { createRecallConversationService } from './recall-conversation-service.js';
import type { RecallPlannedRetrievalQuery } from './recall-inference-capabilities.js';
import type {
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
import { readNodeErrorCode } from './read-node-error-code.js';

const RECOMMENDED_OPTIONAL_INFERENCE_CONFORMANCE_SCHEMA = Type.Object(
  {
    reranking: Type.Union([
      Type.Null(),
      Type.Object(
        {
          query: Type.String({ minLength: 1 }),
          documents: Type.Array(Type.String({ minLength: 1 }), { minItems: 1 }),
          expectedScores: Type.Array(Type.Number(), { minItems: 1 }),
          maximumAbsoluteDifference: Type.Optional(Type.Number({ minimum: 0 })),
        },
        { additionalProperties: false },
      ),
    ]),
    queryPlanning: Type.Union([
      Type.Null(),
      Type.Object(
        {
          expectedPlan: Type.Array(
            Type.Object(
              {
                type: Type.Union([Type.Literal('lex'), Type.Literal('vec'), Type.Literal('hyde')]),
                query: Type.String({ minLength: 1 }),
              },
              { additionalProperties: false },
            ),
            { minItems: 1 },
          ),
        },
        { additionalProperties: false },
      ),
    ]),
  },
  { additionalProperties: false },
);

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

/** Reads accepted optional-capability fixtures used by production setup and doctor runs. */
export async function readRecommendedOptionalInferenceConformance(
  evidencePath: string,
): Promise<RecommendedOptionalInferenceCandidateOptions> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(evidencePath, 'utf8'));
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return {};
    }
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall optional inference conformance evidence invalid at ${evidencePath}: ${message}`,
      {
        cause: error,
      },
    );
  }
  try {
    const evidence = Value.Parse(RECOMMENDED_OPTIONAL_INFERENCE_CONFORMANCE_SCHEMA, parsed);
    return {
      ...(evidence.reranking ? { rerankingConformance: evidence.reranking } : {}),
      ...(evidence.queryPlanning ? { queryPlanningConformance: evidence.queryPlanning } : {}),
    };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Recall optional inference conformance evidence invalid at ${evidencePath}: ${message}`,
      {
        cause: error,
      },
    );
  }
}

function mapOptionalModelArtifactState(state: string): RecallInferenceArtifactState {
  if (state === String(RecallInferenceArtifactState.VALID)) {
    return RecallInferenceArtifactState.VALID;
  }
  if (state === String(RecallInferenceArtifactState.MISSING)) {
    return RecallInferenceArtifactState.MISSING;
  }
  if (state === String(RecallInferenceArtifactState.PARTIAL)) {
    return RecallInferenceArtifactState.PARTIAL;
  }
  if (state === String(RecallInferenceArtifactState.CORRUPT)) {
    return RecallInferenceArtifactState.CORRUPT;
  }
  if (state === String(RecallInferenceArtifactState.INCOMPATIBLE)) {
    return RecallInferenceArtifactState.INCOMPATIBLE;
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
      return {
        artifactState: RecallInferenceArtifactState.NOT_REQUIRED,
        requiredRepair: null,
      };
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
  const devicePolicy = options.device ?? EmbeddedInferenceDevicePolicy.AUTO;
  const rerankingDevice = createPendingDeviceStatus(devicePolicy);
  const queryPlanningDevice = createPendingDeviceStatus(devicePolicy);
  const queryPlanningBaseUrl =
    options.queryPlanningBaseUrl ?? config.queryPlannerBaseUrl ?? 'http://192.168.0.67:8092/v1';

  const embeddedReranking: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.RERANKING,
    candidateId: 'recommended-qwen-reranker-embedded',
    profileId: rerankingProfile.profileId,
    backend: RecallInferenceBackend.EMBEDDED,
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
          embeddingProfileId: null,
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
    capability: RecallInferenceCapability.RERANKING,
    candidateId: 'recommended-qwen-reranker-http',
    profileId: rerankingProfile.profileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
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
        embeddingProfileId: null,
        measurement: { ...verification.measurement },
      };
    },
  };

  const embeddedQueryPlanning: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.QUERY_PLANNING,
    candidateId: 'recommended-qmd-query-planner-embedded',
    profileId: queryPlanningProfile.profileId,
    backend: RecallInferenceBackend.EMBEDDED,
    adapterId: 'node-llama-cpp-qmd-query-planning-v1',
    endpoint: null,
    device: queryPlanningDevice,
    ...createOptionalArtifactBoundaries(config, queryPlanningProfile),
    async verifyCapabilityConformance() {
      const evidence = options.queryPlanningConformance;
      const provider = createEmbeddedQmdQueryPlanningProvider(queryPlanningProfile, {
        modelCacheDirectory,
        device: devicePolicy,
      });
      try {
        const verification = await createRecallConversationService(config, {
          queryPlanningProfile,
          queryPlanner: provider,
        }).verifyQueryPlanningCapability(evidence ? { expectedPlan: evidence.expectedPlan } : {});
        queryPlanningDevice.computeBackend = provider.executionIdentity.computeBackend;
        queryPlanningDevice.names = [...provider.executionIdentity.deviceNames];
        return {
          profileId: verification.profileId,
          adapterId: verification.executionIdentity.adapterId,
          backend: verification.executionIdentity.backend,
          cacheIdentity: verification.executionIdentity.cacheIdentity,
          embeddingProfileId: null,
          measurement: { ...verification.measurement },
        };
      } finally {
        await provider.dispose();
      }
    },
  };

  const httpQueryPlanning: RecallInferenceConfigurationCandidate = {
    capability: RecallInferenceCapability.QUERY_PLANNING,
    candidateId: 'recommended-qmd-query-planner-http',
    profileId: queryPlanningProfile.profileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: 'llama-cpp-http-query-planning-v1',
    endpoint: queryPlanningBaseUrl,
    device: null,
    ...createHttpHealthBoundaries(),
    async verifyCapabilityConformance() {
      const evidence = options.queryPlanningConformance;
      const provider = createQmdHttpQueryPlanningProvider(queryPlanningProfile, {
        baseUrl: queryPlanningBaseUrl,
      });
      const verification = await createRecallConversationService(config, {
        queryPlanningProfile,
        queryPlanner: provider,
      }).verifyQueryPlanningCapability(evidence ? { expectedPlan: evidence.expectedPlan } : {});
      return {
        profileId: verification.profileId,
        adapterId: verification.executionIdentity.adapterId,
        backend: verification.executionIdentity.backend,
        cacheIdentity: verification.executionIdentity.cacheIdentity,
        embeddingProfileId: null,
        measurement: { ...verification.measurement },
      };
    },
  };

  return [embeddedReranking, httpReranking, embeddedQueryPlanning, httpQueryPlanning];
}

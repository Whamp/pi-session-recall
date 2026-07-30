import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import {
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import {
  loadOctenConversationTokenizer,
  OCTEN_TOKENIZER_IDENTITY,
} from './octen-conversation-tokenizer.js';
import type { RecallBackgroundIndexServiceFactory } from './recall-background-index-build.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationService,
} from './recall-conversation-service.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { createRecallTokenizerManifestIdentity } from './recall-index-manifest.js';
import type { RecallInferenceConfigurationCandidate } from './recall-inference-configuration.js';
import {
  createOctenEmbeddingModelProfile,
  type OctenEmbeddingModelProfile,
} from './recall-model-profiles.js';
import type { ConversationTextTokenizer } from './session-conversation-index.js';

const RECOMMENDED_OCTEN_HTTP_CANDIDATE_PREFIX = 'recommended-octen-http-';
const RECOMMENDED_OCTEN_PROFILE_ID = 'octen-embedding-4b';
const CONFIGURED_BACKGROUND_INDEX_SERVICE_FACTORY: RecallBackgroundIndexServiceFactory = {
  moduleUrl: new URL('./configured-recall-inference-runtime.ts', import.meta.url).href,
  exportName: 'createConfiguredRecallBackgroundService',
};

/** Exact provider, tokenizer, and semantic profile for configured Octen HTTP execution. */
export interface RecommendedOctenHttpEmbeddingRuntime {
  profile: OctenEmbeddingModelProfile;
  provider: RecallEmbeddingProvider;
  tokenizerIdentity: ReturnType<typeof createRecallTokenizerManifestIdentity>;
  loadTokenizer(): Promise<ConversationTextTokenizer>;
  embeddingDimensions: number;
  dispose(): Promise<void>;
}

/** Creates the configured Octen profile without mixing HTTP location into vector identity. */
export function createConfiguredOctenEmbeddingModelProfile(
  config: RecallConversationConfig,
  storedDimensions: number,
): OctenEmbeddingModelProfile {
  return createOctenEmbeddingModelProfile(
    {
      requestModel: config.embeddingModel,
      servedModelId: config.embeddingServedModelId,
      artifact: config.embeddingArtifact,
      dimensions: config.embeddingDimensions,
      quantization: config.embeddingQuantization,
      pooling: config.embeddingPooling,
      normalization: 'l2',
    },
    storedDimensions,
  );
}

/** Returns the persisted candidate ID for one vendor-supported Octen stored width. */
export function createRecommendedOctenHttpCandidateId(storedDimensions: number): string {
  if (!Number.isSafeInteger(storedDimensions) || storedDimensions < 1 || storedDimensions > 2_560) {
    throw new Error(
      `Recall Octen setup stored dimensions invalid: expected an integer from 1 through 2560, received ${storedDimensions}`,
    );
  }
  return `${RECOMMENDED_OCTEN_HTTP_CANDIDATE_PREFIX}${storedDimensions}`;
}

/** Reads a valid configured Octen width from its persisted candidate ID. */
export function readRecommendedOctenHttpStoredDimensions(
  candidateId: string | undefined,
  nativeDimensions: number,
): number | null {
  if (!candidateId?.startsWith(RECOMMENDED_OCTEN_HTTP_CANDIDATE_PREFIX)) {
    return null;
  }
  const storedDimensions = Number(
    candidateId.slice(RECOMMENDED_OCTEN_HTTP_CANDIDATE_PREFIX.length),
  );
  return Number.isSafeInteger(storedDimensions) &&
    storedDimensions >= 1 &&
    storedDimensions <= nativeDimensions
    ? storedDimensions
    : null;
}

/** Creates the provider-only Octen runtime shared by setup and detached reconstruction. */
export function createRecommendedOctenHttpEmbeddingRuntime(
  config: RecallConversationConfig,
  storedDimensions: number,
): RecommendedOctenHttpEmbeddingRuntime {
  const profile = createConfiguredOctenEmbeddingModelProfile(config, storedDimensions);
  return {
    profile,
    provider: createLlamaCppHttpEmbeddingProvider(profile, {
      baseUrl: config.embeddingBaseUrl,
      batchSize: config.embeddingBatchSize,
    }),
    tokenizerIdentity: createRecallTokenizerManifestIdentity(OCTEN_TOKENIZER_IDENTITY),
    loadTokenizer: () =>
      loadOctenConversationTokenizer({ cacheDirectory: config.tokenizerCacheDirectory }),
    embeddingDimensions: profile.identity.dimensions,
    async dispose() {},
  };
}

function createRecommendedOctenHttpServiceRuntime(
  config: RecallConversationConfig,
  storedDimensions: number,
): { service: RecallConversationService; dispose(): Promise<void> } {
  const runtime = createRecommendedOctenHttpEmbeddingRuntime(config, storedDimensions);
  return {
    service: createRecallConversationService(config, {
      embeddingProfile: runtime.profile,
      embeddingProvider: runtime.provider,
      tokenizerIdentity: runtime.tokenizerIdentity,
      loadTokenizer: () => runtime.loadTokenizer(),
      backgroundIndexServiceFactory: CONFIGURED_BACKGROUND_INDEX_SERVICE_FACTORY,
    }),
    dispose: () => runtime.dispose(),
  };
}

/** Creates one selectable, artifact-free Octen llama.cpp HTTP inference candidate. */
export function createRecommendedOctenHttpInferenceCandidate(
  config: RecallConversationConfig,
  storedDimensions: number,
): RecallInferenceConfigurationCandidate {
  createConfiguredOctenEmbeddingModelProfile(config, storedDimensions);
  const candidateId = createRecommendedOctenHttpCandidateId(storedDimensions);

  async function runWithRuntime<T>(
    operation: (runtime: ReturnType<typeof createRecommendedOctenHttpServiceRuntime>) => Promise<T>,
  ): Promise<T> {
    const runtime = createRecommendedOctenHttpServiceRuntime(config, storedDimensions);
    try {
      return await operation(runtime);
    } finally {
      await runtime.dispose();
    }
  }

  return {
    capability: RecallInferenceCapability.EMBEDDING,
    candidateId,
    profileId: RECOMMENDED_OCTEN_PROFILE_ID,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: 'llama-cpp-http-embedding-v1',
    endpoint: config.embeddingBaseUrl,
    device: null,
    artifact: null,
    async inspectHealth() {
      return {
        artifactState: RecallInferenceArtifactState.NOT_REQUIRED,
        requiredRepair: null,
      };
    },
    async verifyCapabilityConformance() {
      return runWithRuntime(async (runtime) => {
        const verification = await runtime.service.verifyEmbeddingCapability();
        return {
          profileId: RECOMMENDED_OCTEN_PROFILE_ID,
          adapterId: 'llama-cpp-http-embedding-v1',
          backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
          cacheIdentity: verification.embeddingProfileId,
          embeddingProfileId: verification.embeddingProfileId,
          measurement: { capabilityVerificationCount: 1 },
        };
      });
    },
    generationService: {
      readIndexGenerationStatus: () =>
        runWithRuntime((runtime) => runtime.service.readIndexGenerationStatus()),
      startBackgroundIndexGeneration: () =>
        runWithRuntime((runtime) => runtime.service.startBackgroundIndexGeneration()),
      resumeBackgroundIndexGeneration: () =>
        runWithRuntime((runtime) => runtime.service.resumeBackgroundIndexGeneration()),
    },
  };
}

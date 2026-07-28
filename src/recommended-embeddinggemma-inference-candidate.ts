import { dirname, join } from 'node:path';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
} from './embedded-embeddinggemma-provider.js';
import { createLlamaCppHttpEmbeddingProvider } from './createLlamaCppHttpEmbeddingProvider.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationService,
} from './recall-conversation-service.js';
import {
  EmbeddedInferenceDevicePolicy,
  RecallInferenceArtifactState,
  RecallInferenceBackend,
  RecallInferenceCapability,
} from './enums.js';
import type {
  RecallInferenceConfigurationCandidate,
  RecallInferenceDeviceStatus,
} from './recall-inference-configuration.js';
import { createRecallModelArtifactCache } from './recall-model-artifact-cache.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';
import { createRecommendedEmbeddingGemmaConversationRuntime } from './recommended-embeddinggemma-conversation-service.js';

function mapRecallModelArtifactState(state: string): RecallInferenceArtifactState {
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
  throw new Error(`Recall EmbeddingGemma artifact state unsupported: ${state}`);
}

/** Creates the built-in embedded EmbeddingGemma setup candidate and its public conformance seam. */
export function createRecommendedEmbeddingGemmaInferenceCandidate(
  config: RecallConversationConfig,
): RecallInferenceConfigurationCandidate {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const artifactCache = createRecallModelArtifactCache({
    cacheDirectory: modelCacheDirectory,
    profile,
  });
  const device: RecallInferenceDeviceStatus = {
    policy: 'auto',
    computeBackend: 'pending',
    names: [],
  };

  async function runWithRuntime<T>(
    operation: (
      runtime: ReturnType<typeof createRecommendedEmbeddingGemmaConversationRuntime>,
    ) => Promise<T>,
  ): Promise<T> {
    const runtime = createRecommendedEmbeddingGemmaConversationRuntime(config);
    try {
      return await operation(runtime);
    } finally {
      await runtime.dispose();
    }
  }

  return {
    capability: RecallInferenceCapability.EMBEDDING,
    candidateId: 'recommended-embeddinggemma-embedded',
    profileId: profile.profileId,
    backend: RecallInferenceBackend.EMBEDDED,
    adapterId: 'node-llama-cpp-embedded-v2',
    endpoint: null,
    device,
    artifact: {
      path: join(modelCacheDirectory, profile.source.artifact),
      repository: profile.source.repository,
      revision: profile.source.revision,
      sha256: profile.source.sha256,
      byteSize: profile.source.byteSize,
    },
    async inspectHealth() {
      const inspection = await artifactCache.inspectArtifact();
      return {
        artifactState: mapRecallModelArtifactState(inspection.status.state),
        requiredRepair: inspection.status.state === 'valid' ? null : inspection.status.repair,
      };
    },
    async prepareArtifact(approved) {
      await artifactCache.downloadArtifact({ approved });
    },
    async repairArtifact(approved) {
      await artifactCache.repairArtifact({ approved });
    },
    async verifyCapabilityConformance() {
      return runWithRuntime(async (runtime) => {
        const verification = await runtime.service.verifyEmbeddingCapability();
        device.computeBackend = runtime.executionIdentity.computeBackend;
        device.names = [...runtime.executionIdentity.deviceNames];
        return {
          profileId: profile.profileId,
          adapterId: runtime.executionIdentity.adapter,
          backend: RecallInferenceBackend.EMBEDDED,
          cacheIdentity: verification.embeddingProfileId,
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

function createRecommendedEmbeddingGemmaHttpServiceRuntime(config: RecallConversationConfig): {
  service: RecallConversationService;
  dispose(): Promise<void>;
} {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const tokenizerProvider = createEmbeddedEmbeddingGemmaProvider(profile, {
    modelCacheDirectory,
    device: EmbeddedInferenceDevicePolicy.CPU,
  });
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: createLlamaCppHttpEmbeddingProvider(profile, {
      baseUrl: config.embeddingBaseUrl,
      batchSize: config.embeddingBatchSize,
    }),
    tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
    loadTokenizer: () => tokenizerProvider.loadConversationTokenizer(),
    backgroundIndexServiceFactory: {
      moduleUrl: import.meta.url,
      exportName: 'createRecommendedEmbeddingGemmaHttpBackgroundService',
    },
  });
  return { service, dispose: () => tokenizerProvider.dispose() };
}

/** Creates the built-in llama.cpp HTTP candidate for the same EmbeddingGemma profile. */
export function createRecommendedEmbeddingGemmaHttpInferenceCandidate(
  config: RecallConversationConfig,
): RecallInferenceConfigurationCandidate {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const modelCacheDirectory = join(dirname(config.manifestPath), 'models');
  const artifactCache = createRecallModelArtifactCache({
    cacheDirectory: modelCacheDirectory,
    profile,
  });

  async function runWithRuntime<T>(
    operation: (
      runtime: ReturnType<typeof createRecommendedEmbeddingGemmaHttpServiceRuntime>,
    ) => Promise<T>,
  ): Promise<T> {
    const runtime = createRecommendedEmbeddingGemmaHttpServiceRuntime(config);
    try {
      return await operation(runtime);
    } finally {
      await runtime.dispose();
    }
  }

  return {
    capability: RecallInferenceCapability.EMBEDDING,
    candidateId: 'recommended-embeddinggemma-http',
    profileId: profile.profileId,
    backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
    adapterId: 'llama-cpp-http-embedding-v1',
    endpoint: config.embeddingBaseUrl,
    device: null,
    artifact: {
      path: join(modelCacheDirectory, profile.source.artifact),
      repository: profile.source.repository,
      revision: profile.source.revision,
      sha256: profile.source.sha256,
      byteSize: profile.source.byteSize,
    },
    async inspectHealth() {
      const inspection = await artifactCache.inspectArtifact();
      return {
        artifactState: mapRecallModelArtifactState(inspection.status.state),
        requiredRepair: inspection.status.state === 'valid' ? null : inspection.status.repair,
      };
    },
    async prepareArtifact(approved) {
      await artifactCache.downloadArtifact({ approved });
    },
    async repairArtifact(approved) {
      await artifactCache.repairArtifact({ approved });
    },
    async verifyCapabilityConformance() {
      return runWithRuntime(async (runtime) => {
        const verification = await runtime.service.verifyEmbeddingCapability();
        return {
          profileId: profile.profileId,
          adapterId: 'llama-cpp-http-embedding-v1',
          backend: RecallInferenceBackend.LLAMA_CPP_HTTP,
          cacheIdentity: verification.embeddingProfileId,
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

/** Reconstructs built-in HTTP EmbeddingGemma for one detached replacement worker. */
export function createRecommendedEmbeddingGemmaHttpBackgroundService(
  config: RecallConversationConfig,
): RecallConversationService {
  return createRecommendedEmbeddingGemmaHttpServiceRuntime(config).service;
}

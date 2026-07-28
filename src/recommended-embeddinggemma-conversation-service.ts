import { dirname, join } from 'node:path';

import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
  type EmbeddedEmbeddingGemmaExecutionIdentity,
  type EmbeddedEmbeddingGemmaProviderOptions,
} from './embedded-embeddinggemma-provider.js';
import type { RecallConversationConfig } from './recall-conversation-config.js';
import {
  createRecallConversationService,
  type RecallConversationService,
} from './recall-conversation-service.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './recall-model-profiles.js';

/** Service plus disposable local provider used by guided first-index setup operations. */
export interface RecommendedEmbeddingGemmaConversationRuntime {
  service: RecallConversationService;
  readonly executionIdentity: Readonly<EmbeddedEmbeddingGemmaExecutionIdentity>;
  dispose(): Promise<void>;
}

/** Creates the recommended embedded EmbeddingGemma service and detached-worker factory identity. */
export function createRecommendedEmbeddingGemmaConversationRuntime(
  config: RecallConversationConfig,
  providerOptions: Partial<EmbeddedEmbeddingGemmaProviderOptions> = {},
): RecommendedEmbeddingGemmaConversationRuntime {
  const profile = createRecommendedEmbeddingGemmaModelProfile();
  const modelCacheDirectory =
    providerOptions.modelCacheDirectory ?? join(dirname(config.manifestPath), 'models');
  const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
    ...providerOptions,
    modelCacheDirectory,
  });
  const service = createRecallConversationService(config, {
    embeddingProfile: profile,
    embeddingProvider: provider,
    tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
    loadTokenizer: () => provider.loadConversationTokenizer(),
    backgroundIndexServiceFactory: {
      moduleUrl: import.meta.url,
      exportName: 'createRecommendedEmbeddingGemmaBackgroundService',
    },
  });
  return {
    service,
    get executionIdentity() {
      return provider.executionIdentity;
    },
    dispose: () => provider.dispose(),
  };
}

/** Reconstructs recommended embedded EmbeddingGemma inside one detached index worker. */
export function createRecommendedEmbeddingGemmaBackgroundService(
  config: RecallConversationConfig,
): RecallConversationService {
  return createRecommendedEmbeddingGemmaConversationRuntime(config).service;
}

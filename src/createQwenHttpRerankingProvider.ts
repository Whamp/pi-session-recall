import {
  createLocalRerankerClient,
  type LocalRerankerClientConfig,
} from './local-reranker-client.js';
import {
  createRecallRerankingExecutionIdentity,
  type RecallIdentifiedRerankingProvider,
} from './recall-inference-capabilities.js';
import { RecallInferenceBackend } from './enums.js';
import type { QwenRerankingModelProfile } from './recall-model-profiles.js';

/** HTTP execution settings that do not contribute to the Qwen model profile identity. */
export type QwenHttpRerankingBackendConfig = Omit<LocalRerankerClientConfig, 'model'>;

/** Creates the Qwen HTTP adapter that returns finite relevance scores in candidate order. */
export function createQwenHttpRerankingProvider(
  profile: QwenRerankingModelProfile,
  backend: QwenHttpRerankingBackendConfig,
): RecallIdentifiedRerankingProvider {
  const client = createLocalRerankerClient({
    ...backend,
    model: profile.model,
  });
  return {
    executionIdentity: createRecallRerankingExecutionIdentity(
      profile.profileId,
      'llama-cpp-http-reranking-v1',
      RecallInferenceBackend.LLAMA_CPP_HTTP,
    ),
    async rerankDocuments(query, documents, signal) {
      const scores = await client.rerankDocuments(query, documents, signal);
      for (const [index, score] of scores.entries()) {
        if (score < profile.scoreRange.minimum || score > profile.scoreRange.maximum) {
          throw new Error(
            `Recall Qwen HTTP reranker score outside profile range at candidate index ${index}: expected ${profile.scoreRange.minimum} through ${profile.scoreRange.maximum}, received ${score}`,
          );
        }
      }
      return scores;
    },
  };
}

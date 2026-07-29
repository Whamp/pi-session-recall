import { createCanonicalIdentity } from './create-canonical-identity.js';
import {
  createLocalRerankerClient,
  DEFAULT_LOCAL_RERANKER_REQUEST_TIMEOUT_MILLISECONDS,
  normalizeLocalRerankerEndpoint,
  type LocalRerankerClientConfig,
} from './local-reranker-client.js';
import {
  createRecallRerankingExecutionIdentity,
  type RecallIdentifiedRerankingProvider,
} from './recall-inference-capabilities.js';
import { RecallInferenceBackend } from './enums.js';
import type { QwenRerankingModelProfile } from './recall-model-profiles.js';

const LLAMA_CPP_HTTP_RERANKING_ADAPTER_ID = 'llama-cpp-http-reranking-v1';
const LLAMA_CPP_HTTP_RERANKING_ADAPTER_VERSION = '1';

/** HTTP execution settings that do not contribute to the Qwen model profile identity. */
export type QwenHttpRerankingBackendConfig = Omit<LocalRerankerClientConfig, 'model'>;

/** Creates the Qwen HTTP adapter that returns finite relevance scores in candidate order. */
export function createQwenHttpRerankingProvider(
  profile: QwenRerankingModelProfile,
  backend: QwenHttpRerankingBackendConfig,
): RecallIdentifiedRerankingProvider {
  const normalizedEndpoint = normalizeLocalRerankerEndpoint(backend.baseUrl);
  const requestTimeoutMilliseconds =
    backend.requestTimeoutMilliseconds ?? DEFAULT_LOCAL_RERANKER_REQUEST_TIMEOUT_MILLISECONDS;
  const client = createLocalRerankerClient({
    ...backend,
    model: profile.model,
  });
  return {
    executionIdentity: createRecallRerankingExecutionIdentity(
      profile,
      LLAMA_CPP_HTTP_RERANKING_ADAPTER_ID,
      createCanonicalIdentity('llama-cpp-http-reranking-config-v1', {
        normalizedEndpoint,
        requestTimeoutMilliseconds,
      }),
      RecallInferenceBackend.LLAMA_CPP_HTTP,
      requestTimeoutMilliseconds,
      LLAMA_CPP_HTTP_RERANKING_ADAPTER_VERSION,
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

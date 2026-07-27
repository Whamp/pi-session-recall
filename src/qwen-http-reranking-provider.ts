import {
  createLocalRerankerClient,
  type LocalRerankerClientConfig,
} from './local-reranker-client.js';
import type { RecallRerankingProvider } from './recall-inference-capabilities.js';
import type { QwenRerankingModelProfile } from './recall-model-profiles.js';

/** HTTP execution settings that do not contribute to the Qwen model profile identity. */
export type QwenHttpRerankingBackendConfig = Omit<LocalRerankerClientConfig, 'model'>;

/** Creates the Qwen HTTP adapter that returns finite relevance scores in candidate order. */
export function createQwenHttpRerankingProvider(
  profile: QwenRerankingModelProfile,
  backend: QwenHttpRerankingBackendConfig,
): RecallRerankingProvider {
  return createLocalRerankerClient({
    ...backend,
    model: profile.model,
  });
}

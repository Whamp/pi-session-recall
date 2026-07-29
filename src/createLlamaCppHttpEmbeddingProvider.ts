import {
  createLocalEmbeddingClient,
  type LocalEmbeddingClientConfig,
} from './local-embedding-client.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { RecallEmbeddingModelProfile } from './recall-model-profiles.js';

/** llama.cpp HTTP execution settings excluded from compatible embedding model semantics. */
export type LlamaCppHttpEmbeddingBackendConfig = Omit<
  LocalEmbeddingClientConfig,
  'model' | 'dimensions'
>;

/** Creates a capability-specific llama.cpp HTTP adapter with profile-owned input prompts. */
export function createLlamaCppHttpEmbeddingProvider(
  profile: RecallEmbeddingModelProfile,
  backend: LlamaCppHttpEmbeddingBackendConfig,
): RecallEmbeddingProvider {
  const client = createLocalEmbeddingClient({
    ...backend,
    model: profile.identity.requestModel,
    dimensions: profile.identity.dimensions,
  });
  return {
    async embedQuery(query, signal) {
      const embeddings = await client.embedTexts([`${profile.queryInputPrefix}${query}`], signal);
      const embedding = embeddings[0];
      if (!embedding) {
        throw new Error('Recall llama.cpp HTTP embedding response missing query vector');
      }
      return embedding;
    },
    embedDocuments(documents, signal) {
      return client.embedTexts(
        documents.map((document) => `${profile.documentInputPrefix}${document}`),
        signal,
      );
    },
  };
}

import {
  createLocalEmbeddingClient,
  type LocalEmbeddingClientConfig,
} from './local-embedding-client.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { OctenEmbeddingModelProfile } from './recall-model-profiles.js';

/** HTTP execution settings that do not contribute to the Octen model profile identity. */
export type OctenHttpEmbeddingBackendConfig = Omit<
  LocalEmbeddingClientConfig,
  'model' | 'dimensions'
>;

/** Creates the Octen HTTP adapter with separate query and document embedding operations. */
export function createOctenHttpEmbeddingProvider(
  profile: OctenEmbeddingModelProfile,
  backend: OctenHttpEmbeddingBackendConfig,
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
        throw new Error('Recall Octen HTTP embedding response missing query vector');
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

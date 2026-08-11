import { createLocalEmbeddingClient } from './local-embedding-client.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import { createStoredRecallEmbedding } from './recall-stored-embedding.js';

/** Direct Octen HTTP settings and the vendor-supported stored-prefix width. */
export interface OctenHttpEmbeddingConfig {
  baseUrl: string;
  model: string;
  nativeDimensions: number;
  storedDimensions: number;
  batchSize?: number;
  requestTimeoutMilliseconds?: number;
}

/** Embeds index documents and search queries through direct Octen HTTP. */
export function createOctenHttpEmbeddingProvider(
  config: OctenHttpEmbeddingConfig,
): RecallEmbeddingProvider {
  const client = createLocalEmbeddingClient({
    baseUrl: config.baseUrl,
    model: config.model,
    dimensions: config.nativeDimensions,
    ...(config.batchSize === undefined ? {} : { batchSize: config.batchSize }),
    ...(config.requestTimeoutMilliseconds === undefined
      ? {}
      : { requestTimeoutMilliseconds: config.requestTimeoutMilliseconds }),
  });

  function convertNativeEmbedding(nativeEmbedding: readonly number[]): number[] {
    return createStoredRecallEmbedding(
      nativeEmbedding,
      config.nativeDimensions,
      config.storedDimensions,
    );
  }

  return {
    async embedQuery(query, signal) {
      const nativeEmbedding = (await client.embedTexts([query], signal))[0];
      if (!nativeEmbedding) {
        throw new Error('Recall Octen embedding response missing query vector');
      }
      return convertNativeEmbedding(nativeEmbedding);
    },
    async embedDocuments(documents, signal) {
      const nativeEmbeddings = await client.embedTexts([...documents], signal);
      return nativeEmbeddings.map(convertNativeEmbedding);
    },
    async close() {},
  };
}

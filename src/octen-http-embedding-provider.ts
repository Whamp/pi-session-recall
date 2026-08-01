import {
  createLlamaCppHttpEmbeddingProvider,
  type LlamaCppHttpEmbeddingBackendConfig,
} from './llama-cpp-http-embedding-provider.js';
import type { RecallEmbeddingProvider } from './recall-inference-capabilities.js';
import type { OctenEmbeddingModelProfile } from './recall-model-profiles.js';

/** HTTP execution settings that do not contribute to the Octen model profile identity. */
export type OctenHttpEmbeddingBackendConfig = LlamaCppHttpEmbeddingBackendConfig;

/** Octen-compatible name for the shared profile-driven llama.cpp HTTP embedding adapter. */
export const createOctenHttpEmbeddingProvider: (
  profile: OctenEmbeddingModelProfile,
  backend: OctenHttpEmbeddingBackendConfig,
) => RecallEmbeddingProvider = createLlamaCppHttpEmbeddingProvider;

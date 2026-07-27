import type { RecallEmbeddingModelIdentity } from './recall-index-manifest.js';

/** Embedding semantics shared by all backends serving one compatible model profile. */
export interface RecallEmbeddingModelProfile {
  identity: Readonly<RecallEmbeddingModelIdentity>;
  queryInputPrefix: string;
  documentInputPrefix: string;
}

/** Reranking semantics shared by all backends serving one compatible model profile. */
export interface RecallRerankingModelProfile {
  model: string;
  scoreMeaning: 'higher-is-more-relevant';
}

/** Immutable Octen embedding semantics, excluding backend URL and adapter execution details. */
export interface OctenEmbeddingModelProfile extends RecallEmbeddingModelProfile {
  queryInputPrefix: '';
  documentInputPrefix: '';
}

/** Immutable Qwen reranking semantics, excluding backend URL and adapter execution details. */
export type QwenRerankingModelProfile = RecallRerankingModelProfile;

/** Creates the legacy-compatible Octen profile whose query and document inputs remain unchanged. */
export function createOctenEmbeddingModelProfile(
  identity: RecallEmbeddingModelIdentity,
): OctenEmbeddingModelProfile {
  return Object.freeze({
    identity: Object.freeze({ ...identity }),
    queryInputPrefix: '',
    documentInputPrefix: '',
  });
}

/** Creates the Qwen profile whose finite scores increase with candidate relevance. */
export function createQwenRerankingModelProfile(model: string): QwenRerankingModelProfile {
  const normalizedModel = model.trim();
  if (!normalizedModel) {
    throw new Error('Recall Qwen reranking model profile invalid: expected a non-blank model name');
  }
  return Object.freeze({
    model: normalizedModel,
    scoreMeaning: 'higher-is-more-relevant',
  });
}

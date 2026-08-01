/** Embeds recall queries and index documents through explicitly distinct model operations. */
export interface RecallEmbeddingProvider {
  /** Embeds one submitted recall query using the model profile's query semantics. */
  embedQuery(query: string, signal?: AbortSignal): Promise<number[]>;

  /** Embeds index documents in input order using the model profile's document semantics. */
  embedDocuments(documents: readonly string[], signal?: AbortSignal): Promise<number[][]>;
}

/** Produces one finite relevance score per candidate document in input order. */
export interface RecallRerankingProvider {
  /** Scores candidate documents against one query without reordering the candidates. */
  rerankDocuments(
    query: string,
    documents: readonly string[],
    signal?: AbortSignal,
  ): Promise<number[]>;
}

/** Search and cache identity for one reranking adapter executing one model profile. */
export interface RecallRerankingExecutionIdentity {
  adapterId: string;
  backend: 'embedded' | 'llama-cpp-http' | 'custom';
  cacheIdentity: string;
  modelProfileId: string;
}

/** Reranking provider whose profile and adapter identity can be verified before use. */
export interface RecallIdentifiedRerankingProvider extends RecallRerankingProvider {
  readonly executionIdentity: Readonly<RecallRerankingExecutionIdentity>;
}

/** Creates cache identity that changes with either reranker profile or adapter policy. */
export function createRecallRerankingExecutionIdentity(
  modelProfileId: string,
  adapterId: string,
  backend: RecallRerankingExecutionIdentity['backend'],
): Readonly<RecallRerankingExecutionIdentity> {
  return Object.freeze({
    adapterId,
    backend,
    cacheIdentity: `${modelProfileId}:${adapterId}`,
    modelProfileId,
  });
}

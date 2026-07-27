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

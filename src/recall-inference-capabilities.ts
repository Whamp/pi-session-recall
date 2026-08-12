/** Embeds recall queries and index documents through explicitly distinct Octen operations. */
export interface RecallEmbeddingProvider {
  /** Embeds one submitted recall query using the configured stored-prefix semantics. */
  embedQuery(query: string, signal?: AbortSignal): Promise<number[]>;

  /** Embeds index documents in input order using the same stored-prefix semantics. */
  embedDocuments(documents: readonly string[], signal?: AbortSignal): Promise<number[][]>;

  /** Releases native resources when the owning process no longer needs this provider. */
  close?(): Promise<void>;
}

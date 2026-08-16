---
status: accepted
---

# Reuse identical embedding inputs within Physical sessions

Recall keeps every Dense recall document and its Evidence occurrence, including documents with identical text but different entry IDs, graph positions, or Source locators. Chunk boundaries and the configured 64-token overlap remain unchanged.

During one Physical session replacement, the indexer sends each distinct Dense document text to the embedding provider once. It assigns that stored recall embedding to every document occurrence with the exact same text. Existing vectors loaded by document ID from the current or Active recall database also populate this file-local map. `newlyEmbeddedChunks` counts distinct texts submitted to the provider; `reusedVectors` includes later identical-text occurrences.

The map uses the complete embedding input string rather than metadata or a sampled digest. The embedding provider receives only this string, so entry identity, Project attribution, branch state, timestamps, source geometry, and other document metadata cannot change the embedding. Recall still writes one vector row per Dense document and atomically replaces the complete Physical session projection.

The map lives only for one Physical session. It crosses the indexer's internal 128-document preparation batches, then becomes unreachable before the next file. Recall adds no persistent embedding cache, invalidation policy, repair path, or run-wide memory population. A provider failure still prevents that Physical session replacement, and retry follows the existing full path.

The HTTP provider does not return bit-identical vectors for every batch position. A five-sample synthetic probe measured a minimum stored-vector cosine similarity of 0.999681 for identical text. Recall already treats an existing vector for unchanged document text as reusable across maintenance runs. This decision applies the same semantic contract within one replacement; it does not promise bit-identical recomputation.

The schema-4 corpus measurement found 119,740,141 Dense document tokens. Exact text repeated within the same Physical session accounted for 43,551,880 tokens, or 36.372%. Compaction summaries accounted for 41,559,968 of those repeated tokens. Intentional neighboring-chunk overlap was a separate 10,389,335 tokens, or 8.677%, and remains intact.

A two-sample alternating benchmark used one 248-document representative Physical session and the production tokenizer and HTTP embedding provider. The base revision embedded all 248 documents in a 28.793-second median. The file-local reuse revision embedded 146 distinct texts, reused 102 vectors, retained all 248 documents and 267 Invocations, and completed in a 14.938-second median, a 48.117% wall-time reduction. Every candidate passed SQLite, foreign-key, FTS, projection-cache, vector-parity, and Project metadata checks. Dense metadata and Invocations matched after normalizing the disposable benchmark path.

Do not add a run-wide LRU or persistent content-addressed embedding cache without a separate measurement and decision. This change removes the demonstrated file-local repetition while preserving the simple rebuild and recovery model required by [ADR 0017](0017-stop-optimizing-routine-index-maintenance.md).

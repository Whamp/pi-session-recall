# Inference profiles and provider conformance

Conversation Recall separates model semantics from execution:

- `RecallEmbeddingModelProfile` contains the embedding identity recorded in the index manifest. It does not contain an HTTP URL, device, timeout, or adapter name. The Octen and EmbeddingGemma profiles specialize this contract.
- `QwenRerankingModelProfile` contains the requested model and states that larger finite scores mean greater relevance. It does not contain HTTP execution settings.
- `RecallEmbeddingProvider` exposes separate `embedQuery` and `embedDocuments` operations.
- `RecallRerankingProvider` returns one relevance score per candidate in candidate order.

The current Octen profile preserves the deployed behavior: both operations send their input text unchanged. The [pinned EmbeddingGemma profile](embeddinggemma-model-artifact.md) applies `task: search result | query: ` to queries and `title: none | text: ` to documents. Both profiles can use the llama.cpp `POST /v1/embeddings` adapter. The Qwen HTTP adapter uses `POST /v1/rerank`. These are capability-specific wire contracts, not a generic OpenAI-compatibility promise.

The [embedded EmbeddingGemma provider](embedded-embeddinggemma.md) uses the same profile through dynamically loaded `node-llama-cpp@3.18.1`. Its execution identity records the embedded adapter and CPU device without changing vector compatibility.

The Octen manifest projection remains byte-for-byte compatible with manifest version 5. Changing only a backend URL does not change the profile or require a vector rebuild. Changing a recorded embedding identity field still makes the generation incompatible.

## Deterministic conformance

Run the shared adapter probes without a model download:

```bash
node --import tsx --test \
  src/recall-inference-conformance.test.ts \
  src/embedded-embeddinggemma-provider.test.ts \
  src/embeddinggemma-recall-conversation-service.test.ts
```

The probes use temporary HTTP servers and fixed vectors or scores. They verify:

- separate query and document requests;
- unchanged Octen input text and document order;
- vector count, dimensions, finite values, profile-required L2 normalization, and fixed expected values;
- reranker score count, candidate order, finite values, and fixed expected values;
- bounded query, document, and reranking request counts;
- query, document, and reranking elapsed time through an injectable monotonic clock.

Custom adapters can call `measureRecallEmbeddingProviderConformance` and `measureRecallRerankingProviderConformance` with profile-specific fixed fixtures. A mismatch fails with the capability and vector dimension or candidate index.

## Evidence boundary

This deterministic path proves adapter and service contract behavior. The embedded tests use an injected native boundary; they do not load the real 333,590,944-byte artifact. The HTTP tests use temporary fixed-response servers. Neither path proves that a live server loaded the claimed artifact or that live vectors and scores match acceptance fixtures.

Real EmbeddingGemma tokenizer comparison, canary vectors, cold start, warm inference, throughput, storage, and device measurements remain pending because this environment has no Gemma distribution approval or operator-approved model download. [Embedded EmbeddingGemma execution](embedded-embeddinggemma.md) lists each missing measurement. The committed Octen quality report remains Octen-profile evidence only.

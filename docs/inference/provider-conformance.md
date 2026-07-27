# Inference profiles and provider conformance

Conversation Recall separates model semantics from execution:

- `OctenEmbeddingModelProfile` contains the embedding identity recorded in the index manifest. It does not contain an HTTP URL, device, timeout, or adapter name.
- `QwenRerankingModelProfile` contains the requested model and states that larger finite scores mean greater relevance. It does not contain HTTP execution settings.
- `RecallEmbeddingProvider` exposes separate `embedQuery` and `embedDocuments` operations.
- `RecallRerankingProvider` returns one relevance score per candidate in candidate order.

The current Octen profile preserves the deployed behavior: both operations send their input text unchanged. The operations remain separate so the [pinned EmbeddingGemma profile](embeddinggemma-model-artifact.md) can own asymmetric query and document prompts without changing indexing or search policy code. The Octen HTTP adapter uses `POST /v1/embeddings`. The Qwen HTTP adapter uses `POST /v1/rerank`. These are capability-specific wire contracts, not a generic OpenAI-compatibility promise.

The Octen manifest projection remains byte-for-byte compatible with manifest version 5. Changing only a backend URL does not change the profile or require a vector rebuild. Changing a recorded embedding identity field still makes the generation incompatible.

## Deterministic conformance

Run the shared adapter probes without a model download:

```bash
node --import tsx --test src/recall-inference-conformance.test.ts
```

The probes use temporary HTTP servers and fixed vectors or scores. They verify:

- separate query and document requests;
- unchanged Octen input text and document order;
- vector count, dimensions, finite values, and fixed expected values;
- reranker score count, candidate order, finite values, and fixed expected values;
- bounded query, document, and reranking request counts;
- query, document, and reranking elapsed time through an injectable monotonic clock.

Custom adapters can call `measureRecallEmbeddingProviderConformance` and `measureRecallRerankingProviderConformance` with profile-specific fixed fixtures. A mismatch fails with the capability and vector dimension or candidate index.

## Evidence boundary

This deterministic path proves adapter and service contract behavior. It does not prove that a live Octen or Qwen server is available, that a server loaded the claimed artifact, or that live model vectors and scores match acceptance fixtures. It also does not measure real cold start, warm inference, throughput, or device use. Those live profile-acceptance and performance measurements remain pending under #28. Issue #34 adds no embedded artifact or distribution terms, so it introduces no new legal-approval claim.

The committed Octen hybrid quality report remains unchanged; this ticket does not reinterpret it as Qwen reranker or live adapter conformance evidence.

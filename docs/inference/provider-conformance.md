# Inference profiles and provider conformance

Conversation Recall separates model semantics from execution:

- `RecallEmbeddingModelProfile` contains the embedding identity recorded in the index manifest. It does not contain an HTTP URL, device, timeout, or adapter name. The Octen and EmbeddingGemma profiles specialize this contract.
- `QwenRerankingModelProfile` contains profile identity, requested model, score range, and score policy; the recommended profile also pins its downloadable artifact. It does not contain HTTP or device execution settings.
- `RecallQueryPlanningModelProfile` contains the planner profile, prompt policy, grammar version, typed-query bounds, sampling policy, and conformance canary; the recommended profile also pins its downloadable artifact.
- `RecallEmbeddingProvider` exposes separate `embedQuery` and `embedDocuments` operations.
- `RecallRerankingProvider` returns one relevance score per candidate in candidate order.
- `RecallQueryPlanningProvider` returns one ordered, bounded collection of `lex`, `vec`, and optional `hyde` planned retrieval queries.

The current Octen profile preserves the deployed behavior: both operations send their input text unchanged. The [pinned EmbeddingGemma profile](embeddinggemma-model-artifact.md) applies `task: search result | query: ` to queries and `title: none | text: ` to documents. Both profiles can use the llama.cpp `POST /v1/embeddings` adapter. The Qwen HTTP adapter uses `POST /v1/rerank`. These are capability-specific wire contracts, not a generic OpenAI-compatibility promise.

The [embedded EmbeddingGemma provider](embedded-embeddinggemma.md), [recommended embedded Qwen reranker](qwen-reranker.md), and [recommended QMD query planner](qmd-query-planner.md) use their profiles through dynamically loaded `node-llama-cpp@3.18.1`. Their execution identities record adapter policy, device policy, probed accelerators, selected compute backend and device names, CPU fallback source, and capability-specific resource bounds without changing vector compatibility.

The Octen manifest projection remains byte-for-byte compatible with manifest version 5. Changing only a backend URL does not change the profile or require a vector rebuild. Changing a recorded embedding identity field still makes the generation incompatible.

## Deterministic conformance

Run the shared adapter probes without a model download:

```bash
node --import tsx --test \
  src/recall-inference-conformance.test.ts \
  src/embedded-embeddinggemma-provider.test.ts \
  src/embeddinggemma-recall-conversation-service.test.ts \
  src/embedded-qwen-reranking-provider.test.ts \
  src/qwen-reranker-recall-conversation-service.test.ts \
  src/qmd-query-planning-provider-conformance.test.ts \
  src/embedded-qmd-query-planning-provider.test.ts \
  src/qmd-query-planner-recall-conversation-service.test.ts
```

The probes use temporary HTTP servers and fixed vectors or scores. They verify:

- separate query and document requests;
- unchanged Octen input text and document order;
- vector count, dimensions, finite values, profile-required L2 normalization, and fixed expected values;
- reranker score count, candidate order, finite values, profile score range, fixed expected values, adapter identity, and profile-plus-adapter cache identity;
- bounded query, document, and reranking request counts;
- query, document, and reranking elapsed time through an injectable monotonic clock;
- automatic accelerator probing and execution identity;
- one-warning same-profile CPU fallback through the public conversation service;
- exact explicit-device failure without fallback;
- one shared model load across concurrent requests and bounded context-pool concurrency; and
- idle disposal, reload, and synchronous tokenizer lifetime safety;
- HTTP and embedded reranker timeout and caller cancellation;
- embedded Qwen score recovery plus rejection of the known uncorrected double-sigmoid behavior;
- planner grammar, typed bounds, ordered fixed output, original-term protection, and recall-intent transport;
- planner profile, adapter, prompt, grammar, timeout, and cache identity; and
- HTTP and embedded planner timeout, cancellation, wrong-model rejection, and same-profile CPU fallback.

Custom adapters can call `measureRecallEmbeddingProviderConformance`, `measureRecallRerankingProviderConformance`, and `measureRecallQueryPlanningProviderConformance` with profile-specific fixed fixtures. Identified reranking and planning providers must expose matching profile, adapter, and cache identity. A mismatch fails with the capability and vector dimension, candidate index, or planned-query index.

Mixed setup persists a candidate only after its capability-specific operation passes. Reranker setup requires independent ordered expected scores. Planner setup requires an independent exact typed plan. A failed custom candidate leaves the prior atomic [inference configuration](mixed-inference-configuration.md) unchanged; setup never substitutes a built-in adapter.

## Evidence boundary

This deterministic path proves adapter and service contract behavior. The embedded tests use an injected native boundary; they do not load the real 333,590,944-byte artifact. The HTTP tests use temporary fixed-response servers. Neither path proves that a live server loaded the claimed artifact or that live vectors and scores match acceptance fixtures.

Real EmbeddingGemma tokenizer comparison, canary vectors, cold start, warm inference, throughput, storage, automatic fallback, native-log isolation, and CPU/accelerator device measurements remain pending because this environment has no Gemma distribution approval or operator-approved model download. Real Qwen reranker artifact verification, cross-adapter fixture scores, cold/warm reranking, throughput, and CPU/accelerator measurements also remain pending because no download or device run was approved. Real QMD planner artifact verification, cross-adapter output fixtures, cold/warm planning, throughput, and CPU/accelerator measurements remain pending for the same reason. [Embedded EmbeddingGemma execution](embedded-embeddinggemma.md), the [recommended Qwen reranker](qwen-reranker.md), and the [recommended QMD query planner](qmd-query-planner.md) list each missing measurement. The committed Octen quality report remains Octen-profile evidence only.

The consolidated [embedded profile acceptance ledger](../evaluation/embedded-profile-acceptance.md) records deterministic conformance, exact cache identities, background interruption coverage, the profile-aware live quality command, and pending external evidence without treating fixture devices as hardware measurements.

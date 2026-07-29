# Recommended Qwen reranker

Conversation Recall supports the pinned Qwen3 0.6B reranker through embedded `node-llama-cpp` execution or the built-in llama.cpp HTTP adapter. Both execute one model profile and produce one probability from `0` through `1` for every candidate in input order. Higher scores mean greater relevance.

## Immutable profile

| Property            | Pinned value                                                       |
| ------------------- | ------------------------------------------------------------------ |
| Profile             | `qwen3-reranker-0.6b-q8-0-v1`                                      |
| Request model       | `qwen3-reranker-0.6b-q8_0`                                         |
| Artifact repository | `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`                           |
| Repository revision | `a02f48bb4f057028298c21fa033da2b30d7742d5`                         |
| GGUF artifact       | `qwen3-reranker-0.6b-q8_0.gguf`                                    |
| Byte size           | `639153184`                                                        |
| SHA-256             | `22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48` |
| Score policy        | `llama-cpp-qwen3-rank-probability-v1`                              |
| Score range         | inclusive `[0, 1]`                                                 |
| License             | Apache License 2.0                                                 |

The download URL contains the immutable repository revision. The profile does not contain an HTTP URL, device, or adapter identifier.

## Operator artifact commands

Inspection and verification never download a model:

```bash
npm run --silent model:qwen-reranker -- inspect
npm run --silent model:qwen-reranker -- status
npm run --silent model:qwen-reranker -- verify
npm run --silent model:qwen-reranker -- doctor
```

Downloads, repairs, and removal require explicit consent:

```bash
npm run --silent model:qwen-reranker -- download --approve
npm run --silent model:qwen-reranker -- repair --approve
npm run --silent model:qwen-reranker -- remove --approve
```

Without `--approve`, mutation fails before transport, directory creation, replacement, or removal. Downloads stage to a unique partial path and activate only after byte-size, SHA-256, and bounded GGUF validation. The default cache root is `~/.pi/agent/recall/models`; `PI_RECALL_MODEL_CACHE_DIRECTORY` overrides it.

The command manages only the artifact. Mixed-capability guided configuration remains assigned to issue #43.

## Built-in execution adapters

### llama.cpp HTTP

`createQwenHttpRerankingProvider` sends `POST /v1/rerank` with:

```json
{
  "model": "qwen3-reranker-0.6b-q8_0",
  "query": "submitted query",
  "documents": ["first candidate", "second candidate"],
  "top_n": 2
}
```

The adapter validates the response schema, unique in-range candidate indexes, complete score coverage, finite values, and the profile's inclusive `[0, 1]` score range. It restores input order even when llama.cpp returns results sorted by score. Requests time out after 60 seconds by default and honor caller cancellation.

### Embedded node-llama-cpp

`createEmbeddedQwenRerankingProvider` dynamically loads exactly `node-llama-cpp@3.18.1`, verifies the pinned artifact before native loading, and creates a bounded pool of ranking contexts. Automatic device selection probes supported accelerators. Accelerator initialization failure retries the same profile once on CPU and emits one warning. Explicit device selection fails closed. Native logs go to stderr, and contexts, model, and runtime support idle and explicit disposal.

The embedded adapter intentionally does not pass through `LlamaRankingContext.rankAll()` scores unchanged. At the pinned native revisions:

- llama.cpp `b8390` applies softmax for the Qwen3 rank pooling output ([source](https://github.com/ggml-org/llama.cpp/blob/b6c83aad55a4ce17ec96fced7770cd1be8758193/src/llama-graph.cpp#L2605-L2608));
- `node-llama-cpp@3.18.1` then applies sigmoid to the first returned value ([source](https://github.com/withcatai/node-llama-cpp/blob/57bea3da9ffa78955e8b25f195ce6cc714980cb5/src/evaluator/LlamaRankingContext.ts#L246-L253)).

That second transform compresses a llama.cpp probability `p` to `sigmoid(p)`. The adapter pins `node-llama-cpp@3.18.1`, requires native scores in `[sigmoid(0), sigmoid(1)]`, and applies `logit(score)` to recover `p`. A runtime with different score behavior fails with a searchable score-semantics error rather than silently changing ranking.

## Search and cache identity

A deep search reports:

- model profile ID;
- adapter ID;
- cache identity composed from profile ID plus adapter ID;
- existing reranker model and ranking-policy version.

Changing the adapter changes search/cache identity. Changing the reranker profile changes both profile and cache identity. Neither change affects the embedding model in the index manifest, so the active vector generation remains compatible. Hybrid search reports `rerankerIdentity: null` and never invokes a reranker.

A reranker request, response, score-range, score-count, timeout, cancellation, model-load, or native score-semantics failure rejects `deep-rerank`. Conversation Recall does not fall back to hybrid or another backend.

## Deterministic conformance and measurement

Run the no-download path:

```bash
node --import tsx --test \
  src/recall-inference-conformance.test.ts \
  src/embedded-qwen-reranking-provider.test.ts \
  src/qwen-reranker-recall-conversation-service.test.ts
```

The shared conformance harness measures elapsed reranking time through an injectable monotonic clock and verifies document ordering, score count, finite values, profile range, fixed fixture scores, model-profile identity, adapter identity, and cache identity. Adapter tests additionally exercise HTTP and embedded timeout/cancellation, automatic GPU-to-CPU fallback, and rejection of uncorrected double-sigmoid scores. The service test builds one temporary real zvec index and searches it through both built-in adapters, then changes reranker profile without rebuilding vectors.

## External evidence still pending

No real model download or real device run was approved in this environment. The following acceptance evidence remains pending and is not represented as passing:

1. Operator-approved download of the exact 639,153,184-byte artifact through `model:qwen-reranker`, proving the published SHA-256 and GGUF structure against the live file.
2. A frozen real-model query/document fixture run through llama.cpp `b8390` HTTP and embedded `node-llama-cpp@3.18.1`, recording exact HTTP probabilities and proving the embedded recovered values match within a declared tolerance. The deterministic fixture proves the adapter math and rejection path, not the model's live numeric outputs.
3. CPU and available accelerated-device measurements for cold model load, warm reranking latency, candidate throughput, selected device identity, context-pool storage, and idle reload.
4. Maintainer release review of the upstream Apache-2.0 declaration and notices. No separate legal approval is claimed here.

The deterministic paths must remain unchanged when those measurements are collected; live evidence supplements rather than weakens the fixed criteria.

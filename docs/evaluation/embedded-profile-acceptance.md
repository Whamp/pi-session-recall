# Embedded profile acceptance evidence

Recorded 2026-07-27 for issue #44.

## Decision

**Release acceptance is blocked.** The deterministic adapter, conversation-service, quality-runner, and background-worker path passes. No pinned model artifact is installed, no real model or device measurement ran, and no maintainer approved distribution terms or notices. The evidence does not relabel fixture results as live model results.

The machine-readable ledger is [`embedded-profile-acceptance.json`](embedded-profile-acceptance.json).

## Quality artifacts and Octen evidence

Issue #44 accepted Octen-profile evidence for these paths. The committed artifacts were later regenerated under `deterministic-fixture-v1` (64 dimensions) in #57, so they no longer attest that Octen pass. Current fingerprints:

| File                                                         | SHA-256                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------------ |
| [`recall-quality-report.md`](recall-quality-report.md)       | `4965b741e66bcb1c0a9f6bb9e4d290415495e13594b61e706ba908be7ab425c5` |
| [`recall-quality-results.json`](recall-quality-results.json) | `390e76e73d9225404536e96ea17732a85cf79d23a64fc10d260ee76d121f1800` |

The ledger marks Octen evidence as superseded. The frozen hybrid candidate policy from that Octen run is still reused for EmbeddingGemma evidence; it does not make these artifacts an Octen quality pass.

## Deterministic evidence

Run:

```bash
npm run evidence:embedded:deterministic
```

The 2026-07-27 run passed 48 of 48 tests in 8,268.6 ms. It exercised:

- EmbeddingGemma query/document conformance, 768 dimensions, L2 normalization, asymmetric prompts, tokenizer identity, repeat canary, embedded/HTTP compatibility, lexical-only tool isolation, and same-profile CPU fallback;
- Qwen ordered scores, finite probability domain, double-sigmoid rejection, embedded/HTTP adapter and cache identities, and public `deep-rerank` behavior;
- QMD planner grammar, typed bounds, protected terms, intent, timeout, embedded/HTTP adapter and cache identities, and public capability verification;
- the frozen corpus loader and profile-aware bounded quality runner; and
- detached build survival, progress, active-generation search, stop/resume, cache reuse, and resumable failures during parsing, embedding, store write, optimization, and pre-activation.

The embedded runtime tests use injected native boundaries. Their CPU and accelerator labels describe fixtures, not hardware measurements.

## Recorded identities

| Capability | Profile                              | Backend / adapter                                            | Cache identity                                                                                                   |
| ---------- | ------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Embedding  | `embeddinggemma-300m-q8-0-v1`        | embedded / `node-llama-cpp-embedded-v2`                      | Pending a canonical live canary fingerprint; the quality command records the complete manifest-derived identity. |
| Embedding  | `embeddinggemma-300m-q8-0-v1`        | capability-specific llama.cpp HTTP                           | Same profile-bound vector identity as embedded execution.                                                        |
| Reranking  | `qwen3-reranker-0.6b-q8-0-v1`        | embedded / `node-llama-cpp-qwen-reranking-logit-recovery-v1` | `qwen3-reranker-0.6b-q8-0-v1:node-llama-cpp-qwen-reranking-logit-recovery-v1`                                    |
| Reranking  | `qwen3-reranker-0.6b-q8-0-v1`        | HTTP / `llama-cpp-http-reranking-v1`                         | `qwen3-reranker-0.6b-q8-0-v1:llama-cpp-http-reranking-v1`                                                        |
| Planning   | `qmd-query-expansion-1.7b-q4-k-m-v1` | embedded / `node-llama-cpp-qmd-query-planning-v1`            | Profile + adapter + `qmd-query-expansion-no-think-v1` + `qmd-typed-query-plan-v1`                                |
| Planning   | `qmd-query-expansion-1.7b-q4-k-m-v1` | HTTP / `llama-cpp-http-query-planning-v1`                    | Profile + adapter + `qmd-query-expansion-no-think-v1` + `qmd-typed-query-plan-v1`                                |

EmbeddingGemma uses 768 dimensions, mean pooling, L2 normalization, query prefix `task: search result | query: `, document prefix `title: none | text: `, the tokenizer inside the pinned GGUF through `node-llama-cpp@3.18.1`, and repeat-cosine query canary policy at `0.9995`. The live command binds these fields to the frozen hybrid candidate policy: RRF v2 with `k=60`, active-branch prior `0.01`, eight candidates per channel, and five final results.

## Live quality and measurement path

After distribution approval and explicit artifact download, run the frozen corpus separately on CPU and an available accelerator:

```bash
npm run evaluate:embeddinggemma -- --device cpu
npm run evaluate:embeddinggemma -- --device cuda # or metal/vulkan on supported hardware
```

The command:

- never downloads a model or reads production sessions;
- writes `docs/evaluation/embeddinggemma-quality-<device>.json` and leaves Octen files untouched;
- verifies EmbeddingGemma through `RecallConversationService` before indexing;
- records profile, backend, adapter, requested and selected device, dimensions, prompts, tokenizer, canary, candidate policy, and manifest-derived embedding-cache identity;
- runs the committed scope, provenance, identifier, tool-evidence, duplicate, branch, and source-preservation corpus; and
- measures cold start, warm query latency, warm document-batch latency, indexing throughput, index size, and embedding-cache size.

The attempted CPU run failed before model load with:

```text
Recall embedded EmbeddingGemma artifact unavailable: state is missing; Run model repair and explicitly approve the pinned model download.
```

This is the expected fail-closed result. It supplies no live quality or performance evidence.

## Acceptance matrix

| Criterion                                            | Current evidence                                                  | Status                                  |
| ---------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------- |
| Preserve Octen quality result as Octen-only evidence | Existing files and SHA-256 values unchanged                       | Pass                                    |
| EmbeddingGemma regression corpus                     | Real runner implemented; artifact absent                          | Pending external                        |
| Embedded Qwen score semantics and deep reranking     | Deterministic conformance and public service integration pass     | Deterministic pass; live parity pending |
| Planner capability conformance                       | Deterministic embedded/HTTP conformance passes                    | Deterministic pass; live parity pending |
| Complete identity evidence                           | Ledger records known fields; live device identity remains unset   | Pending live fields                     |
| Cold/warm, throughput, storage, CPU, accelerator     | Real measurement writer implemented; no model/device run approved | Pending external                        |
| Background interruption and activation               | Child-process tests pass across all required phases               | Pass                                    |
| Distribution terms and notices                       | Review packet records obligations and missing files               | Maintainer approval pending             |
| Full validation                                      | Recorded in the implementing commit's verification                | See final verification                  |

## Exact external evidence still required

1. Maintainer approval of [`embedded-profile-distribution-review.md`](../inference/embedded-profile-distribution-review.md), including Gemma's downstream restrictions, agreement copy, modification notice decision, and exact `Notice` text.
2. Approved downloads and full byte-size, SHA-256, and GGUF validation for all three pinned artifacts.
3. Independent real EmbeddingGemma tokenizer IDs and query/document vectors, then embedded/HTTP parity without changing tolerance or profile identity.
4. A passing CPU and accelerated EmbeddingGemma corpus result from the command above.
5. Independent live Qwen HTTP probabilities and embedded recovered scores for the same fixed documents.
6. Independent live QMD embedded/HTTP output for the same query and intent, passing the existing grammar and bounds.
7. CPU and accelerated cold/warm latency, throughput, storage, selected-device, fallback, idle-reload, and native-log evidence for the applicable models.

Query-planned retrieval quality remains assigned to issue #29.

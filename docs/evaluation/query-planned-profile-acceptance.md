# Query-Planned Recall: Live Profile Acceptance

**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.

This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.

## Bounds and identity

- Recorded against commit: `881c32d05c1150de6d6be6e26dcd92e2119cdabe`
- Private manifest SHA-256: `4b3fb8573bcd52a26d66a2f8e60f5bbba57b098530791b23f55505d3e9a1187e`
- Private corpus: 8 cases, 8 snapshots, 44521 indexed documents
- Retrieval embedding policy for every live matrix row: `deterministic-token-hash-v1`, 256 dimensions
- Planner profile: `qmd-query-expansion-1.7b-q4-k-m-v1` / `qmd-query-expansion-1.7B-q4_k_m`
- Prompt / grammar: `qmd-query-expansion-no-think-v1` / `qmd-bounded-query-plan-v2`
- Reranker profile / score policy: `qwen3-reranker-0.6b-q8-0-v1` / `llama-cpp-qwen3-rank-probability-v1`
- Search policy: RRF v2, k=60, fused-document limits 120, 160; duplicate-group rerank/final limits 40/5

## Committed-corpus EmbeddingGemma evidence

| Evidence                 | Device class | Profile                       | Candidate / final recall | Quality gate | Evidence SHA-256                                                   |
| ------------------------ | ------------ | ----------------------------- | -----------------------: | ------------ | ------------------------------------------------------------------ |
| accepted-hybrid-baseline | baseline     | `octen-embed`                 |            1.000 / 1.000 | pass         | `58a0b23e35d7033c396f179dee022fa55dc66a30018f0b94f37626a4f6a437c2` |
| live-profile-candidate   | cpu          | `embeddinggemma-300m-q8-0-v1` |            0.941 / 0.941 | fail         | `c621e619380d338307554bf45487a4ba7e82b98f116443b4cc20516be564bf9a` |
| live-profile-candidate   | accelerated  | `embeddinggemma-300m-q8-0-v1` |            0.941 / 0.941 | fail         | `b5343fbfa51973d08d474b349153c80b9f8bbf826dd3e9c7c3be73e26b403e44` |

## What the live matrix measures

Every matrix row uses `deterministic-token-hash-v1` test embeddings. The run name, backend, and device describe only where the query planner and reranker execute.

The total search time includes retrieval with those fixed test embeddings plus live planning and live reranking. It does not measure end-to-end production inference with Octen or EmbeddingGemma embeddings. The committed-corpus table above reports EmbeddingGemma separately.

## Live planner and reranker matrix

| Planner/reranker run | Planner/reranker backend                                                                          | Planner/reranker device      | Planner adapter                        | Reranker adapter                                  | Cold planning |    Warm planning | Cold reranking | Warm reranking | Query-planned search min / median / max |
| -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- | ------------------------------------------------- | ------------: | ---------------: | -------------: | -------------: | --------------------------------------: |
| embedded-cpu         | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | cpu / cpu                    | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |    11193.5 ms | 2598.4 ms (pass) |      7866.9 ms |       409.9 ms |         52658.5 / 62147.5 / 125894.4 ms |
| embedded-vulkan      | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | accelerated / vulkan         | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |    10874.9 ms | 1886.6 ms (pass) |      7521.1 ms |       157.7 ms |          13567.0 / 15641.7 / 34185.7 ms |
| http-cpu             | llama-cpp-http (llama.cpp b8390 bundle b10e98a CPU planner-ctx-2048 reranker-ctx-8192 batch-2048) | cpu / AMD Ryzen 7 8845HS CPU | `llama-cpp-http-query-planning-v1`     | `llama-cpp-http-reranking-v1`                     |     1272.9 ms | 1231.7 ms (pass) |       393.5 ms |       362.1 ms |         31688.3 / 44916.4 / 110477.3 ms |

## Fallback characterization

- New candidate admissions beyond normal and retrieval-work-matched original-query controls: 0
- Ranking-only promotions: 1
- Preserved existing successes across profile runs: 4
- Planner fallbacks: 0
- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.
- Live new-candidate admission observed: no
- Existing successes preserved across profiles: no
- Existing-success regression profiles: embedded-vulkan, http-cpu

## Candidate work by opaque case

| Planner/reranker run / case | Plan source | Normal               | Equal-work control   | Query-planned        | Candidate work (admitted / allowed) |    Planning / reranking / total |
| --------------------------- | ----------- | -------------------- | -------------------- | -------------------- | ----------------------------------: | ------------------------------: |
| embedded-cpu / case-001     | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  12678.3 / 65570.3 / 78336.1 ms |
| embedded-cpu / case-002     | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   3007.6 / 49606.9 / 52658.5 ms |
| embedded-cpu / case-003     | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   3825.8 / 51716.2 / 55589.3 ms |
| embedded-cpu / case-004     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 6688.7 / 119141.2 / 125894.4 ms |
| embedded-cpu / case-005     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   6022.8 / 57419.1 / 63490.2 ms |
| embedded-cpu / case-006     | planner     | candidate union miss | candidate union miss | candidate union miss |                           102 / 120 |   2636.3 / 55123.8 / 57798.6 ms |
| embedded-cpu / case-007     | planner     | success              | success              | success              |                           142 / 160 |   4391.6 / 58409.2 / 62847.7 ms |
| embedded-cpu / case-008     | planner     | success              | success              | success              |                           160 / 160 |   4089.9 / 57313.5 / 61447.3 ms |
| embedded-vulkan / case-001  | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  10943.4 / 17125.6 / 28156.6 ms |
| embedded-vulkan / case-002  | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2860.1 / 10921.9 / 13844.4 ms |
| embedded-vulkan / case-003  | planner     | final rank miss      | final rank miss      | final rank miss      |                           142 / 160 |   4517.1 / 11133.9 / 15723.2 ms |
| embedded-vulkan / case-004  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   8038.4 / 26023.5 / 34185.7 ms |
| embedded-vulkan / case-005  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   7587.1 / 10341.7 / 17983.1 ms |
| embedded-vulkan / case-006  | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   3141.2 / 10643.5 / 13834.6 ms |
| embedded-vulkan / case-007  | planner     | success              | success              | success              |                           142 / 160 |   3016.3 / 10500.7 / 13567.0 ms |
| embedded-vulkan / case-008  | planner     | success              | success              | success              |                           160 / 160 |   4893.6 / 10623.1 / 15560.2 ms |
| http-cpu / case-001         | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |   1376.4 / 32871.4 / 34295.9 ms |
| http-cpu / case-002         | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   1445.2 / 40313.8 / 41802.0 ms |
| http-cpu / case-003         | planner     | final rank miss      | final rank miss      | final rank miss      |                           102 / 120 |   2547.2 / 29101.2 / 31688.3 ms |
| http-cpu / case-004         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 8419.6 / 101993.7 / 110477.3 ms |
| http-cpu / case-005         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   4870.4 / 45965.7 / 50894.8 ms |
| http-cpu / case-006         | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   1752.4 / 45378.1 / 47184.2 ms |
| http-cpu / case-007         | planner     | success              | success              | success              |                           142 / 160 |   4304.4 / 38294.2 / 42648.6 ms |
| http-cpu / case-008         | planner     | success              | success              | success              |                           160 / 160 |   4437.0 / 56277.6 / 60761.1 ms |

## Failure, tool, and privacy semantics

- Privacy audit: 59 private values checked, 0 leaks
- Planner fallback through public service: pass
- Reranker failure through public service: pass
- Pi tool contract and policy evidence: pass

## Limitations

- Approval applies only as an explicit fallback after hybrid recall misses, with the accepted Octen embedding baseline and recorded planner, reranker, adapter, grammar, score, and search-policy identities.
- Live candidate admissions and preservation of queries already answered by hybrid are reported as fallback characterization, not release gates.
- EmbeddingGemma live candidates remain separate and are not approved when their committed-corpus quality gate fails.
- The committed corpus is synthetic-but-session-shaped; the private corpus is bounded and does not establish broad superiority.
- Private queries, plans, source text, session paths, and model artifacts remain outside Git.
- Hybrid remains the default search mode.

# Query-Planned Recall: Live Profile Acceptance

**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.

This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.

## Bounds and identity

- Recorded against commit: `52b84c8be6a8425fefb149144a5da2143a66e030`
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
| accepted-hybrid-baseline | baseline     | `deterministic-fixture-v1`    |            1.000 / 1.000 | pass         | `4ec75569bf29be698a824c563af36a81b80a9f56bb3e8735be44b9c94341b1a6` |
| live-profile-candidate   | cpu          | `embeddinggemma-300m-q8-0-v1` |            0.941 / 0.941 | fail         | `c621e619380d338307554bf45487a4ba7e82b98f116443b4cc20516be564bf9a` |
| live-profile-candidate   | accelerated  | `embeddinggemma-300m-q8-0-v1` |            0.941 / 0.941 | fail         | `b5343fbfa51973d08d474b349153c80b9f8bbf826dd3e9c7c3be73e26b403e44` |

## What the live matrix measures

Every matrix row uses `deterministic-token-hash-v1` test embeddings. The run name, backend, and device describe only where the query planner and reranker execute.

The total search time includes retrieval with those fixed test embeddings plus live planning and live reranking. It does not measure end-to-end production inference with Octen or EmbeddingGemma embeddings. The committed-corpus table above reports EmbeddingGemma separately.

## Live planner and reranker matrix

| Planner/reranker run | Planner/reranker backend                                                                          | Planner/reranker device      | Planner adapter                        | Reranker adapter                                  | Cold planning |    Warm planning | Cold reranking | Warm reranking | Query-planned search min / median / max |
| -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- | ------------------------------------------------- | ------------: | ---------------: | -------------: | -------------: | --------------------------------------: |
| embedded-cpu         | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | cpu / cpu                    | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2345.8 ms | 2106.8 ms (pass) |       381.4 ms |       378.3 ms |         51285.1 / 59268.0 / 129312.5 ms |
| embedded-vulkan      | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | accelerated / vulkan         | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2256.6 ms | 2169.2 ms (pass) |       155.2 ms |       147.6 ms |          12728.8 / 15208.8 / 26684.4 ms |
| http-cpu             | llama-cpp-http (llama.cpp b8390 bundle b10e98a CPU planner-ctx-2048 reranker-ctx-8192 batch-2048) | cpu / AMD Ryzen 7 8845HS CPU | `llama-cpp-http-query-planning-v1`     | `llama-cpp-http-reranking-v1`                     |     1049.4 ms | 1074.4 ms (pass) |       282.2 ms |       265.5 ms |          32665.1 / 41345.1 / 90804.0 ms |

## Fallback characterization

- New candidate admissions beyond normal and retrieval-work-matched original-query controls: 0
- Ranking-only promotions: 3
- Preserved existing successes across profile runs: 5
- Planner fallbacks: 0
- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.
- Live new-candidate admission observed: no
- Existing successes preserved across profiles: no
- Existing-success regression profiles: embedded-cpu

## Candidate work by opaque case

| Planner/reranker run / case | Plan source | Normal               | Equal-work control   | Query-planned        | Candidate work (admitted / allowed) |    Planning / reranking / total |
| --------------------------- | ----------- | -------------------- | -------------------- | -------------------- | ----------------------------------: | ------------------------------: |
| embedded-cpu / case-001     | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  11178.1 / 49887.0 / 61214.2 ms |
| embedded-cpu / case-002     | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2918.1 / 50643.9 / 53737.6 ms |
| embedded-cpu / case-003     | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   3811.7 / 47298.2 / 51285.1 ms |
| embedded-cpu / case-004     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 4513.0 / 124603.8 / 129312.5 ms |
| embedded-cpu / case-005     | planner     | candidate union miss | candidate union miss | candidate union miss |                           103 / 120 |   2959.2 / 58329.8 / 61464.3 ms |
| embedded-cpu / case-006     | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   4394.6 / 46773.6 / 51341.5 ms |
| embedded-cpu / case-007     | planner     | success              | success              | final rank miss      |                           142 / 160 |   4047.5 / 53096.3 / 57321.8 ms |
| embedded-cpu / case-008     | planner     | success              | success              | success              |                           160 / 160 |   3283.8 / 69842.1 / 73303.0 ms |
| embedded-vulkan / case-001  | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  10236.4 / 15356.7 / 25743.7 ms |
| embedded-vulkan / case-002  | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2538.1 / 10039.3 / 12728.8 ms |
| embedded-vulkan / case-003  | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |    3188.0 / 9709.0 / 13027.2 ms |
| embedded-vulkan / case-004  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   5588.7 / 20950.4 / 26684.4 ms |
| embedded-vulkan / case-005  | planner     | candidate union miss | candidate union miss | candidate union miss |                           103 / 120 |   6136.2 / 11972.6 / 18239.0 ms |
| embedded-vulkan / case-006  | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   2929.3 / 10526.9 / 13583.6 ms |
| embedded-vulkan / case-007  | planner     | success              | success              | success              |                           142 / 160 |   3775.9 / 10331.3 / 14240.6 ms |
| embedded-vulkan / case-008  | planner     | success              | success              | success              |                           160 / 160 |   5056.2 / 10988.1 / 16177.1 ms |
| http-cpu / case-001         | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |   1438.4 / 33549.1 / 35122.8 ms |
| http-cpu / case-002         | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   1297.5 / 37074.2 / 38504.2 ms |
| http-cpu / case-003         | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   2425.6 / 30238.3 / 32800.2 ms |
| http-cpu / case-004         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   4198.8 / 86453.1 / 90804.0 ms |
| http-cpu / case-005         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   2075.9 / 43539.5 / 45754.1 ms |
| http-cpu / case-006         | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   1269.1 / 31264.4 / 32665.1 ms |
| http-cpu / case-007         | planner     | success              | success              | success              |                           142 / 160 |   2436.6 / 41615.7 / 44186.0 ms |
| http-cpu / case-008         | planner     | success              | success              | success              |                           160 / 160 |   1654.3 / 49828.0 / 51617.0 ms |

## Failure, tool, and privacy semantics

- Privacy audit: 59 private values checked, 0 leaks
- Planner fallback through public service: pass
- Reranker failure through public service: pass
- Pi tool contract and policy evidence: pass

## Limitations

- Approval applies only as an explicit fallback after hybrid recall misses, with the accepted committed hybrid baseline and recorded planner, reranker, adapter, grammar, score, and search-policy identities.
- Live candidate admissions and preservation of queries already answered by hybrid are reported as fallback characterization, not release gates.
- EmbeddingGemma live candidates remain separate and are not approved when their committed-corpus quality gate fails.
- The committed corpus is synthetic-but-session-shaped; the private corpus is bounded and does not establish broad superiority.
- Private queries, plans, source text, session paths, and model artifacts remain outside Git.
- Hybrid remains the default search mode.

# Query-Planned Recall: Live Profile Acceptance

**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.

This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.

## Bounds and identity

- Recorded against commit: `ab260aa62322994e50f9387a159f9a132feb0466`
- Private manifest SHA-256: `4b3fb8573bcd52a26d66a2f8e60f5bbba57b098530791b23f55505d3e9a1187e`
- Private corpus: 8 cases, 8 snapshots, 44521 indexed documents
- Retrieval embedding policy for every live matrix row: `deterministic-token-hash-v1`, 256 dimensions
- Planner profile: `qmd-query-expansion-1.7b-q4-k-m-v1` / `qmd-query-expansion-1.7B-q4_k_m`
- Prompt / grammar: `qmd-query-expansion-no-think-v1` / `qmd-bounded-query-plan-v2`
- Reranker profile / score policy: `qwen3-reranker-0.6b-q8-0-v1` / `llama-cpp-qwen3-rank-probability-v1`
- Search policy: RRF v2, k=60, fused-document limits 160; duplicate-group rerank/final limits 40/5

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
| embedded-cpu         | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | cpu / cpu                    | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |    20351.5 ms | 2101.1 ms (pass) |      8085.2 ms |       452.2 ms |         62247.5 / 75866.3 / 163380.5 ms |
| embedded-vulkan      | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | accelerated / vulkan         | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |    12813.9 ms | 2327.2 ms (pass) |      9820.4 ms |       159.0 ms |          11319.1 / 17286.9 / 33917.3 ms |
| http-cpu             | llama-cpp-http (llama.cpp b8390 bundle b10e98a CPU planner-ctx-2048 reranker-ctx-8192 batch-2048) | cpu / AMD Ryzen 7 8845HS CPU | `llama-cpp-http-query-planning-v1`     | `llama-cpp-http-reranking-v1`                     |     1344.6 ms | 1200.7 ms (pass) |       327.6 ms |       378.9 ms |         37269.3 / 55081.6 / 113159.2 ms |

## Fallback characterization

- New candidate admissions beyond normal and retrieval-work-matched original-query controls: 0
- Ranking-only promotions: 2
- Preserved existing successes across profile runs: 3
- Planner fallbacks: 0
- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.
- Live new-candidate admission observed: no
- Existing successes preserved across profiles: no
- Existing-success regression profiles: embedded-cpu, embedded-vulkan, http-cpu

## Candidate work by opaque case

| Planner/reranker run / case | Plan source | Normal               | Equal-work control   | Query-planned        | Candidate work (admitted / allowed) |    Planning / reranking / total |
| --------------------------- | ----------- | -------------------- | -------------------- | -------------------- | ----------------------------------: | ------------------------------: |
| embedded-cpu / case-001     | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  13952.0 / 68570.2 / 82610.9 ms |
| embedded-cpu / case-002     | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   3214.5 / 66851.6 / 70113.9 ms |
| embedded-cpu / case-003     | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   3871.1 / 64128.9 / 68053.4 ms |
| embedded-cpu / case-004     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 8209.3 / 155087.2 / 163380.5 ms |
| embedded-cpu / case-005     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   4555.7 / 57636.8 / 62247.5 ms |
| embedded-cpu / case-006     | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   4269.0 / 65285.2 / 69606.0 ms |
| embedded-cpu / case-007     | planner     | success              | success              | success              |                           142 / 160 |   5021.9 / 76539.5 / 81618.8 ms |
| embedded-cpu / case-008     | planner     | success              | success              | success              |                           160 / 160 | 3346.3 / 102762.3 / 106168.7 ms |
| embedded-vulkan / case-001  | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  16578.2 / 17199.4 / 33917.3 ms |
| embedded-vulkan / case-002  | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   6823.2 / 11295.9 / 18292.7 ms |
| embedded-vulkan / case-003  | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |    4143.3 / 7122.2 / 11319.1 ms |
| embedded-vulkan / case-004  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   8769.1 / 20641.3 / 29477.5 ms |
| embedded-vulkan / case-005  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   4261.3 / 12002.1 / 16314.8 ms |
| embedded-vulkan / case-006  | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   3139.4 / 11754.8 / 14949.9 ms |
| embedded-vulkan / case-007  | planner     | success              | success              | success              |                           142 / 160 |   6952.1 / 10999.1 / 18016.1 ms |
| embedded-vulkan / case-008  | planner     | success              | success              | success              |                           160 / 160 |   3874.1 / 12637.0 / 16557.8 ms |
| http-cpu / case-001         | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |   1532.9 / 44165.0 / 45748.6 ms |
| http-cpu / case-002         | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   1460.8 / 62494.8 / 64010.1 ms |
| http-cpu / case-003         | planner     | final rank miss      | final rank miss      | final rank miss      |                           142 / 160 |   2201.3 / 35020.5 / 37269.3 ms |
| http-cpu / case-004         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 7147.5 / 105935.9 / 113159.2 ms |
| http-cpu / case-005         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   3094.5 / 48556.1 / 51715.5 ms |
| http-cpu / case-006         | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   2783.2 / 56255.5 / 59097.9 ms |
| http-cpu / case-007         | planner     | success              | success              | success              |                           142 / 160 |   2774.2 / 48000.5 / 50847.8 ms |
| http-cpu / case-008         | planner     | success              | success              | success              |                           160 / 160 |   1802.9 / 56596.3 / 58447.6 ms |

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

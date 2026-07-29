# Query-Planned Recall: Live Profile Acceptance

**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.

This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.

## Bounds and identity

- Recorded against commit: `f4f7227dcf70fb014ceebdb9dd5251d61806a729`
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
| embedded-cpu         | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | cpu / cpu                    | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2136.4 ms | 8749.2 ms (pass) |       420.7 ms |       395.5 ms |         45693.5 / 60869.3 / 126536.9 ms |
| embedded-vulkan      | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | accelerated / vulkan         | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2449.3 ms | 1972.6 ms (pass) |       155.4 ms |       145.8 ms |          11991.3 / 13906.4 / 26931.5 ms |
| http-cpu             | llama-cpp-http (llama.cpp b8390 bundle b10e98a CPU planner-ctx-2048 reranker-ctx-8192 batch-2048) | cpu / AMD Ryzen 7 8845HS CPU | `llama-cpp-http-query-planning-v1`     | `llama-cpp-http-reranking-v1`                     |     5467.9 ms | 1237.9 ms (pass) |       472.3 ms |       430.3 ms |         31873.9 / 41111.5 / 101667.5 ms |

## Fallback characterization

- New candidate admissions beyond normal and retrieval-work-matched original-query controls: 1
- Ranking-only promotions: 3
- Preserved existing successes across profile runs: 6
- Planner fallbacks: 0
- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.
- Live new-candidate admission observed: yes
- Existing successes preserved across profiles: yes
- Existing-success regression profiles: none

## Candidate work by opaque case

| Planner/reranker run / case | Plan source | Normal               | Equal-work control   | Query-planned        | Candidate work (admitted / allowed) |    Planning / reranking / total |
| --------------------------- | ----------- | -------------------- | -------------------- | -------------------- | ----------------------------------: | ------------------------------: |
| embedded-cpu / case-001     | planner     | candidate union miss | candidate union miss | final rank miss      |                           152 / 160 |  15027.3 / 48071.4 / 63174.7 ms |
| embedded-cpu / case-002     | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2934.0 / 55586.2 / 58563.9 ms |
| embedded-cpu / case-003     | planner     | final rank miss      | final rank miss      | success              |                           102 / 120 |   2519.3 / 43133.5 / 45693.5 ms |
| embedded-cpu / case-004     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 7406.4 / 119069.4 / 126536.9 ms |
| embedded-cpu / case-005     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   5520.4 / 60044.9 / 65613.3 ms |
| embedded-cpu / case-006     | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   3028.7 / 43683.5 / 46754.1 ms |
| embedded-cpu / case-007     | planner     | success              | success              | success              |                           142 / 160 |   2921.2 / 53202.8 / 56171.3 ms |
| embedded-cpu / case-008     | planner     | success              | success              | success              |                           160 / 160 |   6935.8 / 66703.7 / 73681.7 ms |
| embedded-vulkan / case-001  | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  10982.6 / 15872.1 / 26931.5 ms |
| embedded-vulkan / case-002  | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |    2471.5 / 9476.0 / 11991.3 ms |
| embedded-vulkan / case-003  | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |    3682.6 / 9538.0 / 13268.5 ms |
| embedded-vulkan / case-004  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   3158.2 / 22051.4 / 25273.0 ms |
| embedded-vulkan / case-005  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   3376.1 / 10615.9 / 14039.9 ms |
| embedded-vulkan / case-006  | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |    2991.8 / 9288.8 / 12323.0 ms |
| embedded-vulkan / case-007  | planner     | success              | success              | success              |                           142 / 160 |   2775.2 / 10951.1 / 13772.9 ms |
| embedded-vulkan / case-008  | planner     | success              | success              | success              |                           160 / 160 |   6069.5 / 12647.0 / 18774.8 ms |
| http-cpu / case-001         | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |   3321.5 / 35812.3 / 39178.5 ms |
| http-cpu / case-002         | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   1413.7 / 41585.7 / 43044.5 ms |
| http-cpu / case-003         | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   1882.7 / 29941.4 / 31873.9 ms |
| http-cpu / case-004         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |  9267.4 / 92337.0 / 101667.5 ms |
| http-cpu / case-005         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   2562.5 / 43959.4 / 46570.7 ms |
| http-cpu / case-006         | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   1177.6 / 31884.9 / 33108.0 ms |
| http-cpu / case-007         | planner     | success              | success              | success              |                           142 / 160 |   2557.1 / 30963.2 / 33567.2 ms |
| http-cpu / case-008         | planner     | success              | success              | success              |                           160 / 160 |   1406.9 / 48492.5 / 49942.8 ms |

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

# Query-Planned Recall: Live Profile Acceptance

**Decision: Approved as an explicit fallback after hybrid misses.** Hybrid remains the default.

This evidence measures live query planning and reranking with fixed test embeddings. It does not measure one end-to-end production inference profile or claim broad retrieval superiority.

## Bounds and identity

- Recorded against commit: `feeb18b3d89544210a167be65d05fb9b29b219bb`
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

The private-corpus total search time includes retrieval with fixed test embeddings plus live planning and live reranking. The same live planner/reranker profiles also run over the checksum-fixed committed corpus with deterministic embeddings; this does not measure end-to-end production inference with EmbeddingGemma embeddings.

## Live planner and reranker matrix

| Planner/reranker run | Planner/reranker backend                                                                          | Planner/reranker device      | Planner adapter                        | Reranker adapter                                  | Cold planning |    Warm planning | Cold reranking | Warm reranking | Query-planned search min / median / max |
| -------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------- | ------------------------------------------------- | ------------: | ---------------: | -------------: | -------------: | --------------------------------------: |
| embedded-cpu         | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | cpu / cpu                    | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2264.9 ms | 2574.4 ms (pass) |       385.8 ms |       377.5 ms |         40566.2 / 61187.7 / 128147.8 ms |
| embedded-vulkan      | embedded (node-llama-cpp@3.18.1 / llama.cpp b8390)                                                | accelerated / vulkan         | `node-llama-cpp-qmd-query-planning-v1` | `node-llama-cpp-qwen-reranking-logit-recovery-v1` |     2187.2 ms | 2327.0 ms (pass) |       155.2 ms |       144.8 ms |          12542.2 / 14922.6 / 29276.7 ms |
| http-cpu             | llama-cpp-http (llama.cpp b8390 bundle b10e98a CPU planner-ctx-2048 reranker-ctx-8192 batch-2048) | cpu / AMD Ryzen 7 8845HS CPU | `llama-cpp-http-query-planning-v1`     | `llama-cpp-http-reranking-v1`                     |     1195.3 ms | 1052.1 ms (pass) |       303.4 ms |       270.8 ms |          30354.3 / 36020.4 / 87774.2 ms |

## Live planner/reranker quality on the committed corpus

Correctness uses the frozen committed-corpus retrieval and provenance gates. Median and p95 latency are recorded as explicit-fallback characterization; the hybrid 2-second latency gate does not apply to this mode.

| Profile run     | Cases | Candidate pool recall | Final recall | Context | Source occurrences | Session origins | Evidence relations | Contributing entries | Branches | Planner / reranker calls |         Median / p95 |
| --------------- | ----: | --------------------: | -----------: | ------: | -----------------: | --------------: | -----------------: | -------------------: | -------: | -----------------------: | -------------------: |
| embedded-cpu    |    17 |                100.0% |       100.0% |  100.0% |             100.0% |          100.0% |             100.0% |               100.0% |   100.0% |                  17 / 17 | 14642.3 / 22036.4 ms |
| embedded-vulkan |    17 |                100.0% |       100.0% |  100.0% |             100.0% |          100.0% |             100.0% |               100.0% |   100.0% |                  17 / 17 |   5915.5 / 7872.9 ms |
| http-cpu        |    17 |                100.0% |       100.0% |  100.0% |             100.0% |          100.0% |             100.0% |               100.0% |   100.0% |                  17 / 17 | 11978.8 / 15973.3 ms |

## Fallback characterization

- New candidate admissions beyond normal and retrieval-work-matched original-query controls: 0
- Ranking-only promotions: 3
- Preserved existing successes across profile runs: 5
- No improvement: 16
- Planner fallbacks: 0
- Live admissions and existing-success preservation are characterization, not release gates, because query-planned recall is invoked only after hybrid misses.
- Live new-candidate admission observed: no
- Existing successes preserved across profiles: no
- Existing-success regression profiles: embedded-vulkan

## Candidate work by opaque case

| Planner/reranker run / case | Plan source | Normal               | Equal-work control   | Query-planned        | Candidate work (admitted / allowed) |    Planning / reranking / total |
| --------------------------- | ----------- | -------------------- | -------------------- | -------------------- | ----------------------------------: | ------------------------------: |
| embedded-cpu / case-001     | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  14746.5 / 55658.7 / 70546.3 ms |
| embedded-cpu / case-002     | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2895.8 / 51388.3 / 54422.1 ms |
| embedded-cpu / case-003     | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   3711.9 / 36710.4 / 40566.2 ms |
| embedded-cpu / case-004     | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 | 6717.9 / 121274.4 / 128147.8 ms |
| embedded-cpu / case-005     | planner     | candidate union miss | candidate union miss | candidate union miss |                           103 / 120 |   2561.1 / 69437.6 / 72134.5 ms |
| embedded-cpu / case-006     | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   3246.4 / 44874.2 / 48253.5 ms |
| embedded-cpu / case-007     | planner     | success              | success              | success              |                           142 / 160 |   2972.2 / 50942.6 / 54052.7 ms |
| embedded-cpu / case-008     | planner     | success              | success              | success              |                           160 / 160 |   2532.8 / 65284.0 / 67953.4 ms |
| embedded-vulkan / case-001  | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |  10352.5 / 16346.9 / 26847.0 ms |
| embedded-vulkan / case-002  | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   2375.2 / 10084.7 / 12609.6 ms |
| embedded-vulkan / case-003  | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |    2709.5 / 9681.5 / 12542.2 ms |
| embedded-vulkan / case-004  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   6204.5 / 22908.9 / 29276.7 ms |
| embedded-vulkan / case-005  | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   6283.2 / 11944.0 / 18376.8 ms |
| embedded-vulkan / case-006  | planner     | candidate union miss | candidate union miss | candidate union miss |                           102 / 120 |   1972.6 / 10731.7 / 12848.4 ms |
| embedded-vulkan / case-007  | planner     | success              | success              | final rank miss      |                           142 / 160 |    4402.0 / 9158.0 / 13708.9 ms |
| embedded-vulkan / case-008  | planner     | success              | success              | success              |                           160 / 160 |   2693.1 / 13296.2 / 16136.2 ms |
| http-cpu / case-001         | planner     | candidate union miss | candidate union miss | candidate union miss |                           152 / 160 |   3097.1 / 32589.9 / 35801.8 ms |
| http-cpu / case-002         | planner     | candidate union miss | final rank miss      | candidate union miss |                           160 / 160 |   1158.6 / 33516.3 / 34788.0 ms |
| http-cpu / case-003         | planner     | final rank miss      | final rank miss      | success              |                           142 / 160 |   1832.7 / 28407.7 / 30354.3 ms |
| http-cpu / case-004         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   6431.4 / 81213.4 / 87774.2 ms |
| http-cpu / case-005         | planner     | candidate union miss | candidate union miss | candidate union miss |                           143 / 160 |   2640.5 / 41227.8 / 43988.1 ms |
| http-cpu / case-006         | planner     | candidate union miss | candidate union miss | candidate union miss |                           142 / 160 |   1370.3 / 34759.2 / 36239.0 ms |
| http-cpu / case-007         | planner     | success              | success              | success              |                           142 / 160 |   2667.3 / 31076.2 / 33858.6 ms |
| http-cpu / case-008         | planner     | success              | success              | success              |                           160 / 160 |   1534.1 / 42693.9 / 44343.1 ms |

## Failure, tool, and privacy semantics

- Privacy audit: 59 private values checked, 0 leaks
- Planner fallback through public service: pass
- Reranker failure through public service: pass
- Pi tool contract and policy evidence: pass

## Limitations

- Approval applies only as an explicit fallback after hybrid recall misses, with the accepted committed hybrid baseline and recorded planner, reranker, adapter, grammar, score, and search-policy identities.
- Live candidate admissions and preservation of queries already answered by hybrid are reported as fallback characterization, not release gates.
- Committed-corpus query-planned correctness must pass; its latency is recorded as explicit-fallback characterization rather than compared with the hybrid latency gate.
- EmbeddingGemma live candidates remain separate and are not approved when their committed-corpus quality gate fails.
- The committed corpus is synthetic-but-session-shaped; the private corpus is bounded and does not establish broad superiority.
- Private queries, plans, source text, session paths, and model artifacts remain outside Git.
- Hybrid remains the default search mode.

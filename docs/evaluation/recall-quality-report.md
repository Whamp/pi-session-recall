# Recall quality evaluation before backfill

Generated 2026-07-25T13:47:05.512Z from corpus `recall-quality-hybrid-bounded-v2`.

## Decision

**Automated gate: PASS**

Selected **512/64 tokens/overlap**, **8 candidates/channel**, and **5 final results**. This is the smallest measured candidate count, then the smallest final count, that passes every frozen hybrid-search gate; p95 query latency breaks ties.

**Full corpus backfill remains blocked pending human approval.** The command evaluated only the committed bounded corpus and did not read the configured production sessions directory.

## Frozen quality gate

| Metric                         | Frozen threshold |
| ------------------------------ | ---------------: |
| Candidate-pool recall          |         ≥ 100.0% |
| Fused top-N recall             |          ≥ 90.0% |
| Context usefulness             |          ≥ 90.0% |
| Source-occurrence preservation |         ≥ 100.0% |
| Final duplicate-result rate    |           ≤ 0.0% |
| Query p95                      |      ≤ 2000.0 ms |

The gate and count grid live in `evaluation/recall-quality-cases.json`; the run does not alter them after seeing results.

## Bounded work

| Work                               | Executed |             Hard maximum |
| ---------------------------------- | -------: | -----------------------: |
| Session files/index                |        8 |                        8 |
| Evaluation cases                   |       10 |                       10 |
| Temporary index runs               |        1 |                        1 |
| Search requests, including warmups |       44 |                       44 |
| Reranker requests                  |        0 |                        0 |
| Chunk-embedding HTTP batches       |       10 | bounded by 8 files/index |
| Maximum fused candidates/search    |       96 |                      200 |

Run duration: 8425.8 ms. Work data stayed under `.recall-data/recall-quality-evaluation/` and used only 8 checksum-fixed JSONL files.

## Metric definitions

- **Candidate-pool recall:** fraction of cases whose declared source appears anywhere in the complete bounded fused pool before duplicate grouping.
- **Fused top-N recall:** fraction of cases whose declared source appears in the first _N_ deterministic hybrid result groups.
- **Duplicate-result rate:** slots duplicating an earlier exact cross-session copy or overlapping source span, divided by all slots. Candidate-pool measurement reconstructs raw candidates; final measurement uses visible result groups.
- **Context usefulness:** fraction of cases whose first _N_ matching displayed results contain every independently declared context fragment. Neighbor-expanded text is used when present.
- **Source-occurrence preservation:** fraction of cases retaining the required count of distinct declared source locations, including suppressed duplicate occurrences.
- **Query latency:** wall time for the full read-only hybrid service search. Tables report nearest-rank median and p95 across 10 fixed cases after 1 warmup request per configuration.

## Chunk-policy index comparison

|  Chunk | Stored documents | Scanned/indexed sessions | New embedded documents | Embedding batches | Index time |
| -----: | ---------------: | -----------------------: | ---------------------: | ----------------: | ---------: |
| 512/64 |               96 |                      8/8 |                     92 |                10 |  5295.4 ms |

## Quality and latency matrix

|  Chunk | Candidates/channel | Final | Pool recall | Final recall | Pool duplicates | Final duplicates | Context | Sources |     Query p50/p95 | Gate |
| -----: | -----------------: | ----: | ----------: | -----------: | --------------: | ---------------: | ------: | ------: | ----------------: | ---- |
| 512/64 |                  8 |     5 |      100.0% |       100.0% |            5.0% |             0.0% |  100.0% |  100.0% | 68.5 ms / 72.0 ms | PASS |
| 512/64 |                 16 |     5 |      100.0% |       100.0% |            5.3% |             0.0% |  100.0% |  100.0% | 70.9 ms / 73.0 ms | PASS |
| 512/64 |                 24 |     5 |      100.0% |        90.0% |            8.1% |             0.0% |   90.0% |   90.0% | 70.3 ms / 72.8 ms | FAIL |
| 512/64 |                 32 |     5 |      100.0% |        90.0% |            9.3% |             0.0% |   90.0% |   90.0% | 71.2 ms / 75.8 ms | FAIL |

## Fixed cases and independent source evidence

| Case                              | Category                | Query                                                                      | Expected source evidence                                                               | Required context                                                                                              |
| --------------------------------- | ----------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| semantic-crash-safe-outbox        | semantic_paraphrase     | How did we make sure queued deliveries survive a worker crash?             | semantic-context.jsonl#queue-answer (active)                                           | append-only SQLite outbox; remote acknowledgement                                                             |
| semantic-image-retry-budget       | semantic_paraphrase     | What backoff and attempt limit did we choose for failed picture ingestion? | long-boundary.jsonl#image-retry-answer (active)                                        | decorrelated jitter capped at forty-two seconds; amber-orbit-17                                               |
| exact-read-node-error-code        | exact_identifier        | readNodeErrorCode src/read-node-error-code.ts EPERM                        | identifiers-tools.jsonl#identifier-answer (active)                                     | readNodeErrorCode; src/read-node-error-code.ts                                                                |
| tool-evidence-lease-token         | tool_evidence           | ELOCKED cobalt-7f31 lease-owner.ts                                         | identifiers-tools.jsonl#lease-tool-result (tool_result, active)                        | ELOCKED; cobalt-7f31                                                                                          |
| context-europe-rollout-conditions | context_dependent_reply | Under what conditions did we say yes to the rollout in Europe?             | semantic-context.jsonl#rollout-answer (active)                                         | European rollout during the maintenance window; draining the Paris queue below twenty jobs; Frankfurt replica |
| branch-abandoned-redis-group      | branch                  | What consumer group name belonged to the discarded Redis approach?         | branches.jsonl#redis-abandoned (abandoned)                                             | Redis Streams; glacier-lantern                                                                                |
| branch-active-offline-queue       | branch                  | Which queue backend did we finally choose for offline laptops?             | branches.jsonl#sqlite-active (active)                                                  | local SQLite WAL queue; idempotent upload acknowledgements                                                    |
| branch-summary-hosted-services    | summary                 | Why did the branch reject hosted queue services?                           | branches.jsonl#hosted-branch-summary (branch_summary, branch, active)                  | rejected hosted queue services; without a network connection                                                  |
| compaction-copper-finch           | summary                 | What recovery steps does Copper Finch stand for?                           | summaries.jsonl#recovery-compaction (compaction_summary, compaction, active)           | Copper Finch; snapshot delta-29; rotate the device certificate                                                |
| duplicate-meridian-release        | duplicate_content       | Which checksum and rollback tag were approved for Release Meridian?        | duplicate-a.jsonl#release-copy-a (active)<br>duplicate-b.jsonl#release-copy-b (active) | sha256:4c91d7e2; meridian-safe-3                                                                              |

## Per-case outcome

Shown for 512/64, 8 candidates/channel, and 5 final results.

| Case                              | Category                | Pool | Final | Context | Sources | Raw/grouped |   Query |
| --------------------------------- | ----------------------- | ---- | ----- | ------- | ------- | ----------: | ------: |
| semantic-crash-safe-outbox        | semantic_paraphrase     | hit  | hit   | useful  | 1 kept  |       12/12 | 64.5 ms |
| semantic-image-retry-budget       | semantic_paraphrase     | hit  | hit   | useful  | 1 kept  |       16/15 | 68.4 ms |
| exact-read-node-error-code        | exact_identifier        | hit  | hit   | useful  | 1 kept  |       10/10 | 72.0 ms |
| tool-evidence-lease-token         | tool_evidence           | hit  | hit   | useful  | 1 kept  |        10/8 | 68.7 ms |
| context-europe-rollout-conditions | context_dependent_reply | hit  | hit   | useful  | 1 kept  |       13/12 | 71.8 ms |
| branch-abandoned-redis-group      | branch                  | hit  | hit   | useful  | 1 kept  |       14/14 | 70.6 ms |
| branch-active-offline-queue       | branch                  | hit  | hit   | useful  | 1 kept  |       11/11 | 69.1 ms |
| branch-summary-hosted-services    | summary                 | hit  | hit   | useful  | 1 kept  |       10/10 | 68.5 ms |
| compaction-copper-finch           | summary                 | hit  | hit   | useful  | 1 kept  |       10/10 | 66.6 ms |
| duplicate-meridian-release        | duplicate_content       | hit  | hit   | useful  | 2 kept  |       14/12 | 68.5 ms |

## Reproduce

Prerequisites: the pinned Octen tokenizer assets and configured local embedding endpoint must be available. The optional reranker is not called. The command deletes and recreates only the dedicated ignored evaluation work directory.

```bash
npm run evaluate:recall
```

The command rewrites:

- `docs/evaluation/recall-quality-report.md`
- `docs/evaluation/recall-quality-results.json`

Environment:

- Git commit: `cc3022abb14f83f281280e0ffb97cdb63da5a13e`
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- Embedding: `octen-embed` → `Octen/Octen-Embedding-4B`, `Octen-Embedding-4B.Q8_0.gguf`, 2560 dimensions at `http://192.168.0.67:8090/v1`
- Optional deep reranker, not used by this evaluation: `qwen3-rerank` at `http://192.168.0.67:8091/v1`
- Specification: `/home/will/projects/pi-session-recall/.worktrees/token-aware-reranked-recall/evaluation/recall-quality-cases.json`
- Specification SHA-256: `ea5ba76c777d4b7a1dc3d0c671a935659fa427de718860dab1b4c194a0db6503`

Corpus file checksums:

- `semantic-context.jsonl`: `da042db32fa0fe1c210fc5f060471678623ba0f823abbb58498f0be579a69ebc`
- `identifiers-tools.jsonl`: `aea256ea49e471c9029cfc3358be625c37cc5209fe987a1b05e2ede2ec2553f6`
- `branches.jsonl`: `c0f706ae4138e880702679246ceec00bd99cda4444abd2cbfd573818faf9abdd`
- `summaries.jsonl`: `ce199336000b428686853fd0d605a13df12cd31e83f7b3bfe5ad5f8ba6c07c8b`
- `duplicate-a.jsonl`: `c2972f408868ffad68a2a104c552fff781338b43cdb00e24371b1679c7f22458`
- `duplicate-b.jsonl`: `8b54153ad7a1749beb8492155ecae64c25a270c4d752a8e7a239a040dd63dce0`
- `long-boundary.jsonl`: `49072e98ccab44523d0ba747f3ff537b50c8e4a412ac782d062dd0707437698c`
- `distractors.jsonl`: `6d63fefcb2e6fda2580985bbef4cf39bd073a3e15c9d2df7b67d2000847e4194`

## Limits of this evidence

- The corpus is a committed synthetic-but-session-shaped fixture, not a sample of private production logs. It covers every required retrieval class and includes 48 distractors plus a long boundary case, but it cannot estimate all real-corpus failure modes.
- Latency uses one measured request per case after one warmup, so it compares configurations on this host rather than establishing a capacity benchmark.
- A passing automated gate supports a candidate policy; it does not authorize the full corpus backfill. Human review of this report remains the approval boundary.

# Recall quality evaluation before backfill

Generated 2026-07-25T07:44:31.134Z from corpus `recall-quality-bounded-v1`.

## Decision

**Automated gate: FAIL**

**Evidence version: 1 (stale).** Required-source matching and per-search canary timing changed after this run. Rerun `npm run evaluate:recall` to produce v2 evidence before approval.

No candidate or final-result count passed every frozen gate.

- 512-64, 4 candidates/channel, 3 final: query p95 4719.2 ms exceeds 2000.0 ms
- 512-64, 4 candidates/channel, 3 final: reranker p95 4624.2 ms exceeds 1500.0 ms

**Full corpus backfill remains blocked pending human approval.** The command evaluated only the committed bounded corpus and did not read the configured production sessions directory.

## Frozen quality gate

| Metric                            | Frozen threshold |
| --------------------------------- | ---------------: |
| Pre-rerank recall                 |         ≥ 100.0% |
| Post-rerank recall                |          ≥ 90.0% |
| Context usefulness                |          ≥ 90.0% |
| Source-occurrence preservation    |         ≥ 100.0% |
| Post-rerank duplicate-result rate |           ≤ 0.0% |
| Query p95                         |      ≤ 2000.0 ms |
| Reranker p95                      |      ≤ 1500.0 ms |

The gate and count grid live in `evaluation/recall-quality-cases.json`; the run does not alter them after seeing results.

## Bounded work

| Work                               | Executed |             Hard maximum |
| ---------------------------------- | -------: | -----------------------: |
| Session files/index                |        8 |                        8 |
| Evaluation cases                   |       10 |                       10 |
| Temporary index runs               |        3 |                        3 |
| Search requests, including warmups |      132 |                      132 |
| Reranker requests                  |      132 |                      132 |
| Chunk-embedding HTTP batches       |       28 | bounded by 8 files/index |
| Maximum fused candidates/search    |       96 |                      200 |

Run duration: 1469694.5 ms. Work data stayed under `.recall-data/recall-quality-evaluation/` and used only 8 checksum-fixed JSONL files.

## Metric definitions

- **Pre-rerank recall:** fraction of cases whose declared source appears anywhere in the complete bounded fused pool before duplicate grouping and reranking.
- **Post-rerank recall:** fraction of cases whose declared source appears in the first _N_ reranked result groups.
- **Duplicate-result rate:** slots duplicating an earlier exact cross-session copy or overlapping source span, divided by all slots. Pre-rerank uses reconstructed raw candidates; post-rerank uses visible result groups.
- **Context usefulness:** fraction of cases whose first _N_ matching displayed results contain every independently declared context fragment. Neighbor-expanded text is used when present.
- **Source-occurrence preservation:** fraction of cases retaining the required count of distinct declared source locations, including suppressed duplicate occurrences.
- **Query latency:** wall time for the full read-only service search. **Reranker latency:** wall time inside the local reranker request. Tables report nearest-rank median and p95 across 10 fixed cases after 1 warmup request per configuration.

## Chunk-policy index comparison

|    Chunk | Stored documents | Scanned/indexed sessions | New embedded documents | Embedding batches | Index time |
| -------: | ---------------: | -----------------------: | ---------------------: | ----------------: | ---------: |
|   512/64 |               96 |                      8/8 |                     92 |                10 |  6650.7 ms |
|   768/96 |               84 |                      8/8 |                     80 |                 9 |  4654.8 ms |
| 1024/128 |               80 |                      8/8 |                     76 |                 9 |  4394.0 ms |

## Quality and latency matrix

|    Chunk | Candidates/channel | Final | Pre recall | Post recall | Pre duplicates | Post duplicates | Context | Sources |           Query p50/p95 |        Reranker p50/p95 | Gate |
| -------: | -----------------: | ----: | ---------: | ----------: | -------------: | --------------: | ------: | ------: | ----------------------: | ----------------------: | ---- |
|   512/64 |                  4 |     3 |     100.0% |      100.0% |           6.9% |            0.0% |  100.0% |  100.0% |    969.1 ms / 4719.2 ms |    877.6 ms / 4624.2 ms | FAIL |
|   512/64 |                  4 |     5 |     100.0% |      100.0% |           6.9% |            0.0% |  100.0% |  100.0% |    969.1 ms / 4719.2 ms |    877.6 ms / 4624.2 ms | FAIL |
|   512/64 |                  4 |     8 |     100.0% |      100.0% |           6.9% |            0.0% |  100.0% |  100.0% |    969.1 ms / 4719.2 ms |    877.6 ms / 4624.2 ms | FAIL |
|   512/64 |                  8 |     3 |     100.0% |      100.0% |           5.0% |            0.0% |  100.0% |  100.0% |   2729.2 ms / 8380.2 ms |   2636.0 ms / 8289.4 ms | FAIL |
|   512/64 |                  8 |     5 |     100.0% |      100.0% |           5.0% |            0.0% |  100.0% |  100.0% |   2729.2 ms / 8380.2 ms |   2636.0 ms / 8289.4 ms | FAIL |
|   512/64 |                  8 |     8 |     100.0% |      100.0% |           5.0% |            0.0% |  100.0% |  100.0% |   2729.2 ms / 8380.2 ms |   2636.0 ms / 8289.4 ms | FAIL |
|   512/64 |                 16 |     3 |     100.0% |      100.0% |           5.3% |            0.0% |  100.0% |  100.0% |  6891.4 ms / 14803.8 ms |  6800.1 ms / 14713.2 ms | FAIL |
|   512/64 |                 16 |     5 |     100.0% |      100.0% |           5.3% |            0.0% |  100.0% |  100.0% |  6891.4 ms / 14803.8 ms |  6800.1 ms / 14713.2 ms | FAIL |
|   512/64 |                 16 |     8 |     100.0% |      100.0% |           5.3% |            0.0% |  100.0% |  100.0% |  6891.4 ms / 14803.8 ms |  6800.1 ms / 14713.2 ms | FAIL |
|   512/64 |                 32 |     3 |     100.0% |      100.0% |           9.3% |            0.0% |  100.0% |  100.0% | 14545.6 ms / 17641.5 ms | 14445.9 ms / 17548.3 ms | FAIL |
|   512/64 |                 32 |     5 |     100.0% |      100.0% |           9.3% |            0.0% |  100.0% |  100.0% | 14545.6 ms / 17641.5 ms | 14445.9 ms / 17548.3 ms | FAIL |
|   512/64 |                 32 |     8 |     100.0% |      100.0% |           9.3% |            0.0% |  100.0% |  100.0% | 14545.6 ms / 17641.5 ms | 14445.9 ms / 17548.3 ms | FAIL |
|   768/96 |                  4 |     3 |     100.0% |      100.0% |           3.4% |            0.0% |  100.0% |  100.0% |   838.9 ms / 12591.5 ms |   775.3 ms / 12519.9 ms | FAIL |
|   768/96 |                  4 |     5 |     100.0% |      100.0% |           3.4% |            0.0% |  100.0% |  100.0% |   838.9 ms / 12591.5 ms |   775.3 ms / 12519.9 ms | FAIL |
|   768/96 |                  4 |     8 |     100.0% |      100.0% |           3.4% |            0.0% |  100.0% |  100.0% |   838.9 ms / 12591.5 ms |   775.3 ms / 12519.9 ms | FAIL |
|   768/96 |                  8 |     3 |     100.0% |      100.0% |           2.5% |            0.0% |  100.0% |  100.0% |  4710.6 ms / 20692.8 ms |  4637.5 ms / 20616.5 ms | FAIL |
|   768/96 |                  8 |     5 |     100.0% |      100.0% |           2.5% |            0.0% |  100.0% |  100.0% |  4710.6 ms / 20692.8 ms |  4637.5 ms / 20616.5 ms | FAIL |
|   768/96 |                  8 |     8 |     100.0% |      100.0% |           2.5% |            0.0% |  100.0% |  100.0% |  4710.6 ms / 20692.8 ms |  4637.5 ms / 20616.5 ms | FAIL |
|   768/96 |                 16 |     3 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 12451.5 ms / 23326.1 ms | 12379.2 ms / 23260.7 ms | FAIL |
|   768/96 |                 16 |     5 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 12451.5 ms / 23326.1 ms | 12379.2 ms / 23260.7 ms | FAIL |
|   768/96 |                 16 |     8 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 12451.5 ms / 23326.1 ms | 12379.2 ms / 23260.7 ms | FAIL |
|   768/96 |                 32 |     3 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 22352.2 ms / 23969.3 ms | 22271.6 ms / 23890.0 ms | FAIL |
|   768/96 |                 32 |     5 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 22352.2 ms / 23969.3 ms | 22271.6 ms / 23890.0 ms | FAIL |
|   768/96 |                 32 |     8 |     100.0% |      100.0% |           6.3% |            0.0% |  100.0% |  100.0% | 22352.2 ms / 23969.3 ms | 22271.6 ms / 23890.0 ms | FAIL |
| 1024/128 |                  4 |     3 |     100.0% |      100.0% |           5.2% |            0.0% |  100.0% |  100.0% |   815.8 ms / 22286.8 ms |   748.5 ms / 22212.3 ms | FAIL |
| 1024/128 |                  4 |     5 |     100.0% |      100.0% |           5.2% |            0.0% |  100.0% |  100.0% |   815.8 ms / 22286.8 ms |   748.5 ms / 22212.3 ms | FAIL |
| 1024/128 |                  4 |     8 |     100.0% |      100.0% |           5.2% |            0.0% |  100.0% |  100.0% |   815.8 ms / 22286.8 ms |   748.5 ms / 22212.3 ms | FAIL |
| 1024/128 |                  8 |     3 |     100.0% |      100.0% |           3.3% |            0.0% |  100.0% |  100.0% | 10864.5 ms / 32421.3 ms | 10783.2 ms / 32350.1 ms | FAIL |
| 1024/128 |                  8 |     5 |     100.0% |      100.0% |           3.3% |            0.0% |  100.0% |  100.0% | 10864.5 ms / 32421.3 ms | 10783.2 ms / 32350.1 ms | FAIL |
| 1024/128 |                  8 |     8 |     100.0% |      100.0% |           3.3% |            0.0% |  100.0% |  100.0% | 10864.5 ms / 32421.3 ms | 10783.2 ms / 32350.1 ms | FAIL |
| 1024/128 |                 16 |     3 |     100.0% |      100.0% |           6.4% |            0.0% |  100.0% |  100.0% | 17159.2 ms / 28438.3 ms | 17080.6 ms / 28354.3 ms | FAIL |
| 1024/128 |                 16 |     5 |     100.0% |      100.0% |           6.4% |            0.0% |  100.0% |  100.0% | 17159.2 ms / 28438.3 ms | 17080.6 ms / 28354.3 ms | FAIL |
| 1024/128 |                 16 |     8 |     100.0% |      100.0% |           6.4% |            0.0% |  100.0% |  100.0% | 17159.2 ms / 28438.3 ms | 17080.6 ms / 28354.3 ms | FAIL |
| 1024/128 |                 32 |     3 |     100.0% |      100.0% |           4.6% |            0.0% |  100.0% |  100.0% | 29730.1 ms / 35814.5 ms | 29648.4 ms / 35742.5 ms | FAIL |
| 1024/128 |                 32 |     5 |     100.0% |      100.0% |           4.6% |            0.0% |  100.0% |  100.0% | 29730.1 ms / 35814.5 ms | 29648.4 ms / 35742.5 ms | FAIL |
| 1024/128 |                 32 |     8 |     100.0% |      100.0% |           4.6% |            0.0% |  100.0% |  100.0% | 29730.1 ms / 35814.5 ms | 29648.4 ms / 35742.5 ms | FAIL |

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

Shown for 512/64, 4 candidates/channel, and 3 final results.

| Case                              | Category                | Pre | Post | Context | Sources | Raw/grouped |     Query |  Reranker |
| --------------------------------- | ----------------------- | --- | ---- | ------- | ------- | ----------: | --------: | --------: |
| semantic-crash-safe-outbox        | semantic_paraphrase     | hit | hit  | useful  | 1 kept  |         5/5 |  869.1 ms |  790.1 ms |
| semantic-image-retry-budget       | semantic_paraphrase     | hit | hit  | useful  | 1 kept  |         8/7 | 4719.2 ms | 4624.2 ms |
| exact-read-node-error-code        | exact_identifier        | hit | hit  | useful  | 1 kept  |         6/6 | 3366.2 ms | 3270.8 ms |
| tool-evidence-lease-token         | tool_evidence           | hit | hit  | useful  | 1 kept  |         6/5 | 2538.2 ms | 2443.7 ms |
| context-europe-rollout-conditions | context_dependent_reply | hit | hit  | useful  | 1 kept  |         6/5 |  827.0 ms |  732.6 ms |
| branch-abandoned-redis-group      | branch                  | hit | hit  | useful  | 1 kept  |         6/6 |  969.1 ms |  877.6 ms |
| branch-active-offline-queue       | branch                  | hit | hit  | useful  | 1 kept  |         5/5 |  802.6 ms |  711.2 ms |
| branch-summary-hosted-services    | summary                 | hit | hit  | useful  | 1 kept  |         5/5 |  799.7 ms |  709.5 ms |
| compaction-copper-finch           | summary                 | hit | hit  | useful  | 1 kept  |         5/5 | 2515.9 ms | 2422.3 ms |
| duplicate-meridian-release        | duplicate_content       | hit | hit  | useful  | 2 kept  |         6/5 | 1521.3 ms | 1428.2 ms |

## Reproduce

Prerequisites: the pinned Octen tokenizer assets and the configured local embedding and reranker endpoints must be available. The command deletes and recreates only the dedicated ignored evaluation work directory.

```bash
npm run evaluate:recall
```

The command rewrites:

- `docs/evaluation/recall-quality-report.md`
- `docs/evaluation/recall-quality-results.json`

Environment:

- Git commit: `ee69002d4e6b75df7dc8a7023d337d48718c50d0`
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- Embedding: `octen-embed` → `Octen/Octen-Embedding-4B`, `Octen-Embedding-4B.Q8_0.gguf`, 2560 dimensions at `http://192.168.0.67:8090/v1`
- Reranker: `qwen3-rerank` at `http://192.168.0.67:8091/v1`
- Specification: `/home/will/projects/pi-session-recall/.worktrees/token-aware-reranked-recall/evaluation/recall-quality-cases.json`
- Specification SHA-256: `56c69765256622169a6da4bf5d2b78c1d76de1e621784a1c5a1c371e3eaa6b3c`

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
- The measured grid has no discriminating quality variance across gated recall, context, source-preservation, and visible-duplicate metrics; it can compare latency but cannot rank policy quality.
- Latency uses one measured request per case after one warmup, so it compares configurations on this host rather than establishing a capacity benchmark.
- A passing automated gate supports a candidate policy; it does not authorize the full corpus backfill. Human review of this report remains the approval boundary.

# Project-scoped recall quality evaluation before backfill

Generated 2026-07-25T20:14:40.373Z from corpus `recall-quality-project-scoped-bounded-v3`.

## Decision

**Automated gate: PASS**

Selected **512/64 tokens/overlap**, **8 candidates/channel**, and **5 final results**. This is the smallest measured candidate count, then the smallest final count, that passes every frozen hybrid-search gate; p95 query latency breaks ties.

**Full corpus backfill remains blocked pending human approval.** The command evaluated only the committed bounded corpus and did not read the configured production sessions directory.

## Evaluation identity

- Default scope: `project` (policy v1)
- Project identity policy: v4; metadata schema v3
- Project lineage policy: v1; digest `02023e9934990f61a81488144cb1b27d9002e0af750c38a1b83adc1de367bc4b`
- Hybrid ranking: fusion v1, RRF k=60, active prior +0.0100
- Candidate limits: dense 8, lexical 8, identifier 8; final results 5

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

| Work                                       | Executed | Hard maximum |
| ------------------------------------------ | -------: | -----------: |
| Session files/index                        |       15 |           15 |
| Evaluation cases                           |       17 |           17 |
| Temporary index runs                       |        1 |            1 |
| Search requests, including warmups         |       20 |           20 |
| Reranker requests                          |        0 |            0 |
| Chunk-embedding HTTP batches               |       17 |           20 |
| Maximum fused candidates/search            |       24 |          200 |
| Production repository identity resolutions |        5 |            5 |

Run duration: 15479.0 ms. Work data stayed under `evaluation/.recall-data/recall-quality-evaluation/` and used only 15 checksum-fixed JSONL files.

## Metric definitions

- **Candidate-pool recall:** fraction of cases whose declared source appears anywhere in the complete bounded fused pool before duplicate grouping.
- **Fused top-N recall:** fraction of cases whose declared source appears in the first _N_ deterministic hybrid result groups.
- **Duplicate-result rate:** slots duplicating an earlier exact cross-session copy or overlapping source span, divided by all slots. Candidate-pool measurement reconstructs raw candidates; final measurement uses visible result groups.
- **Context usefulness:** fraction of cases whose first _N_ matching displayed results contain every independently declared context fragment. Neighbor-expanded text is used when present.
- **Source-occurrence preservation:** fraction of cases retaining the required count of distinct declared source locations, including suppressed duplicate occurrences.
- **Query latency:** wall time for the full read-only hybrid service search, measured and gated independently for project and global scope. Tables report nearest-rank p95 across 17 fixed cases after 1 warmup request per represented scope and configuration.

## Chunk-policy index comparison

|  Chunk | Stored documents | Scanned/indexed sessions | New embedded documents | Embedding batches | Index time |
| -----: | ---------------: | -----------------------: | ---------------------: | ----------------: | ---------: |
| 512/64 |              128 |                    15/15 |                    124 |                17 | 11213.0 ms |

## Quality and latency matrix

|  Chunk | Candidates/channel | Final | Pool recall | Final recall | Pool duplicates | Final duplicates | Context | Sources | Provenance | Project p95 | Global p95 | Gate |
| -----: | -----------------: | ----: | ----------: | -----------: | --------------: | ---------------: | ------: | ------: | ---------: | ----------: | ---------: | ---- |
| 512/64 |                  8 |     5 |      100.0% |       100.0% |            2.2% |             0.0% |  100.0% |  100.0% |     100.0% |     73.5 ms |  1447.5 ms | PASS |

## Pre-limit channel proof

| Case                                          | Channel    | Project source admitted | Global source displaced | Polluters inside limit | Proof |
| --------------------------------------------- | ---------- | ----------------------- | ----------------------- | ---------------------: | ----- |
| project-main-retrieves-worktree-before-limits | dense      | yes                     | yes                     |                      8 | PASS  |
| project-main-retrieves-worktree-before-limits | lexical    | yes                     | yes                     |                      8 | PASS  |
| project-main-retrieves-worktree-before-limits | identifier | yes                     | yes                     |                      8 | PASS  |

## Fixed cases and independent source evidence

| Case                                          | Category                | Scope   | Invocation                               | Query                                                                      | Expected source evidence                                                                                                                                                                                                                                               | Excluded sessions               | Required context                                                                                              |
| --------------------------------------------- | ----------------------- | ------- | ---------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| semantic-crash-safe-outbox                    | semantic_paraphrase     | global  | none                                     | How did we make sure queued deliveries survive a worker crash?             | semantic-context.jsonl#queue-answer (active, unrestricted_global_evidence, origin /evaluation/fulfillment, contributors queue-answer)                                                                                                                                  | none                            | append-only SQLite outbox; remote acknowledgement                                                             |
| semantic-image-retry-budget                   | semantic_paraphrase     | global  | none                                     | What backoff and attempt limit did we choose for failed picture ingestion? | long-boundary.jsonl#image-retry-answer (active, unrestricted_global_evidence, origin /evaluation/media, contributors image-retry-answer)                                                                                                                               | none                            | decorrelated jitter capped at forty-two seconds; amber-orbit-17                                               |
| exact-read-node-error-code                    | exact_identifier        | global  | none                                     | readNodeErrorCode src/read-node-error-code.ts EPERM                        | identifiers-tools.jsonl#identifier-answer (active, unrestricted_global_evidence, origin /evaluation/runtime, contributors identifier-answer)                                                                                                                           | none                            | readNodeErrorCode; src/read-node-error-code.ts                                                                |
| tool-evidence-lease-token                     | tool_evidence           | global  | none                                     | ELOCKED cobalt-7f31 lease-owner.ts                                         | identifiers-tools.jsonl#lease-tool-result (tool_result, active, unrestricted_global_evidence, origin /evaluation/runtime, contributors lease-tool-result)                                                                                                              | none                            | ELOCKED; cobalt-7f31                                                                                          |
| context-europe-rollout-conditions             | context_dependent_reply | global  | none                                     | Under what conditions did we say yes to the rollout in Europe?             | semantic-context.jsonl#rollout-answer (active, unrestricted_global_evidence, origin /evaluation/fulfillment, contributors rollout-answer)                                                                                                                              | none                            | European rollout during the maintenance window; draining the Paris queue below twenty jobs; Frankfurt replica |
| branch-abandoned-redis-group                  | branch                  | global  | none                                     | What consumer group name belonged to the discarded Redis approach?         | branches.jsonl#redis-abandoned (abandoned, unrestricted_global_evidence, origin /evaluation/sync, contributors redis-abandoned)                                                                                                                                        | none                            | Redis Streams; glacier-lantern                                                                                |
| branch-active-offline-queue                   | branch                  | global  | none                                     | Which queue backend did we finally choose for offline laptops?             | branches.jsonl#sqlite-active (active, unrestricted_global_evidence, origin /evaluation/sync, contributors sqlite-active)                                                                                                                                               | none                            | local SQLite WAL queue; idempotent upload acknowledgements                                                    |
| branch-summary-hosted-services                | summary                 | global  | none                                     | Why did the branch reject hosted queue services?                           | branches.jsonl#hosted-branch-summary (branch_summary, branch, active, unrestricted_global_evidence, origin /evaluation/sync, contributors hosted-branch-summary)                                                                                                       | none                            | rejected hosted queue services; without a network connection                                                  |
| compaction-copper-finch                       | summary                 | global  | none                                     | What recovery steps does Copper Finch stand for?                           | summaries.jsonl#recovery-compaction (compaction_summary, compaction, active, unrestricted_global_evidence, origin /evaluation/desktop, contributors recovery-compaction)                                                                                               | none                            | Copper Finch; snapshot delta-29; rotate the device certificate                                                |
| duplicate-meridian-release                    | duplicate_content       | global  | none                                     | Which checksum and rollback tag were approved for Release Meridian?        | duplicate-a.jsonl#release-copy-a (active, unrestricted_global_evidence, origin /evaluation/releases, contributors release-copy-a)<br>duplicate-b.jsonl#release-copy-b (active, unrestricted_global_evidence, origin /evaluation/releases, contributors release-copy-b) | none                            | sha256:4c91d7e2; meridian-safe-3                                                                              |
| project-main-retrieves-worktree-before-limits | project_scope           | project | /evaluation/repository/main              | SCOPE_LIMIT_PROBE SCOPE_LIMIT_IDENTIFIER channel budget                    | project-worktree.jsonl#worktree-answer (active, same_repository, origin /evaluation/repository/worktrees/feature, contributors worktree-answer)                                                                                                                        | unrelated-similar-project.jsonl | cedar-worktree-41; project evidence                                                                           |
| project-worktree-retrieves-main               | project_scope           | project | /evaluation/repository/worktrees/feature | What reverse worktree bridge marker did the main checkout choose?          | project-main.jsonl#main-answer (active, same_repository, origin /evaluation/repository/main, contributors main-answer)                                                                                                                                                 | unrelated-similar-project.jsonl | obsidian-main-52; reverse worktree bridge                                                                     |
| project-equivalent-clone-origin               | project_scope           | project | /evaluation/repository/clones/a          | Which marker proves the equivalent clone shares canonical origin identity? | equivalent-clone.jsonl#clone-answer (active, same_repository, origin /evaluation/repository/clones/b, contributors clone-answer)                                                                                                                                       | unrelated-similar-project.jsonl | cobalt-clone-63; canonical origin identity                                                                    |
| project-configured-lineage                    | project_scope           | project | /evaluation/repository/main              | What historical prototype marker did configured project lineage preserve?  | historical-prototype.jsonl#lineage-answer (active, configured_project_lineage, origin /evaluation/historical/quality-prototype/phase-one, contributors lineage-answer)                                                                                                 | unrelated-similar-project.jsonl | linen-lineage-74; successor repository                                                                        |
| project-similar-name-remains-unrelated        | project_scope           | project | /evaluation/repository/main              | Who owns similarity fence token aurora-same-19?                            | project-main.jsonl#similarity-answer (active, same_repository, origin /evaluation/repository/main, contributors similarity-answer)                                                                                                                                     | unrelated-similar-project.jsonl | aurora-same-19; nearby names do not establish project identity                                                |
| project-exact-non-git-origin                  | project_scope           | project | /evaluation/non-git/exact                | What exact non-Git session origin marker excludes nearby directories?      | non-git-exact.jsonl#non-git-answer (active, same_session_origin, origin /evaluation/non-git/exact, contributors non-git-answer)                                                                                                                                        | non-git-nearby.jsonl            | quartz-origin-85; no nearby directory                                                                         |
| global-explicit-cross-project                 | project_scope           | global  | /evaluation/repository/main              | Which foreign project marker may explicit global recall reuse?             | unrelated-similar-project.jsonl#global-foreign-answer (active, unrestricted_global_evidence, origin /evaluation/repository/quality-fixture-similar, contributors global-foreign-answer)                                                                                | none                            | violet-global-96; deliberately broadens scope                                                                 |

## Per-case outcome

Shown for 512/64, 8 candidates/channel, and 5 final results.

| Case                                          | Scope   | Boundary | Pool | Final | Context | Sources | Origin | Relation | Contributors | Branch | Raw/grouped |     Query |
| --------------------------------------------- | ------- | -------- | ---- | ----- | ------- | ------- | ------ | -------- | ------------ | ------ | ----------: | --------: |
| semantic-crash-safe-outbox                    | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       13/13 |   73.4 ms |
| semantic-image-retry-budget                   | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       16/15 |   71.6 ms |
| exact-read-node-error-code                    | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       10/10 |   72.9 ms |
| tool-evidence-lease-token                     | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |        10/9 |   70.1 ms |
| context-europe-rollout-conditions             | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       14/13 | 1447.5 ms |
| branch-abandoned-redis-group                  | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       14/14 |  258.5 ms |
| branch-active-offline-queue                   | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       12/12 |   73.4 ms |
| branch-summary-hosted-services                | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       11/11 |   73.5 ms |
| compaction-copper-finch                       | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       12/12 |   72.3 ms |
| duplicate-meridian-release                    | global  | pass     | hit  | hit   | useful  | 2 kept  | pass   | pass     | pass         | pass   |       14/13 |   73.9 ms |
| project-main-retrieves-worktree-before-limits | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |         8/8 |   72.0 ms |
| project-worktree-retrieves-main               | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |         8/8 |   73.5 ms |
| project-equivalent-clone-origin               | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       11/11 |   71.2 ms |
| project-configured-lineage                    | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |         9/9 |   72.1 ms |
| project-similar-name-remains-unrelated        | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |         8/8 |   72.4 ms |
| project-exact-non-git-origin                  | project | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |         3/3 |   72.9 ms |
| global-explicit-cross-project                 | global  | pass     | hit  | hit   | useful  | 1 kept  | pass   | pass     | pass         | pass   |       10/10 |   71.0 ms |

## Reproduce

Prerequisites: the pinned Octen tokenizer assets and configured local embedding endpoint must be available. The optional reranker is not called. The command deletes and recreates only the dedicated ignored evaluation work directory.

```bash
npm run evaluate:recall
```

The command rewrites:

- `docs/evaluation/recall-quality-report.md`
- `docs/evaluation/recall-quality-results.json`

Environment:

- Git commit: `38f9d505d594d6ad16e9eeb18487b6c75eda693b`
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- Embedding: `octen-embed` → `Octen/Octen-Embedding-4B`, `Octen-Embedding-4B.Q8_0.gguf`, 2560 dimensions at `http://192.168.0.67:8090/v1`
- Hybrid ranking identity: fusion v1, RRF k=60, active prior +0.0100
- Optional deep reranker, not used by this evaluation: `qwen3-rerank` at `http://192.168.0.67:8091/v1`
- Specification: `/home/will/projects/pi-session-recall/.worktrees/project-scoped-recall/evaluation/recall-quality-cases.json`
- Specification SHA-256: `b5209236a48feb2a8ffca5358ef62ac609ad82630d28d079466985470cbe39d9`

Corpus file checksums:

- `semantic-context.jsonl`: `da042db32fa0fe1c210fc5f060471678623ba0f823abbb58498f0be579a69ebc`
- `identifiers-tools.jsonl`: `aea256ea49e471c9029cfc3358be625c37cc5209fe987a1b05e2ede2ec2553f6`
- `branches.jsonl`: `c0f706ae4138e880702679246ceec00bd99cda4444abd2cbfd573818faf9abdd`
- `summaries.jsonl`: `ce199336000b428686853fd0d605a13df12cd31e83f7b3bfe5ad5f8ba6c07c8b`
- `duplicate-a.jsonl`: `c2972f408868ffad68a2a104c552fff781338b43cdb00e24371b1679c7f22458`
- `duplicate-b.jsonl`: `8b54153ad7a1749beb8492155ecae64c25a270c4d752a8e7a239a040dd63dce0`
- `long-boundary.jsonl`: `49072e98ccab44523d0ba747f3ff537b50c8e4a412ac782d062dd0707437698c`
- `distractors.jsonl`: `6d63fefcb2e6fda2580985bbef4cf39bd073a3e15c9d2df7b67d2000847e4194`
- `project-worktree.jsonl`: `d265edfb754072020869ef1ba49f472e6a4f81f25a18be96335a5f87edf6db80`
- `project-main.jsonl`: `dafa31d86195f920bc7ce6d5c21a679320815fa683ed4c5e93a77e428d4bea92`
- `equivalent-clone.jsonl`: `21dad11c36b18a971acd49a891967bfe1a1c6269c938d5e4c0ef31876ada19c1`
- `historical-prototype.jsonl`: `173a2e479ae517d6f7935d711b0e483fcedcd46adde443986a89fad3e765b5b9`
- `unrelated-similar-project.jsonl`: `59aaf838436d857fc67073ffbe8de0d64cd5cbf894fd2f16f0004bdf7bfd1756`
- `non-git-exact.jsonl`: `d41df315b0a25e3e0e6cd5a14c58837e5ee492e68a15e84e87b411384176aebd`
- `non-git-nearby.jsonl`: `8e576068c09966c5a286cd04a56a486c5f16a42a46670cfe83328e61657d9ddc`

## Limits of this evidence

- The corpus is a committed synthetic-but-session-shaped fixture, not a sample of private production logs. It covers the required retrieval and project-identity classes but cannot estimate all real-corpus failure modes.
- Latency uses one measured request per case after one warmup, so it compares configurations on this host rather than establishing a capacity benchmark.
- A passing automated gate supports a candidate policy; it does not authorize the full corpus backfill. Human review of this report remains the approval boundary.

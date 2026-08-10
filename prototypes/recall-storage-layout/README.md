# Recall storage layout prototype

**Throwaway prototype. Do not ship this code.**

## Question

Can compact recall eliminate routine compaction while preserving acceptable latency, which store should own Invocation search, and can one SQLite database with FTS5 plus sqlite-vec supersede the draft flat-Zvec-plus-SQLite architecture?

## Candidate

The prototype:

- copies only dense-searchable conversation, summary, and turn-context documents;
- reuses production embeddings instead of calling the document embedding endpoint;
- stores no tool rows, zero vectors, FTS fields, or HNSW index;
- retains the current non-tool provenance fields;
- compares flat dense results with the production HNSW results;
- scans canonical session JSONL for exact tool/result evidence;
- builds a compact SQLite FTS index containing tool names, bounded locator arguments, and direct bash commands, but no tool results or bash output;
- compares that SQLite index with a second Zvec collection containing the same Invocation records, one FTS field, and no vector fields;
- builds one SQLite 3.53 database with session state, Invocation FTS5, all dense metadata, and released sqlite-vec 0.1.9 FP32 vectors;
- tests unpartitioned, per-project-partitioned, and 16-project-bucket vector layouts;
- exercises explicit rollback, concurrent-reader visibility, SIGKILL recovery, representative updates, and 100 replacement cycles.

It never writes to the production collection. Scratch data lives at:

```text
~/.pi/agent/recall-debug/prototype-dense-only
~/.pi/agent/recall-debug/prototype-sqlite-vec
```

Each builder stops if scratch allocation exceeds 6 GiB or free space falls below 240 GiB.

## Run

```bash
npm run prototype:recall-storage-layout -- run --reset
```

Individual phases:

```bash
npm run prototype:recall-storage-layout -- build --reset
npm run prototype:recall-storage-layout -- benchmark-dense
npm run prototype:recall-storage-layout -- build-invocations
npm run prototype:recall-storage-layout -- benchmark-invocations
npm run prototype:recall-storage-layout -- compare-invocation-stores --reset
npm run prototype:recall-storage-layout -- benchmark-source

npm run prototype:sqlite-vec -- build --reset
npm run prototype:sqlite-vec -- benchmark-before-churn
npm run prototype:sqlite-vec -- atomicity
npm run prototype:sqlite-vec -- update
npm run prototype:sqlite-vec -- churn --cycles=100
npm run prototype:sqlite-vec -- benchmark-after-churn
npm run prototype:sqlite-vec -- build-partitioned
npm run prototype:sqlite-vec -- benchmark-partitioned
npm run prototype:sqlite-vec -- build-bucketed
npm run prototype:sqlite-vec -- benchmark-bucketed
```

Results are written to:

```text
~/.pi/agent/recall-debug/prototype-dense-only/prototype-report.json
~/.pi/agent/recall-debug/prototype-sqlite-vec/sqlite-vec-report.json
```

## Decision gate

Candidate A is viable only if it meets all of these provisional thresholds:

- scratch collection below 5 GiB;
- flat dense search p95 below 500 ms;
- no material top-result regression in the five production benchmark queries;
- exact source search finds tool/result evidence with source provenance;
- no routine full-collection optimization is required.

## Result

Candidate A passed.

| Measurement                                     |                        Result |
| ----------------------------------------------- | ----------------------------: |
| Dense-only flat Zvec allocation after writeback |                      2.02 GiB |
| Dense documents                                 |                       340,736 |
| Omitted lexical-only rows                       |                     1,109,045 |
| Flat dense latency                              |   44.5 ms median; 48.2 ms p95 |
| Dense top-result agreement with production HNSW |                   5/5 queries |
| Dense top-eight overlap                         |                         7–8/8 |
| Compact invocation index                        |  118 MiB; 218,764 invocations |
| Invocation query latency                        | 0.093 ms median; 0.122 ms p95 |
| Replace one session's 772 invocation rows       |        2.07 MiB device writes |
| Full raw JSONL scan                             |           21.7 s; zero writes |

A follow-up comparison fed the same 219,734 Invocation records to SQLite FTS5 and a vectorless Zvec FTS collection:

| Invocation measurement             | SQLite FTS5 | Zvec FTS before optimize |  Zvec FTS after optimize |
| ---------------------------------- | ----------: | -----------------------: | -----------------------: |
| Allocated storage                  |     124 MiB |                  270 MiB |                  197 MiB |
| Initial build time                 |      1.95 s |                  20.38 s |                        — |
| Initial build device writes        |    66.9 MiB |                390.3 MiB |                        — |
| Replace 886 records: elapsed       |     0.096 s |                  0.285 s |                        — |
| Replace 886 records: device writes |    2.48 MiB |                 3.91 MiB |                        — |
| Search median                      |    0.343 ms |                 0.177 ms |                 0.130 ms |
| Search p95                         |     3.18 ms |                  4.90 ms |                  1.57 ms |
| One Zvec optimization              |           — |                        — | 1.55 s; 104.9 MiB writes |

Both stores pass the functional, write, and latency gates. Zvec is a viable Invocation-only store and does not require fake vectors. SQLite remains smaller and also owns transactional per-session state. That comparison reopened the storage decision but did not settle the dense store.

### Unified SQLite plus sqlite-vec

The final prototype copied the certified v7 generation into one SQLite database containing 3,719 session states, 218,139 Invocations with FTS5, 341,036 dense documents with all 46 metadata fields, and 1,024-dimensional FP32 sqlite-vec vectors.

| Measurement                                       |    Unified SQLite result |   Draft flat Zvec + SQLite |
| ------------------------------------------------- | -----------------------: | -------------------------: |
| Projected storage, recommended routing            |                 3.87 GiB |                   2.28 GiB |
| Warm global dense p95                             |                   388 ms |                      69 ms |
| Warm project dense p95                            |                    34 ms |                      21 ms |
| Best-effort cold global                           |                   1.63 s |                     1.37 s |
| Best-effort cold project                          |                   1.75 s |                     1.16 s |
| Matching top results                              |       5/5 in both scopes |                          — |
| Top-eight overlap                                 |  global 7–8; project 5–8 |                          — |
| Invocation FTS p95 after churn                    |                  2.31 ms |                          — |
| Representative dual-vector update                 |  333 ms; 2.61 MiB writes |    5.73 MiB measured in PR |
| 100 replacement cycles                            | 3.06 MiB/cycle; 0 growth |                          — |
| Reader visibility, rollback, and SIGKILL recovery |                    exact | cross-store retry protocol |

One unpartitioned vec0 table gave the best global search but project p95 of about 160 ms. Exact per-project partitioning reduced project p95 to 28 ms but raised global p95 to 650 ms. The recommended production shape keeps two vector copies in the same SQLite transaction: global search uses the unpartitioned table, while default project search uses a 16-bucket table plus exact project metadata. This projects to 3.87 GiB, below the 5 GiB gate.

The unified candidate passed explicit rollback and SIGKILL tests on the full database. A concurrent reader saw the complete old session while the writer had deleted its state, FTS rows, dense metadata, and both vector copies. Recovery restored the exact prior hash, `integrity_check` remained `ok`, and foreign-key checks found no violations. One hundred committed replacement cycles caused no file growth or free-page backlog.

**Verdict:** unified SQLite plus sqlite-vec should supersede the draft flat-Zvec-plus-SQLite storage architecture. Default project latency remains near flat Zvec, explicit global latency stays near the original 500 ms warm gate, incremental writes remain small, and one transaction removes the cross-store recovery problem. Production work must pin pre-1.0 sqlite-vec 0.1.9, route by scope as measured, retain staged activation and canonical JSONL, and verify the published macOS x64 and arm64 packages before release.

The stable design is now one derived SQLite database, canonical JSONL for complete payloads, no HNSW, no zero-vector tool rows, and no routine whole-database optimization. Sanitized measurements are in [`results.json`](results.json).

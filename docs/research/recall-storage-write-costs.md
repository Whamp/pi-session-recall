# Recall storage and write-cost evidence

## Purpose

This report records why `pi-session-recall` must settle its storage layout before changing optimization policy. It separates incremental indexing from full Zvec compaction and attributes their computation and writes to concrete operations.

Measurements came from the production collection on 2026-08-09. The hourly index timer was paused during controlled runs. No measurement rebuilt or destroyed production data.

## Current collection

The collection held about 1.45 million documents. Only about 23.5% were dense-searchable conversation, summary, or turn-context documents. Tool names, arguments, results, direct bash commands, and bash output made up the remaining 76.5%.

The production Zvec collection schema requires every row in that collection to carry its vector field. Lexical-only tool rows therefore store a 1,024-float zero vector even though dense queries filter them out. Zvec 0.6 also supports a separate collection with no vector fields; the zero vectors were a consequence of combining dense and lexical records in one collection, not a Zvec-wide requirement. Every production tool row also carries scalar provenance and participates in two FTS indexes.

This unified layout makes lexical tool evidence contribute to scalar, vector, and FTS maintenance costs.

## Incremental `psr index`

A controlled run indexed one changed physical session without optimization.

| Measurement                     |    Result |
| ------------------------------- | --------: |
| Wall time                       |   41.16 s |
| CPU time                        |   43.05 s |
| Peak memory                     |   1.9 GiB |
| Process-attributed writes       | 73.56 MiB |
| NVMe writes observed during run | 53.48 MiB |
| New dense embeddings            |         6 |
| Reused dense vectors            |       642 |
| Net new documents               |        81 |

### Time attribution

The command spent about three seconds opening the collection, loading index state, discovering 3,721 physical session files, and planning the workset.

It then spent about 36.5 seconds reparsing, rebuilding, rechunking, retokenizing, fetching, and comparing the entire changed session. The Zvec update and six embedding requests completed in about 1.1 seconds.

The changed session contained 4,470 indexed documents:

| Evidence       | Documents |  Tokens |
| -------------- | --------: | ------: |
| Tool results   |     2,331 | 894,343 |
| Tool arguments |       778 | 112,950 |
| Tool names     |       713 |     724 |
| Conversation   |       319 |  31,687 |
| Summaries      |       238 | 107,679 |
| Turn context   |        91 |  34,972 |

Tool evidence accounted for 85.5% of documents and 85.3% of tokens in that session. Appending to the session caused the indexer to reconstruct that unchanged evidence before checksum reconciliation.

### Write attribution

Before the final state checkpoint, the process had written only 2.48 MiB. Changed Zvec allocation was:

- vector: 1.35 MiB;
- FTS: 0.57 MiB;
- scalar: 0.22 MiB;
- small id-map and manifest updates.

The final atomic checkpoint rewrote the entire 70 MiB `index-state.json`, raising process-attributed writes to 73.56 MiB. Changing one session therefore rewrites state for all indexed sessions.

## Full `psr optimize`

A steady-state optimization ran after 2,433 documents had accumulated outside the optimized segment.

| Phase                       |    Wall time | Process-attributed writes |
| --------------------------- | -----------: | ------------------------: |
| Rewrite scalar rows         |       4.77 s |                  4.14 GiB |
| Copy and merge vector index |      18.12 s |                  5.97 GiB |
| Merge both FTS indexes      |     145.10 s |                  4.92 GiB |
| Commit new segment          |       4.26 s |                  0.03 GiB |
| **Total**                   | **172.50 s** |             **15.06 GiB** |

The process consumed 316.7 CPU-seconds. Temporary allocation peaked at 13.10 GiB.

The vector phase reused the existing HNSW file as its merge base, so it did not reconstruct the graph from raw embeddings. The FTS reducer still merged postings across the collection and dominated elapsed time.

## Problems to solve

1. Raw tool results and bash output duplicate canonical JSONL while dominating document and token counts.
2. Tiny tool-name documents pay one full scalar row, two FTS entries, and one zero vector each.
3. Large argument payloads duplicate source content even when only a path, URL, command, or other locator matters.
4. Appending one session reparses and retokenizes its complete history.
5. Updating one session rewrites the global 70 MiB index-state file.
6. Zvec optimization rewrites the full collection rather than work proportional to pending documents.

## Prototype question

The prototype on branch `prototype/recall-storage-layout` asks whether a simpler shape can remove routine compaction:

- dense-only flat Zvec for conversation, summaries, and turn context;
- no tool rows, zero vectors, FTS fields, or HNSW index;
- exact tool and output retrieval from canonical session JSONL;
- existing embeddings reused for the experiment.

The prototype compares storage, write volume, flat dense latency, top-result overlap, and source-search behavior.

## Prototype verdict

The dense-only candidate stored 340,736 real-vector documents in 2.02 GiB after filesystem writeback. Flat dense search measured 44.5 ms median and 48.2 ms p95. It returned the same top result as production HNSW for all five queries, with seven or eight shared results in each top eight.

The first compact SQLite FTS prototype stored 218,764 tool calls and direct bash commands in 118 MiB. It retained tool names, bounded locator arguments, commands, and source locators while excluding tool results, bash output, and large payload arguments. Query latency measured 0.093 ms median and 0.122 ms p95. Replacing the current session's 772 invocation rows wrote 2.07 MiB to the NVMe device.

A follow-up on 2026-08-10 compared SQLite FTS5 with a second, vectorless Zvec FTS collection. Both stores received the same 219,734 Invocation records extracted once from 3,725 session files.

| Invocation measurement             | SQLite FTS5 | Zvec before optimize |      Zvec after optimize |
| ---------------------------------- | ----------: | -------------------: | -----------------------: |
| Allocated storage                  |     124 MiB |              270 MiB |                  197 MiB |
| Build time                         |      1.95 s |              20.38 s |                        — |
| Build device writes                |    66.9 MiB |            390.3 MiB |                        — |
| Replace 886 records: elapsed       |     0.096 s |              0.285 s |                        — |
| Replace 886 records: device writes |    2.48 MiB |             3.91 MiB |                        — |
| Search median                      |    0.343 ms |             0.177 ms |                 0.130 ms |
| Search p95                         |     3.18 ms |              4.90 ms |                  1.57 ms |
| One Zvec optimization              |           — |                    — | 1.55 s, 104.9 MiB writes |

The search-result differences were ranking and tokenization differences rather than demonstrated locator loss. Most exact probes returned the same set. SQLite's `unicode61` tokenizer interpreted `brain_query` broadly enough to include `brain-query` paths, while Zvec returned the actual `brain_query` calls plus literal occurrences. Zvec optimization preserved the Zvec result sets while changing tied order.

A zero-write scan of 2.95 GB of canonical JSONL found exact result evidence for all five probes in 21.7 seconds. That path is suitable for rare full-result recovery, not routine invocation search.

### Unified SQLite plus sqlite-vec follow-up

The final follow-up tested whether SQLite could own the whole derived recall projection rather than only Invocation search and state. It copied the certified v7 candidate into SQLite 3.53 with sqlite-vec 0.1.9:

- 3,719 physical-session states;
- 218,139 compact Invocations with FTS5;
- 341,036 dense documents with all 46 metadata fields;
- 1,024-dimensional FP32 cosine vectors.

The first unpartitioned candidate occupied 2.52 GiB. Warm global dense p95 was 389 ms versus 76 ms for flat Zvec; project p95 was 178 ms versus 22 ms. It preserved the top result for every query, with global top-eight overlap of seven or eight and project overlap of five to eight. Invocation FTS p95 remained below 3 ms. Best-effort cold queries took about 1.36 seconds globally and 1.20 seconds for the metadata-filtered project path.

Project partitioning changed the tradeoff. Exact per-project partitions reduced project p95 to 28 ms but raised global p95 to 650 ms. Sixteen coarse project buckets produced project p95 around 34 ms and global p95 around 596 ms. The recommended shape therefore stores two vector copies in the same SQLite database and transaction: use an unpartitioned table for explicit global search and a 16-bucket table plus exact project metadata for the default project scope.

| Final comparison                    | Unified SQLite |  Draft flat Zvec + SQLite |
| ----------------------------------- | -------------: | ------------------------: |
| Projected allocation                |       3.87 GiB |                  2.28 GiB |
| Warm global dense p95               |         388 ms |                     69 ms |
| Warm project dense p95              |          34 ms |                     21 ms |
| Best-effort cold global             |         1.63 s |                    1.37 s |
| Best-effort cold project            |         1.75 s |                    1.16 s |
| Representative update device writes |       2.61 MiB |                  5.73 MiB |
| 100 replacement cycles              | 3.06 MiB/cycle |                         — |
| Growth after 100 cycles             |        0 bytes |                         — |
| Cross-store commit gap              |           none | idempotent retry required |

The full candidate passed the key transaction test. One transaction deleted session state, Invocation rows and FTS entries, dense metadata, and both vector copies. A concurrent reader still saw the prior complete session; explicit rollback and a SIGKILL both restored the exact prior hash. SQLite reported `integrity_check=ok` and no foreign-key violations. A hundred committed replacements produced no file or free-page growth and no post-churn latency regression.

The unified candidate should supersede the draft flat-Zvec-plus-SQLite layout. Its common project-scoped path is only about 13 ms slower at p95, while one database transaction removes the recovery protocol between stores. Explicit global search is about 319 ms slower but remains near the original 500 ms warm gate. The 1.59 GiB storage premium remains below the 5 GiB prototype limit and far below the legacy database.

Production conditions are to pin pre-1.0 sqlite-vec 0.1.9, keep both vector tables in one transaction, route searches by scope, preserve staged activation and canonical JSONL, and verify the published macOS x64 and arm64 packages before release. The resulting stable direction is one derived SQLite database, canonical source fallback, no HNSW, no zero-vector tool rows, and no routine whole-database optimization.

The sanitized prototype measurements live on branch `prototype/recall-storage-layout` under `prototypes/recall-storage-layout/results.json`.

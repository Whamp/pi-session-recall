# Recall storage and write-cost evidence

## Purpose

This report records why `pi-session-recall` must settle its storage layout before changing optimization policy. It separates incremental indexing from full Zvec compaction and attributes their computation and writes to concrete operations.

Measurements came from the production collection on 2026-08-09. The hourly index timer was paused during controlled runs. No measurement rebuilt or destroyed production data.

## Current collection

The collection held about 1.45 million documents. Only about 23.5% were dense-searchable conversation, summary, or turn-context documents. Tool names, arguments, results, direct bash commands, and bash output made up the remaining 76.5%.

Zvec 0.6 requires every row to carry its vector field. Lexical-only tool rows therefore store a 1,024-float zero vector even though dense queries filter them out. Every tool row also carries scalar provenance and participates in two FTS indexes.

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

A compact SQLite FTS index stored 218,764 tool calls and direct bash commands in 118 MiB. It retained tool names, bounded locator arguments, commands, and source locators while excluding tool results, bash output, and large payload arguments. Query latency measured 0.093 ms median and 0.122 ms p95. Replacing the current session's 772 invocation rows wrote 2.07 MiB to the NVMe device.

A zero-write scan of 2.95 GB of canonical JSONL found exact result evidence for all five probes in 21.7 seconds. That path is suitable for rare full-result recovery, not routine invocation search.

The evidence supports this production direction:

- flat dense Zvec for conversation, summaries, and turn context;
- SQLite FTS for compact invocation evidence and per-session index state;
- canonical JSONL for complete tool results, bash output, and omitted argument payloads;
- no HNSW, Zvec FTS, zero-vector tool rows, or scheduled optimization.

The sanitized prototype measurements live on branch `prototype/recall-storage-layout` under `prototypes/recall-storage-layout/results.json`.

# Recall storage layout prototype

**Throwaway prototype. Do not ship this code.**

## Question

Can a dense-only flat Zvec collection plus source-backed exact retrieval eliminate routine compaction while preserving acceptable recall latency?

## Candidate

The prototype:

- copies only dense-searchable conversation, summary, and turn-context documents;
- reuses production embeddings instead of calling the document embedding endpoint;
- stores no tool rows, zero vectors, FTS fields, or HNSW index;
- retains the current non-tool provenance fields;
- compares flat dense results with the production HNSW results;
- scans canonical session JSONL for exact tool/result evidence;
- builds a compact SQLite FTS index containing tool names, bounded locator arguments, and direct bash commands, but no tool results or bash output.

It never writes to the production collection. Scratch data lives at:

```text
~/.pi/agent/recall-debug/prototype-dense-only
```

The builder stops if scratch allocation exceeds 6 GiB or free space falls below 240 GiB.

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
npm run prototype:recall-storage-layout -- benchmark-source
```

Results are written to:

```text
~/.pi/agent/recall-debug/prototype-dense-only/prototype-report.json
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

The measured shape supports flat dense Zvec plus an incrementally updated SQLite invocation index. HNSW, Zvec FTS, zero-vector tool rows, and routine optimization do not earn their cost on this corpus. Full tool results and bash output remain source-backed and can use the slower raw scan when needed.

Sanitized measurements are in [`results.json`](results.json).

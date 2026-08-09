# Production recall index value benchmark

## Question

Does zvec optimization save enough query time to justify repeatedly processing the full recall collection? How does indexed recall compare with giving a fresh agent the same question and asking it to search raw Pi session JSONL?

## Production data

Measured on 2026-08-09 with:

- 1,443,367 searchable documents;
- 95.19% `embedding` index completeness;
- approximately 69,493 documents not merged into the configured vector index;
- 1,024 stored FP32 dimensions;
- zvec 0.6.0;
- HNSW query `ef` 300.

The benchmark made no production writes and did not run optimization. The [raw vector-search observations](production-recall-index-value-benchmark.json) retain every measured sample and candidate-overlap count.

## HNSW versus linear vector search

Zvec's `isLinear: true` query option forced brute-force vector search while preserving the same production vectors, filters, and query embeddings. This measures the principal query-time benefit of the HNSW index without rebuilding or changing the collection.

Five representative queries covered optimization failures, tool availability, ignored corrupt sessions, database storage, and maintenance scheduling. Each query ran in project and global scope. Each mode received one warmup per query and scope, followed by five measured repetitions: 25 observations per table row.

| Scope   | Mode   | Median wall time | p95 wall time | Median CPU time | p95 CPU time |
| ------- | ------ | ---------------: | ------------: | --------------: | -----------: |
| Project | HNSW   |         11.20 ms |      19.40 ms |        43.27 ms |     52.79 ms |
| Project | Linear |        262.73 ms |     273.29 ms |       288.86 ms |    308.17 ms |
| Global  | HNSW   |          8.32 ms |      10.40 ms |        40.22 ms |     59.47 ms |
| Global  | Linear |        379.92 ms |     388.37 ms |       409.28 ms |    417.04 ms |

Linear search added about 252 ms in project scope and 372 ms globally at the median. Every comparison returned the same first result. Eight of ten query-and-scope comparisons returned the same top-eight candidate set; the other overlaps were five of eight and seven of eight. Linear search is exact while HNSW is approximate, so candidate differences do not by themselves show a quality loss in linear mode.

A separate full hybrid project-scope measurement used the normal service path: query embedding, dense retrieval, ordinary full-text retrieval, identifier retrieval, fusion, and evidence loading. After one warmup, five identical requests had a 1,478.96 ms median and 1,759.36 ms p95. Adding the measured dense-channel difference suggests an all-linear hybrid request near 1.7–2.0 seconds on this host. That range is an estimate, not a direct all-linear hybrid measurement.

This forced-linear test is conservative for a collection that simply stops future optimization. Existing HNSW data remains usable while new documents stay in the linear buffer, so that mixed collection should remain faster than forcing every vector through linear search.

## Indexed recall versus a fresh raw-JSONL agent

Both paths received this query:

> Why have recent pi-session-recall optimization attempts failed?

The recall tool's full hybrid service took 1.48 seconds at the median. Five returned results did not contain the explanation. With the maximum ten results, the exact explanation appeared at rank ten.

A fresh GPT-5.6 Sol agent received only the query. It was forbidden from using PSR, the recall database, observational memory, or the answer written in the current conversation. It searched raw files under `~/.pi/agent/sessions/`, examined 54 project JSONL files, and correctly identified both the stale temporary-directory failure and the malformed full-text postings failure in 94.43 seconds. The workflow reported 141,682 tokens plus 852,480 cached tokens and a $1.29 cost.

For this query, indexed recall was about 64 times faster and far cheaper, but its default five-result ranking missed the answer. Raw search was slower and expensive but more reliable. The result supports an index that narrows source-backed evidence; it does not show that current ranking is sufficient.

## Limits

- One host, one collection, five vector queries, and one raw-agent question cannot establish universal capacity or quality.
- Warmed latency does not measure first-open latency or concurrent load.
- `isLinear: true` bypasses HNSW at query time but leaves the existing physical collection and full-text layout intact. It does not reproduce a database that has never compacted any full-text segment.
- The raw agent narrowed its search by project path rather than parsing every physical session file.
- Optimization may become valuable at larger collections, higher query concurrency, or stricter latency targets.

## Decision supported by this evidence

Index maintenance should not optimize by default. `psr index`, rebuilds, and the default automatic schedule should update evidence without optimization. `psr optimize` remains available explicitly, and automatic daily optimization remains an opt-in for users whose measured query latency or workload justifies its cost.

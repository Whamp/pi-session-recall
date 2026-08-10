---
status: accepted
---

# Keep index optimization explicit

`psr index` and `psr index --rebuild` update searchable evidence without calling zvec optimization. The retained `--no-optimize` flag is a compatibility spelling for update-only indexing; callers no longer need it. `psr optimize` is the sole manual command that requests collection optimization.

`psr auto-index install` installs only the update schedule by default. Reinstalling this default also disables and removes an optimization schedule created by an older release. `psr auto-index install --optimize-daily` explicitly adds the separate 23:00 local-time optimization schedule. Uninstall continues to remove both schedule types.

The [production index-value benchmark](../research/production-recall-index-value-benchmark.md) forced linear vector search across 1,443,367 documents. Median dense retrieval remained below 263 ms in project scope and 380 ms globally. HNSW saved about 252–372 ms per query but required collection-wide optimization that wrote near-collection-sized temporary output. On this workload, optimization cost and failure risk exceeded the measured interactive benefit.

Optimization remains available for larger collections, higher query concurrency, or stricter latency requirements. It also merges FTS segments. That merge can change BM25 scores and ranking without changing the indexed evidence. Users should enable optimization from measured latency, ranking, or resource evidence, not because indexing changed data. Future incremental optimization support in zvec could justify another decision.

A production repair and optimization trial later found one malformed FTS segment and recovered it through Zvec's supported FTS index rebuild. The repaired 1,446,592-document collection optimized successfully twice. The first run changed sampled lexical rankings; the second preserved the sampled results, order, and scores exactly. [The investigation](../research/zvec-fts-segment-repair.md) records the evidence and unresolved cause.

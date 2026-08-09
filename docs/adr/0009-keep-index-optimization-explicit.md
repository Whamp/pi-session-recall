---
status: accepted
---

# Keep index optimization explicit

`psr index` and `psr index --rebuild` update searchable evidence without calling zvec optimization. The retained `--no-optimize` flag is a compatibility spelling for update-only indexing; callers no longer need it. `psr optimize` is the sole manual command that requests collection optimization.

`psr auto-index install` installs only the update schedule by default. Reinstalling this default also disables and removes an optimization schedule created by an older release. `psr auto-index install --optimize-daily` explicitly adds the separate 23:00 local-time optimization schedule. Uninstall continues to remove both schedule types.

The [production index-value benchmark](../research/production-recall-index-value-benchmark.md) forced linear vector search across 1,443,367 documents. Median dense retrieval remained below 263 ms in project scope and 380 ms globally. HNSW saved about 252–372 ms per query but required collection-wide optimization that wrote near-collection-sized temporary output. On this workload, optimization cost and failure risk exceeded the measured interactive benefit.

Optimization remains available for larger collections, higher query concurrency, or stricter latency requirements. Users should enable it from measured query latency or resource evidence, not because indexing changed data. Future incremental optimization support in zvec could justify another decision.

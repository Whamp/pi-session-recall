---
status: accepted
---

# Ingest recall evidence incrementally outside Pi

Pi processes publish tiny durable lifecycle markers, while short-lived external workers derive and commit newly eligible append deltas. Recall evidence becomes immutable after crossing the recall horizon, mutable physical- and logical-session state lives in a separate projection, and search stays read-only against the last durable generation. Full repair, confirmed deletion reconciliation, migration, optimization, and rebuild remain explicit maintenance.

This separates expensive parsing, tokenization, embedding, and zvec writes from Pi's interactive path. It also accommodates zvec's exclusive process-level ownership without a permanent daemon: workers prepare before bounded write windows, markers make replay at least once, and searches never wait for ingestion completion. Materially stale or failed eligible work is visible as scalar backlog state instead of becoming a freshness barrier.

We rejected whole-session lifecycle reconciliation, search-triggered ingestion, filesystem watchers, process-liveness leases, permanent workers, dual-generation incremental writes, transactional staging around per-document zvec writes, and automatic optimization. These alternatives either repeated expensive work, coupled recall availability to maintenance, or introduced coordination machinery without measured benefit.

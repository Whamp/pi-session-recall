---
status: accepted
---

# Ingest recall evidence incrementally outside Pi

## Decision

Pi publishes small immutable lifecycle markers. A short-lived external worker reads bounded append deltas, applies the recall-eligibility policy, prepares embeddings before locking zvec, and commits immutable evidence plus mutable session projections in batches of at most 32 documents.

Search opens only the active durable generation. It never starts ingestion and never waits for the worker queue. It may wait up to 500 ms for the current write window. Interrupted writer state requires a write-capable recovery open before read-only search resumes.

Recall eligibility grows after compaction, branch exit, clean departure, or session quiescence. Context-exit summaries are eligible immediately. Confirmed source deletion is the only retraction path. One missing observation never deletes evidence, and suspicious mass loss stops deletion reconciliation.

## Storage and lifecycle

The recall data root contains generation-independent coordination state and self-contained generations:

```text
recall/
├── active-generation.json
├── generation-registry.json
├── backlog-summary.json
├── operation.lock
├── incremental-worker.lock
├── incremental-diagnostics.jsonl
├── markers/
│   ├── pending/
│   ├── quarantine/
│   └── control/
├── embedding-cache/
├── tokenizers/
└── generations/<generation-id>/
    ├── zvec/
    ├── session-projections/
    ├── index-state.json
    └── index-manifest.json
```

A marker carries one runtime sequence and physical-session identity. The worker orders and coalesces activity while preserving every branch exit. It performs metadata-only recovery sweeps, validates the append cursor and its 4 KiB boundary fingerprint, and reads only appended bytes during normal processing. Projection payloads contain scalar IDs, links, boundaries, and spans; payloads above 1 MiB require explicit reconciliation.

The worker exits when no eligible work remains. A detached coalesced successor waits when marker publication races with an owned worker interval, so work published after the running worker's snapshot cannot be stranded. Bounded metadata sweep continuations schedule their next slice without requiring another lifecycle event. The worker is not a daemon, watcher, process lease, or source of global leaf authority. Concurrent runtimes contribute a monotonic union of observed context exits.

## Rebuild, recovery, and rollback

A rebuild freezes incremental commits but not marker publication or search. It captures approved physical sources, logical sessions, and eligible contributors, then fails closed unless the replacement reproduces every approved source and contributor from the live corpus. It builds and optimizes a replacement generation beside the active generation. Validated readiness is published only inside the short lock that protects the checksummed pointer replacement, preventing recovery from retiring a live replacement between readiness and cutover. The ordinary worker then replays retained generation-independent markers against the replacement. A quarantined marker blocks replay completion until resolved. The former active generation remains as bounded rollback material.

There are no dual-generation incremental writes. Failed or cancelled replacement builds wake marker processing after the old generation becomes writable again; if the registry cannot be cleared immediately, the scalar `rebuild_failed` backlog state lets ordinary recovery finish that transition durably. Post-swap and backlog failures also wake ordinary recovery or replay. `/pi-session-recall-index --rollback` explicitly restores the retained generation, republishes retained marker work, and wakes replay after releasing the write window, including partial marker restoration or later scalar backlog publication failure. `/pi-session-recall-index --collect-retired` removes only generations that are no longer active, building, replay-pending, or retained for rollback. Exact version-5 legacy layout adoption is explicit through `--adopt-legacy`; adoption serves the old generation read-only until a replacement build succeeds.

A stale write-window marker is not safe for read-only zvec open. Recovery uses a write-capable open, deterministic replay, projection checkpoint commit, close, and marker acknowledgement. File shrinkage, cursor loss, boundary mismatch, unsupported layout, and projection overflow enter explicit reconciliation instead of textual repair.

## Measured host policy

The following values are target-host candidates, not universal zvec guarantees:

- marker publication plus detached spawn: p95 at most 25 ms;
- metadata sweep: at most 10,000 files or 500 ms per resumable slice;
- projection payload: at most 1 MiB;
- evidence batch: at most 32 documents;
- write window: p95 target at most 300 ms;
- search wait for the current window: at most 500 ms;
- immediate eligibility quiet period: 60 seconds;
- transfer above 32 prepared documents: wait for 5 minutes without growth;
- quiescence-only crash recovery: 30 minutes without growth.

Diagnostics version 3 records marker age, sweep work, bounded append and parse counts, eligible documents, tokenizer and embedding-cache work, embedding requests, write-window phases, generation and recovery state, deletion safeguards, and scalar backlog state. A missed target returns to design review with raw records; operators do not tune these values silently.

## Consequences

Parsing, tokenization, embedding, and zvec writes stay outside Pi's interactive hooks. Active tails remain in model context until they cross the recall horizon. Search can remain available while ingestion or a rebuild is stale, and scalar backlog warnings expose failed or overdue eligible work without conversation text or source paths.

Full repair, confirmed deletion reconciliation, migration, optimization, rebuild, rollback, and retired-generation cleanup remain explicit maintenance. Production rollout follows [`docs/operations/incremental-recall-rollout.md`](../operations/incremental-recall-rollout.md) and requires human approval.

We rejected whole-session lifecycle reconciliation, search-triggered ingestion, filesystem watchers, process-liveness leases, permanent workers, dual-generation writes, transactional staging around per-document zvec writes, heuristic text repair, and automatic optimization.

Prototype commit `870d148a7cebc71f3371fd965717aeefac818a3c`, `TRANSITIONS.md`, and `GENERATION-CUTOVER.md` remain design evidence. Production tests exercise public seams; the prototype harness is not part of production.

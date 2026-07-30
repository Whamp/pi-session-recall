---
status: accepted
---

# Ingest recall evidence incrementally outside Pi

## Decision

Pi publishes small immutable recall work markers. A short-lived external worker reads bounded append deltas, applies recall eligibility, prepares embeddings before locking the active generation, and commits immutable evidence plus mutable session projections in batches of at most 32 documents.

Search opens only the active durable generation. It never starts ingestion and never waits for worker backlog. It may wait up to 500 ms for the current write or recovery window. An interrupted write requires verified write-capable recovery before read-only search resumes.

Recall freshness is best-effort. Foreground agent and machine performance take priority, so ingestion has no completion deadline or maximum token lag. A large backlog means that search is stale, not that the active generation is invalid.

Recall eligibility grows after compaction, branch exit, clean departure, or session quiescence. Context-exit summaries are eligible immediately. Confirmed source deletion is the only retraction path. One missing observation never deletes evidence, and suspicious mass loss stops deletion reconciliation.

## Storage and lifecycle

The recall data root contains generation-independent coordination state and self-contained generations:

```text
recall/
├── active-generation.json
├── generation-registry.json
├── backlog-summary.json
├── background-index-status.json
├── operation.lock
├── operation.lock.rebuild-owner
├── incremental-worker.lock
├── incremental-diagnostics.jsonl
├── markers/
│   ├── pending/
│   ├── quarantine/
│   └── control/
├── tokenizers/
└── generations/<generation-id>/
    ├── lexical-source/
    ├── dense/
    ├── session-projections/
    ├── write-recovery.json
    ├── replay-snapshots/
    ├── validation-receipt.json
    └── index-manifest.json
```

The three generation stores are independent. Lexical/source evidence is the authoritative catalog, dense evidence is its exact embedded subset, and session projections are the sole mutable per-source ingestion account. The generation has no persistent embedding cache and no live reference to another generation.

A marker carries one runtime sequence and physical-session identity. The worker orders and coalesces activity while preserving every branch exit. It performs metadata-only recovery sweeps, validates the append cursor and its 4 KiB boundary fingerprint, and reads only appended bytes during normal processing. Projection payloads contain scalar IDs, links, boundaries, and spans; payloads above 8 MiB require explicit reconciliation.

The worker exits when no eligible work remains. A detached coalesced successor waits when marker publication races with an owned worker interval, so work published after the running worker's snapshot cannot be stranded. Bounded metadata sweep continuations and first missing-source observations schedule follow-up work without another lifecycle event. The worker is not a daemon, watcher, process lease, or source of global leaf authority. Concurrent runtimes contribute a monotonic union of observed context exits.

Every cross-store write follows ADR 0007: prepare before locking; persist recovery-required state; add lexical/source rows before dense rows; delete dense rows before lexical/source rows; save logical projections and the physical session projection last; close, reopen, and verify; then acknowledge covered markers. Search may use each completed coherent batch while later batches remain pending.

## Rebuild, recovery, and rollback

A rebuild freezes incremental commits but not marker publication or search. It captures one starting snapshot of approved physical sources, logical sessions, source boundaries, and eligible contributors. During its existing source pass, it records expected exact output membership. It builds and optimizes an independent replacement beside the active generation, validates reopened stores without a second import or tokenization pass, and writes an immutable validation receipt. A failed or cancelled replacement never changes the active pointer.

Validated activation writes one immutable generation replay snapshot of pending and quarantined marker IDs inside the short cutover window. The replacement becomes searchable immediately after pointer cutover, while the ordinary worker replays only that snapshot. Newer markers remain ordinary best-effort backlog and do not delay replay completion. The former active generation remains as bounded rollback material. There are no dual-generation incremental writes.

A generation-specific recovery-required record makes an interrupted batch unsafe for read-only open. Recovery re-derives the recorded batch, fetches and verifies existing IDs and checksums, inserts missing rows, repairs only isolated damage, verifies the physical session projection, closes and reopens stores, and clears recovery state before acknowledging markers. Damage that cannot be isolated requires explicit rollback or rebuild rather than textual repair.

The standalone `pi-session-recall` CLI owns rebuild control, recovery, rollback, exact legacy adoption, and retired-generation cleanup. Rollback performs the quick integrity check in ADR 0007 and switches to the retained generation without scanning session files or rebuilding vectors. Cleanup removes only generations that are no longer active, building, replay-pending, or retained for rollback.

## Measured host policy

The following values protect foreground work on the target host; they are not universal zvec guarantees or recall freshness objectives:

- marker publication plus detached spawn: p95 at most 25 ms;
- metadata sweep: at most 10,000 files or 500 ms per resumable slice;
- projection payload: at most 8 MiB;
- evidence batch: at most 32 documents;
- write window: p95 target at most 300 ms; and
- search wait for the current window: at most 500 ms.

Diagnostics record marker age, sweep work, bounded append and parse counts, eligible documents, tokenizer work, vector transfer and build-local reuse, embedding requests, write-window phases, generation and recovery state, deletion safeguards, and scalar backlog state. Missing a foreground-work bound returns to design review with raw records. Operators do not silently convert those bounds into freshness deadlines.

## Consequences

Parsing, tokenization, embedding, and store writes stay outside Pi's interactive work. Active tails remain in model context until they cross the recall horizon. Search remains available against coherent durable evidence while ingestion or a rebuild is stale. The CLI may report scalar backlog and failures; the Pi TUI receives no maintenance progress or status messages.

Full repair, confirmed deletion reconciliation, migration, optimization, rebuild, rollback, and retired-generation cleanup remain explicit maintenance. Production rollout follows [`docs/operations/incremental-recall-rollout.md`](../operations/incremental-recall-rollout.md) and requires human approval.

We rejected whole-session lifecycle reconciliation, search-triggered ingestion, filesystem watchers, process-liveness leases, permanent workers, dual-generation writes, a persistent embedding cache, transactional staging around per-document zvec writes, heuristic text repair, automatic optimization, and a freshness service objective.

Prototype commit `870d148a7cebc71f3371fd965717aeefac818a3c`, `TRANSITIONS.md`, and `GENERATION-CUTOVER.md` remain design evidence. Production tests exercise public seams; prototype harnesses are not part of production.

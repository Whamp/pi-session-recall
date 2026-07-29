---
status: accepted
---

# Run replacement generation builds in detached workers

A first or replacement generation build runs in one detached Node child process. The invoking process writes a versioned request, starts the child with ignored standard streams, records its PID, and releases its handle. The child reconstructs the configured inference runtime and acquires the crash-released rebuild ownership lock. This design adds neither a resident daemon nor a general job scheduler.

One atomically replaced `background-index-status.json` record reports the build ID, generation and embedding-profile identity, PID, process state, scan progress, latest durable physical-session checkpoint, and latest actionable error. Paths and errors are bounded. A status read checks whether an active PID still exists. If the process died without writing a terminal state, status becomes `crashed`; the registry-owned building generation becomes resumable once no process owns the rebuild lock.

`/pi-session-recall-index --rebuild` starts a new replacement and returns after launch. `--status`, `--stop`, `--resume`, and `--discard` inspect or control it. Stop sends `SIGTERM`. The worker aborts through the public rebuild operation, closes stores, marks the registry entry failed, and releases its locks. Resume starts a new detached process for the same generation ID and embedding profile. It reuses the generation index state, idempotent zvec upserts, and the profile-bound embedding cache.

Search continues to open only the active generation. Replacement parsing, embedding, zvec writes, and optimization happen outside that store. The operation lock protects only bounded freeze, cutover, and recovery windows. Parsing, embedding, store-write, optimization, and cutover failures therefore leave the old active generation searchable.

A child cannot inherit injected JavaScript functions. Built-in configuration is reconstructable by default. A caller that injects an embedding profile, provider, tokenizer, store, or project resolver must supply a named service-factory module. Background launch fails before spawning if the child cannot reconstruct the same semantics.

Embedding replacement setup validates and persists a pending selection before worker launch. The named factory reconstructs that pending profile for the replacement, while ordinary configured runtimes keep using the active profile. The generation registry finalizes the pending selection only after the matching generation activates.

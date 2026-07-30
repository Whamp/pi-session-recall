---
status: accepted
---

# Run replacement generation builds in detached workers

A first or replacement generation build runs in one detached Node child process. The invoking process writes a versioned request, starts the child with ignored standard streams, records its PID, and releases its handle. The child reconstructs the configured inference runtime and acquires the crash-released rebuild ownership lock. This design adds neither a resident daemon nor a general job scheduler.

One atomically replaced `background-index-status.json` record reports the build ID, generation and embedding-profile identity, PID, process state, scan progress, latest durable physical session projection, and latest actionable error. Paths and errors are bounded. A status read checks whether an active PID still exists. If the process died without writing a terminal state, status becomes `crashed`; the registry-owned building generation becomes resumable once no process owns the rebuild lock. This record is diagnostic, not generation or progress authority.

The standalone `pi-session-recall` CLI starts, inspects, stops, resumes, and discards replacement work. Starting a rebuild returns after launch. Stop sends `SIGTERM`; the worker aborts through the public rebuild operation, closes stores, marks the registry entry failed, and releases its locks. Resume starts a new detached process for the same generation ID and embedding profile. It reuses exact rows verified in that generation and may transfer compatible vectors from another validated generation. It never depends on a persistent embedding cache.

Build status and progress belong to the CLI. The Pi extension registers no maintenance slash command and sends no build status or progress messages to the TUI.

Search continues to open only the active generation. Replacement parsing, embedding, store writes, optimization, and validation happen against independent replacement stores. The operation lock protects only bounded freeze, cutover, and recovery windows. Parsing, embedding, store-write, optimization, validation, and pre-cutover failures therefore leave the old active generation searchable.

A child cannot inherit injected JavaScript functions. Built-in configuration is reconstructable by default. A caller that injects an embedding profile, provider, tokenizer, store, or project resolver must supply a named service-factory module. Background launch fails before spawning if the child cannot reconstruct the same semantics.

Embedding replacement setup validates and persists a pending selection before worker launch. The named factory reconstructs that pending profile for the replacement, while ordinary configured runtimes keep using the active profile. The generation registry finalizes the pending selection only after the matching generation activates.

---
status: accepted
---

# Run staging index builds in detached workers

A full first or replacement build runs in one detached Node child process. The invoking Pi process writes a versioned worker request, starts the child with ignored standard streams, records its PID, and releases its handle. The child reconstructs the configured conversation service, owns the staging generation and writer locks, and removes its request after reaching a terminal state. No resident daemon or general job scheduler is introduced.

One atomically replaced `background-index-status.json` record reports the build ID, generation and embedding-profile identity, PID, process state, current scan progress, latest durable physical-session checkpoint, and latest actionable error. Paths and errors are length-bounded. Status reads check whether an active PID still exists. A dead PID without a terminal record becomes `crashed`, and its selected staging generation becomes resumable.

`/pi-session-recall-index --rebuild` starts new staging work and returns after launch. `--status`, `--stop`, `--resume`, and `--discard` inspect or control that work. Stop sends `SIGTERM`; the worker aborts the public rebuild operation, which preserves staging state and releases locks. Resume starts a new detached process for the same profile-bound staging generation. It reuses per-session state and the embedding cache. Repeated zvec upserts remain idempotent when a process dies after a store write but before its state checkpoint.

Search still resolves only the active index generation. A background writer cannot acquire or remove the active generation's lock. Parsing, embedding, store-write, optimization, and pre-activation interruption therefore leave active recall unchanged and staging recoverable.

A child process cannot inherit injected JavaScript functions. The default worker can reconstruct the built-in configuration. A caller that injects an embedding profile, provider, tokenizer, store, or project resolver must provide a named service-factory module for the worker. Starting background work fails before launch when those dependencies cannot be reconstructed; the service never switches profiles or adapters silently.

Embedding replacement setup validates and persists a pending embedding selection before starting this worker. The named factory reconstructs that pending selection for staging, while ordinary configured runtimes retain the active selection until the matching generation activates.

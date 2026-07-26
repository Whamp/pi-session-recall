---
status: accepted
---

# Reconcile recall from Pi's session lifecycle

Conversation Recall treats zvec as a live derived projection of Pi's JSONL session store. The initial generation remains explicit and quality-gated, but ordinary freshness no longer depends on a person running the index command.

The extension starts a bounded full catch-up on interactive TUI `session_start`. Persistent TUI and RPC runtimes reconcile the active session on `agent_settled` and await a final active-session reconciliation on `session_shutdown`; RPC skips startup corpus catch-up. Print and JSON runtimes skip automatic lifecycle ingestion so short-lived helper processes cannot monopolize the global writer lock or index disposable sessions. A recall tool call in any mode still applies an active-session freshness barrier before searching. That barrier uses the trusted path from `ctx.sessionManager`; the model cannot choose a session file.

Targeted reconciliation reprocesses one complete session instead of appending only its latest line. Resume, branch, compaction, and fork activity can change provenance attached to earlier documents. Stable document IDs, checksums, and the embedding cache keep that full-session reconciliation incremental in storage and model work.

Automatic maintenance uses the existing process-local serializer and PID-owned writer lock. Background hooks wait at most 250 milliseconds for another writer and then defer to the next lifecycle event. Session shutdown cancels an unfinished corpus catch-up before queuing the final active-session reconciliation, so quit, reload, resume, and fork do not wait for unrelated sessions. A tool's freshness barrier waits under the tool cancellation signal rather than silently searching stale active-session evidence. The manual command remains the repair, optimization, and incompatible-generation rebuild path.

The index excludes `pi-session-recall` calls and their linked results. Those records contain the search query and derived search output; indexing them would feed recalled evidence back into later retrieval. Other tool calls and results remain lexical evidence.

We rejected per-message indexing, append-only session updates, filesystem watchers, and full-corpus scans before every search. Per-message work runs before a session settles, append-only updates preserve stale branch metadata, watchers duplicate Pi's authoritative lifecycle, and query-time full scans make interactive latency depend on corpus size.

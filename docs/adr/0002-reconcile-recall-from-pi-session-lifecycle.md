---
status: superseded
---

# Reconcile recall from Pi's session lifecycle

This decision was superseded after diagnostics showed that a changed active session was reparsed, retokenized, and rewritten in Pi's process after every settled turn. See ADR 0003.

Conversation Recall treats zvec as a live derived projection of Pi's JSONL session store. The initial generation remains explicit and quality-gated, but ordinary freshness no longer depends on a person running the index command.

No Pi runtime scans the full corpus during `session_start`. Persistent TUI and RPC runtimes reconcile the active session on `agent_settled`. Shutdown and reload do not start recall maintenance because complete-session preparation and zvec writes run inside Pi's process and can delay lifecycle completion for large conversations. Print and JSON runtimes skip automatic lifecycle ingestion so short-lived helper processes cannot monopolize the global writer lock or index disposable sessions. A recall tool call in any mode still applies an active-session freshness barrier before searching. That barrier uses the trusted path from `ctx.sessionManager`; the model cannot choose a session file. The manual index command catches up inactive sessions.

Targeted reconciliation reprocesses one complete session instead of appending only its latest line. Resume, branch, compaction, and fork activity can change provenance attached to earlier documents. Stable document IDs, checksums, and the embedding cache keep that full-session reconciliation incremental in storage and model work.

Automatic maintenance uses the existing process-local serializer and PID-owned writer lock. Settled-turn hooks wait at most 250 milliseconds for another writer and then defer to the next lifecycle event. Changes not captured before shutdown are reconciled after a later settled turn, before a recall search, or by the manual index command. A tool's freshness barrier waits under the tool cancellation signal rather than silently searching stale active-session evidence. The manual command remains the full catch-up, repair, optimization, and incompatible-generation rebuild path.

The index excludes `pi-session-recall` calls and their linked results. Those records contain the search query and derived search output; indexing them would feed recalled evidence back into later retrieval. Other tool calls and results remain lexical evidence.

We rejected per-message indexing, append-only session updates, filesystem watchers, and full-corpus scans during startup or before search. Per-message work runs before a session settles, append-only updates preserve stale branch metadata, and watchers duplicate Pi's authoritative lifecycle. Startup and query-time full scans make interactive responsiveness depend on corpus size; Promise scheduling does not move CPU-heavy zvec work off Pi's event loop.

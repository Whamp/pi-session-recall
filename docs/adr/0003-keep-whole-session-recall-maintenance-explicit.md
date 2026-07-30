---
status: accepted
---

# Keep whole-session recall maintenance explicit

Interactive Pi operations must not perform whole-session recall maintenance. Startup, settled turns, shutdown, reload, recall search, and source-neighborhood expansion read the existing active generation without reconciling the active session.

The Pi extension serves model-facing recall reads and publishes small immutable recall work markers. It registers no index-maintenance slash command and sends no maintenance status or progress messages to the TUI. A standalone `pi-session-recall` CLI is the sole operator control surface for setup, status, explicit catch-up, rebuild control, recovery, rollback, and cleanup. It creates managed generations by rebuilding immutable Pi session sources; it does not adopt a legacy recall layout. The current slash command and exact legacy-adoption path are transitional implementation debt, not accepted interfaces; the CLI migration removes them rather than retaining wrappers. ADR 0004's short-lived external worker handles deferred incremental ingestion outside Pi.

Diagnostics on an 18.5 MB active session measured 84.5 seconds for one settled-turn reconciliation. Physical session preparation took 64.5 seconds and zvec writes took 7.3 seconds. Although the then-current embedding cache limited remote model work to 36 new embeddings, the indexer still reparsed and retokenized the complete session, resolved vectors for its dense documents, and upserted 9,099 documents. Even unchanged targeted updates parsed global index state before checking one session.

The active model already receives uncompacted conversation content from Pi's session context. Rebuilding that same content after every turn duplicates work while blocking the process that owns the terminal. Promise scheduling does not move synchronous parsing, tokenization, or store work off Pi's event loop.

Incremental ingestion therefore follows Pi's saved append and compaction boundaries through cheap markers and a separate worker. It processes bounded deltas, persists physical and logical session projections, and writes only added or confirmed-deleted evidence. Foreground agent and machine performance take priority over recall freshness; search neither starts ingestion nor waits for the backlog.

We rejected settled-turn reconciliation, search-time freshness, lifecycle timeouts, maintenance controls in Pi, and moving the same complete-session operation to another hook. Timeouts cannot interrupt long synchronous preparation reliably, changing the trigger does not remove duplicated work, and TUI progress adds noise without controlling detached maintenance.

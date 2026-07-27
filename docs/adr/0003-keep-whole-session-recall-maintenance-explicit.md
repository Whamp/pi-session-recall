---
status: accepted
---

# Keep whole-session recall maintenance explicit

Interactive Pi operations must not perform whole-session recall maintenance. Startup, settled turns, shutdown, reload, and recall search read the existing generation without reconciling the active session. `/pi-session-recall-index` is the only current path for changed, new, removed, or incompatible session evidence.

Diagnostics on an 18.5 MB active session measured 84.5 seconds for one settled-turn reconciliation. Physical session preparation took 64.5 seconds and zvec writes took 7.3 seconds. Although the embedding cache limited remote model work to 36 new embeddings, the indexer still reparsed and retokenized the complete session, resolved cached vectors for its dense documents, and upserted 9,099 documents. Even unchanged targeted updates parse the global index state before checking one session.

The active model already receives uncompacted conversation content from Pi's session context. Rebuilding that same content after every turn duplicates work while blocking the process that owns the terminal. Promise scheduling does not move synchronous parsing, tokenization, or zvec work off Pi's event loop.

A future live-ingestion design should use Pi's saved compaction boundary to process only evidence that has left the active model context. It must persist append and compaction watermarks, avoid reparsing the global index state for one active session, and upsert only added, changed, or removed documents. That design requires separate quality and lifecycle validation; this decision does not approximate it with whole-session work.

We rejected retaining settled-turn reconciliation, retaining search-time freshness, adding timeouts, and moving the same complete-session operation to another lifecycle hook. Timeouts cannot interrupt long synchronous preparation reliably, and changing the trigger does not remove duplicated work.

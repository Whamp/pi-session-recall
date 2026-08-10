---
status: accepted
---

# Store incremental recall state in SQLite

One WAL-mode Recall catalog stores each Physical session file's size, modification time, document identities, and compact Invocation records. Replacing a changed session deletes and inserts only that session's catalog rows in one transaction. Missing, ignored, or malformed indexed sessions are removed from the active recall store and the catalog together during explicit Index maintenance.

Invocation full-text search uses SQLite FTS5 and supports exact Project identity filtering or unrestricted global scope. Stored Invocation rows contain bounded projected locator arguments and Source locators. Complete tool results, bash output, and omitted payload values remain only in canonical session JSONL.

The first normal index after upgrade imports valid `index-state.json` session knowledge into a new catalog. Migrated sessions receive one Invocation backfill pass, after which unchanged sessions are skipped from file metadata without parsing. Normal indexing never rewrites the legacy corpus-wide JSON file. Rebuild candidates create a fresh catalog and must contain it before atomic activation.

Catalog schema or import-policy incompatibility fails with an explicit `psr index --rebuild` action instead of attempting an in-place schema migration.

---
status: accepted
---

# Store recall in one SQLite database

Manifest version 8 stores the complete derived Recall database in one WAL-mode `recall.sqlite` file. SQLite owns Physical session state, compact Invocation metadata and FTS5, Dense recall metadata, and one 16-bucket vec0 table. A changed Physical session updates every projection in one transaction. Readers see either the previous complete session or the next complete session.

The database pins sqlite-vec 0.1.9. It stores each normalized 1,024-dimension FP32 embedding once and uses cosine distance. Project scope queries the bucket selected by `project_key % 16` and filters the exact project key before applying the candidate limit. Global scope searches the same table across all buckets. The measured warm global p95 of about 0.6 seconds is acceptable; a second vector copy is not worth doubling vector storage to reduce it to about 0.4 seconds.

Canonical session JSONL remains authoritative. The Recall database contains bounded Invocation locators, not complete tool results, bash output, or omitted payload arguments. Explicit Source search reads those full payloads from JSONL and writes no persistent data.

Each completed generation contains `recall.sqlite`, the version 8 manifest, and maintenance status. Candidate construction and staged activation prevent a partial database from serving search. Obsolete database layouts are neither opened nor migrated. Rebuild the current database from canonical JSONL instead.

ADR-0011 and ADR-0012 are superseded because their split catalog and dense-store architecture could commit separately. ADR-0013's retirement of public and scheduled optimization remains accepted; only its version 7 storage wording is superseded.

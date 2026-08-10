---
status: accepted
---

# Store recall in one SQLite database

Manifest version 8 stores the complete derived Recall database in one WAL-mode `recall.sqlite` file. SQLite owns Physical session state, compact Invocation metadata and FTS5, Dense recall document metadata, one unpartitioned global vec0 table, and one 16-bucket project vec0 table. A changed Physical session updates every projection in one transaction. Readers see either the previous complete session or the next complete session.

The database pins sqlite-vec 0.1.9. It stores normalized 1,024-dimension FP32 embeddings and uses cosine distance. Global scope queries the unpartitioned vec0 table. Project scope queries the bucket selected by `project_key % 16` and filters the exact project key before applying the candidate limit. Callers choose scope; they do not choose a vector table.

Canonical session JSONL remains authoritative. The Recall database contains bounded Invocation locators, not complete tool results, bash output, or omitted payload arguments. Explicit Source search reads those full payloads from JSONL and writes no persistent data.

Each completed generation contains `recall.sqlite`, the version 8 manifest, and maintenance status. Staged version 7 flat-Zvec-plus-SQLite generations are incompatible and cannot activate. ADR-0011 and ADR-0012 are superseded because their split catalog and dense-store architecture can commit separately. ADR-0013's retirement of public and scheduled optimization remains accepted; only its version 7 storage wording is superseded.

The actual version 6 Zvec database and `index-state.json` remain available only through the clearly named legacy-v6 search, update, vector-reuse, and rollback adapter. This temporary adapter remains until Will explicitly approves ending the rollback window. No version 8 indexing or search path imports or calls Zvec.

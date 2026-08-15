---
status: accepted
---

# Cache tokenized conversation projection inputs

SQLite schema version 4 records each Physical session's exact imported byte length and SHA-256. When a file's size is unchanged but its modification time differs, `psr index` hashes the complete file. An equal digest refreshes only source metadata and skips parsing, graph validation, tokenization, embedding, and projection replacement. A different digest follows the normal full-import path.

Changed bytes still undergo complete format detection, strict graph validation, Invocation reconstruction, Project attribution, vector checksum checks, and one atomic `replacePhysicalSession` transaction. The importer may reuse token and character geometry only when a stable projection-input ID and exact input checksum match a cache row from the previous projection. It rebuilds current leaf, session name, parent/child, branch, active-context, compaction, tool, Project, and source metadata from the newly validated graph. A missing or mismatched cache row runs the current tokenizer and chunker.

The projection-input cache stores IDs, checksums, document order, and references to Dense document rows. It stores no conversation text beyond the existing Dense documents and no complete tool results, bash output, omitted arguments, thinking, images, or unknown records. Cache rows commit and delete with their owning Physical session.

Canonical JSONL remains authoritative. Schema 4 databases are rebuilt rather than migrated from schema 3. Cache state has no repair or upgrade path: incompatible databases rebuild from JSONL, and unusable per-input cache entries fall back to tokenization. The existing full importer remains the correctness oracle and fallback.

This decision deliberately keeps full graph validation and full Physical session replacement. The measured pre-change workload spent about 125–126 seconds in chunk tokenization, 24–26 seconds in graph validation, and 3.6–4.7 seconds in SQLite replacement. Reusing token geometry and replacing quadratic parent-cycle validation with a linear walk reduced an 18.8 KB representative tail update to 8.9 seconds, including 0.21 seconds of graph validation. The result passes the 18.7-second gate, so no graph checkpoint or row-level projection delta is added.

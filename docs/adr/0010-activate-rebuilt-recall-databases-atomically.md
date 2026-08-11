---
status: accepted
---

# Activate rebuilt recall databases atomically

`psr index --rebuild` builds a Candidate recall database beside the Active recall database. Normal indexing and search continue to resolve the active pointer. Rebuild work never deletes or writes the active database.

A normal rebuild activates its candidate only after indexing completes with no failed Physical session files, writes maintenance status, and closes the database. Activation renames the completed candidate within the generation directory and atomically replaces one relative symbolic link named `active`.

`psr index --rebuild --stage` stops after it renames the complete candidate to a durable generation. It prints the exact relative database target and leaves `active` unchanged. Staged construction uses a separate construction lock because it writes no Active recall database state. Normal search and scheduled Index maintenance can continue during the build. `psr activate <database-target>` takes the shared recall writer lock, accepts only an exact complete generation beneath the configured generation root, records the current target for rollback, and atomically replaces `active`.

The shared recall writer lock covers normal indexing, activation, and rollback. Search refuses to open a database while that lock exists. A separate lock serializes staged construction without making the Active recall database unavailable for the duration of a production rebuild.

Each activated generation records the target that was active immediately before it. `psr rollback` verifies that target and atomically replaces the active pointer. The system does not automatically delete completed generations, so the previous database remains restorable until the operator removes it. A missing or incomplete previous target fails without changing the active pointer.

The first generation rebuild treats a complete unversioned database in the recall data directory as the previous database. The unversioned files stay in place throughout the rebuild. This lets existing installations keep searching the old database during the build and roll back to it after activation.

Failed and interrupted candidates stay outside the active path. `psr index --rebuild --stage --resume` continues the sole interrupted candidate and preserves its completed per-session catalog transactions and dense rows. Resume refuses zero or multiple candidates rather than guessing. During the compact-layout cutover, `--reuse-active-vectors` can copy a vector from the Active recall database only when the embedding profile, canonical document ID, and document checksum match. This keeps reuse an optimization: changed or unmatched documents still use the embedding provider, and no unreferenced active rows are seeded into the candidate.

A fresh rebuild removes stale candidate directories after acquiring its construction lock. It does not remove completed generations because one can be active, previous, or waiting for certification.

A single pointer replacement keeps activation atomic without a mutable generation registry. Recording previous state inside the activated generation avoids a multi-file active/previous switch whose interruption could lose rollback state.

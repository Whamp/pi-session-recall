---
status: accepted
---

# Activate rebuilt recall databases atomically

`psr index --rebuild` builds a Candidate recall database beside the Active recall database. Normal indexing and search continue to resolve the active pointer. Rebuild work never deletes or writes the active database.

A rebuild activates its candidate only after indexing completes with no failed Physical session files, writes maintenance status, and closes the database. Activation renames the completed candidate within the generation directory and atomically replaces one relative symbolic link named `active`. The shared recall writer lock covers candidate creation, indexing, activation, rollback, and optimization. Search refuses to open a database while that lock exists.

Each activated generation records the target that was active immediately before it. `psr rollback` verifies that target and atomically replaces the active pointer. The system does not automatically delete completed generations, so the previous database remains restorable until the operator removes it. A missing or incomplete previous target fails without changing the active pointer.

The first generation rebuild treats a complete unversioned database in the recall data directory as the previous database. The unversioned files stay in place throughout the rebuild. This lets existing installations keep searching the old database during the build and roll back to it after activation.

Failed and interrupted candidates stay outside the active path. The next rebuild removes stale candidate directories after acquiring the writer lock. It does not remove completed generations because one can be the active or previous database.

A single pointer replacement keeps activation atomic without a mutable generation registry. Recording previous state inside the activated generation avoids a multi-file active/previous switch whose interruption could lose rollback state.

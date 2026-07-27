---
status: accepted
---

# Activate validated index generations atomically

A full first or replacement build writes to one staging index generation. Each managed generation owns its zvec store, incremental session state, manifest, and writer lock under `index-generations/<generation-id>/`. Searches resolve `active-generation.json` once at the start of an operation and open only that generation. A staging writer uses separate locks, so it does not block searches against the active index generation.

The service checkpoints session state after every changed physical session. A resumed build reuses that state, the staging zvec store, and the profile-bound embedding cache. After the main pass, the service scans the corpus again to catch files changed during the build. Any parse or indexing failure leaves the staging generation resumable and leaves active selection unchanged.

Before activation, the service optimizes zvec and validates the manifest, current embedding canary, zvec schema dimensions, session-state schema, document counts, every state-referenced document, every dense-vector count, and every dense vector's dimensions and finite values. It then atomically replaces `active-generation.json` and removes the staging selection. A crash before the selector replacement leaves the old active generation selected. A crash after it leaves the complete validated generation selected.

`staging-generation.json` identifies the one resumable staging generation and its embedding profile. A different profile cannot reuse that work. `discardStagingIndexGeneration()` is the only operation that removes abandoned staging work; it never removes the selected active generation. Completed older active generations remain on disk because automatic garbage collection is outside this decision.

Existing installations without an active selector remain readable through their legacy zvec, state, and manifest paths. Their next explicit rebuild creates and selects a managed generation instead of deleting the legacy generation. Incremental maintenance updates the currently selected active generation. Interactive startup, settled turns, shutdown, reload, and recall search do not start a full build.

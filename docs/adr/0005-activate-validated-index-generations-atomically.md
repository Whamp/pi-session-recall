---
status: accepted
---

# Activate validated recall generations atomically

A first or replacement build writes one self-contained recall generation under `generations/<generation-id>/`. Each generation owns its zvec evidence store, session projection store, index state, and manifest. Search reads the checksummed `active-generation.json` pointer once and opens only that generation.

`generation-registry.json` records the active, building, rollback, failed, replay-pending, and retired states. A building entry also records the configured embedding profile when known. The registry, rather than a second staging pointer, is the sole authority for replacement work. A crash-released rebuild ownership lock distinguishes a live detached worker from an abandoned `BUILDING` entry.

A replacement build checkpoints each physical session in its own index state. Stop, failure, or worker death leaves the generation directory and registry entry available for explicit resume. Resume reopens the same generation ID, zvec store, index state, manifest, and profile-bound embedding cache. A different embedding profile cannot reuse that generation.

The build reproduces every approved physical source, logical session, and eligible contributor from the active projections. Before cutover, it compares the complete staged evidence membership with the approved snapshot: contributor IDs, physical projection IDs, and source paths must match exactly. Counts and projection presence cannot prove membership. Missing, changed, or unexpected evidence fails the replacement. Lifecycle markers remain generation-independent while the build runs. After cutover, the ordinary incremental worker replays retained markers against the replacement; search does not wait for replay.

Before activation, the service optimizes zvec and validates the manifest, embedding canary, document count, projection count, and approved projection coverage. It then replaces the active pointer atomically inside the cutover write window and records the former active generation for bounded rollback. A failure before the pointer swap leaves the old generation selected. Recovery reconciles registry-first or pointer-first crashes without scanning for a “latest” directory.

Embedding replacement setup persists a pending selection before launching the detached build. The building registry entry carries its semantic embedding profile ID. Inference configuration promotes the pending selection only after that registry entry becomes active. Backend or adapter changes that preserve the profile do not rebuild vectors.

`discardStagingIndexGeneration()` removes only a failed or abandoned non-active registry entry after proving no live rebuild owns it. It acquires the coordinated write window, rereads the generation registry inside that window, and applies its filter to the current registry state. No registry read-modify-write may publish a snapshot read before its ownership window. `/pi-session-recall-index --rollback` restores the retained rollback generation and republishes retained markers. `/pi-session-recall-index --collect-retired` removes only generations outside active, building, replay-pending, and rollback retention.

Exact version-5 legacy adoption remains explicit and read-only. Its next rebuild creates a version-6 managed generation. Interactive startup, lifecycle hooks, and recall search never start a full build.

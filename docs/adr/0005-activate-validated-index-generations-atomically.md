---
status: accepted
---

# Activate validated recall generations atomically

A first or replacement build writes one self-contained recall generation under `generations/<generation-id>/`. Each generation owns independent lexical/source, dense, and session projection stores, one fixed manifest, and—after successful validation—one immutable validation receipt. Search reads the checksummed `active-generation.json` pointer once and opens only that generation.

`generation-registry.json` records active, building, rollback, failed, replay-pending, and retired roles. A building entry also records the configured embedding profile when known. The registry, rather than a staging pointer or status file, is the sole authority for generation roles. A crash-released rebuild ownership lock distinguishes a live detached worker from an abandoned building entry.

A replacement build checkpoints each physical session in the session projection store. Stop, failure, or worker death leaves the generation directory and registry entry available for explicit resume. Resume reopens the same generation ID, stores, manifest, and physical session projections. It first reuses rows verified inside that interrupted generation, then transfers compatible vectors from a previously validated generation. It uses no persistent embedding cache, and a different embedding profile cannot reuse the vectors.

The build reproduces one fixed starting snapshot of approved physical sources, logical sessions, source boundaries, and eligible contributors. During its existing source pass it records exact expected IDs, checksums, membership digests, counts, and projections. After closing the stores, validation reopens them and proves exact lexical/source membership, dense-subset membership, anchor and path integrity, projection coverage, manifest compatibility, profile data, and canaries. Validation never needs a second full source import or tokenization pass. A generation cannot become ready without an immutable success receipt for that validation.

Lifecycle markers remain generation-independent while the build runs. Before pointer cutover, the service captures the exact pending and quarantined marker IDs in an immutable generation replay snapshot for that activation. Markers published later are ordinary backlog.

Inside the short cutover window, the named generation transition rereads pointer and registry state, publishes the ready registry, atomically replaces the active pointer, then publishes the activated registry. The former active generation becomes the bounded rollback generation. A failure before pointer replacement leaves the old generation selected. Recovery reconciles registry-first or pointer-first crashes from durable state without scanning for a latest directory.

The replacement becomes searchable after cutover even while replay remains pending. The ordinary incremental worker processes the fixed generation replay snapshot and completes replay when those exact markers are covered by verified physical session projections and any quarantined markers in that snapshot are resolved. Search never waits for replay, and newly published markers do not extend activation replay indefinitely.

Embedding replacement setup persists a pending selection before launching the detached build. The building registry entry carries its semantic embedding profile ID. Inference configuration promotes the pending selection only after that registry entry becomes active. Backend or adapter changes that preserve the profile do not rebuild vectors.

Discard removes only a failed or abandoned non-active registry entry after proving no live rebuild owns it. The standalone `pi-session-recall` CLI owns discard, rollback, and retired-generation cleanup. Rollback applies ADR 0007's quick health check, records a fixed generation replay snapshot for rollback, restores the retained generation through the named registry-first transition, and resumes replay after the write window. Cleanup removes only generations outside active, building, replay-pending, and rollback retention.

The target architecture does not adopt version-5 or other legacy storage layouts. A first or replacement generation is rebuilt from immutable Pi session sources and must pass the same managed-generation validation. The current exact version-5 adoption path is transitional implementation debt removed at cutover. Interactive startup, lifecycle hooks, recall search, and source-neighborhood expansion never start a full build or report build progress in the TUI.

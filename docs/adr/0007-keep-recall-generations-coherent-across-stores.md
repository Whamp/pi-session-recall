---
status: accepted
---

# Keep recall generations coherent across independent stores

One recall generation owns independent lexical/source, dense, and session projection stores. There is no transaction across them. A generation is coherent when every visible evidence occurrence and source link is complete inside that generation. Coherence does not require freshness: search may serve verified completed batches while later source work remains pending.

## Authority and fixed identity

Pi session JSONL remains the source of truth. Recall generations are derived, disposable, and rebuildable.

The lexical/source store is authoritative for indexed evidence and source-neighborhood data. The dense store is only the exact embedded subset of that catalog. The session projection store is the sole mutable per-source ingestion account. In particular, each physical session projection owns its processed source position, logical-session membership, marker coverage, expected store counts and digests, and repair state. No manifest, registry entry, status file, or second index-state database duplicates that per-source progress.

The generation manifest is written once, before store content, and never changed. It identifies the generation, all store schemas and indexes, import and text-processing policies, provenance and project-identity policies, embedding profile and dimensions, validation canaries, and compatibility requirements. It contains no changing progress, backlog, or record counts.

A successful rebuild writes one immutable generation validation receipt after validation. The receipt binds the manifest fingerprint, the rebuild's starting source snapshot, exact membership digests and counts, validation policy, and canary results. A failed validation writes no success receipt. The generation registry owns roles and transition state; a background status record and backlog summary are derived diagnostics, not authorities.

## Incremental write and checkpoint order

Preparation happens before the exclusive recall write window. The worker reads source bytes, applies recall eligibility, tokenizes, derives stable IDs and checksums, prepares entry anchors and projections, and resolves vectors without blocking search.

The worker divides evidence changes into batches of at most 32 documents. Each write window follows this order:

1. Acquire the exclusive write window and persist a generation-specific recovery-required record before the first store mutation.
2. For additions, write lexical/source rows and anchors before their matching dense rows. For confirmed deletion, remove dense rows before lexical/source rows and anchors.
3. In the final window for a physical source, save its logical session projections, then save the physical session projection last.
4. Close every changed store, reopen it, and verify the exact IDs, checksums, profile data, cross-store membership, and final physical session projection covered by that window.
5. Durably remove the recovery-required record and release the write window.
6. Acknowledge only the exact recall work markers covered by the reopened physical session projection.

Search cannot observe a window halfway through. After the window closes, it may observe that completed coherent batch even when later batches or the physical session checkpoint remain pending. Stable IDs make replay safe.

| Interruption point                                                        | Durable meaning                                               | Required action                                                        |
| ------------------------------------------------------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Before the recovery-required record                                       | No store mutation began                                       | Retry preparation or the batch                                         |
| After the record, before or during mutation, or during an uncertain close | Store state is ambiguous                                      | Refuse read-only open and run write-capable recovery                   |
| After verified reopen but before record removal                           | The batch may already be complete                             | Reverify idempotently, then remove the record                          |
| After an earlier batch but before the final physical session projection   | Completed rows are coherent but source progress is incomplete | Resume from the physical session projection and fetch before inserting |
| After the physical session projection but before marker acknowledgement   | Source progress is complete                                   | Retry acknowledgement from its exact marker coverage                   |
| After marker acknowledgement                                              | Transfer is complete                                          | No recovery work                                                       |

Any error after mutation begins, including an uncertain store close, retains recovery-required state.

## Bounded recovery

Recovery opens the affected stores with write capability and reconstructs the recorded batch from the immutable source, retained marker, stable IDs, and checksums. It fetches and verifies existing rows before inserting, inserts missing rows, replaces only isolated damaged rows, and completes deletions idempotently. It never blindly upserts already-valid immutable evidence.

Recovery verifies and clears only the recorded batch. Earlier completed batches remain usable. If the damage cannot be isolated, an active generation remains unavailable until explicit rollback or rebuild; a damaged replacement build fails without affecting the old active generation.

Read-only search never clears recovery state, starts ingestion, or repairs stores. It may wait at most for the current bounded write or recovery window. If recovery-required state remains after that window, search fails with an actionable recovery error.

## Rebuild validation and vector reuse

A rebuild captures one fixed starting snapshot of approved physical sources, logical sessions, source byte boundaries, and eligible contributors. During the build's existing source pass, it records the expected occurrence IDs, anchor IDs, checksums, membership digests, counts, and physical session projections. Validation must not perform another full import or tokenization pass.

An inactive replacement is not searchable while it builds. Its writer may therefore keep all three stores open across several physical sources instead of applying the active generation's per-window reopen protocol. It closes the writable stores when the batch reaches 2,048 generated records, then publishes the covered physical-source checkpoints. This measured bound avoids both one index block per small source and an unbounded HNSW build. It does not add a read-only validation pass after each batch.

After the build closes its final writable batch, validation reopens the replacement and checks actual membership against the recorded evidence. It verifies every lexical/source occurrence and anchor, the exact dense-searchable subset and embedding profile, every cross-store checksum, projection coverage, store schema and indexes, counts and membership digests, and canaries. It also proves the replacement opens without raw session files, another generation, or an embedding cache. Counts and a few sample rows alone do not validate a generation.

A resumable build resolves a vector in this order:

1. Reuse a matching row already verified in that same interrupted generation.
2. Copy a row from a previously validated generation when the embedding profile, dimensions, embedding-input checksum, evidence checksum, and vector checksum match.
3. Reuse an exact embedding input already resolved during the current build.
4. Recompute the vector from the model.

The destination always stores its own vector copy. It never keeps a live reference to another generation. A failed generation cannot supply vectors to a different build unless it is repaired and passes validation first.

## Rebuild observability and stall containment

The detached worker publishes the exact operation before entering any potentially long source, embedding, zvec write, store open, store close, or final validation call. The status names the source, source number, batch position, and record count when they apply. On completion, it records the operation's duration. This ordering matters because a synchronous native call can block the worker before it can report anything afterward.

The worker also publishes a heartbeat every five seconds. If an operation and the heartbeat are both stale for 30 seconds, operator status records a stall diagnosis with the phase and elapsed times. A long asynchronous model request remains distinguishable from a blocked event loop because its heartbeat continues.

Every detached rebuild runs with V8 CPU sampling enabled. The profile log is scoped to the build ID beside the background status file. A separate watchdog process observes the worker. It does not treat a long operation alone as failure. If both the named operation and heartbeat remain frozen for 30 minutes, it writes the active operation, heartbeat, Linux process statistics, and profile path to a stall artifact, then kills the worker. The ordinary crash path leaves the inactive generation resumable, but a run terminated by this watchdog is diagnostic evidence rather than acceptance evidence.

## Activation and replay

All expensive work and the immutable validation receipt are complete before cutover. Inside the short cutover window, the generation transition owner:

1. Rereads and verifies the active pointer and generation registry.
2. Persists an immutable generation replay snapshot for the activation containing the exact pending and quarantined marker IDs present at that boundary.
3. Publishes the READY registry state, atomically replaces the active pointer, then publishes the activated registry state.

A crash before pointer replacement leaves the old generation selected. A crash between durable transition writes leaves recovery-required state; the named cutover recovery transition reconciles the recorded pointer and registry without scanning for a latest directory.

The validated replacement becomes searchable when cutover completes. The former active generation becomes the retained rollback generation. The ordinary incremental worker then processes that generation replay snapshot against the new active generation. Replay completes when those exact marker IDs are covered by verified physical session projections and any quarantined IDs from that snapshot are resolved. Markers published after the snapshot are ordinary best-effort backlog and do not extend activation replay indefinitely. Search never waits for replay or backlog drain, and incremental work never writes both generations.

## Rollback

Rollback is an explicit, quick switch to the retained generation, not a rebuild or recertification. Before taking the cutover window, it performs a bounded health check:

- required generation files and immutable fingerprints still match;
- all three stores open with the declared schemas and indexes;
- store counts agree with totals derived from the physical session projections; and
- deterministic canary lookups selected from those projections return matching IDs, checksums, and profile data.

Rollback does not scan every row, read current session files, compare freshness, tokenize, embed, or optimize. A failed health check refuses rollback.

After the health check, rollback rereads generation roles inside the cutover window, records a fixed generation replay snapshot for rollback from retained and currently pending markers, and uses the named registry-first rollback transition to replace the active pointer. Marker restoration and replay are idempotent and resume from that snapshot after the write window. Session changes newer than the snapshot remain ordinary backlog. The replaced generation becomes the retained rollback generation according to the existing bounded-retention policy.

## Operator boundary

The Pi extension exposes model-facing recall reads and publishes cheap recall work markers. It exposes no index-maintenance slash command and sends no maintenance status or progress messages to the TUI.

A standalone `pi-session-recall` CLI is the sole operator control surface for setup, status, explicit catch-up, rebuild control, recovery, rollback, and retired-generation cleanup. It creates generations from immutable Pi session sources and never adopts a legacy storage layout. The CLI adapts the same service and named generation transitions used by workers; it does not reimplement lifecycle policy.

## Consequences

Foreground agent and machine performance take priority over recall freshness. There is no freshness deadline or maximum backlog. Search remains valid when it is stale, but not when the active generation has unresolved write recovery or fails compatibility checks.

The required durability boundaries are the recovery-required record for every bounded store mutation and the verified physical session projection for each completed source transfer. Additional native flushes or intermediate rebuild checkpoints belong in acceptance work only if measurements show meaningful recovery savings without material complexity, write amplification, or storage growth.

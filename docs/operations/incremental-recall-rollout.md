# Incremental recall rollout checklist

Production cutover requires human approval. Tests and scratch evaluations do not grant permission to open, copy, lock, rename, or mutate production recall or session paths.

Record the operator, host, date, command output, backup location, active generation ID, and rollback generation ID for every completed stage.

## 1. Preflight

- [ ] Confirm the installed extension commit and Node, zvec, manifest, evidence, projection, marker, and eligibility-policy versions.
- [ ] Confirm `flock` works on the recall filesystem.
- [ ] Confirm the active-generation pointer and generation root share one filesystem.
- [ ] Confirm free space can hold the active generation, replacement generation, rollback generation, marker spool, and temporary build overhead.
- [ ] Confirm no rebuild, adoption, rollback, or collection operation is already running.
- [ ] Review `docs/evaluation/recall-quality-report.md`, the focused diagnostics, dependency audit, structural checks, and slop scan.
- [ ] Stop if hook p95 exceeds 25 ms, a 10,000-file sweep exceeds 500 ms, a batch exceeds 32 documents, write-window p95 exceeds 300 ms, search waits longer than 500 ms, or a projection payload exceeds 8 MiB. Return to design review with the raw scalar records.

## 2. Backup and rollback proof

- [ ] Stop automatic worker launches for the maintenance window without deleting pending markers.
- [ ] Back up the active pointer, generation registry, backlog summary, marker control state, marker spool, and complete active generation.
- [ ] Verify checksums on the backup.
- [ ] Write down the rollback command: `/pi-session-recall-index --rollback`.
- [ ] Prove the backup can be read from a separate location. Do not test restoration over the active production directory.

## 3. Explicit legacy adoption, if required

- [ ] Run `/pi-session-recall-index --adopt-legacy` only when the current exact version-5 layout has been identified.
- [ ] Verify adoption leaves the legacy generation read-only and preserves search.
- [ ] Verify the relocation journal completed or can resume after interruption.
- [ ] Stop if any source, manifest, or projection identity differs from the exact adoption contract.

## 4. Replacement build

- [ ] Announce the rebuild and expected stale-but-available search behavior.
- [ ] Run `/pi-session-recall-index --rebuild` after explicit human approval.
- [ ] Confirm markers continue accumulating outside generations while commits are frozen.
- [ ] Confirm searches continue opening the old active generation during build and optimization.
- [ ] Record progress, batch sizes, cache hits and misses, embedding requests, write-window timings, and failures.
- [ ] Stop before cutover if replacement validation fails.

## 5. Pointer and worker verification

- [ ] Verify the replacement evidence and projection stores, index manifest, and checksums before pointer replacement.
- [ ] Verify the atomic pointer names the validated replacement and the registry checksum agrees.
- [ ] Verify the former active generation is the bounded rollback generation.
- [ ] Restart worker launches.
- [ ] If recovery state exists, verify a write-capable reopen clears it before read-only search resumes.
- [ ] Verify retained markers replay against the active generation and are acknowledged only after projection checkpoint coverage.
- [ ] Verify the marker watermark drains or leaves a scalar, explained backlog.

## 6. Smoke search and rollback proof

- [ ] Run project-scoped dense, lexical, and identifier smoke searches.
- [ ] Verify top-five fusion, branch labels, duplicate occurrences, summaries, tool evidence, turn context, and source provenance.
- [ ] Verify default hybrid search makes no reranker request.
- [ ] Verify search remains available during one bounded write window and never starts ingestion.
- [ ] Exercise `/pi-session-recall-index --rollback` only if the approved maintenance plan includes a real rollback drill. Otherwise verify the retained generation and rollback pointer material without changing production.

## 7. Bounded cleanup

- [ ] Keep the rollback generation through the approved retention period.
- [ ] Run `/pi-session-recall-index --collect-retired` only after smoke searches, marker drain, and rollback evidence pass.
- [ ] Verify cleanup refuses active, building, replay-pending, and retained rollback generations.
- [ ] Record the final pointer, registry, backlog, marker counts, generation directories, and free space.

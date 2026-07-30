# Target recall generation rollout checklist

Production operations require human approval. Tests and evaluations do not grant permission to open, copy, lock, rename, delete, or mutate production recall data or original Pi session files.

Record the operator, host, date, installed commit, command output, active generation ID, and rollback generation ID for every completed stage.

## 1. Preflight

- [ ] Review the target read and write acceptance evidence for the installed commit.
- [ ] Confirm Node, zvec, target manifest, store schema, marker, projection, and eligibility-policy versions.
- [ ] Confirm `flock` works on the recall filesystem.
- [ ] Confirm the active pointer and generation root share one filesystem.
- [ ] Confirm free space can hold the active, replacement, and retained rollback target generations plus temporary build overhead.
- [ ] Run `pi-session-recall status` and stop if another build, recovery, rollback, or cleanup owns the operation.
- [ ] Stop if marker publication p95 exceeds 25 ms, a 10,000-file metadata sweep exceeds 500 ms, a batch exceeds 32 documents, write-window p95 exceeds 300 ms, search waits longer than 500 ms, or a projection payload exceeds 8 MiB.

## 2. Fresh target build

- [ ] Confirm the configured embedding profile and stored dimensions.
- [ ] Announce the expected stale-but-available search behavior.
- [ ] Run `pi-session-recall rebuild` after explicit approval.
- [ ] Use `pi-session-recall status` to record generation identity, process state, durable checkpoint, progress, and actionable errors.
- [ ] Confirm markers continue accumulating outside generations while replacement commits are frozen.
- [ ] If the worker stops, use `pi-session-recall resume` for the same generation and snapshot. Do not create a migration or adoption path.
- [ ] Stop before activation if complete reopened-store validation or the immutable validation receipt fails.

## 3. Activation and replay verification

- [ ] Verify the active pointer names the validated target generation and the registry checksum agrees.
- [ ] Verify search and exact source expansion remain available during fixed replay.
- [ ] Verify replay covers only marker IDs captured at activation; newer markers remain ordinary backlog.
- [ ] Run `pi-session-recall catch-up` as needed until the fixed replay completes.
- [ ] If recovery is required, run `pi-session-recall recover` and verify the generation reopens before search resumes.
- [ ] For the first target activation, verify no rollback generation exists.
- [ ] After later target-to-target activation, verify exactly one validated former target generation has the rollback role.

## 4. Smoke checks

- [ ] Run project-scoped dense, lexical, and identifier searches.
- [ ] Run one explicit global search.
- [ ] Verify top-five fusion, branch labels, duplicate occurrences, summaries, lexical-only tool evidence, neighbor context, and exact source provenance.
- [ ] Expand one returned Evidence occurrence ID without reading session JSONL.
- [ ] Verify default hybrid search makes no reranker request.
- [ ] Verify search never starts ingestion or waits for marker backlog.

## 5. Target rollback

- [ ] Run `pi-session-recall rollback` only when the approved plan includes a rollback drill or an actual target-generation fault requires it.
- [ ] Confirm the bounded health check opens all three target stores and verifies declared fingerprints, projection-derived counts, and canaries.
- [ ] Confirm rollback captures its own fixed replay snapshot and does not read session files or recertify all rows.
- [ ] Confirm a second rollback can switch back to the replaced validated target generation when policy retains it.

## 6. Cleanup

- [ ] Keep rollback material through the approved retention period.
- [ ] Run `pi-session-recall cleanup` only after smoke searches and replay evidence pass.
- [ ] Verify cleanup refuses active, building, replay-pending, and retained rollback generations.
- [ ] Record final pointer, registry, backlog, marker counts, generation directories, and free space.

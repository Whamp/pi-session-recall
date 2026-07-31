# Final target-generation acceptance evidence

## Decision

**Historical safe reproducible evidence: PASS. Release evidence: PENDING CLEAN-CANDIDATE RECERTIFICATION, LEGACY CUTOVER, AND FULL-CORPUS MEASUREMENT.**

The complete #132 read matrix passed against historical runtime candidate `cc991d144953791b8cd798fb84b207bcd34624b7`. The #133 write, recovery, lifecycle, and foreground-bound matrix passed against `67abdc772a19be01620c3cb7951561e546385947`, whose only change from the runtime candidate was regenerated read-evidence documentation. Later repair candidate `2552424196434a368e62448c78be7b8d69ff5aa9` passed focused recovery certification, the full suite, and static gates, but the complete read and write evidence matrices were not rerun against that revision. Review repairs after that candidate also require a fresh clean-candidate run. Existing results are historical evidence, not release certification for the repaired HEAD.

Release evidence also remains incomplete because no immutable full-corpus snapshot of the existing Pi session JSONL source corpus has been identified for the disposable build. The committed 15-file quality corpus is approved only as a bounded retrieval evaluation, so full-corpus generation size and rebuild duration remain unmeasured. The required snapshot is source input: it is not the legacy recall database or a new synthetic corpus. The build output will be a new target-format generation in disposable storage. Production rebuild and activation remain separate human-approved operations.

## Candidate and environment

- Runtime candidate commit: `cc991d144953791b8cd798fb84b207bcd34624b7`
- Read-evidence commit: `cc991d144953791b8cd798fb84b207bcd34624b7`
- Write-evidence commit: `67abdc772a19be01620c3cb7951561e546385947` (read-evidence documentation only after the runtime candidate)
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: `0.6.0`
- Quality specification: `evaluation/recall-quality-cases.json`
- Quality specification SHA-256: `6208cfc632c8ff53815567dd5385297bb6cc513f62e3d50a5bfa8ae687c34439`
- Bounded corpus: 15 checksum-fixed JSONL files, 44,784 source bytes, 17 evaluation cases
- Bounded starting snapshot: `d7386a54a1ac11c2be0d86e184907bfc501bf129bc2553055b4d1b0bfb431cd7`
- Write-fixture snapshot: `25863553fddd9c86f11935cf80a1a48d31eafadb22d9ee2235674134ad7e1c64`

## Commands

```bash
npm run evidence:target-reads
npm run evidence:target-writes
npm test
npm run typecheck
npm run lint
npm run format:check

slop-scan delta \
  --base /home/will/projects/pi-session-recall \
  --head /home/will/projects/pi-session-recall/.worktrees/issue-122-coherent-generations \
  --fail-on added,worsened
```

`npm test` passed against the clean final reviewed runtime candidate. It reported 605 tests: 601 passed, 0 failed, and 4 expected skips.

## Read and retrieval evidence

`npm run evidence:target-reads` passed its frozen quality gate and all 89 composed target-read tests.

| Obligation                                                                                           | Result |
| ---------------------------------------------------------------------------------------------------- | ------ |
| Reopened-store membership, schema, index, profile, digest, path, and canary faults                   | PASS   |
| Lexical/source store without vectors and exact dense subset                                          | PASS   |
| Stored-width truncation, L2 normalization, repeatability, identity, and canaries                     | PASS   |
| Existing search modes, fusion, scope, duplicates, context, provenance, cancellation, and diagnostics | PASS   |
| Exact source-neighborhood expansion and model-facing tool adapter                                    | PASS   |
| Reads during replay, replacement work, and target-to-target rollback                                 | PASS   |

The fixed policy remained 512/64 tokens/overlap, eight candidates per channel, and five final results. Candidate-pool recall, final recall, context usefulness, source-occurrence preservation, and provenance were 100%; final duplicate rate was 0%. Query p95 was 152.220 ms against the 2,000 ms limit.

The fresh bounded replacement generation contained 208 lexical/source rows, 113 dense rows, and 30 projection rows. Its on-disk generation size was 82,156,578 bytes. Fresh build and activation took 3,676.508 ms; the complete build-and-evaluation command took 6,572.479 ms. Size and rebuild duration are reported values, not pass thresholds.

Detailed artifacts:

- [`recall-quality-report.md`](recall-quality-report.md)
- [`recall-quality-results.json`](recall-quality-results.json)

## Write, recovery, and lifecycle evidence

`npm run evidence:target-writes` passed 103 composed behavior tests plus the marker, metadata-sweep, and close/reopen write-window diagnostics.

The incremental fault matrix passed these boundaries:

1. before the recovery record;
2. after the recovery record;
3. after lexical/source mutation;
4. after dense mutation;
5. after logical projection mutation;
6. after physical projection mutation;
7. after store close;
8. after reopened verification;
9. after recovery clear; and
10. after marker acknowledgement.

The confirmed-deletion matrix passed before and after recovery intent, after dense deletion, after lexical/source deletion, after projection deletion, after close, after reopened verification, and after recovery clear.

A real detached child received `SIGKILL`, resumed the same generation, and completed reopened-store validation. Fixed activation replay, fixed rollback replay, bounded rollback health checks, two-generation rollback, and switch-back passed. The 33-document interruption probe resumed between the first 32-document batch and the final physical projection.

| Foreground bound                           | Measured or enforced |           Limit | Result |
| ------------------------------------------ | -------------------: | --------------: | ------ |
| Marker publication plus detached spawn p95 |             3.316 ms |           25 ms | PASS   |
| Metadata sweep p95 at 10,000 files         |            40.005 ms |          500 ms | PASS   |
| Projection payload                         |             enforced | 8,388,608 bytes | PASS   |
| Evidence batch                             |             enforced |    32 documents | PASS   |
| Close/reopen write-window p95, 20 samples  |            63.358 ms |          300 ms | PASS   |
| Search wait for the current write window   |             enforced |          500 ms | PASS   |

Detailed artifacts:

- [`recall-write-acceptance.md`](recall-write-acceptance.md)
- [`recall-write-acceptance.json`](recall-write-acceptance.json)

## Final paired review

Paired Sol and GLM reviewers independently checked Standards and Spec, followed by a separate Sol synthesis. Standards passed after the synthesis rejected three style or dead-surface suggestions that lacked observable harm or would have expanded cleanup scope. The Spec synthesis retained two bounded defects and rejected no additional implementation scope:

1. Lifecycle and metadata-sweep work markers carried raw logical session IDs instead of sessions-root-relative physical source identity. The final candidate derives marker identity from the source path while retaining the raw logical ID only in trigger provenance.
2. The bounded rollback health check validated dense checksum syntax but did not read the selected vector. The final candidate verifies one projection-selected vector checksum and cross-checks its evidence checksum against the authoritative lexical/source occurrence.

Both fixes were developed red-green. Their affected configured-service, lifecycle, worker, transfer, rollback, tool, and target-generation surface passed 74/74 tests before the final full suite.

## Historical gates

- Target read matrix: 89/89 PASS.
- Target write matrix: 103/103 PASS; all three foreground diagnostics PASS.
- Final-review affected tests: 74/74 PASS.
- TypeScript typecheck: PASS.
- Type-aware oxlint over `src`: PASS.
- oxfmt over the repository: PASS.
- Slop-scan delta from an exact detached base worktree: 0 added, 0 worsened.
- Full `npm test`: PASS — 601 passed, 0 failed, 4 expected skips out of 605 tests.

These gates passed on the historical candidates named above. They do not certify one current repaired revision across the complete read, write, recovery, quality, and foreground-limit matrices.

## Safety declaration

No production recall generation was opened or mutated. No original Pi session file was opened or mutated. The harness copied committed fixtures into disposable roots, removed the copied source before target reads where required, and deleted generated stores after each run.

## Remaining release gates

1. On one clean repaired candidate, rerun `npm run evidence:target-reads`, `npm run evidence:target-writes`, recovery certification, the full suite, typecheck, type-aware lint, formatting, diff checks, and repository-required slop scan. Regenerate this report so every result identifies that candidate.
2. Identify a checksum-bound snapshot of the existing Pi session JSONL source corpus. Do not use the legacy recall database or a new synthetic corpus. Use an existing immutable copy, or explicitly authorize a one-time read-only copy into a disposable sessions root. The replacement build must read only that disposable snapshot and must not open production recall storage or mutate original Pi session files.
3. Run the full replacement build against the approved snapshot and record its total duration and final generation size.
4. After all clean-candidate and full-corpus evidence passes together, remove the obsolete public `index()` writer, legacy rebuild and single-store contracts, persistent embedding-cache runtime/configuration, legacy adoption paths, and stale current-looking guidance. This cutover must not add a migration or compatibility wrapper. Review all evidence and the cutover before any production rebuild or activation.

# Final target-generation acceptance evidence

## Decision

**Repaired clean-candidate evidence: PASS. Release evidence: PENDING FULL-CORPUS MEASUREMENT AND LEGACY RETIREMENT.**

The complete #132 read matrix, #133 write and lifecycle matrix, and #142 production-cardinality recovery certification all passed in a disposable clean worktree at runtime candidate `ebb23ee986dc8c7674b58b6ebc8b5756a0645469`. The recovery run reopened exact membership above the observed 119,662-record failure boundary in every store and matched uninterrupted with SIGKILL/resumed detached terminal membership.

Release evidence remains incomplete because the repaired candidate has not completed a fresh build from the approved immutable snapshot of the existing Pi session JSONL source corpus. The committed 15-file quality corpus and generated production-cardinality corpus do not replace that source snapshot. Full-corpus generation size and rebuild duration remain unmeasured. The build output must be a new target-format generation in disposable storage; the prior mixed-commit generation is not evidence. Production rebuild and activation remain separate human-approved operations.

## Candidate and environment

- Runtime candidate commit: `ebb23ee986dc8c7674b58b6ebc8b5756a0645469`
- Read-evidence candidate: `ebb23ee986dc8c7674b58b6ebc8b5756a0645469`
- Write-evidence candidate: `ebb23ee986dc8c7674b58b6ebc8b5756a0645469`
- Recovery-evidence candidate: `ebb23ee986dc8c7674b58b6ebc8b5756a0645469`
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
PI_RECALL_SLOP_BASE_DIRECTORY=/path/to/clean/b04b350 npm run evidence:generation-recovery
npm test
npm run typecheck
npm run lint
npm run format:check

slop-scan delta \
  --base /home/will/projects/pi-session-recall \
  --head /home/will/projects/pi-session-recall/.worktrees/issue-122-coherent-generations \
  --fail-on added,worsened
```

The recovery certifier ran `npm test` against the clean runtime candidate. It reported 642 tests: 638 passed, 0 failed, and 4 expected skips.

## Read and retrieval evidence

`npm run evidence:target-reads` passed its frozen quality gate and all 101 composed target-read tests.

| Obligation                                                                                           | Result |
| ---------------------------------------------------------------------------------------------------- | ------ |
| Reopened-store membership, schema, index, profile, digest, path, and canary faults                   | PASS   |
| Lexical/source store without vectors and exact dense subset                                          | PASS   |
| Stored-width truncation, L2 normalization, repeatability, identity, and canaries                     | PASS   |
| Existing search modes, fusion, scope, duplicates, context, provenance, cancellation, and diagnostics | PASS   |
| Exact source-neighborhood expansion and model-facing tool adapter                                    | PASS   |
| Reads during replay, replacement work, and target-to-target rollback                                 | PASS   |

The fixed policy remained 512/64 tokens/overlap, eight candidates per channel, and five final results. Candidate-pool recall, final recall, context usefulness, source-occurrence preservation, and provenance were 100%; final duplicate rate was 0%. Project query p95 was 145.4 ms and global query p95 was 141.4 ms against the 2,000 ms limit.

The fresh bounded replacement generation contained 208 lexical/source rows, 113 dense rows, and 30 projection rows. Its on-disk generation size was 82,160,438 bytes. Fresh build and activation took 3.8 seconds; the complete build-and-evaluation command took 6,713.2 ms. Size and rebuild duration are reported values, not pass thresholds.

Detailed artifacts:

- [`recall-quality-report.md`](recall-quality-report.md)
- [`recall-quality-results.json`](recall-quality-results.json)

## Write, recovery, and lifecycle evidence

`npm run evidence:target-writes` passed 105 composed behavior tests plus the marker, metadata-sweep, and close/reopen write-window diagnostics.

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
| Marker publication plus detached spawn p95 |             3.444 ms |           25 ms | PASS   |
| Metadata sweep p95 at 10,000 files         |            48.862 ms |          500 ms | PASS   |
| Projection payload                         |             enforced | 8,388,608 bytes | PASS   |
| Evidence batch                             |             enforced |    32 documents | PASS   |
| Close/reopen write-window p95, 20 samples  |            71.068 ms |          300 ms | PASS   |
| Search wait for the current write window   |             enforced |          500 ms | PASS   |

Detailed artifacts:

- [`recall-write-acceptance.md`](recall-write-acceptance.md)
- [`recall-write-acceptance.json`](recall-write-acceptance.json)

## Production-cardinality recovery evidence

`npm run evidence:generation-recovery` passed from the clean candidate with a separate exact `b04b350` slop-scan base. Reopened membership reached 239,328 lexical/source rows, 119,664 dense rows, and 119,785 session-projection rows. Uninterrupted and twice-resumed in-process builds matched exactly. Matched uninterrupted and SIGKILL/resumed detached workers reached terminal succeeded/ready validation with identical membership digests.

Detailed artifacts:

- [`recall-generation-recovery-preflight.md`](recall-generation-recovery-preflight.md)
- [`recall-generation-recovery-preflight.json`](recall-generation-recovery-preflight.json)

## Review disposition

Paired Sol and GLM reviewers found concrete scale, snapshot-isolation, bootstrap, failure-classification, cumulative-membership, property-strength, and detached-equivalence defects. Issues #136–#142 and review repair commit `ebb23ee986dc8c7674b58b6ebc8b5756a0645469` addressed them red-green. A read-only Cursor second opinion independently confirmed the runtime repairs. Its only accepted finding was that the committed evidence still named the pre-repair candidate; these regenerated artifacts close that bookkeeping gap.

## Clean-candidate gates

- Target read matrix: 101/101 PASS.
- Target write matrix: 105/105 PASS; all three foreground diagnostics PASS.
- Production-cardinality recovery certification: PASS.
- TypeScript typecheck: PASS.
- Type-aware oxlint over `src`: PASS.
- oxfmt over the repository: PASS.
- Slop-scan delta from the exact `b04b350` base worktree: 0 added, 0 worsened.
- Full `npm test`: PASS — 638 passed, 0 failed, 4 expected skips out of 642 tests.

## Safety declaration

No production recall generation was opened or mutated. No original Pi session file was opened or mutated. The harness copied committed fixtures into disposable roots, removed the copied source before target reads where required, and deleted generated stores after each run.

## Remaining release gates

1. Select the approved checksum-bound immutable snapshot of the existing Pi session JSONL source corpus for the final repaired candidate. Do not use the legacy recall database, the prior mixed-commit generation, or a new synthetic corpus. The replacement build must read only the disposable snapshot and must not open production recall storage or mutate original Pi session files.
2. Run a fresh replacement build against that snapshot and record its total duration and final generation size.
3. After clean-candidate and full-corpus evidence pass together, remove the obsolete public `index()` writer, legacy rebuild and single-store contracts, persistent embedding-cache runtime/configuration, legacy adoption paths, and stale current-looking guidance. This cutover must not add a migration or compatibility wrapper. Review all evidence and the cutover before any production rebuild or activation.

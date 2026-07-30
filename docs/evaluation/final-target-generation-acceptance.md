# Final target-generation acceptance evidence

## Decision

**Safe reproducible evidence: PASS. Release evidence: BLOCKED on two owner decisions.**

The complete #132 read matrix passed against repaired runtime candidate `693424def7a1565ddd104395b8eb31f9d2818653`. The #133 write, recovery, lifecycle, and foreground-bound matrix passed against `9db5c434e4d7ae576ce93446db644ddcd38dfeff`, whose only change from the runtime candidate was regenerated read-evidence documentation. The full suite and every executable static or behavioral gate now pass. Both evidence commands used copied repository fixtures, generated test sources, disposable temporary roots, and real zvec stores.

Release acceptance remains blocked for two reasons. First, no owner-approved immutable full-corpus snapshot is named in #135, #122, or the governing #115 decision. The committed 15-file quality corpus is approved only as a bounded retrieval evaluation, so full-corpus generation size and rebuild duration remain unmeasured. Second, #135 says the base-wide CodeGraph signature check must pass, but that predicate means “no declaration line changed” and therefore rejects the new and retired APIs required by #122. Production rebuild and activation remain separate human-approved operations.

## Candidate and environment

- Runtime candidate commit: `693424def7a1565ddd104395b8eb31f9d2818653`
- Read-evidence commit: `693424def7a1565ddd104395b8eb31f9d2818653`
- Write-evidence commit: `9db5c434e4d7ae576ce93446db644ddcd38dfeff` (read-evidence documentation only after the runtime candidate)
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: `0.6.0`
- Quality specification: `evaluation/recall-quality-cases.json`
- Quality specification SHA-256: `6208cfc632c8ff53815567dd5385297bb6cc513f62e3d50a5bfa8ae687c34439`
- Bounded corpus: 15 checksum-fixed JSONL files, 44,784 source bytes, 17 evaluation cases
- Bounded starting snapshot: `3d021beb6918e09b510e394efb55470da8548ac145f6ba6d7559933d969f1176`
- Write-fixture snapshot: `77c9e6ec9415ad0c8aa3d01c31c5c867e0374b5702dc0af8015165aaee683332`

## Commands

```bash
npm run evidence:target-reads
npm run evidence:target-writes
npm test
npm run typecheck
npm run lint
npm run format:check

codegraph build .
codegraph diff-impact b9667308b871faef28c2c8574e2ccf541c2a2cd8 -T
codegraph diff-impact b9667308b871faef28c2c8574e2ccf541c2a2cd8 --include-tests
codegraph cycles -T
codegraph cycles -T --functions
codegraph check 693424def7a1565ddd104395b8eb31f9d2818653 -T --cycles --signatures --boundaries
codegraph check b9667308b871faef28c2c8574e2ccf541c2a2cd8 -T --cycles --signatures --boundaries

slop-scan delta \
  --base /home/will/projects/pi-session-recall \
  --head /home/will/projects/pi-session-recall/.worktrees/issue-122-coherent-generations \
  --fail-on added,worsened
```

`npm test` passed against the clean repaired runtime candidate. It reported 604 tests: 600 passed, 0 failed, and 4 expected skips.

## Read and retrieval evidence

`npm run evidence:target-reads` passed its frozen quality gate and all 88 composed target-read tests.

| Obligation                                                                                           | Result |
| ---------------------------------------------------------------------------------------------------- | ------ |
| Reopened-store membership, schema, index, profile, digest, path, and canary faults                   | PASS   |
| Lexical/source store without vectors and exact dense subset                                          | PASS   |
| Stored-width truncation, L2 normalization, repeatability, identity, and canaries                     | PASS   |
| Existing search modes, fusion, scope, duplicates, context, provenance, cancellation, and diagnostics | PASS   |
| Exact source-neighborhood expansion and model-facing tool adapter                                    | PASS   |
| Reads during replay, replacement work, and target-to-target rollback                                 | PASS   |

The fixed policy remained 512/64 tokens/overlap, eight candidates per channel, and five final results. Candidate-pool recall, final recall, context usefulness, source-occurrence preservation, and provenance were 100%; final duplicate rate was 0%. Query p95 was 160.227 ms against the 2,000 ms limit.

The fresh bounded replacement generation contained 208 lexical/source rows, 113 dense rows, and 30 projection rows. Its on-disk generation size was 82,156,578 bytes. Fresh build and activation took 5,590.861 ms; the complete build-and-evaluation command took 8,859.540 ms. Size and rebuild duration are reported values, not pass thresholds.

Detailed artifacts:

- [`recall-quality-report.md`](recall-quality-report.md)
- [`recall-quality-results.json`](recall-quality-results.json)

## Write, recovery, and lifecycle evidence

`npm run evidence:target-writes` passed 102 composed behavior tests plus the marker, metadata-sweep, and close/reopen write-window diagnostics.

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
| Marker publication plus detached spawn p95 |             3.191 ms |           25 ms | PASS   |
| Metadata sweep p95 at 10,000 files         |            39.305 ms |          500 ms | PASS   |
| Projection payload                         |             enforced | 8,388,608 bytes | PASS   |
| Evidence batch                             |             enforced |    32 documents | PASS   |
| Close/reopen write-window p95, 20 samples  |            64.329 ms |          300 ms | PASS   |
| Search wait for the current write window   |             enforced |          500 ms | PASS   |

Detailed artifacts:

- [`recall-write-acceptance.md`](recall-write-acceptance.md)
- [`recall-write-acceptance.json`](recall-write-acceptance.json)

## Final gates

- Target read matrix: 88/88 PASS.
- Target write matrix: 102/102 PASS; all three foreground diagnostics PASS.
- Review-fix focused tests: 32/32 PASS.
- TypeScript typecheck: PASS.
- Type-aware oxlint over `src`: PASS.
- oxfmt over the repository: PASS.
- CodeGraph: no file-level cycles; only the four pre-existing function-level cycles remain.
- Runtime-candidate-to-evidence CodeGraph cycle, boundary, and signature predicates: PASS.
- Base-wide CodeGraph cycle and boundary predicates: PASS.
- Base-wide CodeGraph signature predicate: EXPECTED FAIL — it reports 278 declaration-line changes because #122 intentionally adds the target-generation contracts and retires the legacy contracts. The predicate detects any declaration change; it does not distinguish a reviewed required API from an accidental incompatible change.
- Base-wide CodeGraph impact review: 858 changed functions, 306 affected callers, 69 application files; no file-level cycles and only the four pre-existing function-level cycles.
- Slop-scan delta from an exact detached base worktree: 0 added, 0 worsened.
- Full `npm test`: PASS — 600 passed, 0 failed, 4 expected skips out of 604 tests.

All executable behavioral and static gates pass. #135 remains blocked because its literal base-wide no-signature-change criterion contradicts the required contract replacement and because no authorized full-corpus snapshot is available.

## Safety declaration

No production recall generation was opened or mutated. No original Pi session file was opened or mutated. The harness copied committed fixtures into disposable roots, removed the copied source before target reads where required, and deleted generated stores after each run.

## Required authorization

To complete release acceptance, the owner must make two explicit decisions:

1. Amend or waive #135's base-wide no-signature-change predicate. The recommended criterion is to accept the reviewed #122 contract delta, retain the passing base-wide cycle and boundary predicates, and require the runtime-candidate-to-evidence signature predicate to stay green.
2. Identify an immutable, checksum-bound full-corpus snapshot and explicitly authorize copying it into a disposable sessions root. The authorization must not permit opening the production recall generation or mutating original Pi session files.

After those decisions, rerun the full replacement build against the authorized snapshot and record its total duration and final generation size. Review that evidence before any production rebuild or activation.

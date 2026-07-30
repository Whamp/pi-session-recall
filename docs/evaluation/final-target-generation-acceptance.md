# Final target-generation acceptance evidence

## Decision

**Safe reproducible evidence: PASS. Release evidence: BLOCKED.**

The complete #132 read matrix and #133 write, recovery, lifecycle, and foreground-bound matrix passed against clean candidate commit `5279c669330bf71297d17ac9270685a8f9b29af0`. Both commands used copied repository fixtures, generated test sources, disposable temporary roots, and real zvec stores.

No owner-approved immutable full-corpus snapshot is named in #135, #122, or the governing #115 decision. The committed 15-file quality corpus is approved only as a bounded retrieval evaluation. A full-corpus generation size and rebuild duration therefore remain unmeasured. Production rebuild and activation remain separate human-approved operations.

## Candidate and environment

- Candidate commit: `5279c669330bf71297d17ac9270685a8f9b29af0`
- Node: `v24.16.0`
- Platform: `linux/x64`
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: `0.6.0`
- Quality specification: `evaluation/recall-quality-cases.json`
- Quality specification SHA-256: `6208cfc632c8ff53815567dd5385297bb6cc513f62e3d50a5bfa8ae687c34439`
- Bounded corpus: 15 checksum-fixed JSONL files, 44,784 source bytes, 17 evaluation cases
- Bounded starting snapshot: `5e756bddee4b3b4df005056359610fa64766e8631706e7d363927a1e52ba22d0`
- Write-fixture snapshot: `802ba8c5fade8ae495eb4770d46e386e0becab5d8d0a9396c8d629ee508d8b09`

## Commands

```bash
npm run evidence:target-reads
npm run evidence:target-writes
npm run typecheck
npm run lint
npm run format:check

codegraph build .
codegraph diff-impact b9667308b871faef28c2c8574e2ccf541c2a2cd8 -T
codegraph diff-impact b9667308b871faef28c2c8574e2ccf541c2a2cd8 --include-tests
codegraph cycles -T
codegraph cycles -T --functions
codegraph check b9667308b871faef28c2c8574e2ccf541c2a2cd8 -T --cycles --signatures

slop-scan delta \
  --base /home/will/projects/pi-session-recall \
  --head /home/will/projects/pi-session-recall/.worktrees/issue-122-coherent-generations \
  --fail-on added,worsened
```

`npm test` was not run in this implementation phase. The workflow requires the full suite to run once after review.

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

The fixed policy remained 512/64 tokens/overlap, eight candidates per channel, and five final results. Candidate-pool recall, final recall, context usefulness, source-occurrence preservation, and provenance were 100%; final duplicate rate was 0%. Query p95 was 128.085 ms against the 2,000 ms limit.

The fresh bounded replacement generation contained 208 lexical/source rows, 113 dense rows, and 30 projection rows. Its on-disk generation size was 82,156,578 bytes. Fresh build and activation took 3,450.306 ms; the complete build-and-evaluation command took 6,019.872 ms. Size and rebuild duration are reported values, not pass thresholds.

Detailed artifacts:

- [`recall-quality-report.md`](recall-quality-report.md)
- [`recall-quality-results.json`](recall-quality-results.json)

## Write, recovery, and lifecycle evidence

`npm run evidence:target-writes` passed 101 composed behavior tests plus the marker, metadata-sweep, and close/reopen write-window diagnostics.

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
| Marker publication plus detached spawn p95 |             3.710 ms |           25 ms | PASS   |
| Metadata sweep p95 at 10,000 files         |            35.203 ms |          500 ms | PASS   |
| Projection payload                         |             enforced | 8,388,608 bytes | PASS   |
| Evidence batch                             |             enforced |    32 documents | PASS   |
| Close/reopen write-window p95, 20 samples  |            74.326 ms |          300 ms | PASS   |
| Search wait for the current write window   |             enforced |          500 ms | PASS   |

Detailed artifacts:

- [`recall-write-acceptance.md`](recall-write-acceptance.md)
- [`recall-write-acceptance.json`](recall-write-acceptance.json)

## Final gates

- Focused measurement and report tests: 4/4 PASS.
- TypeScript typecheck: PASS.
- Type-aware oxlint over `src`: PASS.
- oxfmt over the repository: PASS after removing one accumulated trailing blank line.
- CodeGraph: no file-level cycles; only the four pre-existing function-level cycles remain.
- Base-wide CodeGraph cycle and boundary predicates: PASS. Its signature predicate reports the intentional accumulated #122 contract changes rather than a new #135 regression.
- Slop-scan delta: 0 added, 0 worsened.
- Exact search found no retained `EmbeddingVectorCache`, `LocalEmbeddingClient`, `LocalRerankerClient`, `pi-session-recall-index`, `adoptLegacy`, or `exactVersion5` implementation path outside negative tests.
- Full `npm test`: DEFERRED until after review by workflow instruction.

## Safety declaration

No production recall generation was opened or mutated. No original Pi session file was opened or mutated. The harness copied committed fixtures into disposable roots, removed the copied source before target reads where required, and deleted generated stores after each run.

## Required authorization

To complete the blocked full-corpus evidence, the owner must identify an immutable, checksum-bound full-corpus snapshot and explicitly authorize copying that snapshot into a disposable sessions root. The authorization must not permit opening the production recall generation or mutating original Pi session files. After review, run the full test suite once against the final branch. Review of the resulting evidence must precede any production rebuild or activation.

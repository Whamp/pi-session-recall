# Repaired generation recovery preflight

**Result:** PASS

- Candidate commit: `c492fa352d3f20c64de41f94f0cf2525a61710f9`
- Completed: 2026-08-01T00:53:15.529Z
- Generated source snapshot: `7de03f3ead3522a622d9f24bd6843ebcebfa871f30797432e0243678b04cc5e2`
- Certification inputs: `7de61944ec1b2cf9378759da40894300ea29c8e65da41fe16bf224523a88adbe`
- Runtime: v24.16.0 on linux/x64
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: 0.6.0
- Deterministic embedding profile: `embedding-profile-b792b064200a2f84447527b7ecc8b076866e170003ae72c79c642b04608f068e`
- Manifest fingerprint: `21c7e2340521ed228f97ea90cde46a3bfc90797a5c1ef90c13b69ba97e8c3863`
- Starting snapshot fingerprint: `920078034a816c92b8d79b0e1dce82178e7dc085619c159a4b7e60851b139182`

## Production cardinality

Complete reopened validation crossed the observed 119,662-record failure boundary in every real-zvec store.

| Store              | Reopened count | Membership digest                                                  |
| ------------------ | -------------: | ------------------------------------------------------------------ |
| Lexical/source     |        239,328 | `36e0f2283963ae65bb23cdebd6ecea79ef9c7889dc7e4cd87726608ca4ee9e23` |
| Dense              |        119,664 | `2bdff5f46d8a1d7271f17df1a4817bd6fa0c9263463246f1d3a60b5de88c9e64` |
| Session projection |        119,785 | `409d4f9ce3fac2622ead51e14bb48959840a852475d425b1dec6e9df58050e27` |

The uninterrupted and twice-resumed in-process builds used the same candidate commit, generated source snapshot, generation ID, manifest, profile, and source snapshot fingerprint. Their immutable comparable validation receipts and all membership digests agree.

## Detached terminal equivalence

Matched uninterrupted and SIGKILL/resumed detached workers used generated source snapshot `2d25bcd5b88f32967d8e3dc753faa839692f85ec5a00830b6a1342db09f5f4c8`. Both reached terminal succeeded/ready validation. Their compatible embedding profiles, snapshot cardinalities, validation policy, canary results, store counts, and all membership digests agree.

| Store              | Uninterrupted digest                                               | SIGKILL/resumed digest                                             |
| ------------------ | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Lexical/source     | `32364d50a34eb7237bb57225b5593117f51543e90f118fa372c9309596112826` | `32364d50a34eb7237bb57225b5593117f51543e90f118fa372c9309596112826` |
| Dense              | `d64277f348367626358bba4741af51650d1882822810781fca99fdf22aac10fc` | `d64277f348367626358bba4741af51650d1882822810781fca99fdf22aac10fc` |
| Session projection | `d0786a9fa1ca7d359c5c55558b1629800daa2218fe914ab3594ba1f2c108163b` | `d0786a9fa1ca7d359c5c55558b1629800daa2218fe914ab3594ba1f2c108163b` |

## Recovery matrix

| Check                                                                                      | Result |
| ------------------------------------------------------------------------------------------ | ------ |
| Original high-cardinality fixture removed after snapshot capture                           | PASS   |
| Original retained fixture changed after snapshot capture                                   | PASS   |
| Resume after bootstrap snapshot capture interruption                                       | PASS   |
| Resume after durable physical-source checkpoint interruption                               | PASS   |
| Structurally malformed source skipped while later healthy source indexed                   | PASS   |
| Injected operational failure remained fatal                                                | PASS   |
| Injected implementation failure remained fatal                                             | PASS   |
| Generated incremental append/replay/branch/deletion schedule matched fresh rebuild         | PASS   |
| Detached worker was interrupted, replaced, and reached terminal succeeded/ready validation | PASS   |
| Standalone CLI stop/resume reached terminal succeeded/ready validation                     | PASS   |

## Measurements

- Disposable uninterrupted generation size: 3,137,647,268 bytes
- Disposable interrupted generation size: 3,137,647,268 bytes
- Production-cardinality preflight duration: 773120.301 ms

These values are reported without release thresholds.

## Reproduction

Create one clean worktree at candidate commit `c492fa352d3f20c64de41f94f0cf2525a61710f9`. Create a separate clean worktree at `b04b350939de11ae56b67f8d1e8cce9ab0b12ec8` for the slop-scan base, then run from the candidate worktree:

```bash
PI_RECALL_SLOP_BASE_DIRECTORY=/path/to/clean/b04b350939de11ae56b67f8d1e8cce9ab0b12ec8 npm run evidence:generation-recovery
```

The certifier ran:

- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-reporter=tap --test-name-pattern=replacement generation bootstrap interruption model src/build-recall-fixed-snapshot-generation.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-reporter=tap --test-name-pattern=malformed-source skips once|parser-looking operational failures fatal|non-source failure category fatal src/recall-physical-source-generation.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-reporter=tap --test-name-pattern=generated incremental append, replay, branch, and deletion schedules match fresh rebuild membership src/transfer-incremental-recall-work-plan.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-reporter=tap --test-name-pattern=crashed workers at every staging phase remain resumable and idempotent src/recall-background-index-conversation-service.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-reporter=tap --test-name-pattern=standalone rebuild stops, resumes the same snapshot, and discards inactive work src/pi-session-recall-cli.test.ts`
- `npm test`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- `git diff --check b04b350939de11ae56b67f8d1e8cce9ab0b12ec8...c492fa352d3f20c64de41f94f0cf2525a61710f9`
- `slop-scan delta --base <exact-base-worktree> --head "$PWD" --fail-on added,worsened`
- `npm run evidence:generation-recovery`

All source files and stores were generated beneath disposable temporary roots. The run did not open original Pi session JSONL, a live or existing acceptance generation, the Octen endpoint, or production activation.

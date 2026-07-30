# Target write acceptance evidence

**Result:** PASS

- Candidate commit: `9db5c434e4d7ae576ce93446db644ddcd38dfeff`
- Completed: 2026-07-30T13:47:54.488Z
- Source snapshot: `77c9e6ec9415ad0c8aa3d01c31c5c867e0374b5702dc0af8015165aaee683332`
- Runtime: v24.16.0 on linux/x64
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: 0.6.0

## Recovery and lifecycle

The configured service fault matrix passed 10 incremental interruption stages, including the pre-intent boundary, every cross-store mutation boundary, uncertain close, reopened verification, recovery clearing, and marker acknowledgement. A 33-document transfer also stopped between its 32-document first batch and final physical projection, then resumed idempotently.

A real detached child received SIGKILL and resumed the same generation identity through complete reopened-store validation. Fixed activation replay, rollback replay, bounded rollback health checks, two-generation rollback, and switch-back passed. Append, no-op replay, duplicate delivery, branch exit, compaction, quiescence, context-exit summary, confirmed deletion, and suspicious mass-loss protection passed.

## Foreground bounds

| Bound                                      | Measured or enforced |           Limit | Result |
| ------------------------------------------ | -------------------: | --------------: | ------ |
| Marker publication plus detached spawn p95 |             3.191 ms |           25 ms | PASS   |
| Metadata sweep p95 at 10,000 files         |            39.305 ms |          500 ms | PASS   |
| Projection payload                         |             enforced | 8,388,608 bytes | PASS   |
| Evidence batch                             |             enforced |    32 documents | PASS   |
| Close/reopen write-window p95 (20 samples) |            64.329 ms |          300 ms | PASS   |
| Search wait for current write window       |             enforced |          500 ms | PASS   |

## Reproduction

Run:

```bash
npm run evidence:target-writes
```

The command runs these subprocess checks:

- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test src/recall-generation-mutation-ownership.test.ts src/transfer-incremental-recall-work-plan.test.ts src/recall-background-index-conversation-service.test.ts src/pi-session-recall-cli.test.ts src/activate-validated-recall-generation.test.ts src/rollback-recall-generation.test.ts src/run-recall-incremental-worker.test.ts src/coordinate-recall-marker-replay.test.ts src/scan-recall-session-metadata.test.ts src/recall-session-projection.test.ts src/coordinate-recall-write-window.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test src/publish-recall-work-marker.diagnostic.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test src/scan-recall-session-metadata.diagnostic.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test src/commit-incremental-recall-transfer.diagnostic.test.ts`

The source snapshot hashes committed fixtures and deterministic generated-test sources listed in the JSON evidence. Every storage check uses a disposable temporary root and real zvec stores. The run accessed neither the production recall database nor original Pi session files. No extra native flush or intermediate checkpoint was required.

# Repaired generation recovery preflight

**Result:** PASS

- Candidate commit: `2552424196434a368e62448c78be7b8d69ff5aa9`
- Completed: 2026-07-31T02:49:42.361Z
- Generated source snapshot: `7de03f3ead3522a622d9f24bd6843ebcebfa871f30797432e0243678b04cc5e2`
- Certification inputs: `901d181bab9011cf21fd0bccb878de3275db15bd01f933daab80d0ee80359deb`
- Runtime: v24.16.0 on linux/x64
- CPU: AMD Ryzen 7 8845HS w/ Radeon 780M Graphics
- zvec: 0.6.0
- Deterministic embedding profile: `embedding-profile-b792b064200a2f84447527b7ecc8b076866e170003ae72c79c642b04608f068e`
- Manifest fingerprint: `56c0013a66586febb119dc1870015e1e406605ab2b1ff53187a540b421a53017`
- Starting snapshot fingerprint: `06eadab30ccbf4d51bc97a029874b298bb08eed3e76b633c68baf4d75d58295a`

## Production cardinality

Complete reopened validation crossed the observed 119,662-record failure boundary in every real-zvec store.

| Store              | Reopened count | Membership digest                                                  |
| ------------------ | -------------: | ------------------------------------------------------------------ |
| Lexical/source     |        239,328 | `36e0f2283963ae65bb23cdebd6ecea79ef9c7889dc7e4cd87726608ca4ee9e23` |
| Dense              |        119,664 | `2bdff5f46d8a1d7271f17df1a4817bd6fa0c9263463246f1d3a60b5de88c9e64` |
| Session projection |        119,785 | `409d4f9ce3fac2622ead51e14bb48959840a852475d425b1dec6e9df58050e27` |

The uninterrupted and twice-resumed builds used the same candidate commit, generated source snapshot, generation ID, manifest, profile, and source snapshot fingerprint. Their immutable comparable validation receipts and all membership digests agree.

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

- Disposable uninterrupted generation size: 3,137,622,909 bytes
- Disposable interrupted generation size: 3,137,637,866 bytes
- Production-cardinality preflight duration: 325735.774 ms

These values are reported without release thresholds.

## Reproduction

Create a clean worktree at `b04b350939de11ae56b67f8d1e8cce9ab0b12ec8`, then run:

```bash
PI_RECALL_SLOP_BASE_DIRECTORY=/path/to/clean/base-worktree npm run evidence:generation-recovery
```

The certifier ran:

- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-name-pattern=replacement generation bootstrap interruption model src/build-recall-fixed-snapshot-generation.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-name-pattern=malformed-source skips once|parser-looking operational failures fatal|non-source failure category fatal src/recall-physical-source-generation.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-name-pattern=generated incremental append, replay, branch, and deletion schedules match fresh rebuild membership src/transfer-incremental-recall-work-plan.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-name-pattern=crashed workers at every staging phase remain resumable and idempotent src/recall-background-index-conversation-service.test.ts`
- `/home/will/.local/share/mise/installs/node/24.16.0/bin/node --import tsx --test --test-name-pattern=standalone rebuild stops, resumes the same snapshot, and discards inactive work src/pi-session-recall-cli.test.ts`
- `npm test`
- `npm run typecheck`
- `npm run lint`
- `npm run format:check`
- `git diff --check`
- `slop-scan delta --base <exact-base-worktree> --head "$PWD" --fail-on added,worsened`
- `npm run evidence:generation-recovery`

All source files and stores were generated beneath disposable temporary roots. The run did not open original Pi session JSONL, a live or existing acceptance generation, the Octen endpoint, or production activation.

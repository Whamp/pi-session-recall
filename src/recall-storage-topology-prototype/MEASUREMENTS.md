# PROTOTYPE measurements — predictable recall storage topology

Generated 2026-07-29T16:57:15.505Z with `@zvec/zvec` 0.6.0 on disposable scratch data.

## Verdict

The three-store split is the simplest coherent topology, and its lookup, traversal, replay, sizing, validation, and resumable deletion protocols work on scratch evidence. Zvec cannot reliably certify its own crash recovery, but that does not block a rebuildable index. After a crash, verify and reuse valid embeddings, recompute only missing or damaged rows, and reserve a full rebuild for damage that cannot be isolated.

The smallest useful topology is three zvec collections plus application-owned generation metadata:

`lexical-source/` stores vector-free lexical evidence and immutable entry anchors; `dense/` stores only embedded occurrence IDs; `projections/` isolates mutable ingestion state. Stable occurrence IDs join lexical and dense evidence. Entry anchors provide direct source-neighborhood traversal without reopening JSONL.

This topology is acceptable for a rebuildable index. Activate only a reopened and validated generation, and keep the previous valid generation until activation succeeds. After an interrupted build, replay the source, reuse rows whose occurrence ID, embedding profile, and content checksum still match, and re-embed only missing or damaged rows. Rebuild the whole generation only when the damage cannot be isolated.

## Representative fixture

- Sessions: 28
- Immutable entry anchors: 672
- Lexical evidence occurrences: 784
- Dense evidence occurrences: 532
- Mutable projection rows: 56
- Stored embedding dimensions: 256
- Production metadata reference (filesystem/index-state only): 3,500 physical files, 2,346,218,035 source bytes, 1,175,836 logical chunks, 1995.36 source bytes per chunk.

The fixture reproduces the essential scalar/FTS provenance shape, 512-token-scale text, lexical-only tool evidence, dense-eligible conversation evidence, physical/logical projections, and a main plus abandoned branch. It is schema-representative, not a capacity forecast for private corpus text.

## Lifecycle

- Initial build: 68.5 ms
- Incremental append: 15.6 ms
- Optimize: 121.7 ms
- Close: 58.4 ms; return types: `undefined,undefined,undefined`
- Reopen: 32.3 ms
- Immutable replay rows skipped after checksum verification: 256
- Peak whole-generation bytes observed during optimize: 14,706,554

### After initial build

| Component        | Files | Apparent bytes | Allocated bytes |
| ---------------- | ----: | -------------: | --------------: |
| lexical-source   |    44 |      2,664,224 |       2,756,608 |
| dense            |    13 |      5,351,017 |       1,675,264 |
| projections      |    23 |         78,866 |         131,072 |
| whole generation |    82 |      8,094,367 |       4,571,136 |

### After append

| Component        | Files | Apparent bytes | Allocated bytes |
| ---------------- | ----: | -------------: | --------------: |
| lexical-source   |    46 |      3,082,886 |       3,182,592 |
| dense            |    16 |     10,632,727 |       2,863,104 |
| projections      |    25 |        105,613 |         159,744 |
| whole generation |    89 |     13,821,489 |       6,213,632 |

### After optimize

| Component        | Files | Apparent bytes | Allocated bytes |
| ---------------- | ----: | -------------: | --------------: |
| lexical-source   |    61 |      3,007,820 |       3,129,344 |
| dense            |    16 |      4,033,725 |       1,863,680 |
| projections      |    34 |        147,700 |         221,184 |
| whole generation |   113 |      7,189,509 |       5,222,400 |

## Validation

| Check                                                | Result |
| ---------------------------------------------------- | ------ |
| lexical/source logical rows                          | 1456   |
| dense logical rows                                   | 532    |
| projection logical rows                              | 56     |
| ordinary FTS canary                                  | true   |
| case-preserving identifier FTS canary                | true   |
| exact occurrence fetch                               | true   |
| dense nearest-neighbor canary                        | true   |
| dense membership/checksum subset of lexical evidence | true   |
| dense index completeness                             | 1      |

## Source neighborhood

Selected endpoint: `s0-main-17`

Path entries: `s0-main-5` → `s0-main-6` → `s0-main-7` → `s0-main-8` → `s0-main-9`

The expansion fetched 6 exact evidence occurrences, stayed in one logical session: true, and rejected an anchor from an unrelated branch: true.

## Replay amplification

“Physical write versions” counts successful engine write operations because zvec exposes only live logical `docCount`, not stale physical row count.

| Strategy            | Live logical rows | Successful physical write versions | Apparent bytes after initial + 3 replays    | Bytes after optimize |
| ------------------- | ----------------: | ---------------------------------: | ------------------------------------------- | -------------------: |
| blind_upsert        |               256 |                               1024 | 577,722 → 1,155,675 → 1,719,881 → 2,277,130 |              586,322 |
| fetch_verify_insert |               256 |                                256 | 577,722 → 584,770 → 584,771 → 584,771       |              584,776 |

Fetch-and-verify plus insert-if-absent keeps immutable replay source-driven. Blind upsert preserves logical IDs but writes another physical version on every replay.

## Crash and recovery faults

Each variant began as the same SIGKILL residue containing 32 successful writes.

| WAL variant | Open returned success | Logical rows | Exact rows fetched | Thrown error |
| ----------- | --------------------- | -----------: | -----------------: | ------------ |
| intact      | true                  |           32 |                 32 | none         |
| truncated   | true                  |           31 |                 31 | none         |
| flipped     | true                  |           15 |                 15 | none         |
| missing     | true                  |            0 |                  0 | none         |
| unreadable  | true                  |            0 |                  0 | none         |

The intact process-crash path recovered. Truncated, CRC-flipped, missing, and unreadable WAL variants can return a successfully opened collection with partial or zero rows. Therefore a successful zvec open cannot by itself certify a generation. On restart, compare stored rows with the generation manifest and source identities, reuse valid embeddings, and recompute only missing or mismatched rows.

## Resumable split-store deletion

- Target physical session projection: `physical-0`
- Dense rows removed first: 19
- Lexical evidence and entry-anchor rows removed second: 52
- Logical and physical projection rows removed last: 2
- A complete replay removed nothing: true
- Whole generation rename-and-delete completed: true
- Completion check: **reopen_and_verify** — Advance the checkpoint only after reopen verifies the expected IDs are absent. If broader validation fails, replay the source and reuse valid rows; rebuild the whole generation only when the damage cannot be isolated.

## Lightweight durability policy

Native durability established: **false**. The session JSONL files protect the data; durability work here protects the time spent embedding it.

Required application policy: Build outside the active generation, checkpoint bounded completed work, and record expected store counts and validation canaries in the manifest. On restart, reuse rows whose occurrence ID, embedding profile, and content checksum match, then embed only missing or mismatched rows. Keep the previous valid generation until replacement activation succeeds; rebuild the whole generation only when damage cannot be isolated.

Optional native improvement: A status-returning flush or close and fail-open WAL recovery could reduce restart verification and re-embedding time. Adopt it only if zvec can expose it without adding a second recovery protocol or meaningful storage overhead.

Deferred unless a concrete need justifies them: power-loss certification; disk-full recovery for every internal write phase; custom WAL repair; native binary attestation.

## Decision implication

Adopt the three-store topology with zvec as the incumbent. Preserve these crash probes. Add cheap checkpoints or native durability only when they clearly reduce restart verification or re-embedding time without materially increasing code complexity or index size. Do not build a second recovery system to protect data that already exists in the session JSONL files.

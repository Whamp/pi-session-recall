# PROTOTYPE measurements — predictable recall storage topology

Generated 2026-07-29T16:29:26.554Z with `@zvec/zvec` 0.6.0 on disposable scratch data.

## Verdict

The three-store split is the simplest coherent topology and its lookup, traversal, replay, sizing, validation, and resumable deletion protocols work on scratch evidence. Zvec 0.6.0 still fails the production durability threshold because close success is unobservable and corrupt recovery can silently open incomplete state.

The smallest useful topology is three zvec collections plus application-owned generation metadata:

`lexical-source/` stores vector-free lexical evidence and immutable entry anchors; `dense/` stores only embedded occurrence IDs; `projections/` isolates mutable ingestion state. Stable occurrence IDs join lexical and dense evidence. Entry anchors provide direct source-neighborhood traversal without reopening JSONL.

This topology is **not production-acknowledgeable with the current binding**. The retrieval and lifecycle shape works, but the persistence boundary does not.

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

- Initial build: 71.8 ms
- Incremental append: 16.6 ms
- Optimize: 126.4 ms
- Close: 59.0 ms; return types: `undefined,undefined,undefined`
- Reopen: 31.0 ms
- Immutable replay rows skipped after checksum verification: 256
- Peak whole-generation bytes observed during optimize: 17,303,418

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
| blind_upsert        |               256 |                               1024 | 577,722 → 1,149,018 → 1,719,680 → 2,277,130 |              586,322 |
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

The intact process-crash path recovered. Truncated, CRC-flipped, missing, and unreadable WAL variants can return a successfully opened collection with partial or zero rows. Native logs may report the corruption, but the application receives no failed open status.

## Resumable split-store deletion

- Target physical session projection: `physical-0`
- Dense rows removed first: 19
- Lexical evidence and entry-anchor rows removed second: 52
- Logical and physical projection rows removed last: 2
- A complete replay removed nothing: true
- Whole generation rename-and-delete completed: true
- Marker acknowledgement: **blocked** — Node closeSync() returns no native persistence status, and corrupt/truncated/missing WAL recovery can open successfully with partial or zero rows.

## Required native boundary

User-space durability guarantee established: **false**.

Required capability: A native status-returning flush/close boundary covering WAL fsync, scalar/vector/ID-map/index writes, atomically published and fsynced manifest files, and parent-directory fsync; recovery must fail open on WAL open/CRC/truncation and propagate every replayed operation status.

Still uncovered by this user-space prototype: power loss and volatile-device cache behavior; disk-full during each store and manifest phase; failed ID-map or vector/FTS index writes; interrupted recovery replay; native binary attestation to reviewed source.

## Decision implication

Keep zvec as the incumbent topology candidate, but do not select it for production acknowledgement. The next decision is binary: obtain the native persistence/recovery capability above and rerun these preserved probes, or compare another embedded daemon-free engine under the same three-store generation protocol.

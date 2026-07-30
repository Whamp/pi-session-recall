# PROTOTYPE measurements — rebuild recall without a persistent embedding cache

Generated 2026-07-30T00:34:38.673Z from commit `16236255e6b90aeb58c8049f8f04262a67e98abf` in `full` mode.

## Question

Under the proposed split-store generation, does a persistent embedding cache save enough time to justify its extra disk use, file count, and lifecycle?

This report does not apply a fixed acceptable-duration threshold. It compares relative phase time and storage on one reproducible size-stratified sample.

## Sample

- Physical source population: 3,552 files / 2.31 GiB
- Selected sample: 356 files / 219 MiB
- File sampling rate: 10.02%
- Byte sampling rate: 9.25%
- Sample identity: `553e40b04b59ee3106e79c2183169742642f1210d9ab4cc70a8fd00e635369c5`
- Seed: `pi-session-recall-issue-112-no-cache-v1`

| Imported physical format | Accepted files |
| ------------------------ | -------------: |
| canonical_jsonl          |            347 |
| pi_v1_linear             |              7 |
| pi_session_reuse_history |              2 |

## Lane time

| Lane                         |     Total |    Import | Projection preparation | Vector resolution | Lexical write | Dense write | All optimize |
| ---------------------------- | --------: | --------: | ---------------------: | ----------------: | ------------: | ----------: | -----------: |
| text_only                    | 35.83 min | 16.93 min |              16.95 min |            0.2 ms |      1.64 min |      0.7 ms |      11.92 s |
| cold_no_cache                | 58.33 min | 16.48 min |              16.64 min |         23.11 min |      1.62 min |      2.37 s |      13.51 s |
| previous_generation_transfer | 35.19 min | 16.52 min |              16.55 min |            2.36 s |      1.59 min |      2.41 s |      12.65 s |
| warm_shared_cache            | 35.26 min | 16.51 min |              16.51 min |           11.09 s |      1.55 min |      2.37 s |      12.57 s |

## Logical work

| Lane                         | Accepted sources | Logical sessions | Anchors | Lexical evidence | Dense evidence | Model requests | Transferred vectors | Cache hits |
| ---------------------------- | ---------------: | ---------------: | ------: | ---------------: | -------------: | -------------: | ------------------: | ---------: |
| text_only                    |              356 |              358 |   50239 |           123185 |          22854 |              0 |                   0 |          0 |
| cold_no_cache                |              356 |              358 |   50239 |           123185 |          22854 |           1378 |                   0 |          0 |
| previous_generation_transfer |              356 |              358 |   50239 |           123185 |          22854 |              0 |               22854 |          0 |
| warm_shared_cache            |              356 |              358 |   50239 |           123185 |          22854 |              0 |                   0 |      22854 |

## Allocated storage after optimize

| Lane                         | Lexical/source |    Dense | Projections | Whole generation | Peak generation during optimize |
| ---------------------------- | -------------: | -------: | ----------: | ---------------: | ------------------------------: |
| text_only                    |        715 MiB | 1.12 MiB |    28.1 MiB |          744 MiB |                        1.24 GiB |
| cold_no_cache                |        715 MiB |  108 MiB |    28.1 MiB |          851 MiB |                        1.43 GiB |
| previous_generation_transfer |        715 MiB |  108 MiB |    28.2 MiB |          851 MiB |                        1.43 GiB |
| warm_shared_cache            |        715 MiB |  108 MiB |    28.1 MiB |          851 MiB |                        1.43 GiB |

Shared cache after complete seed: 151 MiB across 19,322 files.

Cache seeding itself took 17.48 min total, including 16.37 min source import, 2.19 s dense reads, and 1.08 min production-format cache resolution/writes.

## Relative comparisons

- Cold no-cache total / text-only total: **1.628×**
- Previous-generation transfer total / warm shared-cache total: **0.998×**
- Previous-generation transfer vector resolution / warm-cache vector resolution: **0.213×**
- Warm-cache vector resolution / previous-generation transfer: **4.705×**
- Shared-cache allocated bytes / dense-store allocated bytes: **1.394×**
- Shared-cache allocated bytes per dense occurrence: **6.76 KiB**
- Shared-cache allocated bytes per unique embedding input: **8.00 KiB**

## Validation

| Lane                         | All canaries | Lexical/source rows | Dense rows | Projection rows after reopen | Physical projections |
| ---------------------------- | -----------: | ------------------: | ---------: | ---------------------------: | -------------------: |
| text_only                    |         true |              173424 |          0 |                          714 |                  356 |
| cold_no_cache                |         true |              173424 |      22854 |                          714 |                  356 |
| previous_generation_transfer |         true |              173424 |      22854 |                          714 |                  356 |
| warm_shared_cache            |         true |              173424 |      22854 |                          714 |                  356 |

## Controls

- Embedding requests were strictly sequential with request concurrency 1 and batch size 16.
- Only `cold_no_cache` called the served model. Transfer and warm-cache lanes failed on any miss.
- The live 2,560-dimension Octen output was projected to the first 1,024 dimensions and L2-normalized before storage.
- Scratch source files were evicted with file-specific `POSIX_FADV_DONTNEED` before every lane.
- The previous dense store and shared cache were separately evicted before their measured reads.
- Original Pi sessions, the production recall generation, and the production embedding cache were never opened for writing.

## Caveats

- The sample is one deterministic 10% size-stratified draw, not repeated random samples.
- The production projection builder currently performs a second import/tokenization pass; projection preparation reports that cost separately.
- POSIX_FADV_DONTNEED is an advisory, file-scoped cache control and does not evict directory metadata.
- Only one cold model pass was run; model-service load and co-resident GPU activity may affect its absolute time.
- The split lexical schema is based on the accepted prototype plus current provenance fields, but issue 112 has not frozen every final scalar column.
- All sources are treated as eligible for this upper-bound rebuild comparison; the future recall horizon may reduce row counts.

## Measured implication

The persistent cache removed the cold model phase, but it did not improve the normal replacement path. Verified transfer completed the whole replacement in 35.19 min, while the warm-cache replacement took 35.26 min. Direct vector transfer was 4.705× faster than reading the cold file cache.

The cache added 151 MiB and 19,322 files—1.394× the allocated size of the searchable dense store it duplicated.

## Recommendation

Do not retain a persistent embedding cache in the new topology. Use the active or interrupted generation as the verified vector source, deduplicate cold builds against vectors already written to their dense store, and recompute from immutable sessions when no valid vector source survives.

Reconsider only if future evidence shows that cold source-only rebuilds happen often enough for their avoided model time to outweigh a second vector corpus and its lifecycle.

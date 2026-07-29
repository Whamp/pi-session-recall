# Zvec 0.6.0 for predictable split-store recall generations

**Issue:** [#108](https://github.com/Whamp/pi-session-recall/issues/108), under map [#107](https://github.com/Whamp/pi-session-recall/issues/107)  
**Research date:** 2026-07-29  
**Evaluated implementation:** `@zvec/zvec` 0.6.0, locked by `package.json` and `package-lock.json`; Node tag [`96fe5e0`](https://github.com/zvec-ai/zvec-node/tree/96fe5e09619369179d783d1c517cac33895c2b81), which pins core zvec commit [`cb42297`](https://github.com/alibaba/zvec/tree/cb422972981a2ed4735e9cfb419397c0a2383f02).

## Verdict

**Fits with architectural conditions.** Zvec can remain the embedded engine if one recall generation owns separate zvec collection directories for lexical/scalar evidence, dense evidence, and mutable session projections, plus generation-scoped manifests and index state. This split is necessary: zvec supports collections with no vectors, but a document in a collection that declares a dense vector cannot omit that vector or set it to null. The current zero-vector workaround therefore violates the map’s “lexical-only evidence must not pay dense-vector storage” constraint.

Zvec itself does not provide a transaction, join, union, or search across collections. The coherent-generation invariant must remain an application protocol: stable evidence occurrence IDs connect stores; writes are deterministic and idempotent; immutable evidence is written before the physical session projection checkpoint; every per-document status and every close is checked; marker acknowledgement follows a read-back of that checkpoint; build validation covers every owned store before the active generation pointer changes. This extends, rather than replaces, the accepted evidence-before-projection protocol in `src/commit-incremental-recall-transfer.ts:212-301,304-419` and the atomic pointer protocol in `src/rebuild-recall-generation.ts:461-596`.

Stable physical growth is credible under the map’s operating model. Inserts append physical rows. Updates, upserts, and deletes retain stale physical rows behind delete state until optimization rewrites segments. Immutable evidence avoids that churn; the small mutable projection collection grows with projection revisions; a clean replacement generation resets accumulated physical history. Zvec optimization can reclaim deleted versions, but it should remain explicit rebuild/maintenance work, as ADR 0004 already requires (`docs/adr/0004-ingest-recall-incrementally-outside-pi.md:69-75`). The current 24.39 GB legacy zvec collection is evidence of the old mixed-store, repeated-rewrite design—not a normal-growth forecast.

Two boundaries must be explicit:

1. **Dense dimensions:** core zvec 0.6.0 accepts dense dimensions only in `(0, 20000]` ([source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/common/schema.cc#L87-L106)). Conversation Recall must not invent a universal MRL cap. It should test the user-selected stored dimensions against the selected engine. A profile above 20,000 dimensions makes zvec incompatible and triggers alternative-engine evaluation; it must not be silently truncated.
2. **Acknowledgement durability:** official documentation promises WAL persistence through process crash and power failure ([overview](https://zvec.org/mdx/en/docs/db.md#key-features)), but the public Node API exposes no commit/flush method. Source appends WAL records without flushing at the default threshold and flushes the WAL and stores on collection close ([WAL source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/storage/wal/local_wal_file.cc#L29-L49), [collection flush source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L2217-L2323)). Treat successful close plus checkpoint read-back—not an individual successful status—as Conversation Recall’s durable acknowledgement boundary.

## Evidence classes

- **Documented guarantee** means the official Markdown linked from the complete [`llms.txt`](https://zvec.org/llms.txt) index states the behavior.
- **Source behavior** means zvec Node/core v0.6.0 implements it at the cited commit.
- **Local observation** means a read-only production measurement or disposable `/tmp` probe on this host using the locked package.
- **Inference** means the architecture follows from documented/source behavior but is not a zvec guarantee.
- **Unknown** identifies a behavior that documentation, source inspection, and bounded probes did not establish strongly enough.

All 47 documentation pages in the 2026-07-29 `llms.txt` index were fetched through their `/mdx/en/docs/{path}.md` URLs. The database pages supplied the material evidence; the AI integration pages add no storage guarantees.

## Requirement-by-requirement evidence matrix

| Requirement                                   | Finding and evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Architectural consequence                                                                                                                                                                                                                                                                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Scalar/FTS-only documents                  | **Documented:** “No vector fields are required” and “Zvec supports FTS-only collections” ([FTS guide](https://zvec.org/mdx/en/docs/db/data-operations/query/fts.md#prerequisites)). The Node schema makes `vectors` optional ([Node source](https://github.com/zvec-ai/zvec-node/blob/96fe5e09619369179d783d1c517cac33895c2b81/src/index.d.ts#L1100-L1117)). **Local observation:** a zero-vector-field collection inserted, exact-filtered, BM25-searched, fetched, closed, moved, and reopened successfully.                                                                                                                                                                                                                                                                                                                                                                       | Put lexical evidence, provenance, source-neighborhood fields, and projections in vector-free collections.                                                                                                                                                                                                                                 |
| 2. Missing/null vectors per document          | **Source behavior:** Node accepts an omitted `vectors` object syntactically, then core schema validation rejects a missing required vector. Upstream tests require that rejection ([test](https://github.com/zvec-ai/zvec-node/blob/96fe5e09619369179d783d1c517cac33895c2b81/tests/data/operations.test.ts#L338-L346)). **Local observation:** omitted, `{}`, and `null` each threw `ZVEC_INVALID_ARGUMENT`; no document was inserted.                                                                                                                                                                                                                                                                                                                                                                                                                                               | A mixed collection necessarily gives every lexical-only document a real dense vector. Split collections are mandatory, not an optimization.                                                                                                                                                                                               |
| 3. Lookup, filters, FTS, dense, combination   | **Documented:** `fetch()` is direct ID lookup ([fetch](https://zvec.org/mdx/en/docs/db/data-operations/fetch.md)); SQL-like scalar filters work with or without an index, with indexed fields faster ([filter](https://zvec.org/mdx/en/docs/db/data-operations/query/filter.md)); FTS supplies BM25, phrase, and boolean retrieval and accepts scalar filters ([FTS](https://zvec.org/mdx/en/docs/db/data-operations/query/fts.md)); dense search accepts scalar filters ([vector search](https://zvec.org/mdx/en/docs/db/data-operations/query/single-vector.md#common-parameters)). FTS and vector search are mutually exclusive in one route, and cross-collection queries are unsupported ([FTS constraints](https://zvec.org/mdx/en/docs/db/data-operations/query/fts.md#constraints), [data modeling](https://zvec.org/mdx/en/docs/db/concepts/data-modeling.md#collections)). | Run lexical, identifier, and dense routes separately; merge by stable evidence occurrence ID in application code. This preserves component ranks as ADR 0002 requires (`docs/adr/0002-fuse-hybrid-retrieval-in-application-code.md:1-3`).                                                                                                 |
| 4. Physical layout                            | **Documented:** each collection is an independent, self-contained directory and can be relocated ([persistence](https://zvec.org/mdx/en/docs/db/concepts/data-modeling.md#persistence)). **Local observation:** a two-document scalar/FTS collection contained 36 files: manifest, lock, delete snapshot, ID-map RocksDB, scalar IPC, inverted-index RocksDB, and FTS RocksDB.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | “Single package” and “in-process” are accurate; “single file” is false. A generation must own every collection directory and its JSON metadata.                                                                                                                                                                                           |
| 5. Version behavior and growth                | **Source behavior:** update/upsert mark the old physical document deleted and insert a new one; delete marks the row and removes its primary-key mapping ([segment source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L883-L900)). Optimization compacts segments, filters deleted rows once the delete ratio exceeds 30%, publishes a new manifest, and removes old segments ([optimize source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L779-L914), [threshold](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/common/constants.h#L37-L59)).                                                                                                                                                                    | Insert immutable evidence once. Accept linear, source-driven stale-row growth only in small projections. Rebuild generations deliberately; do not add automatic compaction for development churn.                                                                                                                                         |
| 6. Durability and partial writes              | **Documented:** WAL survives crash/power loss ([overview](https://zvec.org/mdx/en/docs/db.md#key-features)). Batch validation failure updates nothing; after validation, documents are attempted independently and each returns a status ([insert](https://zvec.org/mdx/en/docs/db/data-operations/insert.md#insert-a-batch-of-documents), [upsert](https://zvec.org/mdx/en/docs/db/data-operations/upsert.md#upsert-a-batch-of-documents), [update](https://zvec.org/mdx/en/docs/db/data-operations/update.md#update-a-batch-of-documents)). **Source behavior:** the batch loop returns one status per document and does not roll back earlier operational successes ([collection source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L1420-L1469)).                                                                        | Check status count, order, code, and success for every write. There is no batch transaction and no multi-collection atomicity. Replay from the last physical session projection checkpoint repairs partial progress.                                                                                                                      |
| 7. Concurrency/processes                      | **Documented:** multiple processes may open one collection read-only; writes are single-process exclusive ([overview](https://zvec.org/mdx/en/docs/db.md#key-features), [open](https://zvec.org/mdx/en/docs/db/collections/open.md#parameters)). **Source behavior:** read-only opens take a shared nonblocking file lock; writable opens take an exclusive nonblocking lock ([lock source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L2026-L2051)).                                                                                                                                                                                                                                                                                                                                                                        | The detached worker must own all generation write access during one bounded write window. Search must close/read-wait/reopen around that window; it cannot share a collection with the writer. Separate replacement-generation writes do not block active-generation readers.                                                             |
| 8. Indexing, validation, activation, deletion | **Documented:** new vectors remain in a flat buffer; `optimize()` builds configured indexes, while `stats.docCount` and `stats.indexCompleteness` expose count and per-vector completeness ([optimize](https://zvec.org/mdx/en/docs/db/collections/optimize.md#check-indexing-status)). Destroy removes a collection directory ([destroy](https://zvec.org/mdx/en/docs/db/collections/destroy.md)). Zvec has no generation abstraction.                                                                                                                                                                                                                                                                                                                                                                                                                                              | Validate schema, exact counts/membership, dense completeness `1`, source/provenance coverage, profile metadata, and embedding canary across all stores. Close them, then atomically replace the external checksummed pointer. Retire by deleting the whole inactive generation directory only after reader exclusion and rollback policy. |
| 9. Current corpus                             | **Local observation:** 3,542 JSONL files occupy 2.462 GB. The legacy collection has 1,175,836 logical documents, 2,560-dimensional FP32 vectors, 0.9511 vector-index completeness, 289 files, and 24.395 GB. At least 100,000 documents are lexical-only; the exact count is censored by zvec’s 100,000 `topk` limit.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | The current layout contains at least 1.024 GB of raw zero-vector values before index overhead (`100,000 × 2,560 × 4`). It demonstrates the waste split stores remove, but not steady-state growth because it includes repeated development rebuild/upsert history.                                                                        |
| 10. Lightweight embedded preference           | **Documented:** zvec runs in-process with no server or daemon ([overview](https://zvec.org/mdx/en/docs/db.md)); the installed Node package loads a native addon from the package or platform binding (`node_modules/@zvec/zvec/src/index.js:5-19`). **Observation:** all probes ran as ordinary Node processes with only local directories.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | The preference is satisfied as an operating property, not a guarantee of one file, one transaction, or zero maintenance. Zvec remains acceptable only under the listed conditions.                                                                                                                                                        |

## Detailed findings

### Split-store shape and source-neighborhood support

The minimum coherent generation should own at least:

```text
generations/<id>/
├── lexical/              # all recall evidence, scalar locators, FTS, graph-path data
├── dense/                # dense-searchable evidence only
├── session-projections/  # small mutable physical/logical projections
├── index-state.json
└── index-manifest.json
```

A separate identifier collection is optional. One scalar string field cannot simultaneously carry incompatible inverted and FTS indexes: core rejects replacing one non-vector index type with another ([source](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L460-L495)). Duplicating analyzed ordinary text and case-preserving identifier text inside the lexical collection, as the current integration does (`src/zvec-conversation-store.ts:169-186`), remains simple and supported.

The lexical collection can support source-neighborhood expansion without reopening JSONL. Store exact logical session ID, physical session projection ID/path locator, evidence occurrence ID, entry ID, parent entry ID, selected child/branch relation, source lines/blocks, and contribution IDs as scalars. Index hard equality filters with inverted indexes; use direct `fetch()` once the expansion has stable IDs. Zvec only supplies lookup and filtering. The application must enforce one logical session and one selected graph path. This is an inference from zvec’s exact lookup/filter capabilities and map #107, not a native graph guarantee.

Dense documents should use the same evidence occurrence ID as their lexical counterparts. Dense search returns IDs and scores; one batched fetch from lexical storage restores full provenance. Cross-collection result combination remains deterministic application logic. No source text needs duplication in dense storage unless query output requires it.

### Physical growth

The zero-vector comparison used identical 1,000-document FTS/scalar payloads at the current 2,560 dimensions. The vector-free collection occupied 511,395 bytes after optimization. The zero-vector collection occupied 12,631,538 bytes—24.7 times as much—and its vector index alone was 12,120,064 bytes. This is a bounded local observation, not a universal byte-per-document formula; allocator, segment, index, text, and document-count effects change the ratio.

A projection-shaped scalar collection with 1,000 one-kilobyte documents occupied 1,206,506 bytes. Rewriting all IDs once grew it to 2,316,267 bytes; a second rewrite grew it to 3,438,106 bytes while logical `docCount` stayed 1,000. Explicit optimization reduced it to 1,241,324 bytes. This matches source behavior: updates preserve old physical rows as deleted versions, and optimization can rewrite live rows.

Correct steady state therefore has predictable components:

- lexical physical bytes grow with newly eligible immutable evidence and FTS/inverted-index overhead;
- dense physical bytes grow only with dense-searchable evidence, selected stored dimensions, FP32 values, and vector-index overhead;
- projection bytes grow with projection payload revisions, not with all historical evidence;
- confirmed source deletion creates deleted physical rows until the next explicit rebuild/optimization;
- replacement generation construction temporarily requires old active, new replacement, and optional rollback generations at once.

The engine does not expose a physical-byte estimator. A representative build must measure coefficients for lexical text, dense dimensions, projection revisions, and concurrent generation retention before setup presents a storage estimate. That is a sizing unknown, not a functional blocker.

### Durability, recovery, and coherent commits

Zvec’s source writes each operation to a CRC-protected WAL before mutating in-memory indexes ([operation path](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L903-L971)). On a write-capable reopen, it replays WAL operations and keeps the recovered WAL attached so close or optimization persists the recovered components ([recovery](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L4205-L4320)). A disposable SIGKILL probe observed 32 successful statuses and a WAL; a read-only reopen returned zero documents and logged failed read-only ID-map writes, while a later writable reopen recovered all 32 and a subsequent read-only reopen saw them. This confirms ADR 0004’s rule that interrupted writer state needs write-capable recovery (`docs/adr/0004-ingest-recall-incrementally-outside-pi.md:9-12,45-51`). It also means read-only open is not a recovery operation.

No zvec primitive atomically commits lexical, dense, and projection collections. The safe protocol is:

1. retain generation-independent work markers;
2. under the exclusive write window, idempotently write lexical rows, then dense rows;
3. check every status and close both collections successfully;
4. write logical projections, then the physical session projection checkpoint last;
5. close projection storage;
6. reopen read-only, verify checkpoint coverage and selected evidence IDs/counts, then acknowledge markers.

A crash at any earlier point leaves markers unacknowledged. Replay overwrites the same stable IDs and advances only after the checkpoint. Stale physical versions cost space but do not violate logical coherence. This generalizes the current evidence-before-projection and read-back protocol (`src/commit-incremental-recall-transfer.ts:263-299,326-419`).

The official power-failure statement is broader than the source-level commit boundary visible to Node. Whether the WAL write is force-synced before a successful per-document status is **unknown**. Conversation Recall need not depend on it if marker acknowledgement follows checked close and checkpoint read-back, but a focused kill/power-fault test should remain an engine-upgrade acceptance test.

### Build, validation, activation, rollback, and deletion

Zvec’s `stats` is necessary but insufficient. `docCount` reports live logical documents, and `indexCompleteness` reports only vector indexing. It does not prove cross-store equality, provenance coverage, profile identity, lexical-index correctness, or graph-path completeness. Generation validation must therefore enumerate expected stable IDs from generation-scoped index state/projections and verify:

- exact schemas and schema versions for lexical, dense, and projections;
- exact lexical and dense counts, with dense membership equal to the lexical `isDenseSearchable=true` set;
- `indexCompleteness[embedding] === 1` after optimization;
- direct fetch of deterministic canaries and all expected projection IDs;
- approved physical/logical projection and eligible-contributor coverage;
- manifest fingerprint, user-selected stored dimensions, embedding profile/canary, tokenizer/FTS policy, provenance schema, and source-neighborhood schema;
- no pending partial-write/recovery marker inside the generation.

The current implementation already closes the build before read-only validation, checks evidence count and approved membership, checks physical and logical projections, and fingerprints the manifest (`src/recall-conversation-service.ts:2483-2583`). The destination must widen that validator to every split store.

Activation should remain external to zvec. The repository durably writes checksummed state through temp-file write, file sync, rename, and directory sync (`src/recall-generation-state.ts:467-510`), then search resolves only the pointer-selected generation (`src/recall-generation-state.ts:613-633`). Rebuild closes and validates before the cutover write window, detects concurrent pointer/registry changes, and updates registry-pointer-registry with recovery required after any partial cutover (`src/rebuild-recall-generation.ts:461-596`). This supplies whole-generation activation and rollback even though zvec has no multi-collection transaction.

For deletion, do not call `destroySync()` collection by collection as the generation protocol. First prove the generation is neither active, building, replay-pending, nor retained rollback material; exclude readers; then remove its entire directory. The current collector applies those state checks and recursively removes the generation (`src/collect-retired-recall-generations.ts:46-128`). Whole-directory deletion preserves the invariant better than partially destroying owned collections.

## Conditions and remaining unknowns

Zvec fits only if all of these conditions become specification requirements:

1. Lexical/scalar and dense evidence use separate vector-free and vector-bearing collection directories.
2. Stable evidence occurrence IDs and exact source locators join results in application code; zvec is never treated as a cross-collection query engine.
3. Source-neighborhood data lives in the lexical/scalar index and expansion enforces one logical-session graph path.
4. Every zvec batch status and collection close is checked. The physical session projection checkpoint is written last and read back before marker acknowledgement.
5. Interrupted writers always receive an exclusive write-capable recovery open before search resumes.
6. One writer owns all active-generation collections during the bounded write window; readers use read-only opens outside it.
7. Replacement builds optimize and close every collection before validation. Validation proves cross-store membership and profile/provenance identity, not merely counts.
8. Activation and rollback select one closed, validated generation through the existing checksummed pointer protocol. No collection is activated independently.
9. Retired generations are removed as whole directories only after state and reader exclusion checks.
10. Setup tests the chosen stored dimensions against zvec’s 20,000-dimension engine limit without imposing a project-wide MRL cap or silently truncating.
11. Automatic optimization is not introduced. Immutable evidence and source-driven projection revisions define expected growth; explicit rebuild/maintenance handles exceptional churn.

Remaining unknowns are quantitative: corpus-scale bytes per lexical document, bytes per dense document by selected dimensions/index parameters, projection revision rate, optimize peak temporary space, and close/recovery latency at the target batch size. The visible Node API also leaves pre-close power-loss force-sync semantics unresolved. These require a representative disposable-generation prototype before rollout, but none forces replacement of zvec for profiles at or below 20,000 dimensions.

## Commands used for local observations

All production paths were read only. Disposable writes were confined to `/tmp`.

```bash
# Locked package and complete official documentation
npm ci --ignore-scripts
node -p "require('./node_modules/@zvec/zvec/package.json').version"
curl -fsSL https://zvec.org/llms.txt -o /tmp/zvec-docs/llms.txt
# Each of the 47 /en/docs/ links was fetched as /mdx/en/docs/{path}.md.
git clone --depth 1 --branch v0.6.0 https://github.com/zvec-ai/zvec-node.git /tmp/zvec-node-v0.6.0
git -C /tmp/zvec-node-v0.6.0 submodule update --init --depth 1

# Session corpus and recall-tree metadata; no session content was read
python - <<'PY'
from pathlib import Path
import statistics
r=Path.home()/'.pi/agent/sessions'; f=[p for p in r.rglob('*.jsonl') if p.is_file()]
s=[p.stat().st_size for p in f]
print(len(f), sum(s), int(statistics.median(s)), sorted(s)[int(.95*(len(s)-1))], max(s))
PY
find ~/.pi/agent/recall/zvec -type f -printf '%s\t%P\n' | sort

# Read-only legacy collection snapshot
node --input-type=module <<'JS'
import { ZVecOpen } from '@zvec/zvec';
const c=ZVecOpen(`${process.env.HOME}/.pi/agent/recall/zvec`,{readOnly:true});
console.log(c.stats, c.schema.vectors());
console.log(c.querySync({filter:'isDenseSearchable = false',topk:100000,outputFields:[],includeVector:false}).length);
c.closeSync();
JS

# Disposable behavior/size probes
node /tmp/zvec-0.6.0-probe.mjs
node /tmp/zvec-0.6.0-growth-probe.mjs
node /tmp/zvec-crash-writer.mjs /tmp/zvec-0.6.0-crash-probe &
kill -9 $!
# Then ZVecOpen(path,{readOnly:true}), ZVecOpen(path), close, and read-only reopen.
```

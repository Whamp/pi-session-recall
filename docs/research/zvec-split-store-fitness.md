# Zvec fitness for predictable split-store recall generations

**Ticket:** [#108 — Determine whether zvec can provide predictable split-store recall generations](https://github.com/Whamp/pi-session-recall/issues/108)  
**Map:** [#107 — Design a predictable rebuildable recall storage architecture](https://github.com/Whamp/pi-session-recall/issues/107)  
**Evaluated package:** `@zvec/zvec` 0.6.0  
**Pinned upstream source:** zvec-node [`96fe5e0`](https://github.com/zvec-ai/zvec-node/tree/96fe5e09619369179d783d1c517cac33895c2b81), core zvec [`cb42297`](https://github.com/alibaba/zvec/tree/cb422972981a2ed4735e9cfb419397c0a2383f02)  
**Synthesis basis:** primary research memo at `6ebdf31`, OpenAI source-fidelity review supplied for this synthesis, and GLM source-fidelity review at `1190447`

## Executive answer

Zvec is a **promising candidate, but its fitness is not yet established for production acknowledgement and recovery**.

It satisfies the basic storage shape. Zvec runs embedded and daemon-free; persists each collection in its own relocatable directory; supports vector-free FTS/scalar collections; supports separate dense collections; and provides the lookup, filtering, FTS, dense-search, and collection-local locking primitives needed for an application-managed recall generation. Because a document cannot omit a vector when its collection schema declares one, the map requires separate lexical and dense collections. Stable evidence-occurrence IDs must join their results in application code.

The proposed generation layout is therefore sound:

```text
generations/<id>/
├── lexical/              # immutable evidence, FTS, locators, graph fields
├── dense/                # dense-searchable evidence only
├── session-projections/  # small mutable projections
├── index-state.json
└── index-manifest.json
```

The evidence/projection split already exists in the project. The new architectural change is to replace the mixed evidence collection with separate lexical and dense collections and remove the zero-vector workaround.

Two engine boundaries prevent an unconditional fit verdict:

1. The Node 0.6.0 binding’s `closeSync()` returns no close or flush status. Core destruction calls `Close()` but discards its result. The application therefore cannot implement the primary memo’s requirement that “every collection close is checked.” A post-close read-back can detect some failures, but it cannot prove power-loss durability because reads may be satisfied from the page cache and manifest publication does not visibly sync both the file and parent directory.
2. Recovery is only partly trustworthy. A disposable SIGKILL probe demonstrated one successful process-crash path, but core recovery has paths that treat WAL-open failure as success and ignore statuses from replayed operations. The probe did not exercise truncated or corrupt WAL, disk-full, failed index/ID-map writes, interrupted recovery, or power loss.

The reviews disagree on the severity of these boundaries. GLM judged the primary memo high-fidelity and accepted checked close plus read-back as conservative. OpenAI found that the Node API cannot actually check close and that recovery error swallowing makes the proposed acknowledgement boundary unsafe. The source findings are compatible: close flushes internally, but the binding does not propagate its outcome; recovery works on the tested happy path, but not all failures propagate. Because these gaps govern acknowledgement of source-processing markers, this report adopts the stricter production verdict.

Zvec should remain the preferred engine while a focused prototype tests or removes these blockers. Adoption requires either an upstream/native status-returning flush or close boundary plus recovery fixes, or fault evidence strong enough to define a narrower safe protocol. If that cannot be established, compare another permitted embedded, daemon-free engine under the same generation-contained constraints.

## Confidence and evidence limits

**Confidence in the split-store requirement: high.** Official documentation, Node types/tests, core source, and disposable probes agree that vector-free collections work and vector-bearing collections require vectors on every non-nullable document.

**Confidence in application-managed generation composition: high.** Zvec collections are separate directories and zvec offers no cross-collection transaction, join, union, search, or generation abstraction.

**Confidence in predictable physical growth: medium-low pending a representative build.** Source behavior explains why immutable evidence plus small mutable projections should limit churn. The only controlled size test used 1,000 documents, and the scripts were not preserved in the repository. The 24.4 GB production collection reflects a mixed layout and development history, not a steady-state coefficient.

**Confidence in process-crash recovery: medium for the tested happy path; low for fault coverage.** The SIGKILL sequence recovered 32 rows after a writable reopen. Source inspection found recovery errors that may be swallowed.

**Confidence in power-loss durability: low.** Documentation makes a broad power-failure claim, but the reviewed Node/core boundary does not establish when the application may safely acknowledge durable work.

The source review is of upstream code corresponding to release 0.6.0. The installed package selects a prebuilt native binding. No build attestation or reproducible build tied that binary exactly to the reviewed core commit, so exact native-binary provenance remains unverified.

### Material reviewer disagreements

- **Overall fit:** GLM retained “fits with architectural conditions” and described the memo as high-fidelity. OpenAI judged that fit unestablished because the application cannot observe close failure and recovery suppresses some errors. This synthesis accepts the source findings from both and adopts OpenAI’s stricter production threshold.
- **Acknowledgement boundary:** GLM considered close plus read-back a valid conservative boundary, while noting that `optimizeSync()` also force-persists. OpenAI found close plus read-back unsafe because Node discards close status, manifest publication lacks a demonstrated file-and-directory sync boundary, and read-back may hit page cache. The focused prototype must resolve this disagreement; the report does not treat the boundary as proven.
- **Recovery and research access:** GLM judged the local methods read-only-safe and treated writable recovery as a confirmed load-bearing behavior. OpenAI emphasized that read-only open enters recovery and that replay errors may be swallowed. Both agree that the observed SIGKILL sequence succeeded and that writable recovery is required. They disagree on how far that evidence generalizes. This report limits it to the tested process-crash path.
- **Growth:** GLM called the growth model plausible pending a representative build. OpenAI additionally found that repeat upserts defeat physical idempotence. This report includes both constraints and does not claim corpus-scale predictability.

Production measurements have the same method dispute. Filesystem `stat`, `find`, and `du` are read-only. Direct `ZVecOpen(..., {readOnly: true})` was not proven non-mutating: source enters recovery when a writing segment exists even on read-only open. No production mutation was established, but future research should use a snapshot/copy or first prove that no WAL recovery is pending.

## Requirement matrix

| Requirement                                             | Evidence class and finding                                                                                                                                                                        | Status / consequence                                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedded, in-process, daemon-free                       | **Documented:** zvec is an in-process database with no server.                                                                                                                                    | **Meets.** This does not imply one file or one transaction.                                                                                                             |
| Lexical-only evidence avoids dense storage              | **Documented/source/observed:** FTS-only collections need no vector schema. Missing, empty, or null vectors are rejected when the schema declares a required dense vector.                        | **Meets only with split collections.** Mixed lexical/dense storage fails the requirement.                                                                               |
| User-selected MRL dimensions without a project-wide cap | **Source:** ordinary dense indexes cap dimensions at 20,000; `HNSW_RABITQ` accepts 64–4,095 and constrains dtype/metric; sparse vectors cap at 16,384.                                            | **Conditional.** Validate against the selected index type. An explicit model-support override cannot override the engine/index hard limit. Never truncate silently.     |
| Lexical, identifier, dense, and exact lookup routes     | **Documented:** FTS, scalar filters, direct fetch, and dense search exist. FTS and vector search cannot be combined in one route.                                                                 | **Meets through application fusion.** Preserve component ranks and merge by stable occurrence ID.                                                                       |
| One coherent generation across stores                   | **Documented/source:** collections are independent; locks and commits are collection-local; no cross-collection transaction exists.                                                               | **Application responsibility.** Pin the active pointer and hold the shared application lock across all reads. Activate only a fully closed and validated directory set. |
| Incremental partial-write recovery                      | **Source/observed:** batches are non-transactional; one SIGKILL path recovered after writable reopen. Recovery contains error-swallowing paths.                                                   | **Not established.** Markers must remain generation-independent; writable recovery is necessary but not proven sufficient.                                              |
| Durable acknowledgement                                 | **Documented:** WAL claims crash/power-loss persistence. **Source:** WAL appends do not sync by default; close flushes internally; Node cannot report close failure; manifest sync is incomplete. | **Does not yet meet.** Do not acknowledge work as power-loss durable from per-document status or unchecked close/read-back alone.                                       |
| Bounded write batches                                   | **Source:** a call with more than 1,024 documents is rejected. Project incremental ingest already caps evidence batches at 32.                                                                    | **Conditional.** Replacement builds must chunk writes to at most 1,024 and inspect every returned status.                                                               |
| Predictable growth without automatic compaction         | **Source/observed/inferred:** upsert/update create stale rows; optimization reclaims them; immutable inserts avoid this churn.                                                                    | **Plausible, not demonstrated at corpus scale.** Keep optimization explicit and measure a representative generation.                                                    |
| Source-neighborhood expansion without reopening JSONL   | **Inference:** lexical fields plus fetch/filter can store and retrieve graph locators; zvec provides no graph traversal.                                                                          | **Missing coverage.** Prototype canonical branch endpoints, occurrence cardinality, traversal limits, eligibility, and provenance.                                      |
| Complete pre-activation validation                      | **Documented/source:** stats expose logical count and dense index completeness, not cross-store coherence or FTS correctness.                                                                     | **Application responsibility.** Add membership validation plus ordinary-FTS and identifier-FTS canaries.                                                                |
| Confirmed-source deletion across split stores           | Existing code covers resumable deletion for one evidence store and projections; the primary memo only covered whole-generation retirement.                                                        | **Missing coverage.** Define separate lexical/dense deletion phases, retries, read-back, and acknowledgement order.                                                     |
| Generation retirement and rollback                      | Existing checksummed pointer and whole-directory collector provide the required outer protocol.                                                                                                   | **Meets if preserved.** Never activate, destroy, or retire one owned collection independently.                                                                          |

## Documented zvec behavior

The following statements are **officially documented guarantees or constraints**, not local measurements:

- A collection may contain scalar and FTS fields without any vector field. Zvec describes these as FTS-only collections.
- Each collection persists in a self-contained directory that can be moved and reopened.
- Direct ID fetch, SQL-like scalar filters, BM25/phrase/boolean FTS, and dense-vector search are available. Scalar filters can constrain both FTS and dense routes.
- FTS and vector search are mutually exclusive within one query route. Zvec does not support cross-collection joins, unions, or searches.
- Batch validation failure prevents the batch from starting. After validation, documents execute independently and return ordered per-document statuses; successful earlier operations are not rolled back when a later operation fails.
- Multiple processes may open a collection read-only; writable access is exclusive.
- `optimize()` builds configured vector indexes. `stats.docCount` reports live logical documents, and `stats.indexCompleteness` reports vector-index completeness.
- Zvec documents WAL persistence across crash and power failure.

The last claim is broader than the application-visible behavior established by source. This report does not use it as the acknowledgement rule without a status-propagating persistence boundary and fault evidence.

## Authoritative-source findings

The following are **behaviors found in pinned Node/core source**:

- Core validates every required non-nullable field, including vectors, during insert. A lexical-only document in a vector-bearing schema must carry a real vector.
- A scalar field cannot simultaneously carry incompatible inverted and FTS indexes. Keeping separate analyzed content and case-preserving identifier content in the lexical collection remains the simplest design.
- Update and upsert mark the prior physical row deleted and insert a new row. Stable IDs therefore provide logical idempotence, not physical idempotence. The current integration blindly upserts evidence, so crash replay can create non-source-driven stale rows. Immutable evidence should use insert-if-absent or fetch-and-verify identity/checksum before skipping.
- Optimization rewrites segments and can reclaim deleted versions after the configured delete-ratio threshold. `optimizeSync()` also seals and flushes the writing segment, so it is a possible mid-build persistence checkpoint, though not a substitute for a proven acknowledgement boundary.
- Each write is appended to a CRC-protected WAL before in-memory index mutation. Default appends do not force-sync per document.
- Close flushes the WAL and stores internally, but Node `closeSync()` only releases native pointers and returns `undefined`. Core destruction discards the return status of `Close()`.
- Manifest publication uses `std::ofstream`; the reviewed path does not establish file and parent-directory sync before acknowledgement.
- Read-only open still enters recovery when a writing block exists. It may attempt recovery but cannot reliably complete writes. Interrupted state therefore requires an exclusive writable reopen before search resumes.
- Recovery may return success when WAL open fails and ignores statuses from replayed internal operations. Writable recovery is required but is not proven sufficient.
- Collection locks are local: shared for read-only opens and exclusive for writable opens. Cross-collection coherence requires the project’s application-level operation lock.
- Core rejects write batches above 1,024 documents.
- Ordinary dense dimensions are limited to `(0, 20000]`; `HNSW_RABITQ` is limited to `[64, 4095]` and has additional datatype/metric restrictions. Limits must follow the selected index configuration.

## Read-only local observations and reproducible commands

The host contained 3,542 production session JSONL files totaling 2,462,183,473 apparent bytes. The legacy recall collection contained 289 files totaling 24,394,618,718 apparent bytes (24.395 decimal GB) and approximately 24.257 GB allocated. A prior read-only zvec open reported 1,175,836 live documents, 2,560-dimensional FP32 vectors, and dense index completeness `0.9511445760…`. At least 100,000 rows were lexical-only; the exact query count was capped by zvec’s 100,000 `topk` limit.

These filesystem measurements are reproducible without opening zvec or reading session contents:

```bash
# Count files and sum apparent bytes without reading JSONL contents.
python - <<'PY'
from pathlib import Path
files = [p for p in (Path.home()/'.pi/agent/sessions').rglob('*.jsonl') if p.is_file()]
print({'files': len(files), 'apparent_bytes': sum(p.stat().st_size for p in files)})
PY

# Recall-tree file count, apparent bytes, and allocated bytes.
find ~/.pi/agent/recall/zvec -type f -printf '%s\n' |
  awk '{bytes += $1; files += 1} END {print "files", files; print "apparent_bytes", bytes}'
du -B1 -s ~/.pi/agent/recall/zvec
du -B1 --apparent-size -s ~/.pi/agent/recall/zvec
```

Do not directly open the production collection merely to reproduce zvec stats unless a read-only preflight proves there is no pending WAL/recovery state. Prefer a filesystem snapshot or disposable copy mounted read-only.

Disposable 1,000-document probes observed:

- vector-free layout: 511,395 apparent bytes;
- zero-vector layout at 2,560 dimensions: 12,631,538 apparent bytes, or **24.70×** larger;
- allocated-size reproduction: 610,304 versus 12,361,728 bytes, or **20.26×** larger;
- a 1,000-row projection collection grew from 1,206,506 to 2,316,267 to 3,438,106 apparent bytes after two full rewrites, then fell to 1,241,324 after optimization;
- after SIGKILL, read-only reopen returned zero rows, writable reopen recovered 32, and a later read-only reopen returned 32.

These are **one-shot observations, not presently reproducible from the repository**. The commands named scripts under `/tmp`, but those scripts were not committed. Preserve the probes before relying on them as upgrade gates.

## Stable-growth analysis

A split layout removes a demonstrated structural cost: lexical-only evidence no longer carries a 2,560-element zero vector and associated vector-index data. The 24.70× apparent-length comparison and 20.26× allocated-byte comparison support the qualitative conclusion, but neither ratio is a universal capacity coefficient.

The expected growth model is an **architectural inference** from source behavior:

- lexical bytes grow with newly eligible immutable evidence plus scalar, inverted, and FTS overhead;
- dense bytes grow only with dense-searchable evidence, selected dimensions, FP32 values, and vector-index overhead;
- projection bytes grow with projection revisions and retain stale physical versions until explicit optimization or replacement;
- confirmed deletion leaves deleted physical rows until explicit maintenance;
- a replacement temporarily requires active, replacement, and possibly rollback generations at once;
- blind upsert replay adds avoidable stale rows even when stable IDs preserve logical identity.

Predictability therefore depends on application discipline, not zvec alone. Evidence writes must skip already-verified immutable rows rather than upsert them. Projection churn must remain isolated in the small projection collection. Optimization must remain explicit; development churn must not trigger automatic compaction. Clean replacement generations provide the deterministic reset.

The production 24.395 GB collection cannot establish a steady-state ratio. It combines lexical and dense rows, zero-vector waste, incomplete indexing, and inferred repeated rewrite history. A representative disposable build must measure lexical bytes per evidence row, dense bytes by dimension/index choice, projection revision growth, optimization peak space, close/recovery latency, and concurrent-generation retention before setup can estimate capacity.

## Coherent-generation implications

Zvec cannot make three collections atomic. Conversation Recall must preserve coherence above the engine:

1. Resolve and pin one active generation while holding the shared application lock across every lexical, dense, and projection read.
2. Let one detached worker own exclusive write access to all active-generation collections during a bounded write window.
3. Write immutable lexical rows before their dense counterparts. Check the count, order, and status of every batch result; chunk calls to at most 1,024 rows.
4. Avoid repeat upserts. Fetch and verify stable identity/checksum before treating immutable evidence as already present.
5. Write logical projections after evidence and write the physical session projection checkpoint last.
6. Retain generation-independent markers until the persistence boundary and read-back succeed. The current Node close boundary is insufficiently observable, so final acknowledgement remains blocked pending prototype/upstream work.
7. After interruption, perform writable recovery before search. Treat recovery completion as untrusted until fault tests cover error propagation.
8. Validate exact schemas, index parameters, counts, expected ID membership, dense membership, projection coverage, provenance/profile fingerprints, ordinary-FTS canaries, identifier-FTS canaries, direct-fetch canaries, and dense completeness `1`.
9. Optimize replacement collections explicitly, close them, and validate them before pointer cutover. Activation remains the existing checksummed temp-write, file-sync, rename, and directory-sync protocol.
10. Extend confirmed-source deletion into resumable lexical and dense phases. Each phase needs membership/checkpoint state, retry behavior, persistence/read-back rules, and marker acknowledgement ordering.
11. Retire only whole generation directories after proving they are inactive, not replay-pending, outside rollback retention, and excluded from readers.

Source-neighborhood expansion remains a hypothesis. The lexical collection can store logical session ID, physical projection/path locator, occurrence ID, entry and parent IDs, selected-child relation, source lines/blocks, and contribution IDs. Zvec only supplies exact lookup and filtering; the prototype must prove the application can select one canonical logical-session path and return exact provenance without reopening JSONL.

## Focused prototype requirements

A disposable, representative-generation prototype must answer these questions before adoption:

1. **Persistence boundary:** expose or add a status-returning native flush/close operation. Verify whether `optimizeSync()` can serve as a bounded mid-build checkpoint. Test acknowledgement across process crash and power/filesystem fault; include file and directory durability rather than page-cache read-back alone.
2. **Recovery faults:** test WAL-open failure, CRC failure, truncation, disk-full, failed ID-map/index writes, interrupted replay, and repeated recovery. Require failures to propagate and verify no marker is acknowledged early.
3. **Physical replay:** compare blind upsert with insert-if-absent/fetch-and-verify across repeated crash replay. Confirm that immutable evidence growth remains source-driven.
4. **Representative sizing:** build realistic lexical, dense, and projection collections; report apparent and allocated bytes, peak optimization space, build/close/reopen latency, and active+replacement+rollback retention.
5. **Index constraints:** exercise selected MRL dimensions against each supported index type, especially the 4,095-dimension `HNSW_RABITQ` limit and its dtype/metric constraints. Preserve explicit model-support override only within the selected engine/index hard bound.
6. **Build protocol:** enforce batches of at most 1,024; inspect every status; run exact post-write affected-ID read-back; prove optimize/close/reopen sequencing.
7. **Validation routes:** run deterministic ordinary-FTS, identifier-FTS, dense, filter, and direct-fetch canaries and inspect exact schema/index parameters.
8. **Neighborhood/provenance:** define canonical branch endpoint, entry-to-occurrence cardinality, traversal bounds, eligibility filtering, and exact source handles. Prove expansion stays on one logical-session graph path without JSONL access.
9. **Deletion:** crash each lexical/dense/projection deletion phase and prove retry, checkpoint, read-back, and marker ordering.
10. **Binary provenance and upgrades:** preserve probe sources and pin or attest the native build used by acceptance tests.

## Decisive verdict

**Promising candidate; fit not yet established for durable production acknowledgement and recovery.**

Proceed with zvec as the preferred prototype engine because the split-store shape, embedded operation, retrieval primitives, and generation-contained directories match the map. Do not approve production adoption from the current memo alone. The implementation must first obtain a status-propagating persistence boundary and pass recovery fault tests, representative growth measurements, source-neighborhood validation, and split-store deletion tests.

If upstream/native changes cannot expose close/flush failures or recovery cannot reliably propagate faults, evaluate another embedded, daemon-free engine. Any comparison must retain the map’s constraints: vector-free lexical evidence, user-selected dimensions within engine/index limits, independent rebuildable generation directories, explicit maintenance, stable provenance, and application-controlled atomic activation.

## Unresolved questions

- Can zvec expose a flush or close API whose status covers WAL, indexes, manifest file, and parent-directory durability?
- Does `optimizeSync()` provide a useful, observable checkpoint without imposing unacceptable build time or temporary-space cost?
- How does recovery behave under each untested WAL and filesystem fault, and can upstream fix every swallowed status?
- What are corpus-representative apparent and allocated byte coefficients for lexical rows, dense rows by dimension/index, and projection revisions?
- What is peak disk use while active, replacement, and rollback generations coexist and optimization rewrites segments?
- Can source-neighborhood expansion define one canonical branch path and exact provenance using indexed records alone?
- What resumable phase machine should coordinate confirmed deletion across lexical, dense, and projection collections?
- Can the installed prebuilt native binary be tied reproducibly or attestably to the reviewed source?
- Which alternative embedded engine should be compared if the durability boundary remains unavailable?

## Complete primary-source references

### Requirements and project protocol

- [Issue #108 — zvec split-store fitness](https://github.com/Whamp/pi-session-recall/issues/108)
- [Map #107 — predictable rebuildable recall storage](https://github.com/Whamp/pi-session-recall/issues/107)
- `docs/adr/0002-fuse-hybrid-retrieval-in-application-code.md`
- `docs/adr/0004-ingest-recall-incrementally-outside-pi.md`
- `src/commit-incremental-recall-transfer.ts`
- `src/coordinate-recall-write-window.ts`
- `src/rebuild-recall-generation.ts`
- `src/recall-generation-state.ts`
- `src/recall-conversation-service.ts`
- `src/reconcile-confirmed-session-deletion.ts`
- `src/collect-retired-recall-generations.ts`
- `src/session-conversation-index.ts`
- `src/zvec-conversation-store.ts`
- `src/zvec-session-projection-store.ts`

### Official zvec documentation

- [Documentation index (`llms.txt`)](https://zvec.org/llms.txt)
- [Database overview and operating model](https://zvec.org/mdx/en/docs/db.md)
- [Data modeling: collections, persistence, and cross-collection limits](https://zvec.org/mdx/en/docs/db/concepts/data-modeling.md)
- [Open collection and read-only behavior](https://zvec.org/mdx/en/docs/db/collections/open.md)
- [Optimize and index-completeness statistics](https://zvec.org/mdx/en/docs/db/collections/optimize.md)
- [Destroy collection](https://zvec.org/mdx/en/docs/db/collections/destroy.md)
- [Insert and batch semantics](https://zvec.org/mdx/en/docs/db/data-operations/insert.md)
- [Upsert and batch semantics](https://zvec.org/mdx/en/docs/db/data-operations/upsert.md)
- [Update and batch semantics](https://zvec.org/mdx/en/docs/db/data-operations/update.md)
- [Direct fetch](https://zvec.org/mdx/en/docs/db/data-operations/fetch.md)
- [Scalar filtering](https://zvec.org/mdx/en/docs/db/data-operations/query/filter.md)
- [Full-text search prerequisites and constraints](https://zvec.org/mdx/en/docs/db/data-operations/query/fts.md)
- [Single-vector search and filters](https://zvec.org/mdx/en/docs/db/data-operations/query/single-vector.md)

### Pinned zvec Node/core source

- [zvec-node release source at `96fe5e0`](https://github.com/zvec-ai/zvec-node/tree/96fe5e09619369179d783d1c517cac33895c2b81)
- [Node collection binding: `closeSync()`](https://github.com/zvec-ai/zvec-node/blob/96fe5e09619369179d783d1c517cac33895c2b81/src/binding/collection.cc#L909-L918)
- [Node TypeScript schema and collection API](https://github.com/zvec-ai/zvec-node/blob/96fe5e09619369179d783d1c517cac33895c2b81/src/index.d.ts)
- [Node missing-vector test](https://github.com/zvec-ai/zvec-node/blob/96fe5e09619369179d783d1c517cac33895c2b81/tests/data/operations.test.ts#L338-L350)
- [Core source at `cb42297`](https://github.com/alibaba/zvec/tree/cb422972981a2ed4735e9cfb419397c0a2383f02)
- [Schema validation and dense dimension limits](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/common/schema.cc#L87-L180)
- [Engine constants: batch, dimension, top-k, and compaction limits](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/common/constants.h)
- [Document required-field validation](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/common/doc.cc#L753-L771)
- [Collection close/destruction and status handling](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L302-L360)
- [Conflicting scalar index rejection](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L460-L495)
- [Optimization and segment reclamation](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L779-L914)
- [Batch execution and non-rollback statuses](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L1420-L1469)
- [Collection-local shared/exclusive locks](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/collection.cc#L2026-L2051)
- [Update/upsert physical-row behavior](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L883-L900)
- [WAL-before-mutation operation path](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L903-L971)
- [Segment flush/close path](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L2217-L2323)
- [Recovery open path](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L461-L501)
- [WAL recovery and replay](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/segment/segment.cc#L4205-L4320)
- [WAL append and default flush behavior](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/storage/wal/local_wal_file.cc#L29-L49)
- [Underlying file `fsync`](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/ailego/io/file.cc#L281-L284)
- [Manifest save path](https://github.com/alibaba/zvec/blob/cb422972981a2ed4735e9cfb419397c0a2383f02/src/db/index/common/version_manager.cc#L77-L115)

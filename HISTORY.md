# Storage and Architecture History

This document records how pi-session-recall arrived at its current storage design. It is not a release changelog. It focuses on the database stacks, indexing models, and operational architectures that were shipped, prototyped, rejected, or later removed.

The project changed direction several times in a short period. Those reversals were useful. Each one exposed a requirement that the next design had to preserve or a cost that it had to remove.

## Reading this history

The status of an idea matters:

- **Production** means it served the live recall corpus.
- **Merged** means it reached `master`, even if it was later removed.
- **Prototype** means it produced measurements but never served production.
- **Planned** means the design was documented but not completed on `master`.

Canonical Pi session JSONL has remained the source of truth throughout the project. Every database described below was derived state.

## Current architecture in one page

Since [PR #174](https://github.com/Whamp/pi-session-recall/pull/174), one WAL-mode SQLite database contains:

- Physical session state and document membership;
- compact tool-call and command **Invocation** records indexed with FTS5;
- Dense conversation and summary metadata;
- one 16-bucket `sqlite-vec` table containing each 1,024-dimension FP32 embedding once.

Normal recall searches Dense conversations and compact Invocations together. Complete tool results, bash output, and omitted payload arguments remain only in canonical JSONL and require explicit Source search.

One changed Physical session replaces all of its derived rows in one SQLite transaction. Candidate databases are built beside the active database and become visible through one atomic pointer change only after completion. There is no Zvec dependency, database optimization command, legacy-layout reader, or database rollback protocol. A damaged or incompatible database is rebuilt from JSONL.

The production cutover reduced the derived recall database from about 14 GiB to 3.1 GiB. The certified pre-activation candidate contained 345,907 Dense vectors and 222,145 Invocations. Its storage-engine p95 was 42.8 ms for project search, 366 ms for global search, and 1.8 ms for Invocation search. Live end-to-end checks after activation took about 215 ms for project recall, 558 ms for global recall, and 15.6 seconds for an explicit Source scan across 3,733 files.

`sqlite-vec` 0.1.9 performs exact vector search rather than using an approximate-nearest-neighbor index. Project scope stays fast because it searches one selected bucket. Global scope searches all 16 buckets and is intentionally slower.

The rest of this document explains why this design exists.

## Terms used below

- **Dense search** compares numeric embeddings to find text with similar meaning.
- **Full-text search (FTS)** indexes words and identifiers for exact or lexical matching. BM25 ranks those matches.
- **HNSW** is an approximate vector index that makes Dense search faster by avoiding a comparison with every vector.
- **Invocation** is the current compact record of a tool call or direct bash command. It contains searchable names and locator arguments, not full output.
- **Generation** is a complete candidate or active derived database built from canonical session JSONL.
- **WAL** is SQLite's write-ahead log. It lets readers continue seeing a committed snapshot while a writer prepares a transaction.

## 1. July 24: one Zvec collection and semantic search

**Status: Production, later replaced**

The repository began on July 24, 2026. Commit [`1f45f59`](https://github.com/Whamp/pi-session-recall/commit/1f45f59) added an in-process [Zvec](https://github.com/zvec-ai/zvec) 0.6.0 collection. The first implementation:

- read Pi session JSONL;
- extracted user-visible conversation text;
- called an OpenAI-compatible local embedding endpoint;
- stored vectors and source provenance in one Zvec collection;
- scanned changed session files before searches;
- tracked file fingerprints in `index-state.json`.

The first version deliberately excluded tool calls, tool results, shell output, images, and hidden reasoning. It was a semantic conversation locator, not a self-contained evidence database.

Several features landed within hours:

1. [Issue #2](https://github.com/Whamp/pi-session-recall/issues/2) replaced coarse text with token-bounded conversation chunks.
2. [Issue #3](https://github.com/Whamp/pi-session-recall/issues/3) added a content-addressed embedding cache.
3. [Issue #4](https://github.com/Whamp/pi-session-recall/issues/4) added full-text search for exact filenames, commands, hashes, quoted phrases, and errors, then fused Dense and lexical rankings.
4. [Issue #5](https://github.com/Whamp/pi-session-recall/issues/5) made tool names, arguments, results, commands, errors, URLs, and identifiers searchable without embedding them.
5. [Issue #6](https://github.com/Whamp/pi-session-recall/issues/6) added turn-context documents so replies such as “yes, do it” could be found through the user request that gave them meaning.
6. [Issue #7](https://github.com/Whamp/pi-session-recall/issues/7) added reranking, duplicate suppression, branch preference, and neighbor expansion.

These features established lasting product requirements: semantic recall, exact identifier lookup, searchable tool invocations, branch-aware provenance, and readable context. They also changed the physical shape of the database.

Tool evidence was logically lexical-only but physically lived in the same Zvec schema as Dense conversation rows. That schema had a vector field, so every lexical-only row received a zero-vector sentinel and Dense queries filtered it out. Conversation text was also stored in ordinary and case-preserving FTS fields. Turn-context documents duplicated useful conversational combinations. One source entry could therefore produce several indexed documents.

The expansion was intentional, but its storage cost was not yet visible.

## 2. July 25–27: production backfill and interactive ingestion

**Status: Production, later removed**

The first full production rollout reached more than 1.17 million evidence documents. It validated project scoping, exact tool evidence, active and abandoned branches, provenance, and hybrid search.

The project also tried to keep recall fresh from inside Pi. Startup, settled-turn, shutdown, and pre-search hooks reconciled sessions automatically. That made the interactive process a database maintainer. A headless Pi helper once held the global writer lock for about four minutes, causing real recall calls to fail. A changed 18.5 MB session could take 84.5 seconds to reconcile.

The immediate fix limited lifecycle ingestion by Pi mode, but the larger lesson survived: search and interactive agent work should not own whole-session indexing. That became the explicit writer/read-only split later restored by [PR #143](https://github.com/Whamp/pi-session-recall/pull/143): the Pi tool reads, while a standalone process writes.

## 3. July 27–29: QMD-style planning and managed inference

**Status: Merged, then removed**

The project studied QMD as both a retrieval system and a possible local inference stack. The result requires a careful distinction:

- QMD's **database was never adopted**.
- QMD-style **query planning and model choices were adopted temporarily**.

Research found that QMD indexed Markdown and code-like files from globs, treated one file as one document, and had no suitable general Pi JSONL ingestion path. Its collection storage and file-level chunking did not preserve Pi session graphs, branches, evidence kinds, or exact source geometry. [Issue #27](https://github.com/Whamp/pi-session-recall/issues/27) explicitly kept Pi's importer, project scope, identifier channel, Zvec store, and provenance model.

What the project borrowed was the query pipeline:

- lexical reformulations;
- semantic reformulations;
- an optional hypothetical-answer query;
- weighted reciprocal-rank fusion;
- Qwen reranking;
- local model profiles inspired by QMD.

[PR #58](https://github.com/Whamp/pi-session-recall/pull/58) shipped query-planned recall as an explicit fallback after ordinary hybrid misses. It added more than 21,000 lines across runtime and evaluation code. Live planner/reranker profiles took roughly 6–22 seconds per query. Private-corpus characterization found no newly admitted source, a few ranking-only promotions, and many no-improvement outcomes. Hybrid remained the default.

At the same time, [PR #59](https://github.com/Whamp/pi-session-recall/pull/59) added a much larger operations layer:

- configurable embedded or HTTP inference;
- managed EmbeddingGemma, Qwen, and QMD model artifacts;
- active and staging generations;
- a generation registry and checksummed pointer;
- lifecycle marker files;
- short-lived background workers;
- scalar Zvec session projections;
- incremental append cursors;
- replay, deletion reconciliation, rollback retention, setup, doctor, and repair flows.

This was an attempt to solve two real problems: keep search read-only and keep a working index available during long replacement builds. The implementation worked in tests, but it introduced several overlapping authorities and recovery protocols before the storage topology had settled.

## 4. July 29–August 1: coherent three-Zvec generations

**Status: Planned on `master`, implemented on an archived branch, never activated in production**

By July 29, measurements had exposed the zero-vector problem. Most indexed rows were lexical tool evidence, yet the single mixed collection assigned every row a full fake vector. The project opened [architecture map #107](https://github.com/Whamp/pi-session-recall/issues/107) to design a rebuildable layout with immutable JSONL authority and no vector waste.

A production-shaped [prototype in issue #110](https://github.com/Whamp/pi-session-recall/issues/110) proposed three independent Zvec stores per generation:

```text
lexical-source/   vector-free FTS evidence and source anchors
dense/            real embedded evidence only
projections/      mutable Physical and logical session state
```

The prototype handled builds, appends, replay, lookups, source-neighborhood traversal, validation, resumable deletion, and generation removal. It also found a serious durability ambiguity: corrupted WAL variants could open with partial or zero rows, while Zvec's close API exposed no persistence status.

[ADR 0007 at commit `11f9e0d`](https://github.com/Whamp/pi-session-recall/blob/11f9e0d33746a08bb7e68095b1536d5a779dc7aa/docs/adr/0007-keep-recall-generations-coherent-across-stores.md) defined an application-level transaction protocol across the three stores. It required recovery records, ordered mutations, close-and-reopen verification, source checkpoints, immutable manifests, validation receipts, generation replay snapshots, retained rollback generations, and bounded recovery.

The archived implementation branch contains 60 commits beyond the `master` line it started from. It built activation, rollback, fixed-source snapshots, complete membership enumeration, vector transfer, source-neighborhood expansion, interruption recovery, and production certification. The branch is preserved at tag `archive/issue-122-coherent-generations-20260801`.

The design failed its most important test: production-scale operability. A full-corpus acceptance run stalled at session 904 of 3,563 for more than two hours while consuming a CPU core without writes or errors. Several earlier candidates had already required fixes for unbounded projections, branch-leaf materialization, lock retries, source membership, and recovery certification. Each fix exposed another interaction among independent stores, progress authorities, and replay state.

The key mistake was not choosing three stores. It was trying to manufacture database transactions, snapshots, recovery, and rollback above stores that did not share a transaction boundary. Correctness demanded so much protocol that the derived cache became harder to reason about than the source data.

## 5. August 1: deliberate return to one manual Zvec database

**Status: Production, later replaced by SQLite**

Will stopped the coherent-generation rollout and described the core problem as overcomplication. The desired product was again simple: one Zvec database of Pi sessions, updated only through a CLI until a better update mechanism had evidence behind it.

[PR #143](https://github.com/Whamp/pi-session-recall/pull/143) restored the proven pre-QMD tree, then removed QMD, Qwen, embedded model management, the persistent embedding cache, lifecycle markers, background workers, generation registries, projections, activation, rollback, and automatic reconciliation. The change removed about 83,000 lines.

The resulting architecture was:

- one mixed Zvec collection;
- one standalone writer, `psr index`;
- one read-only Pi search tool;
- direct Octen HTTP embeddings;
- Dense, ordinary lexical, and case-preserving identifier retrieval;
- exact JSONL source locators;
- explicit rebuild for incompatible state.

This reset was successful. It restored an operable product and preserved the retrieval behavior that users already depended on. Later work added visible progress, clearer output, and an opt-in native hourly schedule, but scheduled work still invoked the same standalone writer.

The reset deliberately accepted the mixed collection's zero-vector and duplicated-FTS costs. At that point simplicity and working recall were more valuable than a speculative storage redesign.

## 6. August 8–9: Zvec optimization failures and physical cost

**Status: Production incident, repaired; optimization later retired**

The simple Zvec database grew to about 1.45 million documents. Roughly three quarters were lexical-only tool evidence. That made FTS, scalar storage, and fake vectors the dominant physical costs.

Automatic indexing initially optimized Zvec after changed data. Optimization rewrote most of the collection rather than only the pending rows. The project first separated hourly indexing from daily optimization, then made optimization opt-in.

Two failures forced a deeper investigation:

1. A stale `zvec/0.tmp` directory left an abandoned near-database-sized temporary copy.
2. One immutable FTS shard contained Roaring postings where Zvec's reducer required BitPacked postings, causing every later optimization to fail with `source postings is not BitPacked`.

The malformed shard held live data and could not be dropped. The safe repair rebuilt both collection-level FTS indexes from stored scalar text on a copy, scanned every immutable segment, optimized twice, and atomically cut the repaired collection into production. The exact cause remained unknown; bounded crash reproductions did not recreate it. The full account is in [Zvec FTS segment repair](docs/research/zvec-fts-segment-repair.md).

The repair proved that Zvec could optimize the healthy 1.45-million-document collection in about 3 minutes 44 seconds. It also exposed two non-obvious facts:

- optimization changed BM25 scores and rankings by merging segment-local FTS statistics;
- one steady-state optimization wrote about 15 GiB even when only 2,433 documents were pending.

Measured phase writes were about 4.14 GiB for scalar reduction, 5.97 GiB for vector reduction, and 4.92 GiB for FTS reduction. Hourly optimization would have written roughly 142 TB per year to a 220-TBW SSD. Daily optimization was still about 5.9 TB per year.

A read-only benchmark showed why the index still had value. HNSW Dense search took 8–11 ms median, while forced linear search took 263–380 ms. Indexed recall answered one historical question in about 1.5 seconds; a fresh agent found the same answer by searching raw JSONL in 94 seconds and consumed far more tokens. But the default top five indexed results missed that answer, so the index was fast rather than infallible. See [Production recall index value benchmark](docs/research/production-recall-index-value-benchmark.md).

The lesson was to measure physical writes, ranking effects, and end-to-end retrieval—not infer value from an API named `optimize` or from wall-clock time alone.

## 7. August 9–10: compact storage prototypes

**Status: Prototypes; one draft reached certification but was never activated**

The next redesign started from measured requirements rather than a topology. Normal recall had to preserve:

- semantic conversation search;
- exact tool names, commands, paths, URLs, flags, and issue numbers;
- project scope and exact source provenance;
- access to complete raw tool results when explicitly requested.

It did not need to copy every large tool result into the fast database.

### Candidate A: flat Dense Zvec + SQLite Invocations + JSONL Source

The first prototype removed 1,109,045 lexical-only rows from Zvec and retained only 340,736 Dense documents. It used:

- flat, exact Zvec vector search with no HNSW;
- SQLite FTS5 for compact Invocation records and session state;
- direct JSONL scanning for complete Source evidence.

Results:

| Measurement                    |              Result |
| ------------------------------ | ------------------: |
| Dense-only flat Zvec           |            2.02 GiB |
| Dense p95                      |             48.2 ms |
| Top-result agreement with HNSW |                 5/5 |
| Compact Invocation SQLite      |             118 MiB |
| Invocation p95                 |            0.122 ms |
| Full JSONL scan                | 21.7 s, zero writes |

This became the original version 7 design in [issue #165](https://github.com/Whamp/pi-session-recall/issues/165) and draft [PR #174](https://github.com/Whamp/pi-session-recall/pull/174). A production candidate passed every gate at 2.279 GiB and 49 ms Dense p95. It was never activated and is now recorded as [superseded v7 evidence](docs/research/superseded-v7-compact-production-recall-certification.md).

Its weakness was cross-store consistency. Zvec Dense rows were written before SQLite committed session state and Invocations. Deterministic IDs and repeatable indexing made recovery possible, but the design still needed an application-level retry protocol whenever a process stopped between stores—the same class of problem that had made coherent generations expensive.

### Candidate B: vectorless Zvec for Invocations

A correction during review showed that Zvec 0.6.0 could create a vectorless FTS-only collection. The project benchmarked it against SQLite FTS5 with the same 219,734 Invocation records.

Both worked. Zvec searched as fast or faster after optimization, but SQLite was smaller, built about ten times faster, wrote less, and already owned transactional session state. Zvec also reintroduced optimization and tokenizer/ranking differences. The raw performance difference was not decisive; SQLite's transaction boundary was.

### Candidate C: one SQLite database with sqlite-vec

The project then tested `sqlite-vec` 0.1.9 for exact vector search inside the same SQLite database as state and FTS5. It had no stable approximate-nearest-neighbor index, so the question was whether exact search remained acceptable at about 341,000 vectors.

Three vector layouts were measured:

- one unpartitioned table: good global search, weak project search;
- one partition per project: good project search, poor global search;
- 16 project buckets: good project search, acceptable global search.

A two-table version initially looked best: one unpartitioned global table plus one 16-bucket project table. It gave about 34 ms project p95 and 388 ms global p95, and one transaction covered both vector copies. But the second copy added about 1.35 GiB plus duplicate writes and integrity work.

The final decision stored each vector once in the 16-bucket table. Project search selects one bucket and exact project key. Global search scans all buckets and accepts roughly 0.6-second warm p95. This traded global latency for a smaller and simpler database.

The unified SQLite prototype also proved the property the split designs could not provide: session state, Invocation FTS, Dense metadata, and vectors committed or rolled back together in one database transaction. Transaction rollback, concurrent-reader, and forced-termination tests all preserved a complete old or new session state.

## 8. August 10–11: remove compatibility and ship unified SQLite

**Status: Current production architecture**

The first unified SQLite implementation retained readers and rollback adapters for old Zvec layouts and stored two vector copies. That looked safe but recreated the compatibility and state-machine complexity the project had already rejected.

The final simplification made three decisions:

1. Store one vector copy in one 16-bucket table.
2. Treat canonical JSONL—not an old database—as the recovery source.
3. Keep only staged construction and atomic activation, whose purpose is to prevent a partial current-format database from serving search.

Manifest version 8 and SQLite schema version 3 identify this format. Old v6/v7 databases are neither opened nor migrated. `@zvec/zvec`, layout dispatch, legacy vector reuse, rollback commands, public optimization, and scheduled optimization were removed. [ADR 0014](docs/adr/0014-store-recall-in-one-sqlite-database.md) records the final decision.

A fresh staged database was built from canonical JSONL. The initial build took 6 hours 53 minutes and embedded the Dense corpus from scratch because obsolete Zvec vectors were not reused; the old database was never exposed as the new format. Bounded certification then checked integrity, Dense and Invocation retrieval, Source provenance, storage, latency, transaction rollback, concurrent readers, forced termination, a real selected-session update, device writes, and 100 replacement cycles.

The certified database allocated 3.01 GiB, stayed below every latency gate, reused all 17 expected vectors in its real changed-session probe, made no embedding request in that probe, and showed no growth after 100 replacements. macOS Intel and ARM jobs loaded pinned `sqlite-vec` and exercised both FTS5 and vector search.

[PR #174](https://github.com/Whamp/pi-session-recall/pull/174) merged on August 11 as commit [`8046871`](https://github.com/Whamp/pi-session-recall/commit/8046871). The exact certified candidate was activated, caught up, and exercised through project, global, and Source searches. The hourly update-only timer completed successfully. After explicit approval, the old 14 GiB Zvec database, legacy index state, stray catalog, markers, and stale locks were removed.

## 9. August 11: restore one practical local embedding path

**Status: Current fresh-install default**

The QMD-era architecture had implemented downloaded local model artifacts and embedded inference, but PR #143 removed that whole platform with the planner, reranker, cache, workers, and model registry. The remaining product required a separately running OpenAI-compatible Octen 4B server. That was reasonable for the production machine but a poor fresh-install default for users without a spare model server or hosted embedding budget.

A bounded prototype revisited the requirement without restoring the old platform. It tested 171 deterministic real sessions containing 12,966 Dense documents and 12,836 compact Invocations. Octen 0.6B Q8_0 through llama.cpp preserved 100% of the fixed quality corpus but indexed at 2.14 documents per second and peaked at 5.14 GB RSS. Voyage 4 Nano INT8 ONNX improved throughput and memory but preserved only 81.25% of the fixed corpus. Octen 0.6B SmoothQuant INT8 ONNX preserved 100%, indexed at 3.38 documents per second, peaked at 1.92 GB RSS, and kept sampled search below 58.1 ms p95.

The accepted path uses pinned `onnxruntime-node` 1.27.0 and one project-controlled 1.01 GiB artifact. Independent probes exposed an important semantic detail: the artifact's tokenizer post-processor appends `<|endoftext|>` token `151643`; manually appending the configured `<|im_end|>` EOS token `151645` reduced upstream vector cosine to about 0.64–0.72. The final provider follows the measured tokenizer output, pools its last token, and normalizes the native 1,024-dimensional vector.

`psr setup` now defaults fresh users to local Octen, while an explicit external HTTP profile preserves the 4B server path. Installations without a profile remain HTTP-compatible so upgrades do not reinterpret existing databases. Profile identity lives in the manifest, so changing model or backend requires an explicit rebuild. Download receipts, byte counts, SHA-256 checks, unique partial directories, and atomic activation protect the model cache without introducing a generic provider registry or persistent embedding cache.

See [issue #175](https://github.com/Whamp/pi-session-recall/issues/175), [ADR 0015](docs/adr/0015-use-local-octen-as-the-fresh-install-default.md), the [prototype report](docs/research/local-embedding-runtime-prototype.md), and the [model release](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1).

## What survived every rewrite

The storage engine changed, but several principles endured:

- Pi session JSONL is authoritative.
- Thinking and images are not recall evidence.
- Project scope applies before candidate limits.
- Semantic text and exact identifiers need different retrieval paths.
- Tool results should not be embedded by default.
- Every result needs exact source provenance.
- Abandoned branches remain searchable and labeled.
- Search must not perform hidden maintenance.
- A partial replacement database must never become active.
- Derived recall state must be disposable.

The current design differs mainly in where those principles are enforced. SQLite supplies the transaction and recovery behavior that earlier designs tried to assemble above independent stores.

## Discarded designs at a glance

| Design                                                    | How far it went                              | Why it was discarded                                                                                 |
| --------------------------------------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Semantic-only Zvec                                        | Production                                   | Exact identifiers and tool evidence required lexical retrieval.                                      |
| Interactive Pi lifecycle indexing                         | Production                                   | Session parsing and writer locks blocked interactive and helper processes.                           |
| QMD collection database                                   | Research only                                | File-oriented ingestion did not preserve Pi session structure or provenance.                         |
| QMD-style planner and local model stack                   | Merged                                       | High latency, little measured retrieval gain, and a large model/runtime/evaluation surface.          |
| Managed generation registry and background marker workers | Merged                                       | Too many authorities and recovery paths for a disposable local index.                                |
| Three independent Zvec stores                             | Prototype and archived implementation branch | Production-scale stalls and application-level transaction/recovery complexity.                       |
| One mixed Zvec collection                                 | Production                                   | Zero-vector tool rows, duplicated FTS text, large optimization writes, and operational fragility.    |
| Routine Zvec optimization                                 | Production and scheduled variants            | Rewrote about 15 GiB per run, changed BM25 rankings, and once became blocked by malformed FTS state. |
| Flat Dense Zvec + SQLite FTS5                             | Certified inactive candidate                 | Good size and speed, but changed-session updates crossed two transaction domains.                    |
| Vectorless Zvec Invocation store                          | Prototype                                    | Viable, but larger and more write-heavy than SQLite and lacked the session-state transaction.        |
| Unified SQLite with two vector tables                     | Prototype and draft implementation           | Faster global search did not justify about 1.35 GiB of duplicate vectors and duplicate maintenance.  |
| Legacy v6/v7 readers and database rollback                | Draft implementation                         | Canonical JSONL rebuilds were simpler than permanent obsolete-layout machinery.                      |
| Octen 0.6B Q8_0 GGUF local default                        | Prototype                                    | Quality passed, but indexing was slower and four contexts exceeded the 4 GiB memory gate.            |
| Voyage 4 Nano INT8 ONNX local default                     | Prototype                                    | Runtime cost passed, but fixed-corpus recall reached only 81.25%.                                    |
| Octen 0.6B SmoothQuant ONNX local default                 | Production                                   | Current fresh-install inference path; external Octen HTTP remains available.                         |
| Unified SQLite with one 16-bucket vector table            | Production                                   | Current storage design.                                                                              |

## Lessons for future changes

### Separate requirements from mechanisms

Hybrid retrieval, exact tool lookup, and provenance are product requirements. Zvec FTS, HNSW, QMD planning, and generation registries were mechanisms. Treating a mechanism as permanent made later designs harder to simplify.

### Test the transaction boundary early

The decisive advantage of unified SQLite was not build speed or sub-millisecond FTS. It was the ability to replace one Physical session's complete projection in one transaction. Future storage proposals should identify their atomic unit before comparing query benchmarks.

### Prototype at production cardinality

Small tests did not expose the coherent-generation stalls, the cost of zero-vector rows, or whole-collection optimization writes. The project made its best decisions after using the real number of sessions, documents, vectors, and bytes.

### Measure physical writes and post-command settling

Wall time hid SSD wear, delayed writeback, and near-database-sized temporary output. Future maintenance benchmarks should include block-device writes, peak temporary allocation, memory, and file stabilization after the command returns.

### Keep full payloads at the canonical source

Copying every tool result made exact lookup fast but dominated storage. Compact Invocations retain high-value names, commands, and locators; explicit Source search pays the slow scan only when complete payloads matter.

### Prefer a disposable database over compatibility machinery

The project repeatedly added migration, adoption, rollback, and recovery paths to protect derived data. Once a certified JSONL rebuild existed, those paths cost more than they protected. Staging remains valuable because it prevents partial activation; permanent readers for obsolete layouts do not.

### Complexity is an operational cost

The coherent-generation branch was sophisticated and heavily tested. It was still the wrong product because routine operation and failure diagnosis became harder than rebuilding the cache. Deleting that work was progress, not lost effort: its failures made the final transaction requirement clear.

## Primary evidence

- Initial Zvec implementation: [`1f45f59`](https://github.com/Whamp/pi-session-recall/commit/1f45f59)
- Hybrid and tool evidence: [issue #4](https://github.com/Whamp/pi-session-recall/issues/4), [issue #5](https://github.com/Whamp/pi-session-recall/issues/5), [issue #6](https://github.com/Whamp/pi-session-recall/issues/6), [issue #7](https://github.com/Whamp/pi-session-recall/issues/7)
- QMD-style search: [issue #27](https://github.com/Whamp/pi-session-recall/issues/27), [PR #58](https://github.com/Whamp/pi-session-recall/pull/58)
- Incremental generation runtime: [issue #48](https://github.com/Whamp/pi-session-recall/issues/48), [PR #59](https://github.com/Whamp/pi-session-recall/pull/59)
- Coherent storage map: [issue #107](https://github.com/Whamp/pi-session-recall/issues/107) and child issues #108–#116
- Archived coherent implementation: tag [`archive/issue-122-coherent-generations-20260801`](https://github.com/Whamp/pi-session-recall/tree/archive/issue-122-coherent-generations-20260801)
- Simple recall restoration: [PR #143](https://github.com/Whamp/pi-session-recall/pull/143)
- Zvec performance evidence: [production index-value benchmark](docs/research/production-recall-index-value-benchmark.md)
- Zvec repair: [FTS segment repair](docs/research/zvec-fts-segment-repair.md)
- Superseded split-store candidate: [v7 certification](docs/research/superseded-v7-compact-production-recall-certification.md)
- Unified SQLite prototype: [prototype report](docs/research/unified-sqlite-recall-storage-prototype.md)
- Final specification: [issue #165](https://github.com/Whamp/pi-session-recall/issues/165)
- Final storage decision: [ADR 0014](docs/adr/0014-store-recall-in-one-sqlite-database.md)
- Local embedding decision: [ADR 0015](docs/adr/0015-use-local-octen-as-the-fresh-install-default.md)
- Local embedding prototype: [runtime comparison](docs/research/local-embedding-runtime-prototype.md)
- Production certification: [SQLite certification](docs/research/unified-sqlite-production-recall-certification.md)
- Production implementation: [PR #174](https://github.com/Whamp/pi-session-recall/pull/174), commit [`8046871`](https://github.com/Whamp/pi-session-recall/commit/8046871)

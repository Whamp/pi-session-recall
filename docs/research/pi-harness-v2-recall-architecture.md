# Pi harness v2 lessons for recall architecture

- **Ticket:** [Assess Pi harness v2 as a recall source and storage precedent](https://github.com/Whamp/pi-session-recall/issues/119)
- **Map:** [Map: Design a predictable rebuildable recall storage architecture](https://github.com/Whamp/pi-session-recall/issues/107)
- **Pi revision reviewed:** [`71efc6f0c1909874ec8c944637a9ae7fc0e2d508`](https://github.com/earendil-works/pi/tree/71efc6f0c1909874ec8c944637a9ae7fc0e2d508)

## Executive verdict

Adopt Pi's **logical patterns**, not its operational database.

Stable source-local session and entry identities, immutable parent links, one monotonic session sequence, append-only facts, and a single-writer boundary all strengthen recall's ingestion and validation model. Pi's private branch materialization also validates recall's decision to index source-neighborhood structure instead of rebuilding paths during search.

Do **not** place recall evidence or projections in Pi's SQLite database, read Pi's private tables from generation code, copy Pi `branch_id` values, use lanes as branch identities, or make a Pi writer lease the recall writer lock. Pi's proposed SQLite store is the mutable operational authority for harness execution. A recall generation remains a disposable, independently validated derivative containing lexical/source evidence, dense evidence, and recall-owned session projections.

That boundary does not ban SQLite inside a recall generation. A recall-owned SQLite file could satisfy generation isolation, but Pi's proposal does not justify changing engines: it models an operational conversation tree, not a recall evidence catalog or lexical-search workload. Keep zvec as the incumbent unless acceptance measurements expose a material recovery or lexical-quality problem.

If Pi later ships SQLite-backed sessions, add one narrow **Pi session source adapter** before the canonical graph boundary. The adapter owns all Pi-version and backend knowledge and emits backend-neutral session snapshots or deltas with opaque source identity, stable entry links, source geometry, lane endpoints, facts, and a validated cursor. Recall generation code must not know table names, database files, WAL state, or Pi schema versions.

This is a future compatibility boundary, not a production implementation proposal.

## Evidence status: proposal, implementation, and open questions

The distinctions matter because the apparent SQLite overlap is mostly prospective.

### Proposed design

`harness-v2.md` proposes a session made of an append-only conversation tree, mutable lane pointers, append-only per-lane operation logs, and append-only global facts. One monotonic sequence orders writes across all four parts; entries retain immutable parent chains, while lanes move independently over the tree ([harness-v2 lines 41–75](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L41-L75)). The proposed storage contract assigns `parentId`, `seq`, and timestamp at append, requires identifiers to be unique per session, and assumes one writer per session even when one SQLite database hosts many sessions ([harness-v2 lines 1164–1211](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1164-L1211)).

The proposed SQLite schema is explicitly greenfield. Pi plans to discard existing work-in-progress databases rather than migrate them ([harness-v2 lines 1236–1254](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1236-L1254)). Its `branch_entries` and `branch_tips` tables are explicitly private, rebuildable read caches with no backend-neutral interface ([harness-v2 lines 1256–1274](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1256-L1274)).

### Existing implementation at the reviewed commit

The current SQLite package implements the older single-leaf model. Its schema has `sessions.active_leaf_id`, `session_entries`, an entry-only sequence, `branch_entries`, and materialized summaries; it has no proposed records, lanes, lane moves, facts, branch tips, or leases ([current migration lines 1–59](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/storage/sqlite-node/src/sqlite/migrations/001_initial.sql#L1-L59)). Current appends update the entry, sequence, materialized state, active leaf, and branch cache in one SQLite transaction ([current storage lines 291–336](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/storage/sqlite-node/src/sqlite/storage/index.ts#L291-L336)). Current branch IDs are generated cache identities, not source identities ([current storage lines 146–182](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/storage/sqlite-node/src/sqlite/storage/index.ts#L146-L182)).

The current JSONL implementation reads and writes version 3, with no v4 envelopes or shared sequence in its public entry type ([current JSONL lines 15–23](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/src/harness/session/jsonl-storage.ts#L15-L23), [current entry type lines 375–380](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/src/harness/types.ts#L375-L380)). The design document itself names the current harness and storage code as code to replace, while leaving the testing strategy and implementation sequence as TODOs ([harness-v2 lines 1621–1655](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1621-L1655)).

### Open questions relevant to recall

Pi has not specified an external read-snapshot or export contract, the lifetime and takeover rules of the proposed writer lease, or a stable physical database identity. Replication causality and tree-fork lane handling also remain open or out of scope ([harness-v2 lines 1621–1629](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1621-L1629)). Those gaps prevent a safe direct-ingestion contract today.

## Patterns recall should adopt

### 1. Preserve stable source-local identity and separate it from content

Pi's proposed entry has storage-assigned `seq`, `parentId`, and timestamp but a pre-allocated stable `id`; the storage contract enforces identifier uniqueness per session ([harness-v2 lines 1051–1071](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1051-L1071), [harness-v2 lines 1203–1209](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1203-L1209)). Recovery treats an existing provisioned ID with different content as corruption ([harness-v2 lines 169–184](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L169-L184)).

Recall should keep the same distinction:

- Preserve Pi's raw `session_id` and entry `id` as source-local provenance.
- Keep recall evidence-occurrence and entry-anchor IDs composite over the recall source identity, logical-session occurrence, graph position, evidence part, and source geometry, as approved in [Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112). A raw Pi entry ID is not globally unique across copied databases or distinct physical sources.
- Continue using content and embedding-input checksums to prove immutable-row or vector reuse. Identity names a source location; it does not prove unchanged content.

### 2. Use `parent_id` for topology and `seq` for order and cursor validation

Pi makes the parent chain immutable and lets branches share prefixes ([harness-v2 lines 66–73](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L66-L73)). Its one session sequence covers entries, operation records, facts, and lane moves ([harness-v2 lines 1203–1211](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1203-L1211)).

Recall should therefore:

- Treat `parent_id` as graph authority.
- Treat `seq` as a source-local committed-order and high-water input, not as graph adjacency, branch identity, or a global identifier.
- Allow gaps in entry sequences because non-entry records share the sequence.
- Validate that previously imported anchor entries still have the same ID, parent, type, and checksum before accepting an incremental delta.
- Send sequence regression, duplicate sequence, changed anchors, missing parents, or source-incarnation change to explicit reconciliation rather than repairing textually.

This complements ADR [0004](../adr/0004-ingest-recall-incrementally-outside-pi.md)'s byte cursor and boundary fingerprint. A SQLite adapter needs a sequence-based opaque cursor, not a byte offset or WAL position.

### 3. Keep one writer for recall-owned state

Pi's single-writer rule makes impossible interleavings corruption and serializes lane writes through one harness ([harness-v2 lines 23–39](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L23-L39)). Recall already applies the corresponding pattern through one incremental worker and bounded recall write windows in ADR [0004](../adr/0004-ingest-recall-incrementally-outside-pi.md).

Keep these writer domains separate:

- Pi's writer owns the operational session source.
- The recall worker owns the active recall generation's lexical, dense, and projection writes.
- A replacement worker owns only its replacement generation.
- A recall reader never acquires or impersonates Pi's writer lease.

The proposed `leases` table names a Pi writer claim, but the document gives no external-reader protocol for it ([harness-v2 lines 1236–1249](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1236-L1249)). It cannot serve as recall's operation lock.

### 4. Materialize read paths, but own and validate the materialization

Pi's cache demonstrates the useful pattern: derive path rows from stable parent links, use them for bounded reads, and rebuild them explicitly if damaged. It does not establish a reusable `branch_id` contract. Pi states that no interface exposes the cache and no other backend has it ([harness-v2 lines 1256–1266](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1256-L1266)).

This strengthens [Specify exact source-neighborhood expansion semantics](https://github.com/Whamp/pi-session-recall/issues/111) and [Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112): keep immutable entry anchors and path links in recall's lexical/source evidence store so exact source-neighborhood expansion stays index-only. Rebuild and validate those anchors from normalized source entries. Never ingest Pi `branch_id`, query Pi `branch_entries` during search, or make recall validity depend on Pi's cache health.

### 5. Exclude orchestration records from recall evidence

Pi deliberately separates operation records from the conversation tree. It says deleting operation logs leaves a valid conversation and records must not enter model context, transcripts, branch queries, or forks ([harness-v2 lines 66–75](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L66-L75)).

A future adapter may consume records to understand committed boundaries or lifecycle state, but it must not index them as conversation or tool evidence. Tool evidence continues to come from source conversation entries. Latest global facts and lane positions feed mutable logical session projections, not immutable evidence text.

## Overlap that must remain separate

| Pi operational concern                                   | Recall concern                                 | Required boundary                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Durable harness execution, retries, queues, and recovery | Searchable immutable evidence                  | Do not index operation records or reuse Pi recovery state.                                         |
| SQLite tables shared by many operational sessions        | One self-contained recall generation           | Never put recall stores in Pi's database or retain live cross-database references.                 |
| Current lane pointers and append-only lane-move history  | Recall eligibility and branch provenance       | Normalize endpoints as source facts; do not treat lane names as branch IDs or evidence identities. |
| Private `branch_entries`/`branch_tips` acceleration      | Recall-owned exact source-neighborhood anchors | Derive independently and validate inside each generation.                                          |
| Pi writer claim                                          | Recall incremental/rebuild ownership           | Separate locks and failure domains.                                                                |
| Pi schema evolution                                      | Recall generation compatibility                | Record the adapter contract and canonical policy in the manifest, not Pi table layouts.            |

Reusing Pi's database would violate [Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112): search must open one generation without another generation, an embedding cache, or raw source. It would also make Pi migrations, WAL recovery, and operational deletion part of recall activation and rollback. A self-contained, recall-owned SQLite file inside a generation would not create that coupling. It is still a separate future architecture decision, not an implementation-level engine swap: co-locating stores or adding cross-store transactions would amend the accepted three-independent-store topology and recovery protocol.

The current Pi repository illustrates the coupling risk. Merely opening the SQLite repository configures WAL and synchronous mode and applies migrations ([current repository lines 30–34 and 74–80](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/storage/sqlite-node/src/sqlite/repo.ts#L30-L80)). Recall should not invoke that operational open path as an independent reader.

## Recall-owned SQLite does not yet beat the incumbent

Pi's proposed schema contains operational entries, records, lane state, facts, leases, and a private branch cache; it does not specify an evidence catalog or lexical-search index ([harness-v2 lines 1236–1266](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1236-L1266)). It therefore proves that SQLite can model Pi's conversation workload, not that SQLite preserves recall's ordinary and case-preserving FTS behavior, ranking quality, provenance queries, or corpus-scale storage profile.

A future recall-owned SQLite design could physically co-locate lexical/source evidence, entry anchors, and mutable projections while zvec retained dense vectors. If it transacted across those logical stores, it would replace the current premise—three independent zvec stores with no cross-store transaction—rather than implement it. That could simplify recovery for the SQLite-owned rows. It would not remove cross-engine coherence, and it would leave the expensive embedding corpus in zvec. Any evaluation that reaches this point must explicitly revisit [Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112) and [Define cross-store incremental and generation consistency](https://github.com/Whamp/pi-session-recall/issues/114), not enter as a backend substitution. [Prototype predictable storage topology on representative recall evidence](https://github.com/Whamp/pi-session-recall/issues/110#issuecomment-5120982387) already decided that database durability protects embedding time rather than source data and should become stronger only when measurements show meaningful savings without material complexity or storage growth.

Pi's own durability rule also does not depend on multi-record transactions. It persists each intent and result separately, then recovers an unfulfilled intent by stable provisioned ID ([harness-v2 lines 169–184](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L169-L184)). That pattern supports the recovery-required and verified-replay direction already chosen for recall.

Do not reopen the accepted topology or add a SQLite prototype from this document alone. Reconsider a recall-owned SQLite engine only if [Set acceptance evidence for predictable recall storage and retrieval](https://github.com/Whamp/pi-session-recall/issues/115) finds material zvec recovery cost, unacceptable lexical quality, or a store abstraction that makes an engine comparison cheap. Any later prototype must compare FTS quality, source-neighborhood correctness, crash-state complexity, two-engine cutover and rollback, latency, and allocated bytes against the three-zvec baseline.

## Identity semantics for a SQLite-backed source

| Field              | Meaning recall may rely on                                    | Meaning recall must reject                                           |
| ------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------- |
| `session_id`       | Stable logical session key inside one adapter source          | Physical database identity or globally unique occurrence ID          |
| entry `id`         | Stable graph node key inside that session                     | Content checksum or cross-source occurrence ID                       |
| `parent_id`        | Immutable parent topology                                     | Sequence predecessor                                                 |
| `seq`              | Monotonic committed order and cursor input inside one session | Branch ID, dense document ID, or contiguous entry number             |
| `lane`             | Mutable named endpoint and possible lifecycle observation     | Entry ownership, permanent branch membership, or occurrence identity |
| `branch_id`        | Private cache key only                                        | Importable provenance                                                |
| database path/file | Adapter locator for one container                             | Sufficient session identity or source-incarnation proof              |

Pi proposes one database containing many independently single-writer sessions ([harness-v2 lines 1203–1210](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1203-L1210)). A SQLite recall source therefore cannot be identified by `session_id` alone or by the database path alone. The future identity should be an adapter-owned composite of backend instance/database locator and Pi session ID, plus a revision or incarnation check that detects replacement at the same locator. The exact locator and move/copy policy remains a decision, because Pi has not specified a durable physical database identity.

## Smallest future-proof adapter seam

Place the adapter before strict graph validation and document creation:

```text
Pi JSONL v1/v2/v3/v4 ─┐
                       ├─ Pi session source adapter ─ normalized session snapshot/delta
Pi SQLite session ─────┘                                  │
                                                          ▼
                                          strict session graph boundary
                                                          │
                              eligibility, chunking, recall-owned anchors and stores
```

The adapter contract should expose only:

1. **Opaque source identity:** adapter kind/version, stable source key, raw Pi session ID, and source incarnation.
2. **Consistent, replayable read boundary:** one snapshot high-water token covering entries plus the lane/fact state needed to interpret them. The adapter must reproduce the same normalized input at that boundary after later source writes so bounded recovery can re-derive an interrupted batch.
3. **Normalized entries:** stable entry ID, parent ID, source order, timestamp, type, payload, and discriminated source geometry.
4. **Normalized session state:** session metadata, current lane endpoints, relevant append-only lane moves or equivalent endpoint observations, and latest global facts.
5. **Opaque incremental cursor:** adapter-owned high-water plus anchor digest; generation code compares only adapter identity and compatibility, never SQLite `seq`, JSONL byte offsets, or WAL metadata directly.
6. **Explicit outcomes:** unchanged, appended, deleted, or requires reconciliation. No parser fallback or heuristic repair.

Keep source geometry discriminated. JSONL geometry remains physical line/byte/block/character data. SQLite geometry should use the source key, raw session ID, source sequence, entry ID, and payload-relative block/character positions. Do not invent synthetic JSONL line numbers for SQLite.

The adapter may use a supported Pi API, export, or backend-specific implementation. Its conformance tests own all knowledge of Pi's private schema. Recall's canonical graph, evidence, generation, and search modules consume only normalized output. A Pi schema change should require an adapter update or fail adapter conformance; it must not invalidate already active recall generations.

### Snapshot and cursor requirements

Pi promises committed-entry reads and proposes atomic append of an entry with its lane-leaf move ([harness-v2 lines 1131–1135](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1131-L1135), [harness-v2 lines 1180–1185](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1180-L1185)). It does not promise that several independent external queries share one snapshot. Before SQLite ingestion is accepted, the adapter must prove that entries, high-water, lane endpoints, and facts come from one consistent read boundary.

[Define cross-store incremental and generation consistency](https://github.com/Whamp/pi-session-recall/issues/114) recovers an interrupted batch by re-deriving it from source instead of storing a second prepared evidence payload. A future adapter must therefore reproduce the exact normalized entries and projection inputs at the recorded high-water after later source writes. That high-water is source evidence during preparation and recovery; after verified close and reopen, the physical session projection remains recall's sole mutable per-source checkpoint. If Pi cannot provide replayable boundaries, SQLite-backed ingestion needs a new recovery decision rather than an adapter-only change.

Normal incremental ingestion may then request changes after the opaque cursor. The adapter must force reconciliation when:

- the source incarnation changed or high-water regressed;
- an imported sequence/ID anchor changed;
- parent links changed or a required parent disappeared;
- a duplicate ID or sequence appears;
- the adapter cannot prove one consistent snapshot; or
- the source format or adapter policy is unsupported.

A WAL checkpoint, file size, inode, or modification time is not the logical cursor. Those are backend implementation details and cannot establish source continuity by themselves.

## JSONL v3/v4 compatibility

Pi's proposed compatibility promise is narrow: old coding-agent v3 JSONL must open idle; other current harness and SQLite formats may break without migrations ([harness-v2 lines 1–3](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1-L3)). The proposal's v4 JSONL uses one tagged line per entry, record, lane move, or fact, with sequence equal to line position. A writable v3 session is rewritten once to v4 before its first v4 append; read-only open does not rewrite it ([harness-v2 lines 1217–1234](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L1217-L1234)).

Consequences for recall:

- Keep ADR [0003](../adr/0003-import-historical-sessions-virtually.md)'s existing v1, v2, v3, and reuse-history support. Pi's narrower compatibility policy does not revoke recall's historical import policy.
- Do not claim v4 support until Pi implements and tests the format.
- Treat Pi's v3-to-v4 rewrite as a source rewrite, not an append. Existing byte-boundary validation should route it to explicit reconciliation.
- A v4 adapter must ignore operation records as evidence, use lane/fact lines only for projection state, and validate every shared sequence.
- Record the normalized adapter/import policy in the generation manifest. Do not record or depend on Pi's private SQLite schema version.

## Exact source-neighborhood implications

[Specify exact source-neighborhood expansion semantics](https://github.com/Whamp/pi-session-recall/issues/111) remains correct:

- Earlier entries follow immutable parent links.
- Later entries follow one explicitly selected descendant path.
- A reached fork requires a leaf that contains the anchor; the implementation never defaults to the latest active lane.
- Every traversed conversation-tree entry consumes one count, including structural placeholders.
- Search and expansion use only recall-owned lexical/source evidence and anchors.

Pi's proposed sequence can order entries already proven to be on one path, but it cannot choose a descendant or replace parent traversal. Lane names and leaves are mutable. Two lanes may share an entry and then diverge ([harness-v2 lines 77–95](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L77-L95)). Pi's private reverse branch lookup is therefore an implementation precedent, not a source-neighborhood dependency.

Operation records and global facts are not graph entries and do not consume neighborhood entry counts. Compaction and branch-summary entries remain source entries and retain the behavior approved by [Specify exact source-neighborhood expansion semantics](https://github.com/Whamp/pi-session-recall/issues/111).

## Effect on map decisions

| Decision                                                                                                                      | Effect of harness v2 research                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Specify exact source-neighborhood expansion semantics](https://github.com/Whamp/pi-session-recall/issues/111)                | Strengthened. Keep recall-owned entry anchors and explicit endpoint selection; reject Pi cache and lane defaults.                                                                                                                                                        |
| [Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112)               | Unchanged for this map: three independent zvec stores. A future transactional or co-located SQLite design would reopen this decision and the cross-store protocol; it is not a backend swap.                                                                             |
| [Define cross-store incremental and generation consistency](https://github.com/Whamp/pi-session-recall/issues/114)            | Aligned for current JSONL. A future adapter must make its source boundary replayable for recovery; logical projections still commit before the physical projection, which remains the sole checkpoint authority. Otherwise this decision must be amended before support. |
| [Set acceptance evidence for predictable recall storage and retrieval](https://github.com/Whamp/pi-session-recall/issues/115) | No immediate scope change. Its existing work should reveal whether zvec recovery or lexical quality justifies a later engine comparison.                                                                                                                                 |
| [Reconcile the architecture with existing ADRs and recall work](https://github.com/Whamp/pi-session-recall/issues/116)        | Record the separation from Pi's operational database and the deferred source-adapter boundary. Amend no ADR for an unimplemented Pi format.                                                                                                                              |
| [Key physical session projections by physical source identity](https://github.com/Whamp/pi-session-recall/issues/63)          | Its sessions-root-relative path remains right for JSONL. A future database containing many sessions would need an adapter source key composed with raw `session_id` and an incarnation check.                                                                            |

The accepted many-path provenance, immutable evidence, active-generation pointer, replacement build, rollback, and no-persistent-embedding-cache decisions remain intact.

## Future decisions outside this map

Do not create these as children of the current map. Harness v2 is unimplemented, Pi exposes no supported external SQLite snapshot contract, and the current destination covers the rebuildable recall architecture rather than a new Pi source format. Start a fresh map if Pi ships the design and SQLite or v4 ingestion enters scope.

### Specify the Pi session source adapter and opaque cursor

Decide the backend-neutral source identity, discriminated geometry, snapshot boundary, normalized entry/lane/fact envelope, opaque cursor, mutation detection, deletion semantics, adapter-policy manifest identity, and failure outcomes for JSONL and future SQLite sources. Require a supported Pi API/export or a quarantined adapter module; forbid private table names outside that module.

### Define recall eligibility and provenance for multi-lane sessions

Decide how multiple simultaneous lane leaves affect the singular **effective leaf**, **active branch**, **active context**, branch-exit eligibility, and active-branch preference in the current domain model. Specify lane deletion, lane movement, shared-prefix entries, and concurrent lane activity without turning a lane name into a branch ID.

This decision is necessary before a harness-v2 session can preserve current recall eligibility semantics. Pi explicitly allows multiple lanes to run in parallel and interleave writes while sharing tree entries ([harness-v2 lines 77–95](https://github.com/earendil-works/pi/blob/71efc6f0c1909874ec8c944637a9ae7fc0e2d508/packages/agent/docs/harness-v2.md#L77-L95)).

## Future SQLite-source acceptance evidence

If SQLite-backed Pi ingestion enters scope, require disposable fixtures that prove:

1. Equivalent v4 JSONL and SQLite sources normalize to the same graph, lifecycle facts, chunk content, and source-neighborhood behavior, allowing only adapter-specific source locators and occurrence IDs to differ.
2. A read concurrent with Pi appends observes one committed high-water: no entry without its parent, no lane leaf beyond the captured entries, and no fact from beyond the boundary.
3. After recovery-required state is persisted, later source appends, lane moves, or fact writes do not change replay of the recorded normalized batch; recovery completes that boundary and leaves later work pending.
4. Entry appends interleaved with operation records, facts, and lane moves preserve sequence validation without assuming contiguous entry sequences.
5. Regressed high-water, changed anchor payload, changed parent, duplicate ID/sequence, database replacement at the same path, and the same raw `session_id` in two databases never produce an append delta or alias occurrences.
6. A v3-to-v4 JSONL rewrite forces explicit reconciliation and leaves no stale evidence.
7. Dropping, rebuilding, or changing Pi's private branch cache does not change normalized output.
8. Operation records never become recall evidence; lane and fact changes update only recall-owned projections.
9. A source database unavailable or deleted after ingestion does not prevent search or exact source-neighborhood expansion from the active generation.
10. A Pi schema change fails only adapter conformance; existing generation validation and search remain independent.
11. Multi-lane forks, shared prefixes, lane movement, and lane deletion satisfy the future multi-lane decision without inventing branch IDs.

## Remaining uncertainties

- Harness v2 is a reviewed design, not the current implementation; its test plan, implementation order, and several lane questions remain open.
- Pi has not published a supported external snapshot/export API for SQLite sessions.
- Pi has not defined stable physical database identity or writer-lease takeover/read semantics.
- The final v4 JSONL and greenfield SQLite schemas may change; the stated compatibility policy permits that change.
- Recall's current singular effective-leaf and active-branch vocabulary does not yet define multi-lane semantics.

These uncertainties block SQLite ingestion, not the current map. The safe present decision is to preserve the separate, disposable recall generation and defer all Pi backend coupling to the future source-adapter boundary.

# Incremental processing granularity profile and append-aware design

## Decision

Keep the hourly maintenance schedule. Treat large active-session cost as an incremental-processing granularity problem.

A changed Physical session file currently remains the unit of import, graph validation, document construction, tokenization, vector reuse, and SQLite replacement. The measured workload spent 186.6 seconds reprocessing a 119.6 MB file after only the clone's stored modification time was made stale. Source bytes, derived document content, and embeddings were unchanged.

The recommended next implementation is an append-aware Physical session projector with a strict full-import fallback. It must produce the same Recall database projection as the existing full importer. It must not weaken strict graph validation, Source provenance, all-or-nothing Physical session replacement, or the rule that complete tool results and bash output stay only in canonical JSONL.

This decision is compatible with ADR-0003, which explicitly leaves append-aware ingestion as a separate design problem, and ADR-0014, which requires every changed Physical session projection to become visible in one SQLite transaction.

## Instrumentation

`psr index` now emits one `physical-session-file-profiled` progress event after each successful Physical session replacement. The scheduled CLI writes the event to the journal with the exact file path, planned source-byte change, document counts, and these elapsed phases:

| Phase                              | Boundary                                                                                                                                                                                                                             |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `readParse`                        | Frame, decode, JSON-parse, classify, and virtually convert the complete Physical session file.                                                                                                                                       |
| `graphValidation`                  | Parse canonical graph records; validate headers, IDs, parents, cycles, leaves, compaction/branch links, and tool links; derive graph-wide relationships. Measurements from reused Physical files are summed across logical sessions. |
| `documentConstructionTokenization` | Build entry documents, compact Invocations, turn-context documents, exact-token chunks, checksums, source metadata, and Project attribution. Measurements are summed across logical sessions.                                        |
| `vectorLookup`                     | Fetch current or staged document metadata and vectors and compare canonical checksums in 128-document batches.                                                                                                                       |
| `embedding`                        | Wait for the configured embedding provider for documents whose canonical checksum has no reusable vector. Zero means no embedding request was made.                                                                                  |
| `sqliteReplacement`                | Atomically replace every SQLite projection owned by the Physical session file.                                                                                                                                                       |

`totalElapsedMilliseconds` begins before the prior Physical session state is read and ends after the successful replacement. The six measured phases are intentionally narrower; a small residual includes orchestration and in-memory filtering. Timings use a monotonic clock. The event contains no conversation text, tool arguments, tool output, vector values, or checksums.

The event names the size captured while planning the workset and the size stored by the prior index. A file can grow while indexing, so these values do not claim to describe every byte processed. A positive difference is reported as **additional source bytes at planning**, not appended bytes. Size and modification time alone cannot prove append-only history because Pi can migrate old files by rewriting them.

The measurement mechanism performs two monotonic-clock reads around each phase or vector batch and emits one event outside the measured total. On this host, one million `performance.now()` calls took 44.1 ms at the median. About 400 clock reads therefore cost an estimated 0.018 ms. This is a mechanism estimate, not an end-to-end overhead benchmark; a future optimization PR should still compare profiling enabled and disabled on representative files.

## Measured baseline

The complete sanitized measurements are in [`incremental-processing-granularity-profile.json`](incremental-processing-granularity-profile.json).

Environment:

- base revision `13f7f39d846c291ff63eaed19cc7ebb83e944880`;
- Node 24.16.0 on Linux x86-64;
- production Octen HTTP embedding profile;
- copy-on-write clone of the active Recall database;
- read-only 119,626,979-byte, 20,004-line representative Physical session source;
- 11,352 Dense documents before catch-up.

The production writer was inactive and its operation lock absent before each clone. The source file and active Recall database were never modified. Both clone runs ended with healthy SQLite quick check, foreign keys, Invocation FTS parity, and Dense vector parity.

| Sample                | Source-byte delta | Embedded | Reused |  Total | Read/parse | Graph validation | Documents/tokenization | Vector lookup | Embedding | SQLite replacement |
| --------------------- | ----------------: | -------: | -----: | -----: | ---------: | ---------------: | ---------------------: | ------------: | --------: | -----------------: |
| Catch-up 1            |            18,323 |       12 | 11,341 | 161.6s |      0.35s |            22.6s |                 128.5s |          3.8s |      1.6s |               4.7s |
| Catch-up 2            |            18,323 |       12 | 11,341 | 170.3s |      0.38s |            24.5s |                 133.9s |          4.8s |      1.3s |               5.4s |
| Forced metadata stale |                 0 |        0 | 11,353 | 186.6s |      0.39s |            45.6s |                 134.2s |          2.6s |         0 |               3.7s |

The forced-stale counterfactual first caught up the clone, then decremented only that clone's stored `mtime_ms` by one millisecond. The second pass saw identical source bytes, reused every vector, embedded nothing, and still consumed 186.6 seconds. Document construction/tokenization accounted for 72.0% of that run and graph validation for 24.5%.

The first catch-up process consumed 166.2 CPU seconds and 175.6 wall seconds. The path is CPU-work dominated rather than blocked on the embedding service. Read/parse, vector lookup, embedding, and replacement together accounted for less than 8% of either ordinary catch-up sample.

### What the baseline rejects

- A 30-minute schedule would not halve per-run cost. It would repeat almost all cumulative graph and document work twice as often when the file remains active.
- Faster vector reuse alone cannot produce a material end-to-end improvement. Vector lookup was 1.4–2.8% of measured time.
- Faster JSON framing alone cannot produce a material improvement. Read/parse was about 0.2%.
- Avoiding only embedding requests cannot solve the zero-delta case because that case made no requests.

## Why old records are not wholly immutable

Append-only source bytes do not imply an immutable derived projection. New records can alter metadata or compound documents derived from old entries:

- `currentLeafId` and `sessionName` are copied into every Dense document today.
- `isOnActiveBranch` and `isVisibleInActiveContext` change along the old and new leaf paths.
- Adding a child changes its parent's `childEntryIds`.
- `branchPathLeafIds` changes for ancestors when an old endpoint gains a child or a new endpoint appears.
- A compaction changes `compactedByEntryIds` for summarized ancestors and active-context visibility.
- A tool result completes an earlier tool call relationship and can change compact Invocation error metadata.
- A turn-context document can remain open across tool activity and later assistant entries. An appended assistant can replace the endpoint document for the nearest ancestor user turn.
- A harness `leaf` record can change active state without adding a graph entry.
- A second session header changes a formerly single-session Physical file into reuse history. Prior logical sessions remain valid only if every segment passes the all-or-nothing import policy.

Atomic conversation chunks and summary chunks are more stable. Their IDs depend on logical-session identity, contributing entry IDs, evidence kind/part, text-run index, and character span. Their checksums depend on visible content. Under a verified append, an old record's source text, source line, chunk ID, and vector remain stable. Dynamic graph metadata and open turn contexts are the invalidation frontier.

## Recommended architecture

### 1. Verify the source prefix before taking the append path

Persist a versioned Physical session checkpoint with:

- exact processed byte length and next one-based source line;
- SHA-256 digest of every processed source byte;
- import-policy and projection-schema identities;
- detected physical format and logical-session boundaries;
- source size and modification time used for the committed snapshot.

On a changed file:

1. Open one file descriptor and capture `fstat`.
2. Hash the previously processed prefix and compare it with the checkpoint digest.
3. Read and frame only bytes after the checkpoint through the captured size.
4. Capture `fstat` again. Retry a bounded number of times if the file changed during the snapshot.
5. Use append-aware projection only when the prefix digest, record boundary, policy identity, and logical-session checkpoint all match.
6. Otherwise run the existing full importer and atomically replace the checkpoint with the rebuilt state.

Hashing the old prefix still reads cumulative bytes, but the measured complete read/parse phase was only about 0.4 seconds. The first implementation should pay that bounded correctness cost rather than trust size, `mtime`, inode identity, or sampled windows. A block-hash tree is a later optimization only if prefix hashing becomes measured material work.

Shrink, prefix mismatch, changed import policy, unsupported legacy format, malformed tail, ambiguous header transition, or snapshot races exhaustively falling through their retries must use the full-import path. A same-size file whose digest is unchanged needs only checkpoint metadata refresh; it must not rebuild documents.

### 2. Persist structural graph state without raw payloads

Add SQLite-owned checkpoint tables for the minimum certified state needed to validate tail records and derive invalidations:

- logical-session header identity, cwd, parent-session path, and physical line range;
- entry ID, parent ID, type, timestamp, source line, and child relationships;
- explicit harness leaf state and effective leaf;
- compaction and branch-summary references;
- tool call/result IDs and names needed for uniqueness and pairing;
- latest session name;
- Project identity resolved from each logical-session header;
- bounded visible user/assistant/custom/summary text needed by open turn-context frontiers;
- document-to-contributing-entry dependencies and endpoint/open-turn state.

Do not store complete tool results, bash output, omitted arguments, thinking, images, or unknown raw record JSON. Canonical JSONL remains the only owner of those payloads. Visible text already permitted in Dense recall may be retained only where needed for a still-open derived document.

The old certified graph is the induction base. Parse all tail records before validating them so tail entries may refer to other tail entries exactly as the full parser permits. Check duplicate IDs and tool call/result IDs against both checkpoint and tail state. Check missing parents, cycles, leaves, compaction ancestry, branch-summary references, and tool pairing across the combined graph. A new reuse-history header closes the prior logical segment and starts a separately validated segment.

### 3. Separate stable document content from dynamic graph metadata

The present `metadata_json` duplicates session-wide and graph-dynamic fields into every Dense document. Updating `currentLeafId` or an endpoint in a long linear session can therefore require rewriting most document rows even after tokenization is made incremental.

Normalize these dynamic facts into relational projections:

- session-level current leaf and session name;
- entry parent/child relations;
- active-branch and active-context membership;
- entry-to-endpoint branch memberships;
- entry-to-compaction relations;
- document-to-contributing-entry relations.

Keep stable content, token spans, source provenance, IDs, checksums, sibling links, and vectors on the Dense document. Overlay dynamic fields only for the bounded candidates or documents being returned. Dense nearest-neighbor search still operates on the same vectors and Project routing; candidate presentation must remain byte-for-byte equivalent after normalization.

This avoids an O(total documents) JSON rewrite for session-wide metadata. Updating active paths, branch endpoints, and compaction relationships becomes proportional to affected graph depth and changed endpoints. If normalization proves too invasive for a first slice, patching affected ancestor metadata is a valid prototype, but it is not the recommended final representation for long linear sessions.

### 4. Recompute only dependency-frontier documents

For a verified append:

- create atomic conversation/custom/summary documents only for new visible records;
- preserve old IDs, checksums, token spans, vectors, and source locations;
- update the parent entry's child relation;
- update active branch/context and endpoint relationships along the symmetric difference of old and new paths;
- apply compaction effects only to the summarized ancestor range;
- update the prior tool-call Invocation when a new result completes its pair;
- maintain one open turn-context frontier per endpoint and nearest ancestor user;
- replace a turn-context document only when its contributing assistant entries or endpoint status changes;
- create a new logical checkpoint when a reuse-history header is appended.

A user entry closes the prior open turn at that user boundary and starts a new frontier. Assistant entries extend the nearest open user turn. Tool and other non-user records move the path endpoint without adding visible turn text. Branches create independent frontiers sharing immutable ancestor state.

### 5. Commit checkpoint and projection delta together

Introduce a deep SQLite operation such as `applyPhysicalSessionProjectionDelta`. It should atomically apply:

- checkpoint and structural graph updates;
- stable Dense document inserts/deletes;
- dynamic relation updates;
- vector inserts only for new or checksum-changed documents;
- compact Invocation inserts/updates/deletes;
- session-document ownership changes.

Retain `replacePhysicalSession` as the full-import fallback. Readers must continue to observe either the complete previous Physical session projection or the complete next projection. Embeddings are prepared before the transaction. Cancellation, process death, provider outage, or a late SQLite failure leaves both the previous projection and its checkpoint intact.

Malformed or unsupported changed files must preserve current behavior: report failure and remove the complete indexed projection for that Physical session file. In reuse history, one invalid logical segment invalidates the complete Physical file; previously valid sibling segments must not survive independently.

## Implementation seams

Keep `indexChangedConversationSessions` as the workset orchestrator. Add narrow modules with searchable domain names:

1. `physical-session-append-checkpoint.ts`: prefix verification, bounded source snapshot, checkpoint encoding.
2. `incremental-session-graph.ts`: checkpoint reconstruction, tail validation, effective-leaf and invalidation delta.
3. `incremental-session-documents.ts`: new stable documents and affected turn-context frontier.
4. `sqlite-recall-projection-delta.ts` or an equivalent deep operation inside the existing SQLite module: one transactional delta application.

The full importer remains the independent correctness oracle and fallback. Do not fork its format rules into a weaker parser. Extract shared canonical record validation primitives where both paths need identical semantics.

## Migration and compatibility

The checkpoint, normalized metadata, and delta transaction change the Recall database schema and projection semantics. Increment the SQLite schema and Index manifest identities. Following current policy, obsolete databases are not migrated in place or used with mixed semantics. Build a staged replacement from canonical JSONL, certify it, and atomically activate it.

A lower-risk rollout can separate representation from append processing:

1. introduce normalized dynamic graph metadata while still using full import;
2. certify output parity and search behavior;
3. add checkpoints during full replacements;
4. enable verified append deltas behind an explicit manifest identity;
5. retain full fallback permanently.

The hourly timer, standalone-only writer policy, active-generation pointer, lock behavior, search scope, ranking, and Source search remain unchanged.

## Verification strategy for a future implementation

### Differential oracle

After every generated or explicit append operation, compare an append-aware disposable database with a clean database produced by the current full importer. Equality includes:

- accepted/rejected Physical files and failure text category;
- Physical and logical session ownership;
- every Dense document ID, checksum, content, provenance field, sibling link, and dynamic graph field;
- every vector byte for unchanged checksums and deterministic test embeddings for new checksums;
- every compact Invocation and FTS row;
- project attribution and routing;
- projection counts, foreign keys, FTS parity, vector parity, and search results.

**Property-testing recommendation:** use generated stateful operation sequences because branches, compactions, explicit leaves, tool pairing, and reuse headers form a broad sequence domain with the full importer as a compact independent oracle. Generate linear messages, branch-from-ancestor operations, user/assistant/tool transitions, compactions, branch summaries, session names, harness leaves, and reuse headers. Partition invalid tails for duplicate IDs, missing parents, cycles, invalid leaves, invalid compaction ancestry, malformed framing, and tool mismatches. Persist seeds and minimized counterexamples. The repository does not currently depend on a property-testing framework; obtain approval before adding one, or begin with a deterministic operation corpus and replay runner.

### Named regression examples

Keep explicit tests for:

- unchanged metadata-only dirty detection;
- one linear assistant append;
- a branch appended from an old ancestor;
- explicit harness leaf movement without a new entry;
- compaction with `firstKeptEntryId` and with `retainedTail`;
- a tool result completing an old call;
- a user boundary and an assistant extending an open turn context;
- `session_info` changing session-wide presentation;
- a second header creating reuse history;
- invalid new reuse segment removing the complete Physical projection;
- source shrink and same-size prefix rewrite taking full fallback;
- append during snapshot causing retry or deferred next-pass work;
- interruption before and during the SQLite delta transaction;
- Project lineage or manifest incompatibility requiring rebuild.

### Performance gates

Use the sanitized workload shape in the profile artifact and retain the zero-delta forced-stale counterfactual. A useful first target is:

- at least 10x lower zero-delta and 18 KB-tail wall time on the 119.6 MB/11k-document workload (under 18.7 seconds from the 186.6-second baseline);
- document construction/tokenization proportional to the invalidation frontier rather than total historical documents;
- no embedding request for unchanged checksums;
- no unchanged-vector delete/reinsert;
- no more than 10% regression on full-import fallback or small changed files;
- profiling overhead below 1% when measured enabled versus disabled;
- identical integrity and differential-oracle results.

These are decision gates for a future implementation, not claimed results.

## Risks and open decisions

- Normalizing dynamic metadata changes the search materialization path and needs careful candidate-level joins.
- Strict prefix hashing keeps correctness simple but still reads cumulative bytes.
- Persisted structural state increases schema size and creates another derived representation that must share exact import-policy identity.
- Turn-context frontiers are the most subtle content invalidation boundary.
- Unsupported concurrent writers and whole-file Pi migrations require reliable fallback, not optimistic repair.
- The current importer removes previously indexed evidence when a changed file becomes invalid. Append-aware processing should preserve this behavior unless a separate product decision changes it.

None of these risks supports changing the timer interval. The measured bottleneck is cumulative graph and document reconstruction, and the design should attack that work directly.

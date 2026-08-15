# Plan: improve changed-session indexing without making Recall authoritative

Status: Phase 1 implemented; Phase 2 not triggered

## Outcome

Reduce indexing time for large active Physical session files while keeping canonical session JSONL as the only source of truth.

The Recall database is a disposable projection. When its schema, cache state, or checkpoint state is incompatible, rebuild it from JSONL instead of migrating or repairing it. Keep the hourly schedule, the standalone `psr index` writer, strict session validation, and one atomic Physical session replacement.

Use the existing full importer as both the fallback and the differential correctness oracle.

Measured result: schema 4 projection reuse plus linear parent-cycle validation reduced the representative 18.8 KB tail update to 8.9 seconds and content-identical false-dirty work to 0.12 seconds. Both gates pass, so the plan stops before the coarse append checkpoint.

## Evidence and first target

The measured 119.6 MB session took 186.6 seconds when its content had not changed:

- document construction and tokenization: 134.2 seconds;
- graph validation: 45.6 seconds;
- SQLite replacement: 3.7 seconds;
- vector lookup: 2.6 seconds;
- read and parse: 0.4 seconds;
- embedding: 0 seconds.

The first implementation should remove repeated tokenization before it attempts incremental graph validation or incremental SQLite writes. A useful gate remains less than 18.7 seconds for the representative zero-content and 18 KB-tail cases, but each phase below has a stop gate. Do not build later phases after the measured target is met.

## Constraints

Preserve these contracts:

- Canonical JSONL owns all source evidence.
- `psr index` remains the only evidence writer.
- Readers see either the previous complete Physical session projection or the next one.
- A malformed changed Physical file removes that file's complete indexed projection, as it does today.
- Reuse-history segments remain all-or-nothing within one Physical file.
- Complete tool results, bash output, omitted arguments, thinking, images, and unknown raw records never enter the Recall database.
- Full import remains available for every file and every future database rebuild.

Do not add these without new measurements:

- normalized graph metadata tables;
- row-level projection-delta writes;
- a block-hash tree;
- in-place database migrations;
- cache-repair machinery;
- a daemon, watcher, second writer, or faster schedule;
- a property-testing dependency.

## Phase 0: make the dominant document work visible

Split `documentConstructionTokenization` into internal measurements while retaining the existing public aggregate:

1. pending atomic and summary document construction;
2. turn-context construction and budget splitting;
3. conversation chunk tokenization;
4. checksums, metadata decoration, Invocation construction, and Project attribution.

Run the representative large-session clone twice and add the subphase results to the sanitized profile artifact.

**Stop or redirect:** If tokenization and turn-context budget splitting do not account for most of the 134.2-second phase, revise this plan before changing persistence.

## Phase 1: add one disposable cache schema and reuse tokenized document geometry

Make one schema change for both source identity and projection reuse. Keep full file parsing and strict full graph validation for files whose bytes changed. Avoid repeating tokenizer work for projection inputs whose exact content has not changed.

### Persistence

Store a SHA-256 digest and exact imported byte length on each Physical session. Add one derived cache table owned by that session:

```text
conversation_projection_inputs
  session_path
  projection_input_id
  input_checksum
  ordered_document_ids
```

`projection_input_id` identifies one Dense pre-tokenization input: an atomic message/custom/summary text or one complete turn-context input. Build it from stable logical-session identity, contributing entry IDs, and evidence kind/part. Exclude current leaf, active state, branch endpoints, compaction presentation, session name, and Project attribution.

`input_checksum` hashes the exact canonical input text plus chunk-policy identity. `ordered_document_ids` points to existing Dense document rows, which already hold the resulting content and token/character geometry. Lexical-only evidence and compact Invocations stay on their current reconstruction path.

The table stores no source payload. It is deleted and recreated with the Physical session projection in the existing replacement transaction. A missing, malformed, or mismatched cache entry is a cache miss, not a recovery problem.

Bump the SQLite schema and manifest identity once for both additions. Do not migrate the active database. Build a staged replacement from JSONL, certify it, and activate it through the existing generation flow.

### Content-identical fast path

Compute the digest from the same bounded source snapshot used by import. When size or modification time changes:

1. Open the file and capture its size.
2. Hash exactly that snapshot.
3. Recheck file metadata after the read.
4. Retry once if the file changed during the snapshot; after a second race, use the existing full-import path rather than inventing a new deferred state.
5. If byte length and digest match the committed state, update only source metadata in a small transaction and skip parsing, validation, projection, vectors, and replacement.

Do not trust size, modification time, inode identity, or a sampled digest. Do not add rolling or block hashes unless full-file hashing becomes material; complete read/parse was about 0.4 seconds in the baseline.

### Import path

Extend the document projector behind `readSessionConversationImport`; do not create a second public importer.

For each pending projection input:

1. Derive its stable input ID and exact input checksum without tokenizing it.
2. Look up the prior cache entry and its Dense documents.
3. On an exact match, reuse the prior chunk content, IDs, checksums, character spans, token spans, sibling links, and token counts.
4. Recompute graph-dynamic metadata from the newly validated graph: current leaf, session name, parent/children, branch endpoints, active branch/context, compaction links, tool pairing, and Project attribution.
5. On a cache miss, run the current tokenizer and chunker unchanged.

A changed open turn context gets a different input checksum and is rebuilt. Closed turn contexts, atomic documents, and summaries reuse their prior geometry. Reconstruct compact Invocations with the current full-graph path unless profiling shows that work matters.

### Writer path

Keep the current writer path:

- fetch or embed vectors by document ID and checksum;
- call `replacePhysicalSession` once;
- delete and reinsert that Physical session's rows in one transaction.

Do not add `applyPhysicalSessionProjectionDelta`. SQLite replacement was only 3.7 seconds in the measured case, and the existing operation already provides the required atomicity and interruption behavior.

### Phase 1 proof

Compare an optimized import with a clean full import after each deterministic operation:

- unchanged content with changed metadata;
- one linear assistant append;
- a branch from an old ancestor;
- explicit leaf movement;
- compaction and retained-tail compaction;
- a tool result completing an old call;
- a user boundary and an extended open turn context;
- a session name change;
- a second header creating reuse history;
- same-ID source text rewritten in place;
- malformed tail and malformed reuse segment.

Require equality for acceptance/failure, every Dense document field, every Invocation, vectors for unchanged checksums, Project attribution, FTS parity, vector parity, and representative search results.

Benchmark the same 119.6 MB shape. Require:

- less than 2 seconds for content-identical false-dirty detection;
- no tokenizer calls for unchanged projection inputs;
- no embedding calls for unchanged checksums;
- no more than 10% regression for a full cache miss or small changed file;
- profiling overhead below 1%;
- exact integrity and differential parity.

**Stop:** If both representative cases finish below 18.7 seconds, ship Phase 1 and do not build a graph checkpoint.

## Phase 2: add one coarse append checkpoint only if graph validation remains too slow

Build this phase only when post-Phase-1 profiles show graph validation prevents the 18.7-second target.

Persist one versioned checkpoint blob per Physical session, in the same SQLite transaction as `replacePhysicalSession`. The blob may contain derived structural graph state and visible user/assistant/custom/summary text needed by open turn contexts. It must not contain raw tool results, bash output, omitted arguments, thinking, images, or unknown source records.

The checkpoint records:

- processed byte length, complete prefix SHA-256, and next source line;
- import-policy, chunk-policy, and checkpoint versions;
- Physical format and reuse-history segment boundaries;
- entry IDs, parents, types, source lines, child relations, leaves, compaction and branch references;
- tool call/result identities needed for validation and pairing;
- Project origin, session name, effective leaf, and open turn-context dependencies.

For an apparent append, hash the old prefix, parse the tail, and validate it against the loaded checkpoint. On shrink, prefix mismatch, version mismatch, unsupported format, malformed state, ambiguous reuse-header transition, or a source race, call the full importer. Replace the checkpoint after every full import.

Keep this checkpoint coarse: one encoded value with one version, not a normalized graph schema. It is a disposable acceleration record. If decoding or validation fails, discard its use and rebuild from JSONL. Do not write upgrade or repair code.

Continue to rebuild dynamic document metadata in memory and call `replacePhysicalSession`. The Phase 1 projection cache still avoids historical tokenization.

**Stop:** Ship when the representative append is below 18.7 seconds with exact differential parity. Do not normalize metadata or implement row-level deltas merely because the design document describes them.

## Deferred work and its trigger

Consider normalized dynamic metadata or `applyPhysicalSessionProjectionDelta` only when a fresh profile after Phase 2 shows that metadata reconstruction or SQLite replacement consumes more than half of changed-file time or alone exceeds 10 seconds.

The current evidence is the opposite: replacement took 3.7 seconds. Optimizing it now would add schema, query, and transactional complexity around a minor cost while the database can already be rebuilt from JSONL.

## Delivery sequence

Use one reviewable change per step:

1. document subphase measurements and refreshed profile;
2. source identity, projection-input cache, and full-import differential tests in one schema version;
3. optional coarse append checkpoint in a later schema version only if the Phase 1 gate fails.

Do not reserve unused checkpoint columns. A later schema change is cheap because Recall rebuilds from JSONL instead of migrating derived state.

After each change, run focused tests, the deterministic replay corpus, the representative disposable-clone profile, SQLite integrity checks, `npm test`, `npm run typecheck`, `npm run lint`, `npm run format:check`, and the TypeScript slop checks. Record the new profile beside the existing baseline.

A phase is complete only when its optimized output equals a clean rebuild from the same JSONL. When in doubt, throw away the derived optimization state and run the full importer.

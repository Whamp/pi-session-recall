# Pi Session Recall

`pi-session-recall` gives Pi a `pi-session-recall` tool for searching past conversations by meaning or exact text. It reads Pi session JSONL files, embeds user-visible conversation text through a configured inference provider, stores durable FP32 vectors in a content-addressed cache, and builds dense plus full-text indexes in an in-process [zvec](https://github.com/alibaba/zvec) collection. Search fuses every bounded retrieval channel and suppresses duplicate evidence. Fast deterministic hybrid ranking is the default; Qwen3 reranking is an explicit deep-search option.

## What it indexes

The index contains:

- user messages;
- assistant text;
- turn-context documents that pair user requests with assistant responses;
- visible custom messages;
- compaction summaries;
- branch summaries;
- tool names and argument objects;
- tool result text, including errors, paths, identifiers, and URLs;
- direct bash execution commands and outputs.

The index excludes `pi-session-recall` calls and their results. Recall output is derived evidence; indexing it would create a recursive feedback loop.

Atomic conversation documents come from one visible text run in one session entry. A visible text run contains only adjacent, nonempty text blocks. Thinking, tools, results, images, empty blocks, roles, entries, and summaries all end a conversation run, so boundaries cannot create synthetic text adjacency.

Turn-context documents are secondary evidence. Each starts with visible user text and follows Pi parent links through visible assistant text until the next user entry. A turn may cross tool calls and results, but its text never includes thinking, tool arguments, or raw tool output. Oversized turns split into bounded role-paired documents, so every emitted context retains both user and assistant text. Branched paths produce distinct turn contexts when their contributing entries differ. Each context records all contributing entry IDs; atomic chunks remain available for precise citation.

Tool calls, results, and direct bash executions follow a separate lexical-only evidence path. Names, compact JSON argument objects, result text, commands, and shell output are stored without redaction. Every document stays within one tool source block or bash message field and records its evidence part, tool name, available call linkage, error state, and source path. Thinking and images are never tool evidence.

All evidence uses the exact pinned Octen tokenizer for bounded geometry. The approved policy targets at most 512 tokens with 64 tokens of overlap between adjacent siblings. Tool evidence also contains at most 512 tokens, uses no overlap, and preserves every source character. Splitting prefers Markdown sections, paragraphs, fenced-code boundaries, lines, and sentences before a hard token cut.

Every stored document includes:

- schema version, stable document identity, and content checksum;
- session, parent-session, project, entry, parent-entry, and contributing-entry provenance;
- effective-leaf, active-branch, active-context, and many-path branch membership;
- compaction and branch-summary links;
- source line, block, character, and token spans;
- text-run identity, chunk order, count, and sibling IDs.

Pi does not assign branch IDs. Shared ancestors therefore record every endpoint path that contains them instead of a misleading singular branch ID.

## Session import compatibility

Conversation Recall never rewrites Pi session files. It frames physical JSONL records only at byte `0x0A`, removes one preceding `0x0D`, and preserves U+2028 and U+2029 inside JSON strings. Framing streams across input chunks. A final record without LF still enters JSON parsing, so truncation is reported with its physical path and one-based line.

One import boundary selects one of three disjoint formats before graph validation:

- **Canonical JSONL:** one complete leading session header at supported version 2 or 3, followed by canonical graph records.
- **Unversioned Pi v1:** one complete header without `version`, followed only by supported v1 messages, model changes, thinking-level changes, or compaction records with required v1 metadata and without `id` or `parentId`. Conversion assigns deterministic SHA-256 entry IDs, chains entries in physical order, sets header version 2, and converts `firstKeptEntryIndex` to `firstKeptEntryId`.
- **Pi session-file reuse history:** at least two complete versioned headers, with no pre-header record and at least one record in every segment. Each header starts an independent logical session with its own session ID, cwd, parent-session metadata, graph, and project attribution. All logical sessions retain the shared physical path and original physical lines.

Every canonical representation enters the same strict session graph parser. Detection never selects a converter because graph parsing failed. The parser also requires compaction checkpoints to name an ancestor, branch summaries to name an existing entry or `root`, and tool results to match one unique tool call with the same tool name. Historical tool call/result placeholders whose identifier and tool name are both empty emit no tool evidence; partial-empty or otherwise malformed tool links still fail. Unsupported, ambiguous, malformed, truncated, cyclic, duplicate-ID, invalid-leaf, invalid-reference, and missing-parent inputs fail without searchable documents. If any logical session in a reuse-history file is invalid, the physical import is all-or-nothing: no sibling logical session emits documents. The historical `/new` reuse shape and fix are recorded in [ADR 0003](docs/adr/0003-import-historical-sessions-virtually.md).

Run the guarded compatibility replay against an explicit, non-production corpus root:

```bash
npm run --silent replay:session-import -- --corpus-root /path/to/session-corpus
```

The command reads only `.jsonl` files, refuses any root that overlaps `~/.pi/agent/recall`, and emits one JSON result with per-file outcomes, format counts, logical-session counts, deterministic import digests, and a corpus replay digest. Import digests use corpus-relative source paths so the same source set has the same evidence identity after a read-only relocation; runtime searchable documents still retain their physical source paths. It snapshots source hashes, size, mode, mtime, and inode before and after replay and fails if any source changes.

The privacy-safe frozen expectation lives at `src/fixtures/session-import/historical-corpus-replay-expectation.json`. To enforce its source-set and per-file outcome digests against an available private corpus, run:

```bash
PI_SESSION_IMPORT_CORPUS_ROOT=/path/to/session-corpus \
  node --import tsx --test src/replaySessionImportCorpus.test.ts
```

The documented 121-file failure corpus produces:

| Physical format                    | Accepted files | Logical sessions |
| ---------------------------------- | -------------: | ---------------: |
| Unversioned Pi v1                  |             77 |               77 |
| Pi session-file reuse history      |             33 |               84 |
| Canonical JSONL with U+2028/U+2029 |              9 |                9 |
| **Accepted total**                 |        **119** |          **170** |

Two additional canonical files remain rejected at physical line 212 because entry `8d2b86d9` names missing parent `74da12a2`. Replay does not salvage or rewrite them.

## Install

```bash
pi install /home/will/projects/pi-session-recall
```

Reload a running Pi session with `/reload`.

The committed quality report passes and selects 512/64 chunks, 8 candidates per channel, and 5 final results. Create the initial production index explicitly:

```text
/pi-session-recall-index --rebuild
```

After that initial generation exists, interactive Pi operations read it without performing whole-session maintenance:

- startup, settled turns, shutdown, and reload never reconcile session files;
- `pi-session-recall` searches never reconcile the active session before retrieval;
- `/pi-session-recall-index` is the only path that catches up changed, new, or removed sessions and optimizes the index.

This keeps session parsing, tokenization, embedding-cache checks, and zvec writes out of latency-sensitive Pi lifecycle and search operations. Active conversation content remains in Pi's model context; it becomes searchable recall evidence after explicit maintenance. The incremental state skips unchanged JSONL files, cached vectors prevent unchanged text from reaching the embedding model, and the PID-owned writer lock serializes multiple index processes.

Use `/pi-session-recall-index` for an explicit full catch-up and optimization. Use `--rebuild` to replace an incompatible generation while preserving tokenizer assets and cached vectors. A future compaction-aware incremental path may index only content that has left the active model context; the current implementation deliberately does not approximate that behavior with whole-session work.

## Use

Pi can call the tool with a semantic paraphrase or exact source token:

```text
pi-session-recall({ query: "What did we decide about the job queue?", limit: 5 })
pi-session-recall({ query: "readNodeErrorCode", limit: 5 })
pi-session-recall({ query: "Which queue decision survived later objections?", limit: 5, mode: "deep-rerank" })
```

The extension tells Pi to use `pi-session-recall` when a task depends on a past conversation or a detail absent from current context. Each result contains:

- final ranking score, optional Qwen relevance score, active-branch prior, fused score, and available dense, lexical, and identifier ranks and scores;
- document and summary kind, session name, date, role, branch label, and project directory;
- original candidate text or stitched same-run neighbor context;
- source provenance in `SESSION_FILE#ENTRY_ID` form, plus every contributing entry for turn context;
- every suppressed duplicate occurrence and every chunk used for neighbor context.

## Hybrid retrieval

Each atomic conversation, turn-context, or summary document is stored once with three searchable representations:

- an Octen embedding queried by cosine distance, where lower is better;
- ordinary zvec FTS using the standard tokenizer and lowercase filter, where higher is better;
- case-preserving zvec FTS using the standard tokenizer without filters and requiring every query token, where higher is better.

Tool evidence is stored only in the two FTS representations. Zvec 0.6.0 requires every row to contain a vector, so lexical-only rows receive a fixed zero sentinel generated inside the store; tool text is never sent to the embedding endpoint, and dense queries filter those rows out before ranking.

Search asks each channel for a bounded candidate set, deduplicates identical document IDs, and applies application-side reciprocal rank fusion. A single double-quoted substring uses zvec phrase syntax, so its tokens must be adjacent and ordered even when the query includes surrounding prose. Fusion policy version 2 uses rank constant 60 and excludes dense-only default-hybrid candidates whose cosine distance exceeds `0.5`; any lexical or identifier match remains eligible, and explicit deep reranking still receives the full fused pool. Equal fused scores sort by document ID. Before the gate selects a policy, the search-only fallback is 40 candidates per channel and 5 final results; no channel accepts more than 200 candidates. A clean passing evaluation replaces both fallback counts and the chunk policy with its measured selection. A failed gate approves no policy and blocks production indexing. Each response records the exact fusion version, constant, and channel caps it used.

Application-side fusion is deliberate. Zvec 0.6.0 supports native hybrid RRF through `multiQuerySync()`, but native results omit the component ranks and scores required for evaluation and source-backed diagnostics.

## Ranking and evidence shaping

Search shapes the full fused candidate pool before applying the requested result limit:

1. Merge identical document IDs while retaining each channel rank and score.
2. In default hybrid mode, exclude weak dense-only candidates above cosine distance `0.5`; preserve every lexical or identifier match.
3. Group overlapping reciprocal siblings from the same source run.
4. Group exact-content copies across sessions. Raw evidence never groups with a compaction or branch summary.
5. By default, rank groups deterministically by fused score plus a fixed `0.01` active-branch prior. Abandoned evidence remains eligible and carries an explicit label.
6. In explicit `deep-rerank` mode, send each representative's original text to Qwen3 and rank by its relevance score plus the same active-branch prior. The client requires one unique, in-range, finite score for every submitted index and maps scores by index rather than response order.
7. Apply the final result limit.
8. For winning atomic conversation chunks, fetch at most one immediate sibling on each side. Expansion requires reciprocal pointers and matching session, entry, role, evidence kind, visible text run, source geometry, and overlap text. Turn context, tool evidence, images, thinking, and summaries cannot become neighbors.

Each duplicate group retains every suppressed candidate with its source geometry and fusion components. Neighbor context retains every contributing atomic chunk. A deep-rerank HTTP, JSON, coverage, index, or score failure rejects that deep search; default hybrid search never calls the reranker.

After the quality gate passes, `/pi-session-recall-index` performs a manual full catch-up and optimizes zvec. Use `/pi-session-recall-index --rebuild` when a compatibility error requires a replacement generation. Automatic lifecycle ingestion never creates or replaces an incompatible generation. A search against a missing, locked, or incompatible generation fails without clearing another process's lock.

## Exact Octen tokenizer

The implementation pins:

| Identity                        | Value                                                              |
| ------------------------------- | ------------------------------------------------------------------ |
| Model                           | `Octen/Octen-Embedding-4B`                                         |
| Revision                        | `6e188e3b072c3e3678b235ad84e6e97bcbb71e8f`                         |
| Library                         | `@huggingface/tokenizers@0.1.3`                                    |
| `tokenizer.json` SHA-256        | `83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d` |
| `tokenizer_config.json` SHA-256 | `0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614` |
| Encode options                  | no special tokens; no token-type IDs                               |

The 11 MB `tokenizer.json` is not committed. Explicit indexing downloads a missing asset to a unique temporary file under the recall data directory, verifies its SHA-256, and atomically renames it into the revision cache. Every load verifies both cached files. Corruption and offline cache misses fail without a character-count fallback.

## Compatibility manifest

Each index generation has a separately versioned `index-manifest.json`. It identifies:

- request model, served model ID, artifact, dimensions, quantization, pooling, and a canonical FP32 embedding-canary vector, fingerprint, and cosine floor;
- tokenizer model, immutable revision, asset checksums, library version, and encode options;
- chunk limits, overlap, boundary algorithm, normalization, and policy version;
- conversation and provenance schema versions;
- session import policy version for framing, detection, and virtual conversion;
- project identity, lineage policy versions, and a canonical digest of personal lineage declarations;
- zvec schema, ordinary and case-preserving FTS configuration, FP32 vector storage, and pinned HNSW parameters.

The extension validates the complete manifest before opening or updating zvec. Missing or mismatched manifests are incompatible. The error reports every mismatched field and points to the implemented `/pi-session-recall-index --rebuild` operation. Rebuild removes only zvec, incremental state, and the old manifest under the writer lock; it preserves tokenizer assets and cached vectors. The quality gate must pass before the production command can run.

Embedding text is normalized to Unicode NFC under `unicode-nfc-v1`. Cache identity includes the normalized-text SHA-256; full served-model identity and dimensions; tokenizer revision, assets, library, and encode options; chunk-policy version; and normalization version. A model, text, tokenizer, policy, normalization, or dimension change therefore misses rather than reusing incompatible geometry.

## Inference profiles and HTTP backends

Model profiles define inference semantics; HTTP settings define where those semantics execute. The Octen embedding profile contains the existing manifest identity and preserves raw, unchanged text for both query and document operations. `RecallEmbeddingProvider` keeps those operations separate. The Qwen reranking profile defines ordered, finite, higher-is-more-relevant scores through `RecallRerankingProvider`.

Backend URLs, request timeouts, devices, and adapter implementations are not model-profile identity. Moving the same conforming profile to another HTTP backend does not require a vector rebuild. Changing an embedding identity field recorded in the manifest still requires a rebuild.

See [Inference profiles and provider conformance](docs/inference/provider-conformance.md) for the capability-specific HTTP contracts, deterministic conformance command, measurements, and live-evidence boundary.

The recommended EmbeddingGemma artifact is available only through an explicit operator workflow. Inspecting or checking it never downloads a model:

```bash
npm run --silent model:embeddinggemma -- inspect
npm run --silent model:embeddinggemma -- doctor
```

Download, repair, and removal require `--approve`:

```bash
npm run --silent model:embeddinggemma -- download --approve
npm run --silent model:embeddinggemma -- repair --approve
npm run --silent model:embeddinggemma -- remove --approve
```

This artifact workflow does not yet configure embedded inference or index session text. See [Pinned EmbeddingGemma model artifact](docs/inference/embeddinggemma-model-artifact.md) for the exact revision, size, checksum, prompts, tokenizer and canary policies, cache states, and pending legal and live-model evidence.

## Default local models

The checked-in embedding defaults match `~/.pi/agent/LOCAL-AI.md`:

| Setting         | Default                        |
| --------------- | ------------------------------ |
| Base URL        | `http://192.168.0.67:8090/v1`  |
| Request model   | `octen-embed`                  |
| Served model ID | `Octen/Octen-Embedding-4B`     |
| Artifact        | `Octen-Embedding-4B.Q8_0.gguf` |
| Quantization    | `Q8_0`                         |
| Pooling         | `last`                         |
| Dimensions      | `2560`                         |
| Batch size      | `16`                           |

The embedding endpoint must implement `POST /v1/embeddings` with the OpenAI request and response shape. Read-only search embeds the fixed canary before every query, so an in-process model swap is rejected before zvec opens. On ordinary indexing, an all-hit cache-only rebuild makes no Octen call; the first cache miss validates a fresh canary before any new chunk text reaches the model, preventing new-model vectors from entering an old generation. Explicit `--rebuild` refreshes and preflights the canary before deleting the old generation.

The manifest stores one canonical FP32 canary vector and uses its exact hash as embedding-cache identity. Compatibility compares a fresh canary by cosine similarity with a minimum of `0.9995`. This tolerates the measured geometry variation across llama.cpp parallel slots while rejecting larger drift. A tolerated rebuild retains the persisted canonical hash and can reuse vectors; a canary below the floor creates a new identity and misses the old cache.

Optional deep-rerank defaults are:

| Setting       | Default                       |
| ------------- | ----------------------------- |
| Base URL      | `http://192.168.0.67:8091/v1` |
| Request model | `qwen3-rerank`                |
| Endpoint      | `POST /v1/rerank`             |

In `deep-rerank` mode, the request is `{ model, query, documents, top_n }`. The response must contain `{ model, object, usage, results }`, with one `{ index, relevance_score }` result for every submitted document. Embedding and reranker HTTP requests abort after 60 seconds unless the caller cancels first.

## Configure

Create `~/.pi/agent/recall.json`:

```json
{
  "embeddingBaseUrl": "http://192.168.0.67:8090/v1",
  "embeddingModel": "octen-embed",
  "embeddingServedModelId": "Octen/Octen-Embedding-4B",
  "embeddingArtifact": "Octen-Embedding-4B.Q8_0.gguf",
  "embeddingQuantization": "Q8_0",
  "embeddingPooling": "last",
  "embeddingDimensions": 2560,
  "embeddingBatchSize": 16,
  "rerankerBaseUrl": "http://192.168.0.67:8091/v1",
  "rerankerModel": "qwen3-rerank",
  "denseCandidateLimit": 40,
  "lexicalCandidateLimit": 40,
  "identifierCandidateLimit": 40,
  "sessionsDirectory": "/home/will/.pi/agent/sessions",
  "dataDirectory": "/home/will/.pi/agent/recall",
  "projectLineages": {
    "git-origin:github.com/owner/successor": ["/home/you/projects/historical-prototype"]
  }
}
```

`projectLineages` maps each canonical `git-origin:<host>/<owner>/<repository>` or `git-common-directory:<absolute-path>` identity to one or more absolute historical session-origin roots. A root includes its descendants even when the old directory no longer exists. Lineage overrides Git discovery under that root. Roots assigned to different repository identities must not overlap.

Environment variables override the file settings listed below. `projectLineages` is personal file-only configuration and has no environment-variable or repository-local form.

- `PI_RECALL_CONFIG`
- `PI_RECALL_EMBEDDING_BASE_URL`
- `PI_RECALL_EMBEDDING_MODEL`
- `PI_RECALL_EMBEDDING_SERVED_MODEL_ID`
- `PI_RECALL_EMBEDDING_ARTIFACT`
- `PI_RECALL_EMBEDDING_QUANTIZATION`
- `PI_RECALL_EMBEDDING_POOLING`
- `PI_RECALL_EMBEDDING_DIMENSIONS`
- `PI_RECALL_EMBEDDING_BATCH_SIZE`
- `PI_RECALL_RERANKER_BASE_URL`
- `PI_RECALL_RERANKER_MODEL`
- `PI_RECALL_DENSE_CANDIDATE_LIMIT`
- `PI_RECALL_LEXICAL_CANDIDATE_LIMIT`
- `PI_RECALL_IDENTIFIER_CANDIDATE_LIMIT`
- `PI_RECALL_SESSIONS_DIRECTORY`
- `PI_RECALL_DATA_DIRECTORY`

Changing an embedding, session-import, project-lineage, or index identity recorded in the manifest makes the current generation incompatible. Incremental state also records the import policy, so older state cannot update a new generation. Rebuild explicitly with `/pi-session-recall-index --rebuild`; unchanged text reuses cached vectors because import metadata is excluded from embedding-cache identity. Reranker settings are search-time policy and do not require an index rebuild. Do not delete or rewrite index state through search.

## Storage and concurrency

Default data paths:

```text
~/.pi/agent/recall/zvec/                    zvec collection
~/.pi/agent/recall/index-state.json         incremental session fingerprints
~/.pi/agent/recall/index-manifest.json      generation compatibility identity
~/.pi/agent/recall/tokenizers/<revision>/   checksum-verified tokenizer assets
~/.pi/agent/recall/embedding-cache/v1/      durable content-addressed FP32 vectors
~/.pi/agent/recall/models/                  explicitly approved pinned model artifacts
~/.pi/agent/recall/operation.lock/          explicit-index writer lock
```

A process-local mutex and PID-owned writer lock serialize manual and automatic indexing. Targeted lifecycle ingestion waits at most 250 milliseconds for another writer, then defers quietly until the next lifecycle event. The active-session freshness barrier waits under the tool's cancellation signal because search must not silently return stale evidence from its invoking session. Search never clears or repairs another process's lock. Manual full catch-up checkpoints every 100 changed files; targeted reconciliation checkpoints its one session immediately. Embedding writes use bounded 128-chunk windows.

The embedding cache is a sibling of zvec rather than part of the collection. Each entry has a versioned identity header, FP32 payload, and SHA-256 checksum. Writers fsync a unique temporary file and atomically rename it only after validation. Readers reject identity, dimension, byte-length, checksum, and non-finite-value failures. Rebuilding only zvec and index state leaves the cache available, so unchanged chunks need zero chunk-embedding requests. Index completion reports cache hits, newly embedded chunks, and chunk-embedding request count separately; the model-identity canary request is not a chunk-embedding request.

Conversation text and vectors remain local. The extension sends dense-searchable atomic, turn-context, and summary text only to the configured embedding endpoint. Tool evidence remains lexical-only and is never sent for embedding. Default hybrid search sends nothing to the reranker. Explicit `deep-rerank` sends original text from every representative candidate kind—including tool evidence—to the configured local reranker.

## Evaluate before backfill

Run the fixed quality and latency gate before approving a full corpus backfill:

```bash
npm run evaluate:recall
```

The command reads only the 15 checksum-fixed sessions under `evaluation/corpus/`. It builds one temporary 512/64 index under the ignored `evaluation/.recall-data/recall-quality-evaluation/` directory and measures the approved policy of 8 candidates per channel and 5 final results. Its Git fixtures are temporary repositories in that same guarded directory; the production project identity resolver must derive the declared main-checkout, worktree, clone, and unrelated-repository identities before indexing begins. The 17 cases preserve the established global retrieval classes and add main/worktree, equivalent-clone, configured-lineage, unrelated-similar-project, exact non-Git, and explicit-global coverage. A global control proves dense, lexical, and identifier project restrictions apply before each channel limit. The default hybrid run makes zero reranker requests and writes:

- `docs/evaluation/recall-quality-report.md`
- `docs/evaluation/recall-quality-results.json`

The command never scans the configured production session directory or starts the full backfill. It exits with status 2 when no measured configuration passes every frozen quality and latency threshold. The Pi index command reads the committed result and refuses to scan production sessions unless the run is clean, the automated gate passes, and it selects chunk, candidate, and final-result counts. Invoking the unblocked index command remains the human approval step.

Only clean version-4 evidence can unblock production indexing. The gate binds the result to the current default project scope, repository-identity and lineage policies, lineage digest, hybrid rank-fusion constants, 512/64 chunk geometry, per-channel candidate limits, and final-result count. `/pi-session-recall-index` rejects missing, pre-scope, stale-policy, dirty, failed, or unapproved-policy evidence.

## Develop

All tests use explicit fixture session directories and temporary recall data directories. The inference conformance tests use temporary deterministic HTTP servers; they do not require a live model.

```bash
npm install
node --import tsx --test src/recall-inference-conformance.test.ts
npm test
npm run typecheck
npm run lint
npm run format:check
```

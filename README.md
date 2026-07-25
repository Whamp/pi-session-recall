# Pi Session Recall

`pi-session-recall` gives Pi a `pi-session-recall` tool for searching past conversations by meaning or exact text. It reads Pi session JSONL files, embeds user-visible conversation text with a local OpenAI-compatible model, stores durable FP32 vectors in a content-addressed cache, and builds dense plus full-text indexes in an in-process [zvec](https://github.com/alibaba/zvec) collection. Search fuses every bounded retrieval channel and suppresses duplicate evidence. Fast deterministic hybrid ranking is the default; local Qwen3 reranking is an explicit deep-search option.

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

## Install

```bash
pi install /home/will/projects/pi-session-recall
```

Reload a running Pi session with `/reload`.

The committed quality report currently fails its latency gate, so the extension blocks production indexing. Do not start the full backfill. After a clean bounded evaluation passes, build or update the production index explicitly:

```text
/pi-session-recall-index
```

Replace a missing or incompatible generation while preserving the tokenizer and embedding caches:

```text
/pi-session-recall-index --rebuild
```

Search never starts, resumes, or repairs indexing. It opens only an existing compatible zvec collection in read-only mode.

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

Search asks each channel for a bounded candidate set, deduplicates identical document IDs, and applies application-side reciprocal rank fusion. A single double-quoted substring uses zvec phrase syntax, so its tokens must be adjacent and ordered even when the query includes surrounding prose. Fusion policy version 1 uses rank constant 60. Equal fused scores sort by document ID. Before the gate selects a policy, the search-only fallback is 40 candidates per channel and 5 final results; no channel accepts more than 200 candidates. A clean passing evaluation replaces both fallback counts and the chunk policy with its measured selection. A failed gate approves no policy and blocks production indexing. Each response records the exact fusion version, constant, and channel caps it used.

Application-side fusion is deliberate. Zvec 0.6.0 supports native hybrid RRF through `multiQuerySync()`, but native results omit the component ranks and scores required for evaluation and source-backed diagnostics.

## Ranking and evidence shaping

Search shapes the full fused candidate pool before applying the requested result limit:

1. Merge identical document IDs while retaining each channel rank and score.
2. Group overlapping reciprocal siblings from the same source run.
3. Group exact-content copies across sessions. Raw evidence never groups with a compaction or branch summary.
4. By default, rank groups deterministically by fused score plus a fixed `0.01` active-branch prior. Abandoned evidence remains eligible and carries an explicit label.
5. In explicit `deep-rerank` mode, send each representative's original text to Qwen3 and rank by its relevance score plus the same active-branch prior. The client requires one unique, in-range, finite score for every submitted index and maps scores by index rather than response order.
6. Apply the final result limit.
7. For winning atomic conversation chunks, fetch at most one immediate sibling on each side. Expansion requires reciprocal pointers and matching session, entry, role, evidence kind, visible text run, source geometry, and overlap text. Turn context, tool evidence, images, thinking, and summaries cannot become neighbors.

Each duplicate group retains every suppressed candidate with its source geometry and fusion components. Neighbor context retains every contributing atomic chunk. A deep-rerank HTTP, JSON, coverage, index, or score failure rejects that deep search; default hybrid search never calls the reranker.

After the quality gate passes, run `/pi-session-recall-index` to scan changed sessions and optimize zvec. Use `/pi-session-recall-index --rebuild` when a compatibility error requires a replacement generation. A search against a missing, locked, or incompatible generation fails without changing the index or its lock.

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
- zvec schema, ordinary and case-preserving FTS configuration, FP32 vector storage, and pinned HNSW parameters.

The extension validates the complete manifest before opening or updating zvec. Missing or mismatched manifests are incompatible. The error reports every mismatched field and points to the implemented `/pi-session-recall-index --rebuild` operation. Rebuild removes only zvec, incremental state, and the old manifest under the writer lock; it preserves tokenizer assets and cached vectors. The quality gate must pass before the production command can run.

Embedding text is normalized to Unicode NFC under `unicode-nfc-v1`. Cache identity includes the normalized-text SHA-256; full served-model identity and dimensions; tokenizer revision, assets, library, and encode options; chunk-policy version; and normalization version. A model, text, tokenizer, policy, normalization, or dimension change therefore misses rather than reusing incompatible geometry.

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
  "dataDirectory": "/home/will/.pi/agent/recall"
}
```

Environment variables override the file:

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

Changing an embedding or index identity recorded in the manifest makes the current generation incompatible. Reranker settings are search-time policy and do not require an index rebuild. Do not delete or rewrite index state through search.

## Storage and concurrency

Default data paths:

```text
~/.pi/agent/recall/zvec/                    zvec collection
~/.pi/agent/recall/index-state.json         incremental session fingerprints
~/.pi/agent/recall/index-manifest.json      generation compatibility identity
~/.pi/agent/recall/tokenizers/<revision>/   checksum-verified tokenizer assets
~/.pi/agent/recall/embedding-cache/v1/      durable content-addressed FP32 vectors
~/.pi/agent/recall/operation.lock/          explicit-index writer lock
```

A process-local mutex and PID-owned writer lock serialize explicit indexing. Search does not acquire, clear, or repair the writer lock. It distinguishes a live owner from a stale dead-process lock and directs stale-lock recovery through the explicit index command after the quality gate passes. Changed sessions are checkpointed every 100 files, and embedding writes use bounded 128-chunk windows.

The embedding cache is a sibling of zvec rather than part of the collection. Each entry has a versioned identity header, FP32 payload, and SHA-256 checksum. Writers fsync a unique temporary file and atomically rename it only after validation. Readers reject identity, dimension, byte-length, checksum, and non-finite-value failures. Rebuilding only zvec and index state leaves the cache available, so unchanged chunks need zero chunk-embedding requests. Index completion reports cache hits, newly embedded chunks, and chunk-embedding request count separately; the model-identity canary request is not a chunk-embedding request.

Conversation text and vectors remain local. The extension sends dense-searchable atomic, turn-context, and summary text only to the configured embedding endpoint. Tool evidence remains lexical-only and is never sent for embedding. Default hybrid search sends nothing to the reranker. Explicit `deep-rerank` sends original text from every representative candidate kind—including tool evidence—to the configured local reranker.

## Evaluate before backfill

Run the fixed quality and latency gate before approving a full corpus backfill:

```bash
npm run evaluate:recall
```

The command reads only the eight checksum-fixed sessions under `evaluation/corpus/`. It builds one temporary 512/64 index under the ignored `.recall-data/recall-quality-evaluation/` directory, measures fused top-five quality for the frozen 8, 16, 24, and 32 candidates-per-channel grid, makes zero reranker requests, and writes:

- `docs/evaluation/recall-quality-report.md`
- `docs/evaluation/recall-quality-results.json`

The command never scans the configured production session directory or starts the full backfill. It exits with status 2 when no measured configuration passes every frozen quality and latency threshold. The Pi index command reads the committed result and refuses to scan production sessions unless the run is clean, the automated gate passes, and it selects chunk, candidate, and final-result counts. Invoking the unblocked index command remains the human approval step.

The committed report is valid only when its version-3 rerank-free gate was generated from a clean worktree. Until that gate passes and selects a policy, `/pi-session-recall-index` remains blocked.

## Develop

All tests use explicit fixture session directories and temporary recall data directories.

```bash
npm install
npm test
npm run typecheck
npm run lint
npm run format:check
```

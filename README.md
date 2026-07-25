# Pi Session Recall

`pi-session-recall` gives Pi a `pi-session-recall` tool for searching past conversations. It reads Pi session JSONL files, embeds user-visible conversation text with a local OpenAI-compatible model, and stores vectors in an in-process [zvec](https://github.com/alibaba/zvec) collection.

## What it indexes

The index contains:

- user messages;
- assistant text;
- visible custom messages;
- compaction summaries;
- branch summaries.

Each atomic document comes from one visible text run in one session entry. A visible text run contains only adjacent, nonempty text blocks. Thinking, tools, results, images, empty blocks, roles, entries, and summaries all end a run, so excluded content can never create synthetic text adjacency.

Chunks use the exact pinned Octen tokenizer. Each chunk contains at most 1,024 tokens and overlaps its adjacent siblings by at most 128 tokens. Splitting prefers Markdown sections, paragraphs, fenced-code boundaries, lines, and sentences before a hard token cut.

Every stored document includes:

- session, parent-session, project, entry, and parent-entry provenance;
- effective-leaf, active-branch, active-context, and many-path branch membership;
- compaction and branch-summary links;
- source line, block, character, and token spans;
- text-run identity, chunk order, count, and sibling IDs.

Pi does not assign branch IDs. Shared ancestors therefore record every endpoint path that contains them instead of a misleading singular branch ID.

## Install

```bash
pi install /home/will/projects/pi-session-recall
```

Reload a running Pi session with `/reload`. Build a fresh index explicitly:

```text
/pi-session-recall-index
```

Search never starts or resumes indexing. It opens only an existing compatible zvec collection in read-only mode.

## Use

Pi can call the tool directly:

```text
pi-session-recall({ query: "What did we decide about the job queue?", limit: 5 })
```

The extension tells Pi to use `pi-session-recall` when a task depends on a past conversation or a detail absent from current context. Each result contains:

- semantic distance;
- session name, date, role, and project directory;
- a concise text excerpt;
- source provenance in `SESSION_FILE#ENTRY_ID` form.

Run `/pi-session-recall-index` when you explicitly want to scan changed sessions and optimize zvec. A search against a missing, locked, or incompatible generation fails without changing the index or its lock.

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

- request model, served model ID, artifact, dimensions, quantization, pooling, and a runtime embedding-canary fingerprint;
- tokenizer model, immutable revision, asset checksums, library version, and encode options;
- chunk limits, overlap, boundary algorithm, normalization, and policy version;
- conversation and provenance schema versions;
- zvec schema, FTS configuration, FP32 vector storage, and pinned HNSW parameters.

The extension validates the complete manifest before opening or updating zvec. Missing or mismatched manifests are incompatible. The error reports every mismatched field and points to `/pi-session-recall-index --rebuild`; recall never mixes document geometry silently.

## Default local model

The checked-in defaults match `~/.pi/agent/LOCAL-AI.md`:

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

The endpoint must implement `POST /v1/embeddings` with the OpenAI request and response shape. Each process embeds one fixed canary before opening a manifested index; a changed vector fingerprint makes the generation incompatible.

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
- `PI_RECALL_SESSIONS_DIRECTORY`
- `PI_RECALL_DATA_DIRECTORY`

Changing any manifested identity makes the current generation incompatible. Do not delete or rewrite it through search.

## Storage and concurrency

Default data paths:

```text
~/.pi/agent/recall/zvec/                    zvec collection
~/.pi/agent/recall/index-state.json         incremental session fingerprints
~/.pi/agent/recall/index-manifest.json      generation compatibility identity
~/.pi/agent/recall/tokenizers/<revision>/   checksum-verified tokenizer assets
~/.pi/agent/recall/operation.lock/          explicit-index writer lock
```

A process-local mutex and PID-owned writer lock serialize explicit indexing. Search does not acquire, clear, or repair the writer lock; it refuses to run while the lock exists. Changed sessions are checkpointed every 100 files, and embedding writes use bounded 128-chunk windows.

Conversation text and vectors remain local. The extension sends text only to the configured embedding endpoint.

## Develop

All tests use explicit fixture session directories and temporary recall data directories.

```bash
npm install
npm test
npm run typecheck
npm run lint
npm run format:check
```

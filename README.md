# Pi Session Recall

Pi Session Recall searches past Pi conversations by meaning or exact text. It reads Pi session JSONL files, stores searchable evidence in one local zvec collection, and returns the original file and line range for every result.

Index maintenance is manual. The standalone `psr` command is the only writer. The Pi `pi-session-recall` tool is read-only.

## Install

```bash
npm install
npm link
pi install /path/to/pi-session-recall
```

Reload Pi after installation. Build the first index explicitly:

```bash
psr index --rebuild
```

## Commands

```bash
psr index             # add, update, and remove changed session evidence
psr index --rebuild   # replace incompatible or damaged index state
psr index --compact   # keep the former one-line stdout summary
```

`psr index`:

- recursively scans configured `.jsonl` session files;
- skips files whose size and modification time have not changed;
- reuses matching vectors already stored in zvec;
- calls Octen only for changed dense-searchable evidence;
- removes evidence for deleted session files;
- reports malformed session files and continues with healthy files;
- shows elapsed time and estimates time remaining after a healthy file completes;
- optimizes zvec after a changed pass.

The estimate uses the observed rate of healthy files in the current run. Until enough work completes, the command says that it is calculating the estimate rather than inventing an initial duration. `--compact` preserves the former one-line completed summary and `Failed: ...` lines on stdout; progress remains on stderr.

No startup hook, completed-turn hook, shutdown hook, watcher, background worker, daemon, or search request updates the index.

The indexer checkpoints `index-state.json` after every 100 changed sessions and after the final partial batch. If indexing stops between checkpoints, rerun `psr index`; it safely revisits the uncheckpointed sessions and reuses matching vectors already stored in zvec.

## Search

Pi calls:

```text
pi-session-recall({ query: "What did we decide about the job queue?" })
pi-session-recall({ query: "readNodeErrorCode", limit: 5 })
pi-session-recall({ query: "cross-project decision", scope: "global" })
```

Parameters:

```ts
{
  query: string;
  limit?: number;                 // default 5, maximum 10
  scope?: "project" | "global";  // default project
}
```

Project scope filters each retrieval channel before its eight-candidate limit. Global scope searches the complete collection.

Each search uses three bounded channels:

1. normalized Octen inner-product search;
2. case-insensitive full-text search;
3. case-preserving identifier search.

Application-side reciprocal-rank fusion retains every component rank and score. Ranking suppresses overlapping sibling chunks and exact copies across sessions, applies a `0.01` active-branch preference, and keeps abandoned-branch evidence eligible. A winning atomic conversation chunk can include one exact contiguous sibling on either side. Expansion requires matching session, entry, role, visible text run, source geometry, and reciprocal sibling links.

## Source-backed results

Each model-visible result ends with a locator such as:

```text
Source: /home/you/.pi/agent/sessions/project/session.jsonl:142-146#entry-id
```

Tool details also include:

- physical session path;
- entry and contributing-entry IDs;
- source line start and end;
- source block start and end;
- character start and end;
- duplicate occurrence locations;
- every chunk used for neighbor context.

The agent can read those JSONL lines directly when it needs surrounding source context. Recall does not reopen session files during search and has no separate source-neighborhood database.

## Indexed evidence

The index contains:

- visible user and assistant text;
- user/assistant turn-context documents;
- visible custom messages;
- compaction and branch summaries;
- tool names and argument objects;
- tool result text, including errors, paths, identifiers, and URLs;
- direct bash commands and output.

The index excludes:

- thinking;
- images;
- empty tool placeholders;
- `pi-session-recall` calls and their results.

Tool evidence is lexical-only and never reaches Octen. Zvec 0.6 requires every row to have a vector, so lexical-only rows receive a zero sentinel and dense queries filter them out.

## Session import

The importer never rewrites Pi session files. It supports:

- canonical Pi JSONL sessions at supported versions;
- unversioned Pi v1 sessions through deterministic virtual conversion;
- reused physical session files containing several independently validated logical sessions.

Every accepted representation enters the same strict session graph parser. Unsupported, malformed, truncated, cyclic, duplicate-ID, invalid-leaf, invalid-reference, and missing-parent inputs produce no searchable documents for that physical file. `psr index` records the failure and continues.

The guarded compatibility replay remains available for an explicit non-production corpus:

```bash
npm run --silent replay:session-import -- --corpus-root /path/to/session-corpus
```

## Chunking

The index uses the checksum-pinned `Octen/Octen-Embedding-4B` tokenizer with the frozen policy:

- 512 tokens maximum;
- 64 tokens overlap between adjacent atomic conversation chunks;
- structural boundaries before hard token cuts;
- no overlap for lexical-only tool evidence.

Atomic chunks never cross entries, roles, visible text runs, tools, thinking, images, results, or summaries. Turn-context documents retain both user and assistant text and cite every contributing entry.

## Octen stored dimensions

The default profile uses:

| Setting           | Value                                |
| ----------------- | ------------------------------------ |
| Request model     | `octen-embed`                        |
| Served model      | `Octen/Octen-Embedding-4B`           |
| Native dimensions | 2,560                                |
| Stored dimensions | 1,024                                |
| Transformation    | first N, then local L2 normalization |
| Zvec metric       | inner product                        |

The same transformation applies to document and query vectors. Inner product preserves cosine ordering because both sides are normalized. The feature is vendor-supported prefix storage; this repository does not claim independently verified MRL quality at every cutoff.

Both `psr index` and `pi-session-recall` require the configured Octen HTTP endpoint. This package has no local embedding fallback.

The manifest binds request model, served model, native width, stored width, transformation, tokenizer assets, 512/64 chunking, import policy, project identity policy, and zvec schema. Any change requires:

```bash
psr index --rebuild
```

## Configuration

Configuration defaults to `~/.pi/agent/recall.json`:

```json
{
  "embeddingBaseUrl": "http://192.168.0.67:8090/v1",
  "embeddingModel": "octen-embed",
  "embeddingServedModelId": "Octen/Octen-Embedding-4B",
  "embeddingNativeDimensions": 2560,
  "embeddingStoredDimensions": 1024,
  "embeddingBatchSize": 16,
  "projectLineages": {}
}
```

Environment overrides:

- `PI_RECALL_CONFIG`
- `PI_RECALL_SESSIONS_DIRECTORY`
- `PI_RECALL_DATA_DIRECTORY`
- `PI_RECALL_EMBEDDING_BASE_URL`
- `PI_RECALL_EMBEDDING_MODEL`
- `PI_RECALL_EMBEDDING_SERVED_MODEL_ID`
- `PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS`
- `PI_RECALL_EMBEDDING_STORED_DIMENSIONS`
- `PI_RECALL_EMBEDDING_BATCH_SIZE`

## Local state

Durable recall state contains only:

```text
~/.pi/agent/recall/
├── zvec/
├── index-state.json
└── index-manifest.json
```

The tokenizer loader also keeps checksum-verified tokenizer assets under `tokenizers/`; these are replaceable inference inputs, not recall state. `operation.lock` exists only while `psr` owns the writer lock and is removed when the command exits.

There is no embedding cache, projection database, generation registry, active pointer, activation protocol, replay log, rollback state, marker spool, or model-artifact cache.

## Development validation

```bash
npm test
npm run typecheck
npm run lint
npm run format:check
```

The bounded quality evaluator remains a development command:

```bash
npm run evaluate:recall
```

It reads only the checksum-fixed evaluation corpus, builds disposable indexes, and measures the frozen 512/64, eight-candidates-per-channel, five-final-results policy. Production indexing does not read or gate on generated evaluation files.

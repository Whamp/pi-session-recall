# Pi Session Recall

`pi-session-recall` gives Pi a `recall` tool for searching past conversations. It reads Pi session JSONL files, embeds user-visible conversation text with a local OpenAI-compatible model, and stores vectors in an in-process [zvec](https://github.com/alibaba/zvec) collection.

## What it indexes

The index includes:

- user messages;
- assistant text;
- compaction and branch summaries;
- visible custom messages.

It excludes images, hidden reasoning, tool calls, tool results, and shell output. Every match includes the original session file and entry ID.

Sessions are indexed incrementally. Unchanged files make no embedding requests. Appended entries add only new or changed chunks. Deleting a session removes its chunks on the next pass.

## Install

```bash
pi install /home/will/projects/pi-session-recall
```

Reload a running Pi session with `/reload`, then build the initial index:

```text
/recall-index
```

The first backfill is resumable. Progress is checkpointed every 100 changed sessions in `~/.pi/agent/recall/index-state.json`; chunks already written to zvec are not embedded again after a restart.

## Use

Pi can call the tool directly:

```text
recall({ query: "What did we decide about the job queue?", limit: 5 })
```

The extension tells Pi to use `recall` when a task depends on a past conversation or a detail absent from current context. Each result contains:

- semantic score;
- session name, date, role, and project directory;
- a concise text excerpt;
- source provenance in `SESSION_FILE#ENTRY_ID` form.

The tool checks for changed sessions before every search. Run `/recall-index` when you want an explicit full scan and zvec optimization.

## Default local model

The checked-in defaults match `~/.pi/agent/LOCAL-AI.md`:

| Setting    | Default                       |
| ---------- | ----------------------------- |
| Base URL   | `http://192.168.0.67:8090/v1` |
| Model      | `octen-embed`                 |
| Dimensions | `2560`                        |
| Batch size | `16`                          |

The endpoint must implement `POST /v1/embeddings` with the OpenAI request and response shape.

## Configure

Create `~/.pi/agent/recall.json`:

```json
{
  "embeddingBaseUrl": "http://192.168.0.67:8090/v1",
  "embeddingModel": "octen-embed",
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
- `PI_RECALL_EMBEDDING_DIMENSIONS`
- `PI_RECALL_EMBEDDING_BATCH_SIZE`
- `PI_RECALL_SESSIONS_DIRECTORY`
- `PI_RECALL_DATA_DIRECTORY`

Changing embedding dimensions requires a fresh collection:

```bash
rm -rf ~/.pi/agent/recall
```

Then run `/recall-index` again. The extension detects dimension drift and refuses to query the old collection.

## Storage and concurrency

Default data paths:

```text
~/.pi/agent/recall/zvec/             zvec collection
~/.pi/agent/recall/index-state.json  incremental session fingerprints
~/.pi/agent/recall/operation.lock/   cross-process writer lock
```

A process-local mutex and PID-owned lock serialize indexing and search operations across Pi sessions. A dead or incomplete process lock is removed automatically. Embeddings are written in bounded 128-chunk windows so one large session cannot retain the full vector set in memory.

Conversation text and vectors remain local. The extension sends text only to the configured embedding endpoint.

## Develop

```bash
npm install
npm test
npm run typecheck
npm run lint
npm run format:check
```

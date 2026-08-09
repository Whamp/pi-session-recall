# Pi Session Recall

Pi Session Recall searches past Pi conversations by meaning or exact text. It reads Pi session JSONL files, stores searchable evidence in one local zvec collection, and returns the original file and line range for every result.

The standalone `psr index` command is the only index writer. Run it directly or opt into a native per-user schedule. `psr ignore` writes PSR policy state but never opens the index. Pi lifecycle and the `pi-session-recall` tool remain read-only.

## Why index session history?

Raw session JSONL remains the source of truth, but asking a fresh agent to search it spends time and model context on file discovery, format parsing, branch structure, compaction, and evidence location. Pi Session Recall does that work ahead of time, limits search to the invoking project by default, combines meaning, ordinary text, and identifier retrieval, and returns exact JSONL line citations.

A measured production comparison gave the same question to the recall tool and to a fresh agent restricted to raw JSONL. The full hybrid tool took 1.48 seconds at the median. The raw agent took 94.43 seconds, examined 54 project files, and used 141,682 tokens plus 852,480 cached tokens. The agent found the answer reliably; the tool required its maximum ten results to include the answer at rank ten. Indexed recall was about 64 times faster and far cheaper in this case, while the ranking result exposed work still needed. This is one measured query, not a universal quality or capacity claim. See [Production recall index value benchmark](docs/research/production-recall-index-value-benchmark.md).

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
psr index                                      # add, update, and remove changed session evidence
psr index --rebuild                            # replace incompatible or damaged index state
psr index --compact                            # keep the former one-line stdout summary
psr optimize                                   # explicitly optimize existing zvec data
psr auto-index install                         # update hourly without optimization
psr auto-index install --interval 30m           # replace the update interval
psr auto-index install --optimize-daily         # also optimize daily at 23:00
psr auto-index uninstall                        # remove every installed schedule
psr ignore add path/to/session.jsonl            # exclude one exact physical session path
psr ignore list                                 # print sorted excluded paths
psr ignore remove path/to/session.jsonl         # make one exact path eligible again
```

`psr index`:

- recursively scans configured `.jsonl` session files;
- skips files whose size and modification time have not changed;
- reuses matching vectors already stored in zvec;
- calls Octen only for changed dense-searchable evidence;
- removes evidence for deleted or newly ignored indexed session files;
- skips ignored files before parsing or embedding them;
- reports malformed eligible session files and continues with healthy files;
- shows elapsed time and estimates time remaining after a healthy file completes;
- leaves collection optimization to the explicit `psr optimize` command.

The estimate uses the observed rate of healthy files in the current run. Until enough work completes, the command says that it is calculating the estimate rather than inventing an initial duration. `--compact` preserves the former one-line completed summary and `Failed: ...` lines on stdout; progress remains on stderr. The legacy `--no-optimize` flag remains accepted as a compatibility alias for ordinary update-only indexing.

`psr optimize` does not scan or index sessions. It compacts the existing collection under the same writer lock. Compaction merges FTS segments as well as vector data, so BM25 scores and ranking can change even though the indexed evidence does not. The operation may write near-collection-sized temporary output.

No startup hook, completed-turn hook, shutdown hook, watcher, package daemon, or search request updates the index.

### Ignoring exact physical session paths

Ignore state persists in `~/.pi/agent/recall/physical-session-ignore.json`. Both manual and scheduled `psr index` runs read one snapshot after acquiring the index lock. An ignored new file is not parsed, embedded, stored, or added to `index-state.json`. If the file is already indexed, the next maintenance pass removes its documents and state. Removing the ignore makes the unchanged source eligible as a new file on the next pass.

`add` and `remove` are idempotent. They report `Already ignored` or `Not ignored` and exit successfully when no state changes. `list` prints normalized paths in ordinary string order, one per line, and prints nothing when the list is empty.

Path identity is exact and lexical. Relative command arguments resolve from the command's current working directory with `path.resolve`. Commands do not expand `~`, resolve symlinks, inspect the filesystem, require a `.jsonl` suffix, restrict paths to the configured sessions directory, or interpret glob characters. For example, `*.jsonl` names a path containing a literal asterisk. Symlink aliases remain distinct paths.

`psr index --rebuild` preserves ignore state and rebuilds only eligible files. Invalid ignore state aborts maintenance before rebuild removes the working index.

### Automatic index maintenance

`psr auto-index install` creates one per-user native schedule that updates changed evidence without optimizing zvec. The interval accepts a positive whole number followed by lowercase `m` or `h`; the default is `1h`. Installation never uses `sudo`. Reinstalling replaces the definition and captures absolute paths to the current Node executable and installed package.

Optimization is optional. Add `--optimize-daily` to install a second schedule that runs `psr optimize` every day at 23:00 local time. Reinstalling without that flag disables and removes any older optimization schedule. Enable it only when measured query latency, ranking, or workload evidence justifies collection-wide compaction.

The generated definitions run the equivalent of:

```text
<absolute-node> --import tsx <absolute-package-root>/bin/psr index
# Only with --optimize-daily:
<absolute-node> --import tsx <absolute-package-root>/bin/psr optimize
```

Both commands use the same writer lock, so an update and optional optimization cannot write concurrently. Generated definitions do not copy `PI_RECALL_*` overrides from the installation shell. Scheduled runs use the durable recall configuration file and normal defaults.

On Linux, default installation writes one systemd user service and timer under `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/`. `--optimize-daily` adds a second service and timer. Installation starts the selected timers, then attempts one immediate index run. If that run fails, installation warns but leaves the timers active. Read logs with:

```bash
journalctl --user-unit=pi-session-recall-index.service
journalctl --user-unit=pi-session-recall-optimize.service # opt-in schedule only
```

On macOS, default installation writes one mode-`0600` LaunchAgent under `~/Library/LaunchAgents/`; it uses `RunAtLoad`. `--optimize-daily` adds a calendar LaunchAgent for 23:00. They write separate logs:

```text
~/.pi/agent/logs/pi-session-recall-auto-index.out.log
~/.pi/agent/logs/pi-session-recall-auto-index.err.log
~/.pi/agent/logs/pi-session-recall-auto-optimize.out.log
~/.pi/agent/logs/pi-session-recall-auto-optimize.err.log
```

The macOS path is runtime-untested. No Mac was available to verify plist acceptance; `RunAtLoad`, `StartInterval`, or `StartCalendarInterval` execution; retry after an exit-status-1 run; overlap suppression across both jobs; absolute Node plus `--import tsx`; log appends; or access to the durable recall configuration and embedding endpoint from the LaunchAgent environment. Other platforms fail with an unsupported-platform error.

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
├── index-manifest.json
└── physical-session-ignore.json
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

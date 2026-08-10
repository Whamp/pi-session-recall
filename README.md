# Pi Session Recall

Pi Session Recall searches past Pi conversations and compact tool Invocations. It stores dense conversation documents in a flat local Zvec collection, stores bounded tool-call and command locators in SQLite, and returns the original JSONL file and line range for every result.

The standalone `psr index` command is the only index writer. Run it directly or opt into a native per-user schedule. `psr ignore` writes PSR policy state but never opens the index. Pi lifecycle and the `pi-session-recall` tool remain read-only.

## Why index session history?

Raw session JSONL remains the source of truth, but asking a fresh agent to search it spends time and model context on file discovery, format parsing, branch structure, compaction, and evidence location. Pi Session Recall does that work ahead of time, limits search to the invoking project by default, searches dense conversations and compact Invocations together, and returns exact JSONL line citations.

A measured production comparison gave the same question to the recall tool and to a fresh agent restricted to raw JSONL. The full hybrid tool took 1.48 seconds at the median. The raw agent took 94.43 seconds, examined 54 project files, and used 141,682 tokens plus 852,480 cached tokens. The agent found the answer reliably; the tool required its maximum ten results to include the answer at rank ten. Indexed recall was about 64 times faster and far cheaper in this case, while the ranking result exposed work still needed. This is one measured query, not a universal quality or capacity claim. See [Production recall index value benchmark](docs/research/production-recall-index-value-benchmark.md).

The compact replacement passed its pre-activation storage, retrieval, Source, and incremental-write gates. See [Compact production recall certification](docs/research/compact-production-recall-certification.md).

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
psr index --rebuild                            # build and atomically activate a replacement database
psr index --rebuild --stage                    # build a replacement without activating it
psr index --rebuild --stage --resume           # continue one interrupted staged build
psr index --rebuild --stage --resume --reuse-active-vectors # reuse checksum-matched vectors
psr activate generations/generation-...        # activate one certified staged database
psr rollback                                   # restore the immediately previous database
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
- reuses matching vectors already stored in the dense Zvec collection;
- calls Octen only for changed conversation, summary, and turn-context documents;
- replaces each changed session's compact Invocation rows in SQLite;
- removes evidence for deleted or newly ignored indexed session files;
- skips ignored files before parsing or embedding them;
- reports malformed eligible session files and continues with healthy files;
- shows elapsed time and estimates time remaining after a healthy file completes;
- leaves collection optimization to the explicit `psr optimize` command.

`psr index --rebuild` builds a candidate recall database beside the active database. A fatal error, cancellation, or failed session leaves normal recall on the active database. A successful rebuild closes and verifies the candidate, then atomically makes it active. The replaced database remains available to `psr rollback`. The next rebuild removes failed or interrupted candidates but never removes the previous database.

Add `--stage` when the candidate must pass checks before activation. A staged rebuild uses a separate construction lock, so normal search and scheduled updates keep using the active database. The command prints the exact `generations/generation-...` target and leaves the active pointer unchanged. If a fatal dependency outage interrupts the build, rerun it once with `--rebuild --stage --resume`. Resume requires exactly one interrupted candidate and preserves every completed Physical session. Add `--reuse-active-vectors` during the compact-layout cutover to avoid re-embedding unchanged Dense recall documents. Reuse requires the Active recall database to have the same embedding profile. The indexer also requires each canonical document ID and checksum to match before it copies that vector; changed documents still go to the embedding provider.

Run `psr activate <database-target>` only after that target passes its checks. Activation takes the shared writer lock, verifies the staged database is complete, records the current database for rollback, and replaces the active pointer atomically.

The first rebuild after upgrading safely adopts the current unversioned layout. It remains active while the candidate builds and becomes the previous database after activation. Run `psr rollback` to restore it without rebuilding. Existing installations can keep using the unversioned database until they activate a compact generation.

The estimate uses the observed rate of healthy files in the current run. Until enough work completes, the command says that it is calculating the estimate rather than inventing an initial duration. `--compact` preserves the former one-line completed summary and `Failed: ...` lines on stdout; progress remains on stderr. The legacy `--no-optimize` flag remains accepted as a compatibility alias for ordinary update-only indexing.

`psr optimize` does not scan or index sessions. It compacts the existing flat dense collection under the same writer lock and may write near-collection-sized temporary output. Invocation search lives in SQLite and is not part of this operation.

No startup hook, completed-turn hook, shutdown hook, watcher, package daemon, or search request updates the index.

### Ignoring exact physical session paths

Ignore state persists in `~/.pi/agent/recall/physical-session-ignore.json`. Both manual and scheduled `psr index` runs read one snapshot after acquiring the index lock. An ignored new file is not parsed, embedded, stored, or added to `recall-catalog.sqlite`. If the file is already indexed, the next maintenance pass removes its documents, compact Invocation records, and catalog state. Removing the ignore makes the unchanged source eligible as a new file on the next pass.

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

Indexing, activation, rollback, optimization, and search use the same writer lock. A writer cannot switch databases while another writer is active, and search never opens a database while the lock is held. Staged construction uses a separate lock because it writes only its candidate generation. Generated definitions do not copy `PI_RECALL_*` overrides from the installation shell. Scheduled runs use the durable recall configuration file and normal defaults.

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

The WAL-mode `recall-catalog.sqlite` stores each physical session's size, modification time, dense document identities, and compact Invocation records. Each completed physical session replaces only its own catalog rows in one transaction. If indexing stops, rerun `psr index`; completed sessions remain committed, unfinished sessions are revisited, and matching dense vectors are reused.

The compact layout requires a version 7 manifest. Existing installations must run `psr index --rebuild`; ordinary indexing rejects the old layout instead of mixing schemas. After activation, unchanged sessions are skipped from size and modification time without parsing.

## Search

Pi calls:

```text
pi-session-recall({ query: "What did we decide about the job queue?" })
pi-session-recall({ query: "readNodeErrorCode", limit: 5 })
pi-session-recall({ query: "cross-project decision", scope: "global" })
pi-session-recall({ query: "CT1000P3PSSD8", source: true, scope: "global" })
```

Parameters:

```ts
{
  query: string;
  limit?: number;                 // default 5, maximum 10
  scope?: "project" | "global";  // default project
  source?: boolean;               // default false
}
```

Normal recall searches both fast stores automatically: flat dense search over conversations, summaries, and turn context, plus SQLite full-text search over compact Invocations. It combines both candidate lists before applying `limit`; callers do not select a fast store. The mixed-result policy keeps an Invocation visible when both kinds match without displacing more than one of the first five strong conversation results. Project scope filters both stores before their eight-candidate limits. Global scope searches both complete stores.

Set `source: true` only when you need complete raw tool results, bash output, or omitted invocation payloads. Source search performs a slower, case-insensitive literal scan of the original session JSONL and writes no index or cache data. Project scope scans only logical sessions whose exact project identity matches the trusted Pi working directory. Global scope scans every eligible physical session file. Exact ignored paths remain excluded. Results include the physical path, source line range, entry ID when present, and a bounded matching excerpt. A file that disappears or becomes unreadable during the scan is reported without hiding matches from other files.

Dense conversation ranking suppresses overlapping sibling chunks and exact copies across sessions, applies a `0.01` active-branch preference, and keeps abandoned-branch evidence eligible. A winning atomic conversation chunk can include one exact contiguous sibling on either side. Expansion requires matching session, entry, role, visible text run, source geometry, and reciprocal sibling links. Invocation results carry the tool name, bounded searchable locator fields, error status when known, and exact source locator.

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

The agent can read those JSONL lines directly when it needs surrounding source context. Normal recall does not reopen session files and has no separate source-neighborhood database. Explicit `source: true` search reads canonical JSONL on demand and persists nothing.

## Indexed evidence

The flat dense store contains only:

- visible user and assistant text;
- user/assistant turn-context documents;
- visible custom messages;
- compaction and branch summaries.

The SQLite catalog contains one compact Invocation for each eligible assistant tool call and direct bash execution. It stores tool names, bounded locator arguments or commands, call identity, error status when known, project attribution, and source locators.

Neither store contains thinking, images, complete tool results, bash output, omitted payload arguments, empty tool placeholders, or derived `pi-session-recall` calls and results. Complete output and omitted payloads remain available only through explicit `source: true` search.

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
- structural boundaries before hard token cuts.

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

The version 7 manifest binds request model, served model, fixed 1,024-dimension stored width, transformation, tokenizer assets, 512/64 chunking, import policy, project identity policy, and dense-only FLAT store schema. Any change requires:

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
- `PI_RECALL_EMBEDDING_STORED_DIMENSIONS` (must remain `1024` for the compact layout)
- `PI_RECALL_EMBEDDING_BATCH_SIZE`

## Local state

After the first generation rebuild, durable recall state has this shape:

```text
~/.pi/agent/recall/
├── active -> generations/generation-.../
├── generations/
│   ├── generation-.../              # active recall database
│   └── generation-.../              # retained previous recall database
└── physical-session-ignore.json
```

Each new generation contains `zvec/`, `recall-catalog.sqlite`, `index-manifest.json`, and `index-maintenance-status.json`. Failed or interrupted rebuilds can leave a `candidate-.../` directory. The next rebuild removes stale candidates. A successful `--stage` rebuild leaves a complete `generation-.../` directory that remains inactive until its exact target is passed to `psr activate`. Rebuilds do not remove completed generations.

An existing unversioned `zvec/`, `index-state.json`, and old `index-manifest.json` remain in place during the first compact rebuild. The version 7 compact layout rejects that manifest for normal indexing and directs the operator to `psr index --rebuild`. Rebuild candidates create a fresh catalog from canonical session JSONL. The activated generation records the prior layout as its previous database, so `psr rollback` can atomically point `active` back to it.

The tokenizer loader also keeps checksum-verified tokenizer assets under `tokenizers/`; these are replaceable inference inputs, not recall state. `operation.lock` exists only while `psr` owns the writer lock and is removed when the command exits. There is no embedding cache, generation registry, replay log, marker spool, or model-artifact cache.

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

It reads only the checksum-fixed evaluation corpus, builds disposable compact databases, and measures the frozen 512/64, eight-candidates-per-fast-store, five-final-results policy. `evaluation/compact-recall-cases.json` separately fixes normal Invocation, explicit Source search, and mixed-result cases. Production indexing does not read or gate on generated evaluation files.

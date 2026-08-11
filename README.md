# Pi Session Recall

Pi Session Recall searches past Pi conversations and compact tool Invocations. Manifest version 8 stores the complete derived projection in one WAL-mode `recall.sqlite` database with FTS5 and pinned sqlite-vec 0.1.9. Canonical session JSONL owns complete payloads, and every result identifies its original JSONL file and line range.

The standalone `psr index` command is the only index writer. Run it directly or opt into a native per-user schedule. `psr ignore` writes PSR policy state but never opens the index. Pi lifecycle and the `pi-session-recall` tool remain read-only.

## Why index session history?

Raw session JSONL remains the source of truth, but asking a fresh agent to search it spends time and model context on file discovery, format parsing, branch structure, compaction, and evidence location. Pi Session Recall does that work ahead of time, limits search to the invoking project by default, searches dense conversations and compact Invocations together, and returns exact JSONL line citations.

A measured production comparison gave the same question to the recall tool and to a fresh agent restricted to raw JSONL. The full hybrid tool took 1.48 seconds at the median. The raw agent took 94.43 seconds, examined 54 project files, and used 141,682 tokens plus 852,480 cached tokens. The agent found the answer reliably; the tool required its maximum ten results to include the answer at rank ten. Indexed recall was about 64 times faster and far cheaper in this case, while the ranking result exposed work still needed. This is one measured query, not a universal quality or capacity claim. See [Production recall index value benchmark](docs/research/production-recall-index-value-benchmark.md).

The [unified SQLite prototype](docs/research/unified-sqlite-recall-storage-prototype.md) passed its storage, latency, retrieval-overlap, update-write, churn, and atomicity gates. The old [v7 certification](docs/research/superseded-v7-compact-production-recall-certification.md) is superseded and does not certify v8. Real v8 production certification and activation still remain. The production gates are at most 5 GiB allocated storage, project Dense search below 100 ms p95, and global Dense search below 700 ms p95 on the measured corpus.

## Install

The runtime's built-in `node:sqlite` must include FTS5 and load pinned sqlite-vec 0.1.9. `psr index` cannot create `recall.sqlite` without both.

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
psr index --compact                            # keep the former one-line stdout summary
psr auto-index install                         # update hourly
psr auto-index install --interval 30m           # replace the update interval
psr auto-index uninstall                        # remove every installed schedule
psr ignore add path/to/session.jsonl            # exclude one exact physical session path
psr ignore list                                 # print sorted excluded paths
psr ignore remove path/to/session.jsonl         # make one exact path eligible again
```

`psr index`:

- recursively scans configured `.jsonl` session files;
- skips files whose size and modification time have not changed;
- reuses checksum-matched vectors already stored in the current Recall database when explicitly requested for a staged rebuild;
- calls Octen only for changed conversation, summary, and turn-context documents;
- replaces each changed session's state, compact Invocations, Dense recall metadata, and single vector projection in one SQLite transaction;
- removes evidence for deleted or newly ignored indexed session files;
- skips ignored files before parsing or embedding them;
- reports malformed eligible session files and continues with healthy files;
- shows elapsed time and estimates time remaining after a healthy file completes;
- performs no corpus-wide compaction or optimization.

`psr index --rebuild` builds a candidate recall database beside the active database. A fatal error, cancellation, or failed session leaves normal recall on the active database. A successful rebuild closes and verifies the candidate, then atomically makes it active. The next fresh rebuild removes failed or interrupted candidates.

Add `--stage` when the candidate must pass checks before activation. A staged rebuild uses a separate construction lock, so normal search and scheduled updates keep using the active database. The command prints the exact `generations/generation-...` target and leaves the active pointer unchanged. If a fatal dependency outage interrupts the build, rerun it once with `--rebuild --stage --resume`. Resume requires exactly one interrupted candidate and preserves every completed Physical session. Add `--reuse-active-vectors` only when a compatible current-format Active recall database exists. The indexer requires the embedding profile, canonical document ID, and checksum to match before it copies a vector; changed documents still go to the embedding provider.

Run `psr activate <database-target>` only after that target passes its checks. Activation takes the shared writer lock, verifies the staged database is complete, and replaces the active pointer atomically.

Obsolete database layouts are not opened, migrated, or used for vector reuse. Build the current database from canonical session JSONL with `psr index --rebuild --stage`, certify it, and activate it.

The estimate uses the observed rate of healthy files in the current run. Until enough work completes, the command says that it is calculating the estimate rather than inventing an initial duration. `--compact` preserves the former one-line completed summary and `Failed: ...` lines on stdout; progress remains on stderr.

No startup hook, completed-turn hook, shutdown hook, watcher, package daemon, or search request updates the index.

### Ignoring exact physical session paths

Ignore state persists in `~/.pi/agent/recall/physical-session-ignore.json`. Both manual and scheduled `psr index` runs read one snapshot after acquiring the index lock. An ignored new file is not parsed, embedded, stored, or added to `recall.sqlite`. If the file is already indexed, the next maintenance pass removes its complete Recall database projection. Removing the ignore makes the unchanged source eligible as a new file on the next pass.

`add` and `remove` are idempotent. They report `Already ignored` or `Not ignored` and exit successfully when no state changes. `list` prints normalized paths in ordinary string order, one per line, and prints nothing when the list is empty.

Path identity is exact and lexical. Relative command arguments resolve from the command's current working directory with `path.resolve`. Commands do not expand `~`, resolve symlinks, inspect the filesystem, require a `.jsonl` suffix, restrict paths to the configured sessions directory, or interpret glob characters. For example, `*.jsonl` names a path containing a literal asterisk. Symlink aliases remain distinct paths.

`psr index --rebuild` preserves ignore state and rebuilds only eligible files. Invalid ignore state aborts maintenance before rebuild removes the working index.

### Automatic index maintenance

`psr auto-index install` creates one per-user native schedule that updates changed evidence. The interval accepts a positive whole number followed by lowercase `m` or `h`; the default is `1h`. Installation never uses `sudo`. Reinstalling replaces the definition, captures absolute paths to the current Node executable and installed package, and disables and removes optimization jobs created by older releases.

The generated definition runs the equivalent of:

```text
<absolute-node> --import tsx <absolute-package-root>/bin/psr index
```

Indexing, activation, and search use the same writer lock. A writer cannot switch databases while another writer is active, and search never opens a database while the lock is held. Staged construction uses a separate lock because it writes only its candidate generation. Generated definitions do not copy `PI_RECALL_*` overrides from the installation shell. Scheduled runs use the durable recall configuration file and normal defaults.

On Linux, installation writes one systemd user service and timer under `${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user/`. It removes stale `pi-session-recall-optimize.service` and `.timer` units, starts the index timer, then attempts one immediate index run. If that run fails, installation warns but leaves the timer active. Read logs with:

```bash
journalctl --user-unit=pi-session-recall-index.service
```

On macOS, installation removes the stale `dev.pi-session-recall.auto-optimize.plist` LaunchAgent and writes one mode-`0600` index LaunchAgent under `~/Library/LaunchAgents/`; it uses `RunAtLoad`. Logs are written to:

```text
~/.pi/agent/logs/pi-session-recall-auto-index.out.log
~/.pi/agent/logs/pi-session-recall-auto-index.err.log
```

The macOS path is runtime-untested. No Mac was available to verify plist acceptance; `RunAtLoad` or `StartInterval` execution; retry after an exit-status-1 run; absolute Node plus `--import tsx`; log appends; or access to the durable recall configuration and embedding endpoint from the LaunchAgent environment. Other platforms fail with an unsupported-platform error.

The WAL-mode `recall.sqlite` database stores each Physical session's size and modification time, compact Invocation rows with FTS5, Dense recall metadata, and one 16-bucket vec0 table. Each Dense recall document stores one vector. Each completed Physical session replaces only its own rows across all projections in one transaction. If indexing stops, rerun `psr index`; completed sessions remain committed and unfinished sessions are revisited.

Manifest version 8 and SQLite schema version 3 identify this layout. Obsolete database layouts are incompatible; rebuild them from canonical JSONL. After activation, unchanged sessions are skipped by size and modification time without parsing.

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

Normal recall searches both projections in `recall.sqlite`: sqlite-vec Dense recall and FTS5 compact Invocations. It combines both candidate lists before applying `limit`; callers choose scope, not a storage engine. Project scope searches the selected vec0 bucket and applies the exact project key before its eight-candidate limit. Global scope searches the same vec0 table across all 16 buckets. FTS5 applies the same project or global scope. The mixed-result policy keeps an Invocation visible when both kinds match without displacing more than one of the first five strong conversation results.

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

Dense metadata and vectors in `recall.sqlite` contain only:

- visible user and assistant text;
- user/assistant turn-context documents;
- visible custom messages;
- compaction and branch summaries.

The FTS5 projection contains one compact Invocation for each eligible assistant tool call and direct bash execution. It stores tool names, bounded locator arguments or commands, call identity, error status when known, project attribution, and source locators.

The Recall database contains no thinking, images, complete tool results, bash output, omitted payload arguments, empty tool placeholders, or derived `pi-session-recall` calls and results. Complete output and omitted payloads remain available only through explicit `source: true` search.

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
| sqlite-vec metric | cosine                               |

The same transformation applies to document and query vectors. sqlite-vec compares the stored FP32 vectors with cosine distance. The feature is vendor-supported prefix storage; this repository does not claim independently verified MRL quality at every cutoff.

Both `psr index` and `pi-session-recall` require the configured Octen HTTP endpoint. This package has no local embedding fallback.

The version 8 manifest binds request model, served model, fixed 1,024-dimension FP32 width, cosine distance, sqlite-vec 0.1.9, FTS5, one-table 16-bucket routing, tokenizer assets, 512/64 chunking, import policy, and project identity policy. Any change requires:

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
- `PI_RECALL_EMBEDDING_STORED_DIMENSIONS` (must remain `1024` for manifest version 8)
- `PI_RECALL_EMBEDDING_BATCH_SIZE`

## Local state

After the first generation rebuild, durable recall state has this shape:

```text
~/.pi/agent/recall/
├── active -> generations/generation-.../
├── generations/
│   ├── generation-.../              # active recall database
│   └── generation-.../              # inactive completed generation
└── physical-session-ignore.json
```

Each version 8 generation contains `recall.sqlite`, `index-manifest.json`, and `index-maintenance-status.json`. SQLite may create `recall.sqlite-wal` and `recall.sqlite-shm` while the database is open; they are part of the same transactional database, not separate stores. Failed or interrupted rebuilds can leave a `candidate-.../` directory. The next fresh rebuild removes stale candidates. A successful `--stage` rebuild leaves a complete `generation-.../` directory inactive until its exact target is passed to `psr activate`. Rebuilds do not remove completed generations.

Rebuild candidates create `recall.sqlite` from canonical session JSONL. The current code ignores obsolete root database artifacts and staged legacy generations; remove them after the current database is certified and active.

The tokenizer loader also keeps checksum-verified tokenizer assets under `tokenizers/`; these are replaceable inference inputs, not recall state. `operation.lock` exists only while `psr` owns the writer lock and is removed when the command exits. There is no embedding cache, generation registry, replay log, marker spool, or model-artifact cache.

## Certify a staged version 8 database

The certification command accepts one exact `generations/generation-...` target under an explicitly supplied data root. It opens that candidate read-only and checks the current format directly. It never activates the candidate or opens the Active recall database for writes.

Run the read-only phase first. Omit `--output` until the measurements are ready to commit:

```bash
npm run certify:unified-sqlite-recall -- \
  --data-root /exact/path/to/recall \
  --candidate-target generations/generation-... \
  --project-identity git-origin:github.com/Whamp/pi-session-recall
```

The read-only phase checks the strict version 8 manifest, sqlite-vec 0.1.9 identity and Linux load, SQLite and projection integrity, counts, allocated database/WAL/SHM storage, fixed Dense and Invocation probes, and exact Source provenance. It exits 2 while clone gates remain pending. The PR's SQLite-vec GitHub Actions jobs separately execute runtime loading on macOS x64 and arm64; local certification does not imitate those checks with package metadata.

CAUTION: The next command deletes its per-run clone directory. Supply a dedicated scratch root that is disjoint from the recall data root. The command refuses to copy when scratch allocation would exceed 6 GiB or free space is below 240 GiB. It never mutates the candidate, the Active database, or canonical session JSONL.

Run clone certification with one exact indexed Physical session and the Linux block device used for write measurement:

```bash
npm run certify:unified-sqlite-recall -- \
  --data-root /exact/path/to/recall \
  --candidate-target generations/generation-... \
  --project-identity git-origin:github.com/Whamp/pi-session-recall \
  --scratch-root /exact/path/to/dedicated-certification-scratch \
  --representative-session /exact/path/to/session.jsonl \
  --block-device nvme0n1 \
  --output docs/research/unified-sqlite-production-recall-certification.json
```

The async clone phase makes only the representative clone session stale, then calls the production `indexChangedConversationSessions` path over the real sessions directory with the production tokenizer, exact ignored-path policy, project resolver, and embedding provider. It requires exactly one indexed session, zero failures, expected checksum-vector reuse, available device-write counters, unchanged counts, and an unchanged unrelated Physical session. It separately labels and retains the 100-cycle direct-database churn probe for long-term page and file behavior, plus concurrent-reader isolation, explicit rollback, SIGKILL recovery, post-churn integrity, and latency. `--output` writes sanitized JSON and Markdown only to `docs/research/unified-sqlite-production-recall-certification.*`. Do not commit a report until the real staged candidate has run.

`candidatePreActivationPassed` means every local candidate and disposable-clone gate passed. The PR's required platform jobs supply separate runtime evidence. This certification benchmark is an explicit release operation, not a requirement for ordinary user rebuilds. Staged construction and atomic activation keep partial databases out of service.

After this branch is merged and deployed, the following checks remain pending:

1. Pause automatic indexing for the bounded activation window.
2. Run `psr activate generations/generation-...` with the exact certified target.
3. Run one immediate changed-session index.
4. Run project, global, normal, and Source recall against the Active database.
5. Confirm the hourly timer is active.
6. Remove obsolete database artifacts after the current database passes live checks.
7. Obtain a successful `.github/workflows/sqlite-vec-platform-smoke.yml` PR run. Its stable `SQLite-vec macOS x64 runtime load` job runs on `macos-26-intel` and asserts `process.arch === 'x64'`; its stable `SQLite-vec macOS arm64 runtime load` job runs on `macos-26` and asserts `process.arch === 'arm64'`. Both jobs load pinned sqlite-vec 0.1.9 through `node:sqlite`, call `vec_version()`, query FTS5, and insert/query vec0. Record the real successful workflow URL as release evidence only after it exists.

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

It reads only the checksum-fixed evaluation corpus, builds disposable Recall databases, and measures the frozen 512/64, eight-candidates-per-projection, five-final-results policy. `evaluation/compact-recall-cases.json` separately fixes normal Invocation, explicit Source search, and mixed-result cases. Production indexing does not read or gate on generated evaluation files.

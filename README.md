# Pi Session Recall

Pi Session Recall searches past Pi conversations and compact tool Invocations. Manifest version 8 stores the complete derived projection in one WAL-mode `recall.sqlite` database with FTS5 and pinned sqlite-vec 0.1.9. Canonical session JSONL owns complete payloads, and every result identifies its original JSONL file and line range.

The standalone `psr index` command is the only index writer. Run it directly or opt into a native per-user schedule. `psr ignore` writes PSR policy state but never opens the index. Pi lifecycle and the `pi-session-recall` tool remain read-only.

## Why index session history?

Raw session JSONL remains the source of truth, but asking a fresh agent to search it spends time and model context on file discovery, format parsing, branch structure, compaction, and evidence location. Pi Session Recall does that work ahead of time, limits search to the invoking project by default, searches dense conversations and compact Invocations together, and returns exact JSONL line citations.

That repeated work grows with the session repository. The raw-agent evaluation used 15 synthetic files totaling 44,782 bytes. For one real-world example, the maintainer's personal session history measured on 2026-08-16 contained 3,803 files and 3.476 GiB of raw JSONL—254 times as many files and more than 83,000 times as many bytes. Its largest file was 130.66 MiB. This is one large, long-running history, not a claim about a typical installation. Other users may have much smaller or larger histories. At this scale, a fresh agent cannot put the whole corpus in model context. It must repeatedly narrow paths, search text, parse records, and resolve competing evidence. Recall pays those costs during indexing and reuses the result across questions.

A separate comparison on the maintainer's session repository gave the same question to the recall tool and to a fresh agent restricted to raw JSONL. The full hybrid tool took 1.48 seconds at the median. The raw agent took 94.43 seconds, examined 54 project files, and used 141,682 tokens plus 852,480 cached tokens. The agent found the answer reliably; the tool required its maximum ten results to include the answer at rank ten. Indexed recall was about 64 times faster and far cheaper in this case, while the ranking result exposed work still needed. This is one measured query on one user's history, not a universal quality or capacity claim. See [Production recall index value benchmark](docs/research/production-recall-index-value-benchmark.md).

The [unified SQLite prototype](docs/research/unified-sqlite-recall-storage-prototype.md) passed its storage, latency, retrieval-overlap, update-write, churn, and atomicity gates. The [production certification](docs/research/unified-sqlite-production-recall-certification.md) then passed every pre-activation gate, and the current layout was activated and live-verified on 2026-08-11. The old [v7 certification](docs/research/superseded-v7-compact-production-recall-certification.md) remains only as historical evidence.

## Install

The runtime's built-in `node:sqlite` must include FTS5 and load pinned sqlite-vec 0.1.9. `psr index` cannot create `recall.sqlite` without both.

```bash
npm install
npm link
pi install /path/to/pi-session-recall
```

Reload Pi after installation. Fresh users should run guided setup. Local Octen is the default; it downloads and verifies a 1.01 GiB model before writing configuration.

```bash
psr setup
psr index --rebuild
```

Agents and scripts can run the same flow without prompts:

```bash
psr setup --local --yes
psr index --rebuild
```

To use an existing OpenAI-compatible embedding server instead:

```bash
psr setup --external --yes \
  --base-url http://127.0.0.1:8090/v1 \
  --model octen-embed \
  --served-model-id Octen/Octen-Embedding-4B \
  --native-dimensions 2560
psr index --rebuild
```

### Choose an embedding profile

| Choose                     | Best when                                                                                                   | Tradeoff                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Local Octen 0.6B (default) | You want the simplest setup, offline search, and no conversation or query text sent to an embedding server. | Downloads 1.01 GiB and indexes on your CPU.                                     |
| External HTTP              | You already run a compatible embedding server and want faster indexing.                                     | Sends conversation and query text to that server and adds a service dependency. |

In one fixed 32-query comparison, both profiles found every expected source in the Dense top eight. HTTP Octen 4B kept all 32 in Recall's final five, versus 31 for local Octen 0.6B. On the measured CPU and GPU hosts, HTTP queries took 26.7 ms at the median versus 32.8 ms locally, and a full rebuild was projected at 8.0 hours versus 21.8 hours.

Skipping the index also worked on this small synthetic corpus: 32 fresh agents found and cited the expected evidence in all 32 queries. Each agent reopened all 15 raw files and took 15.7 seconds at the median. One agent answering all 32 questions took 27.1 seconds total because it learned the corpus once; that is an amortized best case, not normal fresh-query behavior.

For the same 15-file suite, 32 fresh raw searches took 520.7 seconds. Building the HTTP index once and running all 32 searches took 5.28 seconds total, saving 515.4 seconds (99%, about 99× faster). The local profile took 12.29 seconds total, saving 508.4 seconds (97.6%, about 42× faster). These small-corpus totals are not a production rebuild break-even estimate. The measurements cover different execution boundaries and depend on the corpus and hardware; see the [complete comparison](docs/research/local-vs-http-embedding-profile-comparison.md).

**Recommendation:** use the local default unless you already have a trusted embedding server. The HTTP profile is the better choice when rebuild speed matters.

## Commands

```bash
psr setup                                      # guided fresh-install setup; local is the default
psr setup --local --yes --index                # noninteractive local setup and initial rebuild
psr model status                               # inspect local artifact state without mutation
psr model download [--yes]                     # explicitly download or repair the local artifact
psr model doctor                               # verify hashes, native runtime, and one embedding
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
- hashes same-size files whose modification time changed and refreshes source metadata without parsing when the SHA-256 is unchanged;
- reuses exact token geometry for unchanged Dense projection inputs while rebuilding graph-dynamic metadata from the current validated graph;
- reuses checksum-matched vectors already stored in the current Recall database when explicitly requested for a staged rebuild;
- calls the configured local or HTTP Octen profile only for changed conversation, summary, and turn-context documents;
- replaces each changed session's state, compact Invocations, Dense recall metadata, and single vector projection in one SQLite transaction;
- removes evidence for deleted or newly ignored indexed session files;
- skips ignored files before parsing or embedding them;
- reports malformed eligible session files and continues with healthy files;
- reports one content-free per-file profile after a successful replacement, covering read/parse, graph validation, document construction/tokenization, vector lookup, embedding, and SQLite replacement;
- shows elapsed time and estimates time remaining after a healthy file completes;
- performs no corpus-wide compaction or optimization.

`psr index --rebuild` builds a candidate recall database beside the active database. A fatal error, cancellation, or failed session leaves normal recall on the active database. A successful rebuild closes and verifies the candidate, then atomically makes it active. The next fresh rebuild removes failed or interrupted candidates.

Add `--stage` when the candidate must pass checks before activation. A staged rebuild uses a separate construction lock, so normal search and scheduled updates keep using the active database. The command prints the exact `generations/generation-...` target and leaves the active pointer unchanged. If a fatal dependency outage interrupts the build, rerun it once with `--rebuild --stage --resume`. Resume requires exactly one interrupted candidate and preserves every completed Physical session. Add `--reuse-active-vectors` only when a compatible current-format Active recall database exists. The indexer requires the embedding profile, canonical document ID, and checksum to match before it copies a vector; changed documents still go to the embedding provider.

Run `psr activate <database-target>` only after that target passes its checks. Activation takes the shared writer lock, verifies the staged database is complete, and replaces the active pointer atomically.

Obsolete database layouts are not opened, migrated, or used for vector reuse. Build the current database from canonical session JSONL with `psr index --rebuild --stage`, certify it, and activate it.

For a hard cutover from an obsolete installation, build and certify the staged database with the new package before replacing the installed package. Then pause scheduled indexing, deploy the new package, and immediately activate the exact certified target. Until activation, the deployed current-only code will reject the obsolete root manifest instead of serving or updating it.

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

The WAL-mode `recall.sqlite` database stores each Physical session's exact imported byte length and SHA-256, size/mtime freshness hints, compact Invocation rows with FTS5, Dense recall metadata, disposable token-geometry cache rows, and one 16-bucket vec0 table. Each Dense recall document stores one vector. Each completed Physical session replaces only its own rows across all projections and cache state in one transaction. If indexing stops, rerun `psr index`; completed sessions remain committed and unfinished sessions are revisited.

Manifest version 8 and SQLite schema version 4 identify this layout. Obsolete database layouts are incompatible; rebuild them from canonical JSONL. After activation, unchanged sessions are skipped by size and modification time without parsing, and same-size metadata-only changes are skipped after a complete SHA-256 read.

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

The index uses the checksum-pinned tokenizer belonging to the selected embedding profile with the frozen policy:

- 512 tokens maximum;
- 64 tokens overlap between adjacent atomic conversation chunks;
- structural boundaries before hard token cuts.

Atomic chunks never cross entries, roles, visible text runs, tools, thinking, images, results, or summaries. Turn-context documents retain both user and assistant text and cite every contributing entry.

## Embedding profiles

Fresh `psr setup` defaults to local Octen:

| Setting                      | Local default                                                                                                                            |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Profile                      | `local-octen-embedding-0.6b-onnx-int8-v1`                                                                                                |
| Model                        | `Octen/Octen-Embedding-0.6B`                                                                                                             |
| Runtime                      | native `onnxruntime-node` 1.27.0 on macOS arm64 and Linux x64 Ryzen; `onnxruntime-web` 1.27.0 WASM on other Linux x64 CPUs and macOS x64 |
| Artifact                     | SmoothQuant INT8 ONNX, 1.01 GiB                                                                                                          |
| Native and stored dimensions | 1,024 FP32                                                                                                                               |
| Transformation               | tokenizer final token, then local L2 normalization                                                                                       |
| sqlite-vec metric            | cosine                                                                                                                                   |

The graph accepts batch size one. Native Pi Session Recall on macOS arm64 and Linux x64 Ryzen runs four bounded concurrent operations against one shared session by default. The WASM path serializes operations because ONNX Runtime Web supports single-threaded WASM under Node. The certified tokenizer post-processor appends `<|endoftext|>` token `151643`; manually appending its configured `<|im_end|>` EOS token `151645` produces incompatible vectors. The [project release](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1) pins every artifact byte and carries the Apache 2.0 license and provenance notice.

After download, local indexing and search work offline. Conversation text stays in the Pi process and is not sent to an embedding server. `psr model status` reads receipt and file sizes. `psr model doctor` additionally hashes every artifact file, loads the native runtime, produces one normalized 1,024-dimensional embedding, and releases the session. `psr model download` requires confirmation or `--yes`; it streams into a unique partial directory, verifies hashes, and only then activates the complete model. If status reports `partial` or `corrupt`, rerun `psr model download`; failed downloads remove their partial directory and do not replace an existing artifact.

The real artifact download, runtime query, fixed-vector conformance, disposable SQLite build, close, reopen, and offline search run in CI on Linux x64 EPYC, macOS arm64, and macOS x64 Intel. macOS arm64 uses native ONNX Runtime; EPYC and Intel runners use the measured WASM fallback because native inference produced incompatible vectors on both. Linux x64 Ryzen uses native ONNX Runtime after a real Ryzen conformance and one-session offline integration check. Unsupported platforms are rejected before the 1.01 GiB download and directed to the HTTP profile.

The execution backend is part of the embedding profile recorded in the Recall manifest. A Ryzen Linux installation that built a local Recall database with the earlier all-x64 WASM policy must run `psr index --rebuild` once after upgrading. Pi Session Recall refuses ordinary indexing against the incompatible manifest instead of mixing vectors from two runtimes.

The explicit HTTP profile retains the previous behavior:

| Setting           | External HTTP default               |
| ----------------- | ----------------------------------- |
| Profile           | `octen-http-v1`                     |
| Request model     | `octen-embed`                       |
| Served model      | `Octen/Octen-Embedding-4B`          |
| Native dimensions | 2,560                               |
| Stored dimensions | first 1,024 FP32 values             |
| Transformation    | prefix, then local L2 normalization |
| sqlite-vec metric | cosine                              |

Installations without `embeddingProfile` continue to select this HTTP profile so an upgrade cannot reinterpret an existing database. `psr setup` writes the local profile for fresh users. There is no silent model or backend fallback.

The version 8 manifest binds request profile, served model, native and stored dimensions, transformation, tokenizer assets, cosine distance, sqlite-vec 0.1.9, FTS5, one-table 16-bucket routing, 512/64 chunking, import policy, and project identity policy. Switching profile or changing any bound setting requires:

```bash
psr index --rebuild
```

## Configuration

`psr setup` writes `~/.pi/agent/recall.json` atomically with mode `0600`. A local setup resembles:

```json
{
  "embeddingProfile": "local-octen-embedding-0.6b-onnx-int8-v1",
  "localModelRootDirectory": "/home/you/.pi/agent/recall-models"
}
```

Optional local tuning fields are `localEmbeddingParallelism` and `localEmbeddingIntraOperationThreads`. The measured defaults are both `4` on a 16-thread AMD Ryzen 7 8845HS. Lower them on smaller machines if native inference contends with other work.

An external configuration resembles:

```json
{
  "embeddingProfile": "octen-http-v1",
  "embeddingBaseUrl": "http://127.0.0.1:8090/v1",
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
- `PI_RECALL_EMBEDDING_PROFILE`
- `PI_RECALL_EMBEDDING_BASE_URL`
- `PI_RECALL_EMBEDDING_MODEL`
- `PI_RECALL_EMBEDDING_SERVED_MODEL_ID`
- `PI_RECALL_EMBEDDING_NATIVE_DIMENSIONS`
- `PI_RECALL_EMBEDDING_STORED_DIMENSIONS` (must remain `1024` for manifest version 8)
- `PI_RECALL_EMBEDDING_BATCH_SIZE`
- `PI_RECALL_LOCAL_MODEL_ROOT_DIRECTORY`
- `PI_RECALL_LOCAL_EMBEDDING_PARALLELISM`
- `PI_RECALL_LOCAL_EMBEDDING_INTRA_OPERATION_THREADS`

## Local state

After the first generation rebuild, durable recall state has this shape:

```text
~/.pi/agent/
├── recall/
│   ├── active -> generations/generation-.../
│   ├── generations/
│   │   ├── generation-.../          # active recall database
│   │   └── generation-.../          # inactive completed generation
│   └── physical-session-ignore.json
└── recall-models/
    └── local-octen-embedding-0.6b-onnx-int8-v1/
        ├── model.int8.onnx
        ├── model.int8.onnx.data
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── model-receipt.json
```

Each version 8 generation contains `recall.sqlite`, `index-manifest.json`, and `index-maintenance-status.json`. SQLite may create `recall.sqlite-wal` and `recall.sqlite-shm` while the database is open; they are part of the same transactional database, not separate stores. Failed or interrupted rebuilds can leave a `candidate-.../` directory. The next fresh rebuild removes stale candidates. A successful `--stage` rebuild leaves a complete `generation-.../` directory inactive until its exact target is passed to `psr activate`. Rebuilds do not remove completed generations.

Rebuild candidates create `recall.sqlite` from canonical session JSONL. The current code ignores obsolete root database artifacts and staged legacy generations; remove them after the current database is certified and active.

The HTTP tokenizer loader keeps checksum-verified tokenizer assets under `recall/tokenizers/`. The local profile reads its tokenizer from the verified model directory. Both are replaceable inference inputs, not recall state. `operation.lock` exists only while `psr` owns the writer lock and is removed when the command exits. There is no embedding cache, generation registry, replay log, marker spool, or background inference worker.

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

The production version 8 candidate passed this certification on 2026-08-11. It was then activated, immediately indexed, checked through project, global, normal, and Source recall, and verified with exact Dense/vector and Invocation/FTS parity. The hourly timer resumed successfully. Approved obsolete Zvec and root artifacts were removed only after those checks, reducing the live recall directory from about 17 GiB to 3.1 GiB at cutover. The durable results are in [the production certification report](docs/research/unified-sqlite-production-recall-certification.md).

`.github/workflows/sqlite-vec-platform-smoke.yml` separately verifies pinned sqlite-vec 0.1.9 on macOS x64 and arm64. `.github/workflows/local-octen-platform-smoke.yml` verifies the local model artifact and selected native-or-WASM inference path on Linux x64, macOS x64, and macOS arm64.

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

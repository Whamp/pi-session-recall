# Pi Session Recall

`pi-session-recall` gives Pi one model-facing tool for searching past conversations and expanding an exact result into nearby source entries. Pi session JSONL files remain immutable source data. Recall storage is disposable and rebuildable.

## Runtime boundaries

The Pi extension does only three things:

- registers the `pi-session-recall` model tool;
- publishes cheap immutable work markers from lifecycle events; and
- disposes inference resources on shutdown.

The extension does not index sessions, open maintenance progress in the TUI, or register a maintenance slash command. Use the standalone `pi-session-recall` executable for every operator action.

## Recall generations

Each recall generation is self-contained under `generations/<generation-id>/` and owns:

- a scalar-only lexical/source store with evidence occurrences and entry anchors;
- a dense store containing only dense-searchable evidence;
- a scalar-only session-projection store;
- a fixed generation manifest; and
- an immutable validation receipt after complete validation.

Search opens only the checksummed pointer-selected target generation. It does not open raw session files, another generation, or an embedding cache. A generation with missing artifacts, mismatched identities, or unresolved recovery state is unavailable until the operator recovers, rolls back, or rebuilds it.

There is no migration or adoption path for older storage layouts. Build a fresh generation from the immutable Pi session sources.

## Install

```bash
pi install /home/will/projects/pi-session-recall
```

Reload Pi with `/reload` after installing or updating the extension.

The package exposes the standalone operator executable through its `bin` entry. From this checkout, use:

```bash
npm exec -- pi-session-recall setup
npm exec -- pi-session-recall status
```

`setup` reports supported stored widths and their evidence status. Configure and verify the selected inference capabilities before starting a build. See [inference configuration](docs/inference/mixed-inference-configuration.md) and [first-generation setup](docs/inference/first-index-guided-setup.md).

## Operator CLI

```text
pi-session-recall setup
pi-session-recall status
pi-session-recall catch-up
pi-session-recall rebuild
pi-session-recall stop
pi-session-recall resume
pi-session-recall discard
pi-session-recall recover
pi-session-recall rollback
pi-session-recall cleanup
```

- `setup` presents supported embedding widths and evidence status.
- `status` reports readiness, generation roles, replay, recovery, backlog, and detached process state.
- `catch-up` runs one explicit short-lived incremental worker pass.
- `rebuild` launches a detached fresh target-generation build and returns.
- `stop` requests cooperative termination of the current replacement build.
- `resume` continues the same generation and fixed source snapshot.
- `discard` removes abandoned non-active replacement work.
- `recover` repairs an isolated interrupted write or named cutover.
- `rollback` switches to the one retained validated target generation.
- `cleanup` removes only generations made collectible by policy.

The first target activation has no legacy rollback generation. Rollback becomes available only after two validated target-format generations have existed.

## Model tool

Search uses exactly one request form:

```text
pi-session-recall({ query: "What did we decide about the job queue?", limit: 5 })
pi-session-recall({ query: "readNodeErrorCode", scope: "global", limit: 5 })
pi-session-recall({ query: "Which choice survived review?", mode: "deep-rerank" })
```

Search defaults to project scope and deterministic hybrid ranking. It preserves dense, lexical, and case-preserving identifier evidence, project admission before channel limits, fusion, reranking, duplicate groups, same-run neighbor context, branch labels, cancellation, diagnostics, provenance, and output ceilings.

Every result exposes an **Evidence occurrence ID**. Use that exact value for source-neighborhood expansion:

```text
pi-session-recall({
  expandSourceNeighborhood: {
    evidenceOccurrenceId: "occurrence_…",
    previousEntryCount: 2,
    nextEntryCount: 2
  }
})
```

Expansion follows indexed parent links backward and one selected descendant path forward. It reads only lexical/source evidence and entry anchors. It never searches, embeds, reranks, reads projections, or reopens session JSONL.

## Indexed evidence

Recall indexes source-backed:

- user, assistant, and visible custom messages;
- compaction and branch summaries;
- turn-context documents for retrieval;
- tool names, arguments, results, and errors; and
- direct bash commands and output.

Hidden thinking, images, and derived `pi-session-recall` calls and results are excluded. Tool evidence stays lexical-only and receives no fake vector. The dense store contains exactly the dense-searchable subset of lexical/source evidence.

Physical source identity comes from the normalized sessions-root-relative path. Logical session occurrences add complete-header position, so copied files and repeated raw session IDs remain distinct. Evidence occurrence IDs describe exact source geometry rather than text content.

## Incremental ingestion

Pi lifecycle hooks publish immutable markers without reading session bodies. A short-lived external worker processes bounded append deltas and exits when no eligible work remains. It is not a daemon or filesystem watcher.

The worker prepares parsing, tokenization, project identity, evidence, anchors, projections, and vectors before the exclusive write window. Each write window:

1. persists generation-local recovery intent;
2. adds lexical/source rows before dense rows, or deletes dense rows before lexical/source rows;
3. writes logical projections and the physical projection last;
4. closes, reopens, and verifies changed stores;
5. clears recovery state; and
6. acknowledges only markers covered by the verified physical projection.

Evidence batches contain at most 32 documents. Search may wait for the current bounded write window, but never for marker backlog or a replacement build.

## Rebuild, activation, and rollback

A rebuild captures one fixed source snapshot and builds beside the active generation. It records expected membership during the source pass, closes and reopens all stores, validates exact membership and identities, then writes the immutable validation receipt.

Activation captures one fixed replay snapshot, publishes registry state first, swaps the active pointer, and completes registry publication. Search can serve the coherent new generation while its fixed marker replay remains pending. Newer markers remain ordinary backlog.

Rollback performs a bounded target-generation health check. It does not read session files, recertify all rows, migrate bytes, or dual-write generations.

Follow the [operator rollout checklist](docs/operations/incremental-recall-rollout.md). Production rebuild, activation, rollback, and cleanup require explicit human approval.

## Storage layout

```text
~/.pi/agent/recall/
├── active-generation.json
├── generation-registry.json
├── backlog-summary.json
├── background-index-status.json
├── inference-configuration.json
├── markers/{pending,quarantine,control}/
├── operation.lock/
├── incremental-worker.lock
└── generations/<generation-id>/
    ├── lexical-source/
    ├── dense/
    ├── session-projections/
    ├── index-manifest.json
    ├── validation-receipt.json
    ├── write-recovery.json
    └── replay-snapshots/
```

## Evaluation and development

The quality evaluation uses only committed fixtures and disposable roots:

```bash
npm run evaluate:recall
```

Development checks:

```bash
npm test
npm run typecheck
npm run lint
npm run format:check
```

Tests and evaluations must never open the production recall root or original Pi session files. Use copied fixtures and disposable session and generation roots.

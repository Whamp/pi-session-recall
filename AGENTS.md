## Agent skills

### Issue tracker

Issues are tracked in `Whamp/pi-session-recall` on GitHub. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the five canonical triage labels without aliases. See `docs/agents/triage-labels.md`.

### Domain docs

Use the single-context layout. See `docs/agents/domain.md`.

## Cursor Cloud specific instructions

`pi-session-recall` is a TypeScript Pi extension (a library/CLI, not a web app). Dev
dependencies are pure JS/TS; there is no service to boot. The standard dev commands live
in the README "Develop" section: `npm test`, `npm run typecheck`, `npm run lint`,
`npm run format:check`. Dependencies are refreshed automatically on VM startup.

Demonstrate core functionality without any network or model download:

- `npm run evaluate:recall` — builds a temporary 512/64 index over the 15 checksum-fixed
  sessions in `evaluation/corpus/`, runs hybrid retrieval + ranking, and prints the quality
  gate result. Deterministic, in-process, no HTTP. It rewrites `docs/evaluation/recall-quality-report.md`
  and `docs/evaluation/recall-quality-results.json` with fresh timestamps/timings — revert those
  regenerated files unless the change is intentional.
- `npm run --silent setup:recall -- status` — operator entry point; prints JSON setup/readiness
  state. On a fresh VM it reports `state: "unconfigured"`, `recallReady: false` (no model downloaded).

Non-obvious environment caveats:

- Node version: the `@earendil-works/*` peer deps declare `engines.node >= 22.19.0`. The VM's
  Node may be slightly older (e.g. 22.14.0), producing `EBADENGINE` warnings from `npm install`.
  These are warnings only — typecheck, lint, format, the full test suite, and `evaluate:recall`
  all work regardless.
- Three tests fail/flake here purely due to the VM filesystem/CPU, not code changes — leave them
  alone (do not "fix" them for the environment):
  - `src/adopt-legacy-recall-generation.test.ts` ("requires source and generations on one
    filesystem"): the VM uses `fuse-overlayfs`, which reports a different `st_dev` for regular
    files vs directories, so the test's single-filesystem assertion cannot hold.
  - `src/run-recall-quality-evaluation.test.ts` ("query-planned ... pre-rerank scope control"):
    order-dependent — it `mkdtemp`s under `evaluation/.recall-data/` without creating the parent.
    Run `mkdir -p evaluation/.recall-data` first (or run the whole suite) and it passes.
  - `src/embedded-qmd-query-planning-provider.test.ts` (timeout/cancellation subtest): timing
    sensitive on virtualized CPUs and may report `cancelled`.

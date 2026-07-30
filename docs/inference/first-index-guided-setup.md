# First target-generation setup

A fresh installation starts without an active recall generation. Setup selects and verifies inference capabilities; rebuild creates a new target generation from immutable Pi session sources. No command adopts, migrates, or opens an older storage layout.

## Inspect stored-width choices

Run the standalone operator CLI:

```bash
pi-session-recall setup
```

The JSON result reports each profile's native dimensions, default stored dimensions, allowed widths, evidence status, and evidence sources.

Current defaults:

- EmbeddingGemma: 768 stored dimensions; 512, 256, and 128 are verified reductions.
- Octen 4B: 1,024 stored dimensions; any positive width through 2,560 is mechanically supported by vendor-documented prefix truncation.

Select and verify an EmbeddingGemma width through the same standalone setup command. Model download remains an explicit consent gate:

```bash
pi-session-recall setup select-embeddinggemma --stored-dimensions 512 --approve-download
```

The accepted candidate identity persists the selected width, so later standalone rebuild workers reconstruct the same verified profile semantics. A reduced vector keeps the first N native components and then L2-normalizes that prefix. Stored width changes semantic profile and generation identity. Backend URL, device, and adapter location do not change vector compatibility when capability semantics remain unchanged.

## Configure inference

Embedding is required. Reranking and query planning are optional. Verify the selected capability providers before launching a build. See:

- [Mixed inference configuration](mixed-inference-configuration.md)
- [Provider conformance](provider-conformance.md)
- [Embedded EmbeddingGemma](embedded-embeddinggemma.md)

A detached worker must be able to reconstruct the same configured provider and profile identity. Target builds and incremental transfer call `embedDocuments`; search calls `embedQuery`; reranking calls `rerankDocuments`.

## Build the first generation

Inspect status, then launch the detached build:

```bash
pi-session-recall status
pi-session-recall rebuild
pi-session-recall status
```

The build:

1. captures one fixed source snapshot;
2. creates independent lexical/source, dense, and session-projection stores;
3. writes evidence in bounded windows;
4. closes and reopens every store;
5. validates exact membership, identities, projections, profile, and canaries;
6. writes an immutable validation receipt; and
7. activates through registry-first pointer publication with a fixed replay snapshot.

If the worker stops or crashes, resume the same generation:

```bash
pi-session-recall resume
```

Use `pi-session-recall stop` for cooperative termination and `pi-session-recall discard` only for abandoned non-active work. Use `pi-session-recall recover` when status reports an interrupted write or cutover.

The first target activation has no rollback generation. Target-to-target rollback becomes available after a later validated replacement activates.

## Consent and safety

Model download and production rebuild require separate approval. Implementation tests use deterministic providers, copied session fixtures, real disposable zvec stores, and disposable generation roots. They never open production recall data or original Pi session files.

A release still needs owner-approved environment evidence for model download, device selection, full rebuild duration, and final generation size. Those values are reported, not used as pass thresholds.

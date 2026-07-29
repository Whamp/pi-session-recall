# PROTOTYPE — rebuild recall without a persistent embedding cache

This throwaway benchmark answers one question for
[Choose the coherent recall generation storage topology](https://github.com/Whamp/pi-session-recall/issues/112):

> Under the proposed split-store generation, does a persistent embedding cache save enough time to justify its extra disk use, file count, and lifecycle?

The benchmark does not decide against a fixed rebuild-time limit. It compares relative phase time and storage across the same deterministic 10% size-stratified sample, then leaves the design decision to human review.

## Compared lanes

Every lane reads the same frozen sample and builds a new scratch generation sequentially:

1. `text_only` — lexical/source evidence, entry anchors, and session projections; no vectors.
2. `cold_no_cache` — all three stores; unique embedding inputs are sent to the model with no persistent cache.
3. `previous_generation_transfer` — all three stores; vectors are copied only after occurrence ID, content checksum, embedding-input checksum, and profile match the preceding scratch generation.
4. `warm_shared_cache` — all three stores; vectors come from a fully populated scratch copy of the production cache format.

The embedding model is never called concurrently. The cold lane sends one HTTP request at a time. Transfer and warm-cache lanes fail if they would need the model.

## Sample

The sampler:

- lists physical Pi session JSONL files without opening the production recall database;
- sorts files by byte size;
- divides that ordering into one stratum per sampled file;
- chooses one file per stratum with the fixed seed `pi-session-recall-issue-112-no-cache-v1`;
- selects 10% of physical files for a full run;
- copies selected files into a private, read-only scratch snapshot so every lane sees identical bytes.

The committed report contains only aggregate counts, sizes, timings, ratios, and validation results. It never contains source paths or conversation text.

## Run

Validate the harness on a small, size-distributed pilot:

```bash
npm run prototype:no-embedding-cache -- --pilot
```

Run the approved 10% benchmark:

```bash
npm run prototype:no-embedding-cache -- --full
```

Optional flags:

```text
--scratch-root PATH  Put private scratch data below PATH.
--keep-scratch       Keep the private snapshot and stores after the report is written.
```

The default scratch parent is `~/.cache/pi-session-recall-prototypes/issue-112-no-cache`. The command refuses to place scratch data inside the Pi sessions directory, the production recall data directory, or the repository.

## Measurement controls

- All lanes run sequentially.
- One excluded warm-up request verifies the live model and client-side 1,024-dimension prefix-then-L2 projection.
- Before each lane, the harness asks the kernel to evict only the scratch snapshot files from page cache with `posix_fadvise(POSIX_FADV_DONTNEED)`.
- Before transfer and cache reads, it applies the same file-specific eviction to the corresponding scratch store.
- Every lane uses the production importer, tokenizer, chunk policy, projection encoding, and zvec 0.6.0 engine.
- The accepted split-store prototype schema is expanded with the current provenance fields so lexical sizing is not based on a content-only toy row.
- Store writes use checked batches of at most 32 rows. Immutable evidence is inserted, never blindly upserted.
- The model lane sends one request at a time to the configured Octen endpoint. No benchmark lanes overlap.

## Safety

- Original Pi session files are read and copied, never modified.
- The production zvec generation and production embedding cache are never opened.
- All zvec collections and cache files are created below the private scratch run directory.
- Private scratch data is removed after content-free measurements are written unless `--keep-scratch` is explicit.
- Prototype code and results stay on the throwaway prototype branch; production code does not adopt the harness.

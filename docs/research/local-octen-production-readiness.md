# Local Octen production readiness

Status: **release evidence passed**

## Artifact

The production profile uses the project-controlled [`model-octen-embedding-0.6b-onnx-int8-v1`](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1) release through pinned native `onnxruntime-node` 1.27.0 on macOS arm64 and Linux x64 Ryzen processors. Other Linux x64 processors and macOS x64 use pinned `onnxruntime-web` 1.27.0 WASM because native Intel and EPYC runners produced incompatible vectors and stable ONNX Runtime Node packages omit the current Darwin x64 binding. GitHub's reported byte sizes and SHA-256 digests match the implementation constants and downloaded files.

## Bounded prototype

The accepted SmoothQuant ONNX candidate indexed the deterministic 171-session sample with:

- 12,966 Dense documents;
- 100% fixed-corpus recall;
- 3.38 Dense documents per second;
- 1.92 GB peak RSS;
- 58.1 ms sampled search p95.

See [Local embedding runtime prototype](local-embedding-runtime-prototype.md) for the Octen GGUF, Voyage ONNX, and rejected compact Octen ONNX comparisons.

## Clean Linux x64 CLI path

A clean temporary configuration and model root ran the public commands against the exact GitHub release:

1. `psr setup --local --yes --config <scratch>/recall.json --model-root <scratch>/models`
2. `psr model status`
3. `psr model doctor`
4. the real environment-gated integration test

Setup streamed, size-checked, and hash-checked all seven release assets before activating the model directory and writing configuration. Status reported `ready`. Doctor rehashed the artifact, loaded the selected runtime, and returned a normalized 1,024-dimensional vector with norm `1.0000000028391913`.

The integration test independently compared query and document embeddings with upstream Safetensors references, built a disposable SQLite Recall database from one canonical session while `fetch` was disabled, released the provider, created a fresh service, and recovered the expected result through global search. It completed in 4.98 seconds. The complete download, status, doctor, and integration sequence took 42 seconds on an AMD Ryzen 7 8845HS with Node 24.16.0.

All paths were under the dedicated scratch root or the test runner's temporary directory. The active production Recall database and production configuration were not opened or mutated.

## CPU-specific x64 runtime

The portable WASM runtime loaded the same graph and 1.06 GB external weights in Node on the Linux prototype host. It loaded in 740 ms, answered one query in 372 ms, used about 2.50 GB RSS, and reached 0.98489 cosine against the upstream Safetensors reference. Native Intel Linux and macOS runners instead produced the same incompatible 0.736891 query cosine.

A later six-public-text comparison on an AMD Ryzen 7 8845HS measured native ONNX at 19.28 ms warm median, 24.63 ms p95, and 636 MB incremental peak RSS. WASM measured 152.10 ms median, 242.63 ms p95, and 2.31 GB incremental peak RSS. Minimum upstream cosine was 0.9704 native and 0.9653 WASM. This small check used no sessions and wrote no database; it supports runtime routing, not corpus-throughput claims. See [`local-octen-ryzen-backend-comparison.json`](../evaluation/local-octen-ryzen-backend-comparison.json).

Production uses native ONNX on Linux x64 Ryzen and WASM on other Linux x64 processors. GitHub's EPYC runner produced the same incompatible 0.736891 native cosine as Intel, so CPU vendor is not a safe boundary. macOS x64 remains WASM because current stable packages omit its native binding. The manifest records the selected backend, so moving a database between native and WASM platforms requires a rebuild.

## Dependency audit

`onnxruntime-node` 1.27.0 declared vulnerable `adm-zip <0.6.0` in its install-script dependency tree. The lockfile overrides that dependency to patched 0.6.0, matching Microsoft's merged post-1.27 dependency bump. `npm audit --omit=dev` reports zero vulnerabilities, and the real artifact still passed after the override.

Socket reports an obfuscated-code warning for `onnxruntime-web`'s generated/minified runtime bundles. This is the official pinned Microsoft package, not an unknown wrapper; its project and pull-request checks pass, and the exact runtime executes the checksum-pinned model under the conformance gate. The warning is accepted for this required x64 fallback.

## Supported-platform evidence

[GitHub Actions run 31552401490](https://github.com/Whamp/pi-session-recall/actions/runs/31552401490) passed at commit `eead5609b9c3f76bba9dfb001be98bb4ca934d80`:

- [Linux x64](https://github.com/Whamp/pi-session-recall/actions/runs/31552401490/job/93977697017), using WASM;
- [macOS arm64](https://github.com/Whamp/pi-session-recall/actions/runs/31552401490/job/93977697024), using native ONNX Runtime;
- [macOS x64](https://github.com/Whamp/pi-session-recall/actions/runs/31552401490/job/93977696992), using WASM.

Each job downloaded or restored the exact project artifact, ran `psr model doctor`, and executed the real conformance plus offline SQLite build-close-reopen-search test. Unsupported platforms are rejected before download.

Machine-readable evidence: [`local-octen-production-readiness.json`](../evaluation/local-octen-production-readiness.json).

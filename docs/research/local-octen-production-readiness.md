# Local Octen production readiness

Status: **local evidence passed; platform CI pending**

## Artifact

The production profile uses the project-controlled [`model-octen-embedding-0.6b-onnx-int8-v1`](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1) release through pinned `onnxruntime-node` 1.27.0. GitHub's reported byte sizes and SHA-256 digests match the implementation constants and downloaded files.

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

Setup streamed, size-checked, and hash-checked all seven release assets before activating the model directory and writing configuration. Status reported `ready`. Doctor rehashed the artifact, loaded the native runtime, and returned a normalized 1,024-dimensional vector with norm `1.0000000028391913`.

The integration test independently compared query and document embeddings with upstream Safetensors references, built a disposable SQLite Recall database from one canonical session while `fetch` was disabled, released the provider, created a fresh service, and recovered the expected result through global search. It completed in 4.98 seconds. The complete download, status, doctor, and integration sequence took 42 seconds on an AMD Ryzen 7 8845HS with Node 24.16.0.

All paths were under the dedicated scratch root or the test runner's temporary directory. The active production Recall database and production configuration were not opened or mutated.

## Dependency audit

`onnxruntime-node` 1.27.0 declared vulnerable `adm-zip <0.6.0` in its install-script dependency tree. The lockfile overrides that dependency to patched 0.6.0, matching Microsoft's merged post-1.27 dependency bump. `npm audit --omit=dev` reports zero vulnerabilities, and the real artifact still passed after the override.

## Remaining gate

`.github/workflows/local-octen-platform-smoke.yml` must pass on:

- Linux x64;
- macOS arm64;
- macOS x64.

Each job downloads or restores the exact project artifact, runs `psr model doctor`, and executes the real conformance plus offline SQLite build-close-reopen-search test. Windows remains unverified.

Machine-readable evidence: [`local-octen-production-readiness.json`](../evaluation/local-octen-production-readiness.json).

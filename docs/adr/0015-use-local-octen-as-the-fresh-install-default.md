---
status: accepted
---

# Use local Octen as the fresh-install default

Fresh `psr setup` installations default to `Octen/Octen-Embedding-0.6B` running in-process. macOS arm64 and Linux x64 Ryzen processors use pinned native `onnxruntime-node` 1.27.0. Other Linux x64 processors and macOS x64 use pinned single-threaded `onnxruntime-web` 1.27.0 WASM. Native Intel and EPYC runners produced incompatible 0.73689-cosine vectors for the accepted quantized graph while WASM remained conformant. Native ONNX remained conformant on the measured Ryzen host and materially outperformed WASM. The certified artifact is a 1.01 GiB SmoothQuant INT8 ONNX conversion published in the project-controlled GitHub release [`model-octen-embedding-0.6b-onnx-int8-v1`](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1). The CLI verifies every file by byte size and SHA-256 before atomically making the complete model directory available.

The local model returns native 1,024-dimensional vectors. The native provider runs bounded concurrent batch-one operations against one ONNX Runtime session; the WASM provider serializes operations. Both pool the last tokenizer-produced token, and applies explicit L2 normalization before SQLite storage. The certified tokenizer post-processor ends inputs with `<|endoftext|>` token `151643`; manually appending the tokenizer configuration's `<|im_end|>` EOS token `151645` materially changes vectors and is incompatible with this profile.

The existing OpenAI-compatible Octen HTTP profile remains an explicit setup choice. It keeps the current 4B model, 2,560 native dimensions, first-1,024 prefix transformation, and local normalization. Installations with no `embeddingProfile` keep that HTTP behavior so an upgrade does not reinterpret an existing production manifest. New users receive the local default by running `psr setup`.

The manifest binds the request profile, served model, native and stored dimensions, transformation, and tokenizer assets. Local and HTTP profiles are incompatible even when both store 1,024 dimensions. Switching profiles requires an explicit staged or atomic `psr index --rebuild`; no backend or model fallback occurs silently.

The local runtime does not restore the retired configurable-inference platform. Pi Session Recall supports exactly two product profiles: certified local Octen and direct Octen HTTP. It adds no provider plugin registry, background inference worker, persistent embedding cache, reranker, query planner, or model failover.

The bounded 171-session prototype indexed 12,966 Dense documents with 100% fixed-corpus recall, 3.38 documents per second, 1.92 GB peak RSS, and 58.1 ms sampled search p95 on an AMD Ryzen 7 8845HS. Voyage 4 Nano INT8 ONNX used fewer resources but preserved only 81.25% of the fixed corpus. Octen Q8_0 GGUF preserved quality but used 5.14 GB peak RSS and indexed at 2.14 documents per second. A later six-text, zero-session comparison on the Ryzen host measured native ONNX at 19.28 ms warm median and 636 MB incremental peak RSS versus WASM at 152.10 ms and 2.31 GB. Both paths remained conformant. The accepted ONNX artifact matched upstream Safetensors with at least 0.946 cosine in certified batch-one probes and preserved the full practical recall gate.

After download, local indexing and search require no network access and send no conversation text to an embedding service. Linux x64, macOS arm64, and macOS x64 run the real artifact download, runtime doctor, conformance, disposable SQLite build, close, reopen, and offline search in `.github/workflows/local-octen-platform-smoke.yml`. Unsupported platforms are rejected before download and directed to external HTTP.

`onnxruntime-node` 1.27.0 declared vulnerable `adm-zip <0.6.0` for its install script. The package lock overrides that one transitive dependency to patched 0.6.0, matching the dependency bump already merged upstream after 1.27. A clean production audit and real-model runtime test must remain release gates.

This decision preserves ADR-0004's separation of semantic profile from execution backend and ADR-0014's unified SQLite storage. It changes the fresh-install inference default, not the database architecture or canonical JSONL authority.

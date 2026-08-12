# Local embedding runtime prototype

Status: **accepted for implementation**

## Decision

Use `Octen/Octen-Embedding-0.6B` as the default local embedding model for new Pi Session Recall installations. Run the verified SmoothQuant INT8 ONNX artifact through native `onnxruntime-node` 1.27.0 on macOS arm64. Use `onnxruntime-web` 1.27.0 WASM on Linux x64 and macOS x64 with the same graph and weights because native Intel execution produced incompatible vectors. Store native 1,024-dimensional normalized vectors.

Do not use Voyage 4 Nano as the default. Its runtime was smaller and faster, but it recovered only 81.25% of the fixed recall corpus. Do not use the compact full-INT8 Octen ONNX export: its vector similarity fell as low as 0.696 against upstream Safetensors. The Octen Q8_0 GGUF preserved quality but used more memory and indexed more slowly than the accepted ONNX path.

The implementation must keep the existing external Octen HTTP profile. This decision affects fresh local installations only. It does not migrate or rebuild the current production database.

## Question

Can a fresh user run useful recall locally on an ordinary CPU without a separate model server or paid embedding API?

The candidate must:

- preserve at least 90% of the fixed recall corpus;
- work within a two-hour bounded sample run;
- stay below 4 GiB peak RSS;
- return sampled searches within 500 ms;
- use an ungated Apache-2.0 model;
- produce 1,024-dimensional vectors compatible with the current SQLite layout;
- leave the production recall database untouched.

## Corpus and safeguards

All candidates used the same deterministic sample:

| Property                        |                                                              Value |
| ------------------------------- | -----------------------------------------------------------------: |
| Discovered physical sessions    |                                                              3,737 |
| Selected physical sessions      |                                                                171 |
| Represented session directories |                                                                171 |
| Source bytes                    |                                                        170,255,864 |
| Dense documents                 |                                                             12,966 |
| Compact Invocations             |                                                             12,836 |
| Selection                       |                             project round-robin ordered by SHA-256 |
| Selection SHA-256               | `de249b629f25b68651c74ed7a1f801512d634499fea5f3164d3bf9acc069dc70` |

The prototype also ran the checksum-fixed synthetic recall corpus. It wrote only disposable databases under `~/.pi/agent/recall-debug/`. Guards limited each run to 200 sessions, 20,000 Dense documents, two hours, 8 GiB scratch allocation, and a 240 GiB free-space floor. The reports contain aggregate measurements and fixed public probe text; they contain no sampled session paths or content.

Hardware:

- AMD Ryzen 7 8845HS;
- 8 cores and 16 threads;
- Linux x64;
- Node 24.16.0.

## Results

| Candidate                                                   | Fixed recall | Dense docs/s | Sample build | Peak RSS | Search p95 |             Model files | Verdict                                                     |
| ----------------------------------------------------------- | -----------: | -----------: | -----------: | -------: | ---------: | ----------------------: | ----------------------------------------------------------- |
| Octen 0.6B Q8_0 GGUF, `node-llama-cpp` 3.19.1               |         100% |         2.14 |    100.8 min |  5.14 GB |    50.7 ms |                  639 MB | Reject runtime: memory gate failed and indexing was slowest |
| Voyage 4 Nano INT8 ONNX, `onnxruntime-node` 1.27.0          |       81.25% |         4.07 |     53.1 min |  2.40 GB |   119.0 ms |   422 MB plus tokenizer | Reject model: practical recall gate failed                  |
| Octen 0.6B SmoothQuant INT8 ONNX, `onnxruntime-node` 1.27.0 |         100% |         3.38 |     63.9 min |  1.92 GB |    58.1 ms | 1.068 GB plus tokenizer | **Accept**                                                  |

The sampled build includes JSONL parsing, chunking, project attribution, embedding, and transactional SQLite writes. Search includes query embedding and sqlite-vec retrieval.

The block-device write counters were system-wide gross measurements collected across long runs. Unrelated system activity makes them unsuitable for comparing candidates. Scratch allocation and database correctness remained bounded; production implementation still needs focused changed-session write checks.

## Vector conformance

### Accepted Octen SmoothQuant ONNX

Source artifact:

- repository: `cstr/Octen-Embedding-0.6B-ONNX-INT8`;
- revision: `3d68a234435972890cbdf71b6a90f9d3fecc7370`;
- graph SHA-256: `48c4eb1401ba5a5d22d7a7e1fb3e94d63e8ed06231e3d124babc00ead78c8771`;
- weights SHA-256: `1ea5b1a2737474b819a301725cb71381e418d7baa8263769f73486fbe9a74b65`.

Against upstream `Octen/Octen-Embedding-0.6B` revision `d715b32ee68f057b54dff09fc93c23485bc403d3`, ten multilingual and technical probes had median cosine similarity 0.956 and minimum 0.928 in a multi-row run. The fixed retrieval corpus nevertheless remained at 100%. Batch-one probes improved to median 0.983 and minimum 0.946; the graph declares batch size one, so the accepted provider runs concurrent batch-one requests against one shared ONNX session.

Production-provider conformance exposed one tokenizer detail that the aggregate prototype report had not named. With special tokens enabled, the certified tokenizer post-processor ends input with `<|endoftext|>` token `151643`, even though `tokenizer_config.json` names `<|im_end|>` token `151645` as EOS. Manually appending `151645` reduced two upstream-reference cosine scores to 0.716 and 0.637. The accepted provider therefore uses and validates the tokenizer-produced `151643` final token before last-token pooling.

### Octen Q8_0 GGUF

The GGUF had stronger vector parity: median cosine similarity 0.99938 and minimum 0.99780 against upstream Safetensors. Its four llama.cpp embedding contexts exceeded the 4 GiB memory gate, and the sample build was 58% slower than accepted ONNX.

### Voyage 4 Nano ONNX

Voyage had excellent artifact parity: median cosine similarity 0.99991 and minimum 0.99982 at the supported 1,024-dimensional MRL prefix. That proves the 81.25% recall result was not caused by a broken conversion. Voyage uses distinct query and document prompts, bidirectional attention, mean pooling, a learned 1,024-to-2,048 projection, and prefix truncation followed by normalization. The ONNX graph contained those semantics; the GGUF required external projection and runtime overrides, so ONNX was the correct test format.

### Rejected compact Octen ONNX

`cstr/Octen-Embedding-0.6B-ONNX-INT8-FULL` revision `ca4b38a54c6dde80a3dd2f3882f559cb8c6ba3ef` quantized the embedding table as well as matrix multiplications. Independent probes returned median cosine similarity 0.885 and minimum 0.696. The prototype rejected it before a database build.

## Artifact policy

The accepted model and its upstream are ungated and Apache 2.0. The exact verified bytes are published in the project-controlled GitHub release [`model-octen-embedding-0.6b-onnx-int8-v1`](https://github.com/Whamp/pi-session-recall/releases/tag/model-octen-embedding-0.6b-onnx-int8-v1). The release contains the graph, external weights, tokenizer files, an artifact manifest, Apache License 2.0, and a provenance notice. GitHub reports the same SHA-256 digests and byte sizes recorded by the prototype. The production CLI must verify every downloaded file before activation.

The community repository documents SmoothQuant, but its checked-in `quantize_octen_int8.py` still describes the older vanilla dynamic-INT8 path. The project release therefore makes no unsupported reproducibility claim: it mirrors the exact certified community bytes and records both source revisions. Replace it only through a separately versioned and re-certified artifact.

## Production platform amendment

`onnxruntime-node` 1.25.1, 1.26.0, and 1.27.0 were inspected after the initial decision; all ship Darwin arm64 but no Darwin x64 binding. Native Intel Linux with 1.27.0 and native Intel macOS with the older 1.23.2 package both produced the same incompatible 0.736891 query cosine. The accepted model loaded through `onnxruntime-web` 1.27.0 WASM in Node with external weights, reached 0.98489 cosine against the upstream reference, loaded in 740 ms, answered one query in 372 ms, and used about 2.50 GB RSS on the Linux prototype host. Production therefore uses that single-threaded portable backend on both x64 platforms and binds the backend name into the manifest.

## Implementation consequences

The production change should add one profile, not restore the retired inference platform:

- local profile: Octen 0.6B SmoothQuant INT8 through native ONNX Runtime on macOS arm64 or the x64 WASM fallback;
- external profile: existing OpenAI-compatible Octen HTTP provider;
- local artifact cache with partial download, checksums, receipt, atomic activation, status, and diagnosis;
- `psr setup` plus `psr model status`, `download`, and `doctor`;
- one shared ONNX session with four concurrent batch-one operations on the measured 16-thread CPU;
- last-token pooling at the tokenizer-produced `<|endoftext|>` token `151643`, followed by local L2 normalization;
- model, tokenizer, artifact, pooling, normalization, runtime, and vector width in the manifest identity;
- explicit staged rebuild when an existing installation changes embedding profiles;
- no QMD planner, reranker, Zvec, provider registry, daemon, or persistent vector cache.

## Evidence

Machine-readable results:

- [`local-octen-0.6b-prototype-results.json`](../evaluation/local-octen-0.6b-prototype-results.json)
- [`local-octen-0.6b-q8-conformance.json`](../evaluation/local-octen-0.6b-q8-conformance.json)
- [`local-octen-0.6b-onnx-prototype-results.json`](../evaluation/local-octen-0.6b-onnx-prototype-results.json)
- [`local-octen-0.6b-onnx-conformance.json`](../evaluation/local-octen-0.6b-onnx-conformance.json)
- [`rejected-local-octen-0.6b-onnx-full-int8-conformance.json`](../evaluation/rejected-local-octen-0.6b-onnx-full-int8-conformance.json)
- [`voyage-4-nano-prototype-results.json`](../evaluation/voyage-4-nano-prototype-results.json)
- [`voyage-4-nano-int8-conformance.json`](../evaluation/voyage-4-nano-int8-conformance.json)

The throwaway harness lives only on the prototype branch. Production code should carry the measured decision, not the prototype implementation.

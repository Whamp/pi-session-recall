# Pinned EmbeddingGemma model artifact

Conversation Recall recommends one EmbeddingGemma artifact for future embedded inference. This ticket makes the artifact inspectable and safely cacheable; it does not configure an embedding provider, load the model, or index session text.

## Immutable profile

| Property                     | Pinned value                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| Profile                      | `embeddinggemma-300m-q8-0-v1`                                                            |
| Model                        | `google/embeddinggemma-300M`                                                             |
| Artifact repository          | `ggml-org/embeddinggemma-300M-GGUF`                                                      |
| Repository revision          | `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12`                                               |
| GGUF artifact                | `embeddinggemma-300M-Q8_0.gguf`                                                          |
| Byte size                    | `333590944`                                                                              |
| SHA-256                      | `b5ce9d77a3fc4b3b39ccb5643c36777911cc4eb46a66962eadfa3f5f60490d63`                       |
| Quantization                 | `Q8_0`                                                                                   |
| Native and stored dimensions | `768`                                                                                    |
| Pooling                      | mean                                                                                     |
| Normalization                | L2                                                                                       |
| Query format                 | `task: search result \| query: ${query}`                                                 |
| Document format              | `title: none \| text: ${document}`                                                       |
| Tokenizer                    | tokenizer metadata inside the checksum-pinned GGUF artifact                              |
| Canary                       | `repeat-cosine-v1`, 768 L2-normalized dimensions, minimum repeated-query cosine `0.9995` |
| Terms                        | [Gemma Terms of Use](https://ai.google.dev/gemma/terms), distribution review required    |

The model URL contains the immutable repository revision. The tokenizer identity includes the artifact SHA-256, so a moving upstream file cannot retain this profile identity. The query and document formats follow QMD revision `e428df76bc0274d9e93eb7ca3e95673315c42e90`; Conversation Recall fixes the document title to `none`.

## Operator commands

Commands emit one JSON value to stdout. Inspection and status never create the cache directory or start a request.

```bash
npm run --silent model:embeddinggemma -- inspect
npm run --silent model:embeddinggemma -- status
npm run --silent model:embeddinggemma -- verify
npm run --silent model:embeddinggemma -- doctor
```

`inspect` includes the complete profile and current status. `status` and `verify` re-read and re-hash the artifact. `doctor` adds a health conclusion and exact repair action.

Downloads, repairs, and removal require the explicit `--approve` flag:

```bash
npm run --silent model:embeddinggemma -- download --approve
npm run --silent model:embeddinggemma -- repair --approve
npm run --silent model:embeddinggemma -- remove --approve
```

Without `--approve`, these operations fail before transport, directory creation, replacement, or removal. The optional `PI_RECALL_MODEL_CACHE_DIRECTORY` environment variable changes the cache root. The default is `~/.pi/agent/recall/models`.

A download streams into a unique sibling `*.partial-<uuid>` path. It must match the pinned byte size and SHA-256 and contain a bounded, structurally valid GGUF metadata and tensor directory. Only then does an atomic rename activate the artifact and an atomic receipt bind it to the profile. A failed or interrupted download remains partial and cannot replace a valid artifact. Re-running download against a valid artifact performs no transport work.

## Status and repair

| State          | Meaning                                                                                                 | Doctor action                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `missing`      | No final artifact or partial download exists.                                                           | Approve a pinned download.                                                        |
| `partial`      | A staged download exists, but no final artifact activated.                                              | Approve repair; a fresh pinned download is validated before activation.           |
| `corrupt`      | The final path fails file, size, SHA-256, or GGUF validation.                                           | Approve repair to stage and validate a replacement.                               |
| `incompatible` | Pinned bytes validate, but the activation receipt is missing, malformed, or belongs to another profile. | Approve repair to bind the verified bytes to this profile; no download is needed. |
| `valid`        | Artifact bytes, GGUF structure, and profile receipt all validate.                                       | No repair required.                                                               |

Removal deletes only this profile's model-cache directory. It does not alter an index generation, tokenizer cache, embedding-vector cache, or recall configuration.

## Evidence boundary

Committed tests use generated local GGUF fixtures and injected transport. They prove consent gating, immutable URL selection, staged activation, file size, SHA-256 and GGUF rejection, all five status states, transport-free compatible repair, no redownload of valid bytes, and explicit removal. They do not download or execute the 333,590,944-byte model.

The following external acceptance evidence remains pending:

- a legal/distribution review of the Gemma Terms of Use and required notices;
- an operator-approved download of the pinned real artifact and verification of its published size, SHA-256, and GGUF directory through this command;
- model loading, tokenizer conformance, a canonical live canary vector, cold/warm inference, device identity, and throughput measurements, which belong to embedded execution issue #36 and parent issue #28.

These pending items are not represented as passing evidence, and the profile remains marked `review-required`.

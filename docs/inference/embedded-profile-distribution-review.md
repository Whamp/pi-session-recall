# Embedded profile distribution review packet

**Status: maintainer approval required.** This packet records source facts and release actions. It does not approve distribution or claim legal review.

## Artifacts under review

| Capability     | Pinned artifact                                                                      | Declared terms                      | Repository evidence                                                                                                                                                                                                                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------ | ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Embedding      | `embeddinggemma-300M-Q8_0.gguf` at `0f741b5a6585bd53aeb15cd1372c56f2a0f65e12`        | Gemma Terms of Use                  | The [pinned conversion repository](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/tree/0f741b5a6585bd53aeb15cd1372c56f2a0f65e12) contains the model and README but no license file or license card metadata. Its model card names `google/embeddinggemma-300M` as the base model.                                        |
| Reranking      | `qwen3-reranker-0.6b-q8_0.gguf` at `a02f48bb4f057028298c21fa033da2b30d7742d5`        | Apache-2.0 metadata                 | The [pinned conversion model card](https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/blob/a02f48bb4f057028298c21fa033da2b30d7742d5/README.md) declares `apache-2.0` and points to `Qwen/Qwen3-Reranker-0.6B`. The pinned conversion repository contains no `LICENSE` or `NOTICE` file.                                 |
| Query planning | `qmd-query-expansion-1.7B-q4_k_m.gguf` at `7816de0b72572c6c860ca1eddf97ba9e7fb8cc65` | MIT metadata; Apache-2.0 base model | The [pinned fine-tune model card](https://huggingface.co/tobil/qmd-query-expansion-1.7B-gguf/blob/7816de0b72572c6c860ca1eddf97ba9e7fb8cc65/README.md) declares `mit` and names `Qwen/Qwen3-1.7B` as its base model. The fine-tune repository contains no license file. The base-model repository contains an Apache-2.0 `LICENSE`. |

Repository metadata and file lists were checked on 2026-07-27. They identify material for review; metadata alone does not prove that the distributor has met the licenses.

## Gemma release actions

The current [Gemma Terms of Use](https://ai.google.dev/gemma/terms), §3.1, states that a distributor must:

1. include the §3.2 use restrictions as an enforceable provision in the downstream agreement;
2. notify downstream recipients that Gemma and model derivatives remain subject to those restrictions;
3. give every third-party recipient a copy of the agreement;
4. mark modified files with prominent modification notices; and
5. accompany non-hosted distributions with a `Notice` text file containing this exact text:

> Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms

Before release, a maintainer must decide whether the package's download/cache workflow constitutes distribution, approve the downstream agreement and restriction notice, approve the agreement-delivery mechanism, decide whether the converted GGUF needs a modification notice, and approve the exact `Notice` file placement. The repository must not change `distributionStatus` from `review-required` until that decision is recorded.

## Qwen reranker release actions

Before release, a maintainer must:

1. verify the license chain from `Qwen/Qwen3-Reranker-0.6B` through the pinned GGUF conversion;
2. obtain and retain the authoritative Apache License 2.0 text rather than relying only on model-card metadata;
3. determine whether the upstream work carries attribution or `NOTICE` material that must accompany distribution;
4. record where required license and notice files ship; and
5. approve the embedded adapter's documented score-semantics patch separately from license review.

## QMD planner release actions

Before release, a maintainer must:

1. verify that the fine-tune publisher's MIT declaration covers the pinned weights and conversion;
2. retain the authoritative MIT license text and required copyright notice;
3. review the Apache-2.0 obligations inherited from `Qwen/Qwen3-1.7B`;
4. determine whether base-model attribution or `NOTICE` material must accompany distribution; and
5. record where every required license and notice file ships.

## Approval record

A release approval must record:

- reviewer and date;
- exact profile IDs, repository revisions, artifact SHA-256 values, and byte sizes;
- approved distribution mechanism;
- downstream agreement and notice paths;
- all license and attribution files included in the package; and
- any release conditions or rejected distribution modes.

Until that record exists, all three profiles remain `review-required`, model downloads remain operator-approved actions, and [embedded profile acceptance](../evaluation/embedded-profile-acceptance.md) remains blocked.

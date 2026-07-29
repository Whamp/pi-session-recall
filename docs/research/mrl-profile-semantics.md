# MRL truncation semantics for recall profiles

**Ticket:** [#109 — Verify model-supported MRL truncation semantics for recall profiles](https://github.com/Whamp/pi-session-recall/issues/109)  
**Architecture map:** [Map: Design a predictable rebuildable recall storage architecture](https://github.com/Whamp/pi-session-recall/issues/107)  
**Models:** `Octen/Octen-Embedding-4B` and the pinned `embeddinggemma-300m-q8-0-v1` profile  
**Repository baseline:** commit `e44240a1c5b54ea6db9e5bdf45c97649dcd7b820`

## Executive answer

The two models require different setup semantics.

**EmbeddingGemma has verified, model-specific Matryoshka Representation Learning support at 768, 512, 256, and 128 dimensions.** Google’s model card and technical report explicitly identify those widths; the report describes losses over overlapping subdimensions, and its evaluations report all four. A reduced EmbeddingGemma vector must be produced by keeping the leading `N` components and then L2-normalizing that prefix:

```text
model output at 768
→ first N components
→ L2 normalization
→ stored vector
```

Setup may offer 768, 512, 256, and 128 as first-class, model-verified choices. Other positive widths below 768 are mechanically sliceable, but the cited model-owner sources do not explicitly document or evaluate those cutoffs. They may be allowed only through an explicit unverified override. Widths above 768 are invalid because the model cannot produce them.

**Octen-Embedding-4B has no verified reduced-width MRL guarantee for the final Octen checkpoint.** Octen documents a 2,560-dimensional, last-token-pooled, normalized LoRA fine-tune of Qwen3-Embedding-4B and reports results only at 2,560 dimensions. Qwen’s base model supports MRL and user-selected dimensions from 32 through 2,560, but that guarantee cannot automatically be inherited by a later fine-tuned checkpoint. Octen’s training sources do not document nested-dimension losses or reduced-width evaluation.

Setup should therefore treat **2,560 as Octen’s only verified and recommended width**. Any smaller Octen width is an explicit, unverified override. Widths from 32 through 2,559 have relevant base-model evidence but no final-checkpoint verification; widths from 1 through 31 are only mechanically sliceable.

Changing stored dimensions changes vector semantics. It requires a distinct semantic embedding-profile identity, canary, vector-cache identity, index manifest, zvec schema, inference-conformance identity, and recall generation. It must not change the model-artifact identity when the artifact bytes are unchanged.

No universal dimension cap is supported by this evidence, and setup must not impose a mandatory quality gate after a user explicitly accepts an unverified width.

## Confidence and evidence limits

### Confidence

- **High:** EmbeddingGemma’s verified widths, leading-prefix semantics, truncate-then-normalize order, architecture, and retrieval prompts.
- **High:** Octen’s native width, pooling, normalization module, prompt metadata, and absence of reduced-width evidence in the reviewed first-party sources.
- **High:** Mechanical slicing is not proof of trained MRL behavior.
- **High:** Stored width is part of vector compatibility and therefore generation identity.
- **Medium:** Exact parity between the project’s GGUF execution and each model’s reference Sentence Transformers execution. The relevant runtimes support the required operations, but the actual deployed artifacts were not executed or inspected during this research.

### Evidence limits

This report consolidates three completed research outputs: the primary memo and independent OpenAI and GLM same-axis source reviews. No new research pass was performed.

The source review covered public model cards, immutable model configuration revisions, technical documentation, papers, pinned runtime source, and repository code at `e44240a`. It did **not** execute either model, inspect production configuration or recall data, open a production model cache, or measure retrieval quality on the recall corpus.

Consequently, this report establishes:

- documented model semantics;
- pinned implementation behavior;
- repository behavior at the stated baseline;
- architectural consequences of selecting a stored width.

It does not establish:

- real-artifact parity with a reference implementation;
- retrieval quality at any width on the project’s corpus;
- that Octen retained Qwen’s nested-prefix quality after fine-tuning;
- that an arbitrary mechanically sliceable width is useful.

A missing statement is described narrowly as “not documented or evaluated by the cited model-owner sources,” not as proof that no evidence exists anywhere.

## Octen evidence matrix

| Question                           | Evidence state                 | Finding                                                                                                                                                                   |
| ---------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Native dimensions                  | Documented/configured          | 2,560. The checkpoint has `hidden_size: 2560`, and the pooler declares `word_embedding_dimension: 2560`.                                                                  |
| Final-checkpoint MRL training      | Missing coverage               | No reviewed Octen source describes a nested-dimension loss.                                                                                                               |
| Reduced-width evaluation           | Missing coverage               | Octen reports the final checkpoint only at 2,560 dimensions.                                                                                                              |
| Base-model MRL                     | Documented for Qwen, not Octen | Qwen3-Embedding-4B documents MRL and user-defined widths from 32 to 2,560.                                                                                                |
| Semantically verified widths       | Verified                       | 2,560 only for the final Octen checkpoint.                                                                                                                                |
| Other sliceable widths             | Implementation behavior        | Any leading prefix from 1 to 2,559 can mechanically be produced. This does not verify quality.                                                                            |
| Pooling                            | Configured and demonstrated    | Last non-padding token. Correct extraction depends on padding and attention-mask behavior.                                                                                |
| Normalization                      | Configured and demonstrated    | A final Normalize module and explicit L2 normalization in the Transformers example.                                                                                       |
| Reduced-width operation order      | Missing coverage               | Octen does not document reduced-width semantics. If an override is supported, use the same explicit leading-prefix-then-L2 policy rather than adapter-dependent behavior. |
| Similarity                         | Configured                     | Cosine.                                                                                                                                                                   |
| Query prompt                       | Configured, not defaulted      | `Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:`                                                                            |
| Document prompt                    | Configured, not defaulted      | One space, `" "`.                                                                                                                                                         |
| Default prompt                     | Configured                     | `default_prompt_name` is null; the card also demonstrates unprompted encoding.                                                                                            |
| Project input behavior at baseline | Local observation              | Query and document text are sent unchanged.                                                                                                                               |
| Project execution at baseline      | Local observation              | Defaults to Q8_0, last pooling, and 2,560 dimensions, subject to configuration overrides. The remote GGUF repository, revision, and checksum are not pinned.              |
| Setup verdict                      | Synthesis                      | Recommend 2,560. Smaller widths require explicit unverified override.                                                                                                     |

## EmbeddingGemma evidence matrix

| Question                           | Evidence state               | Finding                                                                                                                                          |
| ---------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Native dimensions                  | Documented                   | 768 after pooling and the dense projection stack.                                                                                                |
| MRL training                       | Documented                   | Losses are applied across overlapping subdimensions.                                                                                             |
| Evaluated widths                   | Documented                   | 768, 512, 256, and 128.                                                                                                                          |
| Semantically verified widths       | Verified                     | 768, 512, 256, and 128.                                                                                                                          |
| Other sliceable widths             | Implementation behavior      | Any leading prefix from 1 to 767 can be produced, but other cutoffs are not explicitly documented or evaluated by the cited model-owner sources. |
| Pooling                            | Documented/source-confirmed  | Attention-mask-aware mean pooling.                                                                                                               |
| Projection order                   | Documented                   | Mean pooling → 768→3,072 dense projection → 3,072→768 dense projection.                                                                          |
| Normalization                      | Documented                   | Keep the leading prefix and then L2-renormalize it.                                                                                              |
| Similarity                         | Documented/local observation | Cosine is appropriate; dot product is equivalent only for unit-normalized vectors. The project uses cosine.                                      |
| Query prompt                       | Documented                   | `task: search result \| query: {content}`                                                                                                        |
| Document prompt                    | Documented                   | `title: {title \| "none"} \| text: {content}`                                                                                                    |
| Project input behavior at baseline | Local observation            | Uses the retrieval prompts and fixes document title to `none`.                                                                                   |
| Model context                      | Documented                   | The model card says 2K; the Sentence Transformers guide gives `max_seq_length: 2048`.                                                            |
| Project execution at baseline      | Local observation            | Checksum-pinned Q8_0 GGUF through node-llama-cpp or llama.cpp HTTP; project-side normalization and FP32 stored vectors.                          |
| Setup verdict                      | Synthesis                    | Offer 768, 512, 256, and 128 as first-class choices. Other sub-native widths require explicit unverified override.                               |

## Verified MRL support and dimensions

### EmbeddingGemma

EmbeddingGemma’s model card names a 768-dimensional output and smaller 512-, 256-, and 128-dimensional MRL options. It explicitly tells callers to truncate and then re-normalize.

The technical report is stronger evidence than a generic inference utility: it adapts the contrastive and spread-out losses to overlapping subdimensions, combines those losses without special weighting, identifies 512, 256, and 128 as supported widths in addition to 768, and reports evaluation results at all four widths.

This supports the exact setup classification:

- **768:** verified native width;
- **512:** verified MRL width;
- **256:** verified MRL width;
- **128:** verified MRL width;
- **1–767 except 128, 256, and 512:** mechanically sliceable but unverified at that cutoff;
- **greater than 768:** invalid.

The arbitrary slicing accepted by Sentence Transformers is not additional training evidence. Its `truncate_embeddings` utility simply returns `embeddings[..., :truncate_dim]`.

### Octen

Octen’s card, model configuration, pooling configuration, and benchmark table establish a 2,560-dimensional output. Its first-party training post says the Qwen base was fine-tuned with LoRA over query, key, value, and feed-forward layer families, but it does not identify an MRL loss or reduced-prefix evaluation.

The reviewers differed on one sentence in the primary memo. The primary memo said the modified layers “contribute to every output component.” The OpenAI review considered that overstated because the source does not prove that every component changed or received a nonzero contribution. The GLM review accepted the broader inference. The conservative formulation is:

> The modified layers can alter the hidden representation from which all 2,560 output components are pooled.

That is sufficient for the decision: preservation of Qwen’s nested-prefix quality is plausible, but it is not guaranteed by the base model’s documentation.

Octen’s setup classification is therefore:

- **2,560:** verified final-checkpoint width;
- **32–2,559:** mechanically sliceable, with base-Qwen MRL evidence but no final-Octen verification;
- **1–31:** mechanically sliceable only;
- **greater than 2,560:** invalid.

“MRL unsupported” must be read as **unsupported as a verified claim for the final Octen checkpoint**, not as proof that reduced Octen prefixes are necessarily poor.

## Exact pooling, truncation, and normalization order

### EmbeddingGemma

The conforming reduced-width sequence is:

1. Format the query or document with the profile’s operation-specific prompt.
2. Tokenize using the accepted model/GGUF tokenizer behavior.
3. Execute the model.
4. Mean-pool attended token representations.
5. Apply the 768→3,072 projection.
6. Apply the 3,072→768 projection.
7. Keep the first `N` components.
8. L2-normalize the `N`-component prefix.
9. Store the resulting FP32 unit vector.
10. Compare with cosine, or with dot product only where unit normalization is guaranteed by the store contract.

Sentence Transformers performs truncation before optional output normalization and performs embedding-output quantization after normalization. The model pipeline’s native Normalize module does not eliminate the need to normalize again after slicing: the norm of a prefix is generally below one.

Normalizing the 768-vector before slicing and then normalizing the prefix again produces the same final direction as slicing the unnormalized vector and normalizing once. The prohibited behavior is **normalize 768 → slice → store without re-normalizing**.

At the stated repository baseline, the project normalizes EmbeddingGemma’s returned vector. Reduced-width support should put one shared, adapter-independent operation at that boundary:

```text
native provider output
→ validate native width and finite components
→ leading-prefix slice
→ L2 normalization
→ stored-width validation
```

Both embedded and HTTP execution should use that operation. A server-specific `dimensions` parameter would allow adapters to produce different semantics and should not define the profile.

### Octen

Octen’s reference module graph is:

```text
Transformer
→ last-token pooling
→ Normalize
```

The direct Transformers example selects the last token and applies L2 normalization. Correct last-token pooling depends on padding. Octen’s simple example requests left padding; Qwen’s reference implementation either uses the final token for left-padded batches or computes the last attended token from the attention mask.

Octen does not document reduced-width operation order. If the project allows an explicit reduced-width override, it should define that profile as:

```text
native 2,560 output
→ leading-prefix slice
→ L2 normalization
```

This is a project-defined mechanical policy, not a verified Octen MRL guarantee.

## Prompt and tokenizer semantics

Prompts are part of embedding semantics because they alter every resulting component.

For EmbeddingGemma retrieval:

- queries use `task: search result | query: {content}`;
- documents use `title: {title | "none"} | text: {content}`;
- the project fixes the title to `none`.

These prompts appear in Google’s instructions and training examples. They do not determine which widths were MRL-trained, but they must be included in profile and cache identity.

For Octen, the stored Sentence Transformers configuration provides a named query instruction and a one-space document prompt, while leaving `default_prompt_name` null. Therefore, selecting `prompt_name="query"` is observably different from calling unprompted `model.encode`. The evidence does not establish either mode as the only valid recall profile. The project profile at the stated baseline sends query and document text unchanged, so changing to the named prompt would itself require a new semantic profile and generation.

Chunk tokenization and inference tokenization must not be conflated. The project’s EmbeddingGemma chunk geometry derives tokenizer identity from the checksum-pinned GGUF and calls tokenization without special tokens. node-llama-cpp inference may nevertheless prepend or append model-resolved boundary tokens. Likewise, Octen’s chunk tokenizer is pinned separately and disables special tokens, but that does not prove the remote llama.cpp server uses identical tokenizer metadata or boundary-token behavior.

## GGUF execution implications

The GGUF path is not identical to the reference Sentence Transformers path. It substitutes quantized GGUF weights, llama.cpp pooling and dense-module execution, GGUF tokenizer metadata, and node-llama-cpp or HTTP transport.

llama.cpp supports applying converted Sentence Transformers dense modules after pooling **when the GGUF was converted with the relevant dense-module support**. The source establishes conditional execution order; it does not prove that an uninspected artifact contains the expected tensors.

node-llama-cpp returns the native embedding vector without performing MRL slicing or normalization in its wrapper. QMD likewise formats EmbeddingGemma’s prompts and returns `embedding.vector` without implementing MRL truncation. The project must therefore own the profile-level truncation and final normalization behavior.

The HTTP client at the stated baseline sends only `{model,input}` and validates response count and configured width. It does not request dimensions or normalize output. To preserve execution-independent semantics, HTTP should request native output and apply the same local prefix-and-normalize operation as embedded execution.

ADR-0004 does **not** state that the width is 768. The GLM review correctly identified that attribution as an error in the primary memo. The ADR establishes that one semantic embedding profile owns native and stored dimensions across execution adapters. The profile instance at the stated baseline fixes EmbeddingGemma at 768. Supporting selectable stored widths requires extending that profile architecture without allowing embedded and HTTP adapters to define different vector semantics.

The Q8_0 artifact quantizes model weights, not the project’s stored vectors. The vector cache writes FP32 payloads, and zvec uses `VECTOR_FP32` with cosine similarity.

## Canary capabilities and limits

A dimension-aware deterministic canary can establish:

- returned vector width equals the selected stored width;
- all components are finite;
- final L2 norm is one within tolerance;
- repeated execution meets a stated stability threshold;
- reduced output equals `L2(native[0:N])` within tolerance, if both outputs or an independent reference are available;
- query and document paths apply distinct prompts;
- embedded and HTTP adapters match an independently generated reference under the same artifact, tokenizer, pooling, truncation, and normalization policy.

The conformance harness at the stated baseline checks shape, finite values, normalization, and expected vectors. The profile canary embeds one query twice and requires cosine similarity of at least `0.9995`. The manifest stores a canonical FP32 vector and fingerprint. Existing fixture tests exercise prompt routing and non-repeatability rejection, but injected vectors are not evidence about a real model artifact.

A canary cannot prove:

- training used MRL;
- Octen retained Qwen’s MRL behavior;
- an undocumented cutoff preserves retrieval quality;
- a selected width is acceptable on the recall corpus;
- a stable remote server is serving the intended model;
- GGUF conversion exactly matches the reference checkpoint.

Repeatability proves repeatability, not identity or quality. Prefix equality proves slicing order, not training support. Setup must therefore keep **source evidence state** separate from **runtime conformance state**.

## Setup evidence states and explicit user override

Setup should expose at least these states:

- **Verified:** model-owner training/documentation and cutoff-specific evaluation support the width.
- **Base-model evidence only:** the base checkpoint supports the width, but the selected fine-tune does not verify retained semantics.
- **Mechanical only:** the software can slice the prefix, but there is no model-specific cutoff evidence.
- **Runtime verified:** the selected artifact and adapter passed bounded canary checks.
- **Unknown:** the available sources do not establish the claim.

Recommended choices:

- **EmbeddingGemma:** offer 768, 512, 256, and 128 as first-class choices. Do not make corpus-quality claims until corpus measurements exist.
- **Octen:** recommend 2,560 only.

An unverified choice should require one explicit action and persist that decision:

> `Octen/Octen-Embedding-4B` documents and evaluates 2,560 dimensions. Its Qwen3 base model supports MRL widths from 32 to 2,560, but Octen does not document reduced-width fine-tuning or evaluation. Selecting 512 will keep the first 512 components and L2-normalize them. Shape and execution can be verified; retrieval quality and Octen MRL support cannot. This selection creates a distinct embedding profile and replacement recall generation. Continue with `--allow-unverified-dimensions 512`.

For an undocumented EmbeddingGemma cutoff:

> EmbeddingGemma documents MRL at 768, 512, 256, and 128 dimensions. Selecting 384 is mechanical leading-prefix slicing without model-specific training or evaluation evidence. The prefix will be L2-normalized and stored as a distinct embedding profile and recall generation. Continue with `--allow-unverified-dimensions 384`.

After that explicit decision, setup must not silently restore native width or impose a mandatory quality gate. Quality evaluation remains useful evidence, not a second approval requirement.

Persist the model repository and revision, artifact identity, evidence state, citations, native and stored widths, truncation and normalization policies, prompts, pooling, tokenizer identity, runtime versions, canary result, verification time, and unverified-override state.

## Profile, cache, and generation identity

A stored-width change must alter all vector-compatibility identities:

1. **Semantic embedding profile:** represent `nativeDimensions`, `storedDimensions`, `prefix` truncation, and `l2-after-truncate` normalization explicitly. Include them with prompts and canary policy in the derived `embeddingProfileId`.
2. **Canary:** change expected width, canonical vector, and fingerprint. A separate native-output canary may support prefix-relation checks.
3. **Embedding-vector cache:** include stored width and operation policy. Cached stored vectors at different widths must never alias.
4. **Index manifest and store:** record native/stored widths and operation order. Change the manifest fingerprint and zvec collection dimension.
5. **Inference capability and conformance:** include the semantic profile and evidence provenance. The selected-capability record and its nested conformance record currently hold execution and numeric conformance data but not the full source-evidence provenance.
6. **Recall generation:** change `embeddingProfileId` and `indexManifestFingerprint`. Activation, rollback, and replay must select a complete compatible generation.
7. **Incremental transfer:** bind prepared vectors to the target generation and exact manifest/profile fingerprint. Reject prepared work if either identity changed.
8. **Runtime ownership:** key runtime reuse on the complete effective inference selection.

Model-artifact identity must remain stable when only stored width changes. Repository, immutable revision, file size, checksum, and bytes remain identical.

The reviewers expressed a terminology nuance here. The artifact cache uses the recommended artifact candidate’s `profileId`, while the service separately derives `embeddingProfileId` from vector semantics. The safe implementation is to keep the artifact/candidate `profileId` stable and vary the derived semantic `embeddingProfileId`. A deeper artifact-cache split is required only if dimension selection would otherwise mutate the artifact-cache key; it is not inherently a prerequisite.

The zero-vector fallback at the stated baseline for lexical-only records must not be copied into a new dimension architecture. Lexical-only evidence should not acquire a dimension-dependent fake vector.

## Decisive per-model verdicts

### Octen — reduced-width MRL is unsupported as a verified final-checkpoint claim

Use 2,560 as the only verified and recommended stored width. Permit smaller widths only through an explicit unverified override that distinguishes Qwen base-model evidence from final-Octen evidence.

This verdict does not claim that smaller Octen prefixes are necessarily unusable. It states that the supplied primary sources do not justify presenting them as model-supported MRL choices.

### EmbeddingGemma — MRL is verified at 768, 512, 256, and 128

Offer all four widths as first-class setup choices. Implement one execution-independent `native output → leading prefix → L2 normalization` operation for embedded and HTTP adapters. Permit other positive sub-native widths only through the explicit unverified override.

Do not introduce a universal dimension cap. Do not require a post-selection quality gate. Corpus evaluation may inform defaults and warnings later, but it is not part of the model-supported-width proof.

## Unresolved questions

1. Did Octen’s fine-tuning retain useful nested-prefix behavior, and was any nested-dimension loss used but undocumented?
2. Does the deployed Octen GGUF exactly match the first-party checkpoint, pooling metadata, tokenizer behavior, and normalization semantics?
3. Does the pinned EmbeddingGemma GGUF contain and execute the expected converted dense modules?
4. How closely do embedded and HTTP GGUF outputs match a reference Sentence Transformers execution at 768, 512, 256, and 128?
5. What retrieval-quality tradeoffs do those four EmbeddingGemma widths produce on the recall corpus?
6. If project-specific Octen evaluation supports a reduced width, how should setup label it without misrepresenting model-owner MRL evidence?
7. Should native vectors be retained in a distinct derivation cache, allowing several stored-width profiles to be generated without repeating inference?
8. How should semantic-evidence provenance be represented in the selected-capability and conformance records without conflating it with runtime measurements?

The narrow follow-up for EmbeddingGemma is to freeze reference vectors at the four documented widths and compare both GGUF adapters against them, followed by one unchanged recall-corpus evaluation per width.

The narrow follow-up for Octen is to obtain an Octen-authored statement or training configuration covering nested-dimension losses, then evaluate selected widths against a checksum-pinned artifact. Evaluation alone could establish a project-supported width, but not prove Octen’s training used MRL.

## Complete primary-source references

### Model-owner sources

1. [Octen-Embedding-4B model card, revision `fea468fae3f0caffbae8a12ba792d1c394b6277d`](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/README.md)
2. [Octen model configuration](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/config.json)
3. [Octen module graph](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/modules.json)
4. [Octen pooling configuration](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/1_Pooling/config.json)
5. [Octen normalization configuration](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/2_Normalize/config.json)
6. [Octen Sentence Transformers prompt and similarity configuration](https://huggingface.co/Octen/Octen-Embedding-4B/blob/fea468fae3f0caffbae8a12ba792d1c394b6277d/config_sentence_transformers.json)
7. [Octen training post §2.3.1](https://octen-team.github.io/octen_blog/posts/octen-rteb-first-place/#231-lora-fine-tuning-based-on-qwen3)
8. [Qwen3-Embedding-4B model card, revision `5cf2132abc99cad020ac570b19d031efec650f2b`](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/5cf2132abc99cad020ac570b19d031efec650f2b/README.md)
9. [EmbeddingGemma model card](https://ai.google.dev/gemma/docs/embeddinggemma/model_card)
10. [EmbeddingGemma technical report, architecture §2.1](https://arxiv.org/html/2509.20354v3#S2.SS1)
11. [EmbeddingGemma technical report, MRL §2.2](https://arxiv.org/html/2509.20354v3#S2.SS2.SSS0.Px2)
12. [EmbeddingGemma technical report, evaluation §4](https://arxiv.org/html/2509.20354v3#S4)
13. [Google Sentence Transformers inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers)

### Runtime and implementation sources

14. [Sentence Transformers `truncate_embeddings`, commit `d407492`](https://github.com/huggingface/sentence-transformers/blob/d40749229c2518328335dda01084562691052f22/sentence_transformers/util/tensor.py#L98-L134)
15. [Sentence Transformers encode order: truncation, normalization, and output quantization](https://github.com/huggingface/sentence-transformers/blob/d40749229c2518328335dda01084562691052f22/sentence_transformers/sentence_transformer/model.py#L644-L706)
16. [Sentence Transformers attention-mask-aware mean pooling](https://github.com/huggingface/sentence-transformers/blob/d40749229c2518328335dda01084562691052f22/sentence_transformers/sentence_transformer/modules/pooling.py#L196-L213)
17. [llama.cpp pooling and conditional dense-module execution, tag `b8390`](https://github.com/ggml-org/llama.cpp/blob/b6c83aad55a4ce17ec96fced7770cd1be8758193/src/llama-model.cpp#L8668-L8678)
18. [node-llama-cpp `LlamaEmbeddingContext.getEmbeddingFor`, version 3.18.1](https://github.com/withcatai/node-llama-cpp/blob/57bea3da9ffa78955e8b25f195ce6cc714980cb5/src/evaluator/LlamaEmbeddingContext.ts#L78-L117)
19. [node-llama-cpp `LlamaEmbedding`](https://github.com/withcatai/node-llama-cpp/blob/57bea3da9ffa78955e8b25f195ce6cc714980cb5/src/evaluator/LlamaEmbedding.ts#L10-L63)
20. [QMD EmbeddingGemma prompt formatting, revision `e428df7`](https://github.com/tobi/qmd/blob/e428df76bc0274d9e93eb7ca3e95673315c42e90/src/llm.ts#L89-L114)
21. [QMD embedding return path](https://github.com/tobi/qmd/blob/e428df76bc0274d9e93eb7ca3e95673315c42e90/src/llm.ts#L1297-L1314)

### Repository sources at `e44240a`

22. `src/recall-model-profiles.ts`
23. `src/recall-conversation-config.ts`
24. `src/createLlamaCppHttpEmbeddingProvider.ts`
25. `src/local-embedding-client.ts`
26. `src/octen-conversation-tokenizer.ts`
27. `src/embedded-embeddinggemma-provider.ts`
28. `src/embedding-vector-cache.ts`
29. `src/zvec-conversation-store.ts`
30. `src/recall-inference-conformance.ts`
31. `src/recall-inference-configuration.ts`
32. `src/recall-conversation-service.ts`
33. `src/recall-index-manifest.ts`
34. `src/recall-generation-state.ts`
35. `src/recall-model-artifact-cache.ts`
36. `src/prepare-incremental-recall-transfer.ts`
37. `src/commit-incremental-recall-transfer.ts`
38. `src/recall-extension.ts`
39. `docs/adr/0004-keep-embedding-profile-semantics-independent-of-execution.md`
40. `docs/inference/embedded-embeddinggemma.md`

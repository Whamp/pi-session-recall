# Guided setup for the first index

A fresh guided setup starts unconfigured. `status` recommends the pinned embedded EmbeddingGemma profile but does not select it, download it, load inference, parse sessions, or create an index.

Run each setup step explicitly:

```bash
npm run --silent setup:recall -- status
npm run --silent setup:recall -- select-embeddinggemma --approve-download
npm run --silent setup:recall -- estimate
npm run --silent setup:recall -- estimate --measure --sample-sessions 3
npm run --silent setup:recall -- start --approve-build
```

Every successful command writes one JSON value to stdout. An agent-led setup skill can inspect that value, present it to the operator, and invoke the next command without parsing terminal prose.

## Consent boundaries

`status` reports:

- model purpose;
- immutable repository revision and artifact name;
- Gemma license name, URL, and distribution-review state;
- exact download size;
- target cache path;
- automatic device policy; and
- whether setup selected the profile.

`select-embeddinggemma` requires `--approve-download`. It verifies artifact size, SHA-256, and GGUF structure, then loads the configured provider canary and tokenizer. Setup writes the selected embedding configuration only after those checks succeed.

`estimate` reads file metadata only. It reports the number of physical session files and their total bytes without loading a tokenizer, model, zvec, or session contents.

`estimate --measure` is optional. It requires a verified embedding selection. The setup CLI applies the selected chunk and candidate policy used by the Pi extension, so the measured profile and cache identity remain compatible with the full build. The sample accepts one through ten physical session files and spans the corpus file-size distribution. It reports:

- model and tokenizer cold-start milliseconds;
- sampled sessions, source bytes, and dense documents;
- measured sample milliseconds;
- source-byte and dense-document throughput;
- embedding-cache hits, new embeddings, and embedding request count; and
- an estimated full-build duration range.

The estimate scales measured sample time by the ratio of total corpus bytes to sampled bytes, adds cold start, and reports 80% through 125% of that projection. This range is planning evidence, not a completion deadline.

The measured sample verifies the selected profile, creates its canonical manifest identity in memory, and warms the profile-bound embedding cache without creating generation state. The full build creates a replacement generation with the same profile identity and reuses those vectors. Unchanged sampled documents therefore become cache hits.

`start` requires a stored estimate and the separate `--approve-build` flag. It launches a detached worker and returns its status immediately. The worker creates a registry-owned replacement generation, validates and optimizes it, and activates it atomically. If the worker stops or crashes, `resume` reopens the same generation ID and durable index checkpoint.

Use `defer` to stop after configuration or estimation:

```bash
npm run --silent setup:recall -- defer
```

Deferral preserves the selected embedding configuration, estimate, and cached vectors. Recall remains unavailable until the first generation activates.

Estimate/build state lives at `~/.pi/agent/recall/first-index-setup.json`. Successful embedding selection also writes the authoritative capability record to `~/.pi/agent/recall/inference-configuration.json`; later [mixed inference setup](mixed-inference-configuration.md) may retain the profile while changing its backend or add and repair optional capabilities. Measured estimates, readiness checks, and first-generation launch reconstruct that authoritative selection. A verified HTTP embedding can complete the first-index flow without an embedded-only setup-state record, so changing EmbeddingGemma from embedded to HTTP cannot silently route inference back through the embedded provider. A pre-#43 first-index state without an inference-configuration record retains its documented embedded compatibility fallback. Existing background controls remain available through `/pi-session-recall-index --status`, `--stop`, `--resume`, and `--discard`.

## Deterministic verification

The committed tests use temporary session corpora, deterministic embedding providers, real temporary zvec stores, and the public `RecallConversationService` operations. They prove:

- metadata inspection performs no model, tokenizer, parsing, or store work;
- selection verification exercises the profile canary and tokenizer without indexing documents;
- sampling stays within its requested bound;
- sample vectors are reused by the full rebuild without creating generation state;
- configuration is absent before explicit selection and retained after deferral;
- build approval is separate from download approval and estimation;
- quality-evaluation state does not block measurement or an explicitly approved build;
- a stopped or crashed replacement resumes through the background service boundary; and
- post-selection measurement and build control use the authoritative mixed inference runtime rather than the initial embedded selection.

Run them with:

```bash
node --import tsx --test \
  src/recall-first-index-setup.test.ts \
  src/recall-first-index-setup-command.test.ts \
  src/recall-background-index-conversation-service.test.ts
```

## External evidence still pending

This implementation did not download or execute the 333,590,944-byte model. Gemma distribution and notice review remains pending, so the profile is not release-approved. A release also still needs real CPU and accelerated measurements for model cold start, warm inference, first-index throughput, duration-estimate accuracy, selected device identity, index size, and embedding-cache size. Run those checks against the pinned artifact and unchanged acceptance criteria; do not replace them with deterministic fixture results.

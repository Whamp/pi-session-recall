# Mixed inference configuration and repair

Conversation Recall persists independently verified embedding, reranking, and query-planning selections in `~/.pi/agent/recall/inference-configuration.json`. Embeddings are required. Reranking and query planning remain `null` until explicitly selected and can be removed without affecting the embedding generation.

The earlier `first-index-setup.json` remains the estimate/build workflow state. A successful `select-embeddinggemma --approve-download` now also writes the authoritative embedded embedding selection to `inference-configuration.json`. Existing installations that have only the earlier state continue through the pre-#43 runtime fallback.

## Operator status and doctor

The setup CLI exposes read-only mixed-configuration inspection:

```bash
npm run --silent setup:recall -- inference status
npm run --silent setup:recall -- inference doctor
```

`status` does not load inference. It reports, for every capability:

- whether the capability is required, configured, and healthy;
- model profile, backend, adapter, endpoint, and cache identity;
- local device policy, detected compute backend, and device names where applicable;
- immutable artifact path, repository revision, byte size, SHA-256, and current state;
- the last accepted conformance time and measurements; and
- one exact required repair, including an unavailable exact adapter rather than a substitute.

`doctor` additionally reruns the selected candidate's capability-specific conformance operation. It does not rewrite configuration or repair artifacts.

## Deterministic configuration command

`runRecallInferenceSetupCommand()` is the stable setup seam used by an agent-led setup flow. It emits one JSON value and supports:

```text
inference status
inference doctor
inference configure CAPABILITY CANDIDATE [--approve-artifact] [--approve-replacement]
inference repair CAPABILITY [--approve-artifact]
inference remove CAPABILITY
```

The command receives an explicit candidate catalog. The default CLI catalog contains embedded and llama.cpp HTTP EmbeddingGemma plus the recommended embedded/HTTP reranker and planner candidates. Project-specific custom adapters register through the same `RecallInferenceConfigurationCandidate` contract; there is no arbitrary URL/field mapper.

A candidate is accepted only after `verifyCapabilityConformance()` succeeds and returns the exact candidate profile, backend, adapter, and cache identity. Built-in candidate implementations call the public `RecallConversationService` verification operations. Custom candidates must call the relevant shared harness:

- `measureRecallEmbeddingProviderConformance()` with independent query/document vectors;
- `measureRecallRerankingProviderConformance()` with independent ordered scores; or
- `measureRecallQueryPlanningProviderConformance()` with an independent fixed typed plan.

Failed preparation, artifact validation, identity matching, conformance, or staging launch leaves the previous atomic configuration unchanged.

## Replacement and cache behavior

Changing only the embedding backend or adapter while retaining the same model profile updates configuration without calling an index-generation operation. The active vectors remain compatible.

Changing the embedding profile requires `--approve-replacement`. Setup calls the candidate's public generation service and starts a detached staging build, or resumes staging only when its embedding semantic identity is exact. A staging generation for another profile is never discarded or reused automatically. The active generation remains selected until the replacement validates and activates through the existing atomic-generation boundary.

Reranker and planner selections carry capability-specific cache identity. Replacing either changes only that capability's cache/search-policy identity. It neither opens zvec nor starts a vector rebuild. Removing either optional capability preserves embeddings and the other optional capability. An embedding-only runtime continues to serve hybrid recall and rejects `deep-rerank` with an explicit configuration error.

Repair requires the exact selected candidate. Artifact mutation requires separate approval. Repair reinspects the artifact, reruns conformance, and atomically refreshes only that capability's device/artifact/conformance record. Valid sibling capabilities and artifacts are retained.

## Runtime reconstruction

`createConfiguredRecallInferenceRuntime()` reconstructs exact recommended embedded or llama.cpp HTTP selections. It does not substitute a model, backend, adapter, endpoint, or device policy. The HTTP EmbeddingGemma backend retains the same profile identity and uses the checksum-pinned local GGUF tokenizer. Unknown candidate IDs and custom adapters without project runtime wiring fail explicitly.

Detached embedding replacement uses a candidate-specific named background service factory. The child therefore receives the selected embedding semantics even when the new configuration is committed only after launch succeeds.

## Deterministic evidence

The committed tests prove:

- embeddings alone make mixed configuration ready;
- all three capabilities can independently record embedded, llama.cpp HTTP, or conforming custom execution;
- failed custom conformance cannot replace an accepted selection;
- backend-only embedding movement starts no staging build;
- an approved embedding profile change starts staging and leaves the old selection on launch failure;
- optional profile/cache identity changes do not rebuild vectors;
- repair changes only the selected capability;
- status and doctor expose every configured identity, artifact, device, conformance result, and repair;
- embedding-only hybrid recall works while deep reranking fails clearly; and
- earlier first-index, background generation, Qwen reranker, and QMD planner behavior remains green.

Run the deterministic path with:

```bash
node --import tsx --test \
  src/recall-inference-configuration.test.ts \
  src/qwen-reranker-recall-conversation-service.test.ts \
  src/qmd-query-planner-recall-conversation-service.test.ts \
  src/recall-first-index-setup-command.test.ts \
  src/recall-background-index-conversation-service.test.ts
```

## External evidence still pending

No model download, legal approval, or real device run was available for #43. The default optional built-in candidates therefore fail closed when fixed conformance evidence is absent; their implementations do not derive expected output from the provider under test. Release acceptance still requires:

1. Gemma distribution and notice approval.
2. Independently accepted EmbeddingGemma query/document vectors for embedded and HTTP comparison.
3. Independently accepted Qwen reranker scores proving embedded logit recovery and HTTP score equivalence.
4. An independently accepted QMD planner typed plan for embedded and HTTP comparison.
5. Real CPU and accelerated cold/warm latency, fallback/device identity, throughput, index size, and cache-size measurements listed in the capability documents.
6. Project-specific runtime factory evidence for any custom adapter selected for production.

Supplying those unchanged fixtures to `createRecommendedOptionalInferenceCandidates()` activates the same deterministic setup path. The missing evidence must be produced against the pinned artifacts; the acceptance tolerances and model identities must not be weakened to make a run pass.

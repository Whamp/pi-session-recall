# Mixed inference configuration and repair

Conversation Recall persists independently verified embedding, reranking, and query-planning selections in `~/.pi/agent/recall/inference-configuration.json`. Embeddings are required. Reranking and query planning remain `null` until explicitly selected and can be removed without affecting the embedding generation.

The earlier `first-index-setup.json` remains the estimate/build workflow state. A successful `select-embeddinggemma --approve-download` also writes the authoritative embedded embedding selection to `inference-configuration.json`. Guided measurement and build use any verified embedding selection, including HTTP. On upgrade, an existing legacy index manifest creates `legacy-octen-installation.json`; only that durable marker preserves implicit Octen behavior. A fresh installation without inference state, setup state, or a legacy manifest refuses search and indexing until setup verifies an embedding.

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

The command receives an explicit candidate catalog. The default CLI catalog contains embedded and llama.cpp HTTP EmbeddingGemma plus the recommended embedded/HTTP reranker and planner candidates. Project-specific custom adapters register a candidate catalog and a `RecallInferenceAdapterRegistry` runtime factory; there is no arbitrary URL/field mapper.

A candidate is accepted only after `verifyCapabilityConformance()` succeeds and returns the exact candidate profile, backend, adapter, and cache identity. Embedding candidates also return the semantic `embeddingProfileId` used by index-generation manifests and selectors; this identity may differ from the adapter-specific cache identity. Built-in candidate implementations call the public `RecallConversationService` verification operations. Custom candidates must call the relevant shared harness:

- `measureRecallEmbeddingProviderConformance()` with independent query/document vectors;
- `measureRecallRerankingProviderConformance()` with independent ordered scores; or
- `measureRecallQueryPlanningProviderConformance()` with live grammar, bounds, protected-term, intent, timeout, and cache-identity checks. An independent fixed plan may add strict cross-adapter parity.

Failed preparation, artifact validation, identity matching, conformance, or staging launch leaves the previous atomic configuration unchanged.

## Replacement and cache behavior

Changing only the embedding backend or adapter while retaining the same model profile updates configuration without calling an index-generation operation. The active vectors remain compatible. If the first generation has not activated yet, guided measurement and launch also reconstruct this updated selection instead of reusing the initial embedded setup runtime.

Changing the embedding profile requires `--approve-replacement`. Setup validates the complete selection, persists it as a pending embedding replacement, then calls the candidate's public generation service. It starts a detached replacement or resumes one only when `embeddingProfileId` matches exactly. A replacement generation for another profile is never discarded or reused automatically. Configuration mutations serialize through launch and rollback, so one failed launch cannot erase a later replacement. A configuration-write failure starts no worker, and a launch failure restores the previous state.

The pending replacement remains inactive while staging builds, so configured runtimes continue to use the embedding selection that matches the active generation. Once the matching generation activates, configuration reads resolve the pending selection as active. Setup and runtime reconstruction both honor a configured nondefault active-generation selector path. This selector-based resolution is crash-safe on either side of activation. Version 1 configuration files migrate in memory; their embedding cache identity becomes the explicit generation identity because version 1 required those values to be equal.

The first-index `select-embeddinggemma` action can select a fresh installation or move execution for the same embedding profile. It refuses to replace a different profile or overwrite a pending replacement; use `inference configure ... --approve-replacement` for that transition.

Reranker and planner selections carry capability-specific cache identity. Replacing either changes only that capability's cache/search-policy identity. It neither opens zvec nor starts a vector rebuild. Removing either optional capability preserves embeddings and the other optional capability. An embedding-only runtime continues to serve hybrid recall and rejects `deep-rerank` with an explicit configuration error.

Repair requires the exact selected candidate. Artifact mutation requires separate approval. Repair reinspects the artifact, reruns conformance, and atomically refreshes only that capability's device/artifact/conformance record. Valid sibling capabilities and artifacts are retained.

## Runtime reconstruction

`createConfiguredRecallInferenceRuntime()` reconstructs exact recommended embedded or llama.cpp HTTP selections. It does not substitute a model, backend, adapter, endpoint, or device policy. The HTTP EmbeddingGemma backend retains the same profile identity and uses the checksum-pinned local GGUF tokenizer.

A custom integration registers its setup candidates and one `RecallInferenceAdapterRegistry`. Persisted candidate, profile, backend, adapter, and endpoint identities select that registry after restart. An unavailable registry fails explicitly. The registry creates the complete mixed runtime, so it also owns any custom detached-worker factory required for background indexing.

Detached built-in embedding replacement uses a candidate-specific named background service factory. The child therefore receives the pending embedding semantics without making that selection active before its generation activates.

## Deterministic evidence

The committed tests prove:

- embeddings alone make mixed configuration ready;
- all three capabilities can independently record embedded, llama.cpp HTTP, or conforming custom execution;
- failed custom conformance cannot replace an accepted selection;
- backend-only embedding movement starts no staging build;
- an approved embedding profile change durably records pending semantics before launch, keeps the old selection active while staging, and restores the prior state on launch failure;
- matching replacement work resumes by embedding semantic identity even when cache identity is richer;
- concurrent replacement attempts serialize through launch and rollback;
- first-index selection cannot bypass staging for a different embedding profile;
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

## Production optional-capability evidence

The built-in planner runs live grammar, bounds, protected-term, intent, timeout, and cache-identity checks during `configure` and `doctor`. The reranker also requires independently accepted ordered scores so it cannot certify its own score semantics.

Place accepted evidence at `~/.pi/agent/recall/inference-conformance.json`:

```json
{
  "reranking": {
    "query": "source provenance",
    "documents": ["Relevant document", "Irrelevant document"],
    "expectedScores": [0.9, 0.1],
    "maximumAbsoluteDifference": 0.000001
  },
  "queryPlanning": null
}
```

The CLI loads this file for `configure`, `repair`, and `doctor`. A missing reranking fixture fails closed. The optional `queryPlanning.expectedPlan` field adds exact output parity to the planner's live structural checks.

## External evidence still pending

No approved model download, legal approval, or real device run was available for #43. Release acceptance still requires:

1. Gemma distribution and notice approval.
2. Independently accepted EmbeddingGemma query/document vectors for embedded and HTTP comparison.
3. Independently accepted Qwen reranker scores proving embedded logit recovery and HTTP score equivalence.
4. An independently accepted QMD planner typed plan for embedded and HTTP comparison.
5. Real CPU and accelerated cold/warm latency, fallback/device identity, throughput, index size, and cache-size measurements listed in the capability documents.
6. Project-specific runtime factory evidence for any custom adapter selected for production.

Write the accepted fixtures unchanged to `inference-conformance.json`, or pass them directly to `createRecommendedOptionalInferenceCandidates()`. Produce missing evidence against the pinned artifacts; do not weaken tolerances or model identities to make a run pass.

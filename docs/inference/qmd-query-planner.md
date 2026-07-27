# Recommended QMD query planner

Conversation Recall can configure and verify QMD's pinned Qwen3 1.7B query-expansion model through embedded `node-llama-cpp` execution or the built-in llama.cpp HTTP adapter. This capability produces an ordered query plan for later search integration. It does not execute planned retrieval or ranking.

## Immutable profile

| Property            | Pinned value                                                       |
| ------------------- | ------------------------------------------------------------------ |
| Profile             | `qmd-query-expansion-1.7b-q4-k-m-v1`                               |
| Request model       | `qmd-query-expansion-1.7B-q4_k_m`                                  |
| Artifact repository | `tobil/qmd-query-expansion-1.7B-gguf`                              |
| Repository revision | `7816de0b72572c6c860ca1eddf97ba9e7fb8cc65`                         |
| GGUF artifact       | `qmd-query-expansion-1.7B-q4_k_m.gguf`                             |
| Byte size           | `1282438912`                                                       |
| SHA-256             | `000dfb1c06efa6a049e9f64ba921c3740e2454f62abab6fa10e77bd30bb2bcc0` |
| Prompt policy       | `qmd-query-expansion-no-think-v1`                                  |
| Grammar version     | `qmd-typed-query-plan-v1`                                          |
| License metadata    | MIT                                                                |

The download URL contains the immutable repository revision. Hugging Face's revision API and resolver headers report the same revision, filename, byte size, linked SHA-256, and MIT card metadata. That metadata check did not download or validate the complete GGUF.

The profile carries QMD 2.6.3's `/no_think` prompt, sampling values, 2,048-token context, 600-token output limit, and `lex:`, `vec:`, `hyde:` grammar. Post-generation validation requires one to three `lex` queries, one to three `vec` queries, at most one `hyde` query, unique nonblank single-line text, and preservation of a protected original term during conformance.

Recall intent is appended as a separate `Query intent:` line. It guides planning but is not itself a planned retrieval query.

## Operator artifact commands

Inspection and verification never download a model:

```bash
npm run --silent model:qmd-query-planner -- inspect
npm run --silent model:qmd-query-planner -- status
npm run --silent model:qmd-query-planner -- verify
npm run --silent model:qmd-query-planner -- doctor
```

Downloads, repairs, and removal require explicit consent:

```bash
npm run --silent model:qmd-query-planner -- download --approve
npm run --silent model:qmd-query-planner -- repair --approve
npm run --silent model:qmd-query-planner -- remove --approve
```

Without `--approve`, mutation fails before transport, directory creation, replacement, or removal. Downloads stage to a unique partial path and activate only after byte-size, SHA-256, and bounded GGUF validation. The default cache root is `~/.pi/agent/recall/models`; `PI_RECALL_MODEL_CACHE_DIRECTORY` overrides it.

The command manages only the pinned artifact. Mixed-capability persisted setup, status, and repair remain assigned to issue #43.

## Capability verification

Inject a `queryPlanningProfile` and identified `queryPlanner` into `createRecallConversationService`, then call `verifyQueryPlanningCapability()`. Verification invokes the profile's fixed query and recall intent through the shared conformance harness. It returns:

- profile and request-model identity;
- adapter backend and adapter ID;
- prompt policy and grammar version;
- request timeout;
- cache identity composed from profile, adapter, prompt, and grammar policy; and
- elapsed planning time plus `lex`, `vec`, and `hyde` counts.

The service rejects incomplete or profile-mismatched planner configuration. An unconfigured service remains valid for existing hybrid and deep-rerank behavior, but independent planner verification fails with a configuration error.

Planner profile and adapter identity are not written to the embedding manifest. Replacing either planner leaves the active vector generation compatible. Search still exposes only `hybrid` and `deep-rerank`; issue #29 owns query-planned retrieval, fallback, policy evidence, and ranking.

## Built-in execution adapters

### llama.cpp HTTP

`createQmdHttpQueryPlanningProvider` sends `POST /v1/chat/completions` with the exact request model, one user message, profile grammar, and profile sampling policy. It validates one assistant response, exact response-model identity, output grammar, typed bounds, and uniqueness. Requests time out after 60 seconds by default and honor caller cancellation.

This is a capability-specific llama.cpp contract, not a generic OpenAI-compatibility promise. A wrong served model, malformed response, invalid plan, timeout, or cancellation fails verification without switching models or adapters.

### Embedded node-llama-cpp

`createEmbeddedQmdQueryPlanningProvider` dynamically loads exactly `node-llama-cpp@3.18.1`, verifies the pinned artifact before native loading, shares an in-flight model load, creates a bounded generation context per request, and applies the same profile grammar and sampling policy as HTTP.

Automatic device selection probes supported accelerators. Accelerator initialization failure retries the same planner profile once on CPU and emits one warning. Explicit device selection fails closed. Native logs go to stderr. Request contexts are disposed after every plan; idle model and runtime resources are disposed after five minutes by default or by explicit `dispose()`.

## Deterministic conformance and measurement

Run the no-download path:

```bash
node --import tsx --test \
  src/qmd-query-planning-model-profile.test.ts \
  src/qmd-query-planning-provider-conformance.test.ts \
  src/embedded-qmd-query-planning-provider.test.ts \
  src/qmd-query-planner-recall-conversation-service.test.ts \
  src/manage-recall-qmd-query-planner-model.test.ts
```

The same `measureRecallQueryPlanningProviderConformance` harness exercises the built-in HTTP and embedded adapters. Fixed providers prove ordered output, typed bounds, optional HyDE, recall-intent prompt transport, protected terms, profile/adapter/policy/cache identity, timeout, cancellation, wrong-model rejection, grammar failure, and automatic same-profile CPU fallback. The service test builds one temporary real zvec generation, replaces planner profile and adapter, verifies both, and searches the unchanged generation in hybrid mode.

## External evidence still pending

No operator approved the 1,282,438,912-byte model download or a real CPU/accelerator run in this environment. The following acceptance evidence remains pending and is not represented as passing:

1. Download the exact artifact through `model:qmd-query-planner`, then record full-file byte-size, SHA-256, and GGUF validation. Current upstream header metadata matches these identities but is not full-file evidence.
2. Run one frozen query and recall-intent fixture through the real embedded model and a llama.cpp HTTP server loaded from the same artifact. Record both raw outputs and prove each passes the unchanged shared conformance bounds, grammar, and protected-term checks.
3. Measure cold model load, warm planning latency, generated-token throughput, selected device identity, CPU fallback, context/model storage, and idle reload on CPU and an available supported accelerator.
4. Complete maintainer review of the upstream MIT declaration and any bundled notices before release. Repository metadata is verified; legal approval is not claimed.

Real-model evidence must supplement rather than weaken the deterministic criteria.

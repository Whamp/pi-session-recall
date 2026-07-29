# Issue #28 integration acceptance ledger

This ledger maps every acceptance criterion in #28 and child issues #34–#44 to observable implementation, deterministic tests, or honest release evidence. Issue #27 is the governing specification; it has no section titled “Acceptance Criteria.” Its configurable-inference and index-generation requirements are represented by #28 and these child tickets. Query-planned retrieval remains assigned to #29 and is not claimed here.

Status meanings:

- **Pass — deterministic:** committed tests exercise the behavior without downloading a real model.
- **Implemented; live evidence pending:** the runtime/measurement path exists, but fixture evidence cannot establish real-model or real-device behavior.
- **Blocked — external:** release requires an approved artifact, independent fixture, device run, or maintainer distribution decision that is intentionally absent.

The consolidated external-evidence boundary is [`embedded-profile-acceptance.md`](embedded-profile-acceptance.md) and its machine-readable [`embedded-profile-acceptance.json`](embedded-profile-acceptance.json) ledger.

## #28 — Configurable inference and resumable index generations

| Acceptance criterion                                                                                           | Evidence                                                                                                                                                       | Status                                         |
| -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Existing Octen HTTP indexing and hybrid/deep-rerank behavior remains compatible                                | `recall-inference-conformance.test.ts:33`; `recall-index-manifest.test.ts:75`; full legacy service and quality suite                                           | Pass — deterministic                           |
| Fresh configuration selects/verifies embedded EmbeddingGemma and starts its first generation only with consent | `recall-first-index-setup-command.test.ts:57`; `recall-first-index-setup.test.ts:73,110,180`                                                                   | Pass — deterministic; real model build pending |
| Capabilities independently select embedded or HTTP execution                                                   | `recall-inference-configuration.test.ts:70`; built-in adapter conformance tests                                                                                | Pass — deterministic                           |
| Same verified embedding profile can switch backend without rebuild                                             | `recall-inference-configuration.test.ts:141`; `recall-first-index-setup-command.test.ts:57` proves later estimate/build uses the authoritative changed backend | Pass — deterministic                           |
| Embedding profile change requires a replacement and never mixes vectors                                        | profile-change tests in `recall-inference-configuration.test.ts`; profile identity on `RecallGenerationRegistryEntry`; detached resume integration tests       | Pass — deterministic                           |
| Reranker/planner changes alter affected cache/policy identity without rebuilding vectors                       | `qmd-query-planner-recall-conversation-service.test.ts:24`; `recall-inference-configuration.test.ts:70,141`                                                    | Pass — deterministic                           |
| Automatic GPU failure retries the same profile on CPU with one warning                                         | embedding, reranker, and planner embedded-provider fallback tests                                                                                              | Implemented; live device evidence pending      |
| No operation silently changes profile, backend, or adapter                                                     | exact-identity checks in `configured-recall-inference-runtime.ts`; `recall-inference-configuration.test.ts:395`; explicit-device failure tests                 | Pass — deterministic                           |
| Artifact revision, size, checksum, GGUF, and conformance mismatches fail closed                                | `recall-model-artifact-cache.test.ts:17–184`; capability conformance rejection tests                                                                           | Pass — deterministic; real artifacts pending   |
| Background build survives its invoking Pi session and reports progress                                         | `recall-background-index-conversation-service.test.ts:137,202`                                                                                                 | Pass — child process integration               |
| Stop/restart reuses completed work                                                                             | `recall-background-index-conversation-service.test.ts:261`                                                                                                     | Pass — child process integration               |
| Active generation remains searchable during replacement                                                        | `background rebuild reports progress while active recall remains searchable`                                                                                   | Pass — integration                             |
| Only a complete validated replacement generation activates                                                     | `rebuild-recall-generation.test.ts`; detached crash/resume matrix in `recall-background-index-conversation-service.test.ts`                                    | Pass — integration                             |
| Cold/warm inference, indexing throughput, model/device identity, and storage are measured                      | `evaluate:embeddinggemma` records the required fields; no approved model/device run exists                                                                     | Blocked — external live measurements           |
| Format, type, lint, and full test suites pass                                                                  | Run the exact gates listed under “Integration gates” below                                                                                                     | Verified at integration HEAD                   |

## #34 — Preserve Octen behind model profiles and provider contracts

| Acceptance criterion                                                                        | Evidence                                                                                                  | Status                            |
| ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | --------------------------------- |
| Distinct query/document embedding operations preserve Octen inputs and vectors              | `recall-inference-conformance.test.ts:33`; `recall-conversation-service.test.ts:3021`                     | Pass — deterministic              |
| Reranking uses ordered finite capability-specific scores                                    | `recall-inference-conformance.test.ts:181,392,447`                                                        | Pass — deterministic              |
| Profile identity excludes backend URL, device, and adapter                                  | `recall-model-profiles.ts`; ADR 0004; backend-switch test at `recall-inference-configuration.test.ts:141` | Pass — observable + deterministic |
| Existing Octen configuration/manifest needs no rebuild                                      | `recall-index-manifest.test.ts:75`; ledger records superseded Octen paths after deterministic-fixture regeneration | Pass — deterministic; Octen artifact attestation superseded |
| Built-in HTTP embedding/reranking adapters pass shared conformance                          | `recall-inference-conformance.test.ts:33,181`                                                             | Pass — deterministic              |
| Existing indexing, hybrid, deep-rerank, manifest, cache, and quality behavior remains green | Full `npm test` plus unchanged Octen quality evidence identity                                            | Pass — deterministic              |

## #35 — Download and verify pinned EmbeddingGemma

| Acceptance criterion                                                                                            | Evidence                                                                           | Status                        |
| --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------- |
| Profile pins revision, bytes, SHA-256, GGUF, dimensions, prompts, pooling, normalization, tokenizer, and canary | `embeddinggemma-model-profile.test.ts`; `recall-model-profiles.ts`                 | Pass — deterministic identity |
| Download requires explicit approval                                                                             | `recall-model-artifact-cache.test.ts:17`; model command tests                      | Pass — deterministic          |
| Immutable download activates only after size/checksum/GGUF validation                                           | `recall-model-artifact-cache.test.ts:47,138`; `validateRecallGgufModelArtifact.ts` | Pass — deterministic fixtures |
| Partial/corrupt downloads never replace valid cached artifacts                                                  | `recall-model-artifact-cache.test.ts:47,184`                                       | Pass — deterministic          |
| Status/doctor distinguishes missing, partial, corrupt, valid, incompatible and gives repair                     | `recall-model-artifact-cache.test.ts:81`; model command tests                      | Pass — deterministic          |
| Tests use local fixtures/mocked transport and never download the real model                                     | `recall-model-artifact.test-utils.ts`; artifact tests                              | Pass — deterministic          |

## #36 — Build and search with embedded EmbeddingGemma

| Acceptance criterion                                                                           | Evidence                                                                                     | Status                                                                                            |
| ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Native execution loads dynamically and HTTP search does not require it                         | dynamic imports in embedded providers; `docs/inference/embedded-embeddinggemma.md`           | Pass for search; EmbeddingGemma indexing intentionally uses the local GGUF tokenizer per ADR 0004 |
| Query/document prompts are asymmetric and document title is `none`                             | `recall-inference-conformance.test.ts:115`; `embedded-embeddinggemma-provider.test.ts:112`   | Pass — deterministic                                                                              |
| Tokenizer matches runtime on frozen prose, code, emoji, and multilingual inputs                | `embedded-embeddinggemma-provider.test.ts:112` uses an injected native boundary              | Implemented; independent real-runtime parity pending                                              |
| Manifest records and validates canary, dimensions, pooling, normalization, tokenizer, artifact | `recall-index-manifest.test.ts:75`; `embeddinggemma-recall-conversation-service.test.ts:350` | Pass — deterministic                                                                              |
| Lexical-only tool evidence is never embedded                                                   | legacy incremental test plus EmbeddingGemma acceptance runner                                | Pass — deterministic; real corpus run pending                                                     |
| Same profile moves between embedded and HTTP without rebuild                                   | `embeddinggemma-recall-conversation-service.test.ts:68`; backend-switch configuration test   | Pass — deterministic                                                                              |
| EmbeddingGemma/Octen changes require different generations                                     | manifest/profile identity checks; profile-change configuration and staging tests             | Pass — deterministic                                                                              |
| Service builds/searches temporary 768-dimensional profile generation while Octen remains green | `embeddinggemma-recall-conversation-service.test.ts:68`; full suite                          | Pass — deterministic native fixture; live quality pending                                         |

## #37 — Accelerated embedded inference with safe CPU fallback

| Acceptance criterion                                                  | Evidence                                                                                                        | Status                                                                  |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Automatic probing reports selected backend/device                     | embedded embedding/reranker/planner automatic-selection tests                                                   | Implemented; real devices pending                                       |
| GPU initialization failure retries same profile on CPU and warns once | fallback tests in all three embedded provider suites                                                            | Implemented; real device failure pending                                |
| Explicit device and bounded parallelism overrides exist               | provider option types; `embedded-embeddinggemma-provider.test.ts:12,99,339`                                     | Pass — deterministic                                                    |
| Context pools use conservative hard caps                              | `EMBEDDED_INFERENCE_MAX_PARALLELISM`; bounded-parallelism tests                                                 | Pass — deterministic                                                    |
| Concurrent requests reuse provider runtime/model-load promise         | `embedded-embeddinggemma-provider.test.ts:264`; equivalent single-flight implementations for optional providers | Implemented per provider; global cross-capability sharing is not proven |
| Native logs cannot corrupt stdout                                     | every native logger writes to stderr; child workers ignore stdio                                                | Pass — observable implementation; real native log run pending           |
| Idle contexts/models dispose according to lifecycle                   | `embedded-embeddinggemma-provider.test.ts:374,444`; optional-provider disposal implementations                  | Pass — deterministic                                                    |
| Failure never changes model profile or HTTP backend                   | explicit-device and exact-adapter failures; configured runtime fail-closed test                                 | Pass — deterministic                                                    |

## #38 — Deep-rerank with recommended Qwen

| Acceptance criterion                                                                   | Evidence                                                                                          | Status                                                |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Artifact requires consent and immutable validation                                     | shared artifact cache plus Qwen model-command tests                                               | Pass — deterministic fixtures                         |
| Embedded/HTTP adapters share ordering, range, timeout, cancellation, cache conformance | `recall-inference-conformance.test.ts:181,251`; `embedded-qwen-reranking-provider.test.ts:13,213` | Pass — deterministic                                  |
| Fixture semantics reject known double sigmoid                                          | `recall-inference-conformance.test.ts:418`; `embedded-qwen-reranking-provider.test.ts:13,273`     | Pass — deterministic; independent live parity pending |
| Deep-rerank works through both built-in adapters                                       | `qwen-reranker-recall-conversation-service.test.ts:78`                                            | Pass — deterministic                                  |
| Profile/adapter changes update policy/cache without vector rebuild                     | planner/reranker identity tests and formatted policy evidence                                     | Pass — deterministic                                  |
| Reranker failure is actionable and never changes backend/mode                          | legacy service failure test; embedded/HTTP rejection tests                                        | Pass — deterministic                                  |

## #39 — Configure and verify recommended query planner

| Acceptance criterion                                                                 | Evidence                                                                                               | Status                                                          |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| Artifact requires consent and immutable validation                                   | shared artifact cache plus planner model-command tests                                                 | Pass — deterministic fixtures                                   |
| Embedded/HTTP adapters use one shared conformance contract                           | `qmd-query-planning-provider-conformance.test.ts:30`; `embedded-qmd-query-planning-provider.test.ts:8` | Pass — deterministic                                            |
| Probe enforces bounded `lex`/`vec`/optional `hyde` grammar and intent                | same tests plus `recall-query-planning-policy.ts`                                                      | Pass — deterministic                                            |
| Profile, adapter, prompt, grammar, timeout, cancellation, cache identity inspectable | query-planner conformance and provider tests                                                           | Pass — deterministic                                            |
| Planner changes do not rebuild vectors                                               | `qmd-query-planner-recall-conversation-service.test.ts:24`                                             | Pass — deterministic                                            |
| Setup verifies planner independently                                                 | `recall-inference-configuration.test.ts:70`; optional candidate boundaries                             | Pass — deterministic when independent expected plan is supplied |
| No query-planned retrieval/ranking is added                                          | extension schema still exposes exactly `hybrid` and `deep-rerank`; #29 remains separate                | Pass — observable contract                                      |

## #40 — Build and atomically activate staging generation

| Acceptance criterion                                                          | Evidence                                                                           | Status                            |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------- |
| Generation owns evidence, projections, state, manifest, and embedding profile | `recall-generation-state.ts`; `rebuild-recall-generation.ts`; ADR 0005             | Pass — observable + deterministic |
| Replacement builds while search reads the active pointer                      | detached active-search integration test                                            | Pass — integration                |
| Retained lifecycle markers replay after activation                            | generation cutover and replay tests                                                | Pass — integration                |
| Manifest, canary, document count, and projection coverage validate first      | rebuild validation path in `recall-conversation-service.ts`; rebuild failure tests | Pass — integration                |
| Store optimization precedes one atomic pointer switch                         | `rebuild-recall-generation.test.ts`                                                | Pass — integration                |
| Stop, crash, or failure preserves active search and resumable replacement     | detached stop/resume and crash-phase matrix                                        | Pass — integration                |
| Explicit discard removes only an unowned non-active registry entry            | service discard implementation and index-command control tests                     | Pass — integration                |
| Interactive Pi lifecycle/search does not start whole-session maintenance      | `recall-extension.test.ts`; README explicit-maintenance contract                   | Pass — extension contract         |

## #41 — Resumable detached background processes

| Acceptance criterion                                                                     | Evidence                                                           | Status                           |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------- |
| Detached process survives client and owns staging lock                                   | `recall-background-index-conversation-service.test.ts:202`         | Pass — child process integration |
| Bounded status includes generation/process/progress/checkpoint/error                     | status schema and tests at lines 137,261,351                       | Pass — integration               |
| Status/stop/resume/discard exists without daemon framework                               | `recall-index-command.test.ts:163,205`; one detached worker module | Pass — deterministic             |
| Stop preserves session state and profile cache                                           | `recall-background-index-conversation-service.test.ts:261`         | Pass — child process integration |
| Resume skips completed work, reuses cache, and tolerates repeated upsert                 | tests at lines 261,391                                             | Pass — child process integration |
| Stale/crashed worker records are detected accurately                                     | tests at lines 351,391                                             | Pass — child process integration |
| Parsing, embedding, store-write, and optimization crashes resume; cutover faults recover | detached crash matrix plus `rebuild-recall-generation.test.ts`     | Pass — integration               |
| Active generation remains searchable throughout                                          | test at line 137                                                   | Pass — child process integration |

## #42 — Guided first-index estimate and launch

| Acceptance criterion                                                                               | Evidence                                                        | Status                                      |
| -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------- |
| Fresh status is unconfigured and recommendation is not auto-selected/downloaded                    | `recall-first-index-setup-command.test.ts:57`                   | Pass — deterministic                        |
| Purpose/source/license/size/cache/device shown before approval                                     | same command test and status JSON                               | Pass — deterministic                        |
| Metadata estimate counts sessions/bytes without model work                                         | `recall-first-index-setup.test.ts:73`                           | Pass — deterministic                        |
| Optional bounded sample reports cold start, throughput, duration range                             | `recall-first-index-setup.test.ts:180`                          | Pass — deterministic; live accuracy pending |
| Sample embeddings persist in the profile cache without generation state and full build reuses them | `recall-first-index-setup.test.ts`                              | Pass — integration                          |
| Full build requires separate confirmation after estimate                                           | `recall-first-index-setup-command.test.ts:57`                   | Pass — deterministic                        |
| Approval launches detached worker and returns immediately                                          | command test plus background client-survival test               | Pass — integration                          |
| Deferral retains config and reports recall not ready                                               | command test                                                    | Pass — deterministic                        |
| Small deterministic agent-drivable interface exists                                                | `setup:recall` JSON CLI and `runRecallFirstIndexSetupCommand()` | Pass — observable contract                  |

## #43 — Add and repair mixed inference configurations

| Acceptance criterion                                                           | Evidence                                                                                                         | Status                                                                             |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Embeddings required; reranking/planning optional                               | `recall-inference-configuration.test.ts:70`                                                                      | Pass — deterministic                                                               |
| Capability backends select independently                                       | same test                                                                                                        | Pass — deterministic                                                               |
| Custom candidate must return matching capability conformance before acceptance | `recall-inference-configuration.test.ts:243`; shared harnesses exported for custom candidates                    | Pass — deterministic boundary; production runtime factory remains adapter-specific |
| Same embedding profile can change backend without rebuild                      | test at line 141 plus first-index configured-runtime regression at `recall-first-index-setup-command.test.ts:57` | Pass — deterministic                                                               |
| Embedding profile change starts staging without vector mixing                  | test at line 141 and generation profile checks                                                                   | Pass — deterministic                                                               |
| Optional profile changes affect only cache/policy identity                     | tests at lines 70,141                                                                                            | Pass — deterministic                                                               |
| Repair preserves valid siblings/artifacts                                      | test at line 243                                                                                                 | Pass — deterministic                                                               |
| Status/doctor explains selections, health, conformance, and repair             | tests at lines 243,352                                                                                           | Pass — deterministic                                                               |
| No setup/runtime substitution                                                  | test at line 395; configured-runtime exact checks; new first-index regression                                    | Pass — deterministic                                                               |

## #44 — Publish embedded-profile acceptance evidence

| Acceptance criterion                                                                                          | Evidence                                                                     | Status                                             |
| ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------- |
| Octen quality result unchanged and profile-scoped                                                             | hashes in `embedded-profile-acceptance.json`                                 | Superseded — committed artifacts regenerated with deterministic-fixture-v1 |
| EmbeddingGemma passes committed regression corpus                                                             | runner exists; no real artifact was approved                                 | Blocked — external live quality run                |
| Embedded Qwen conformance/deep-rerank has verified live score semantics                                       | deterministic fixtures pass; independent live parity absent                  | Blocked — external independent scores/model run    |
| Planner capability conformance passes; retrieval quality stays in #29                                         | deterministic embedded/HTTP conformance passes                               | Pass — deterministic; live parity pending          |
| Evidence records profile/backend/adapter/device/dimensions/prompts/tokenizer/canary/policy/cache              | ledger records known values and explicitly marks live fields pending         | Implemented; pending fields are honest blockers    |
| Cold/warm, throughput, storage, CPU, accelerator measured                                                     | `evaluate:embeddinggemma` writer exists; no authorized live run              | Blocked — external measurements                    |
| Background interruption, resumption, active availability, marker replay, validation, and activation exercised | background and generation integration suites                                 | Pass — integration                                 |
| Gemma terms/notices documented without claiming legal approval                                                | `embedded-profile-distribution-review.md`                                    | Pass as review packet; maintainer approval pending |
| Full gates and applicable slop scan pass                                                                      | exact commands below; final result recorded in the integration report/commit | Verified at integration HEAD                       |

## Integration gates

Run from the consolidated foundation worktree:

```bash
npm test
npm run typecheck
npm run lint
npm run format:check
git diff --check origin/master...HEAD
npm run evidence:embedded:deterministic
slop-scan delta --base <clean-dc8c0dc-worktree> --head "$PWD" --fail-on added,worsened
```

The deterministic gates do not override the external release blockers above.

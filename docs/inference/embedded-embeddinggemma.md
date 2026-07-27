# Embedded EmbeddingGemma execution

Conversation Recall can build and search a 768-dimensional index with the pinned `embeddinggemma-300m-q8-0-v1` profile through CPU, Metal, CUDA, or Vulkan execution. The same profile also works through the built-in llama.cpp HTTP adapter. The profile controls prompts, dimensions, pooling, normalization, tokenizer identity, artifact identity, and canary behavior. Device selection does not change vector compatibility.

## Runtime behavior

`createEmbeddedEmbeddingGemmaProvider` loads `node-llama-cpp@3.18.1` only when embedding or tokenization first needs it. HTTP-only search does not import the native runtime. The package is optional, so an unavailable native package does not break installation or the HTTP adapter.

The embedded provider:

- verifies the approved model-cache artifact before importing the runtime;
- defaults to `device: 'auto'`, probes supported compute backends, and selects the first reported accelerator;
- uses CPU when no supported accelerator is reported;
- retries the same model profile once on CPU when automatic accelerator runtime, model, or context initialization fails;
- emits one fallback warning through `onWarning`, or stderr when no callback is supplied;
- treats explicit `cpu`, `metal`, `cuda`, or `vulkan` overrides as exact and never falls back from them;
- records the policy, probed backends, selected compute backend, device names, fallback source, and parallelism in `executionIdentity` after initialization;
- applies `task: search result | query: ` to submitted queries;
- applies `title: none | text: ` to index documents;
- returns 768 finite, L2-normalized dimensions in input order;
- tokenizes chunk input through the tokenizer inside the checksum-pinned GGUF, without special tokens;
- shares one in-flight model load across concurrent requests;
- uses one context by default and accepts an explicit parallelism from 1 through 4;
- caps GPU offload at 32 layers and asks node-llama-cpp to fit the fixed 2,048-token embedding context;
- never sizes the pool from aggregate reported GPU memory;
- disables debug and progress logs and routes error-level native logs to stderr;
- disposes idle contexts, model, and runtime after five minutes by default, then reloads them on demand;
- accepts `idleTimeoutMilliseconds: 0` to disable idle disposal; and
- keeps resources loaded while the synchronous conversation tokenizer is checked out, so `RecallConversationService` never retains a tokenizer backed by a disposed model.

The service runs the profile canary twice through the query operation. The vectors must have cosine similarity of at least `0.9995`. It stores the canonical vector and complete profile identity in the index manifest. Search repeats the canary check before opening zvec.

## Service wiring

The [guided first-index setup](first-index-guided-setup.md) selects embedded execution through the public service dependencies below. Later mixed-capability setup remains separate:

```typescript
import {
  createEmbeddedEmbeddingGemmaProvider,
  createEmbeddingGemmaTokenizerManifestIdentity,
} from './src/embedded-embeddinggemma-provider.js';
import { RecallSearchScope } from './src/enums.js';
import { createRecallConversationService } from './src/recall-conversation-service.js';
import { createRecommendedEmbeddingGemmaModelProfile } from './src/recall-model-profiles.js';

const profile = createRecommendedEmbeddingGemmaModelProfile();
const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
  modelCacheDirectory: '/home/you/.pi/agent/recall/models',
  device: 'auto',
  parallelism: 1,
  onWarning: (warning) => console.error(warning),
});

const service = createRecallConversationService(config, {
  embeddingProfile: profile,
  embeddingProvider: provider,
  tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
  loadTokenizer: () => provider.loadConversationTokenizer(),
});

try {
  await service.index();
  console.error(provider.executionIdentity);
  const result = await service.search('source-backed decision', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
} finally {
  await provider.dispose();
}
```

Use an explicit device to troubleshoot one backend:

```typescript
const provider = createEmbeddedEmbeddingGemmaProvider(profile, {
  modelCacheDirectory,
  device: 'cuda',
  parallelism: 2,
});
```

A failed explicit override fails the operation. It does not switch to CPU, another accelerator, another model profile, or HTTP.

Download approval remains separate:

```bash
npm run --silent model:embeddinggemma -- download --approve
npm run --silent model:embeddinggemma -- verify
```

The provider rejects missing, partial, corrupt, or incompatible cache state. It never downloads or repairs a model.

## HTTP execution with the same profile

Use `createLlamaCppHttpEmbeddingProvider(profile, backend)` with the same profile and tokenizer manifest identity. Backend URL and adapter location do not enter index compatibility. An existing EmbeddingGemma generation therefore stays searchable when execution moves between embedded and HTTP adapters.

Fresh HTTP indexing still needs an exact local chunk tokenizer. The current built-in EmbeddingGemma tokenizer comes from the pinned GGUF through `node-llama-cpp`; the gated upstream standalone tokenizer assets have not received legal approval or independent checksum acceptance. Callers may inject a separately accepted exact tokenizer through `loadTokenizer`.

Changing to Octen changes dimensions, prompts, pooling, tokenizer, and artifact identity. The service rejects the existing generation and requires an explicit rebuild.

## Deterministic conformance and measurement

Run the committed path without a model download:

```bash
node --import tsx --test \
  src/embedded-embeddinggemma-provider.test.ts \
  src/embeddinggemma-recall-conversation-service.test.ts \
  src/recall-inference-conformance.test.ts
```

The tests use an injected node-llama-cpp boundary, a temporary HTTP server, and a real temporary zvec store. They prove:

- lazy dynamic import and artifact-verification order;
- automatic accelerator probing and selected device reporting;
- explicit CPU execution and exact accelerator override failure;
- one-warning CPU fallback through `RecallConversationService.index()`;
- unchanged model profile identity across fallback;
- bounded context options and a maximum pool size of four;
- one shared runtime/model load and two-way concurrency with a two-context fixture;
- native logger isolation from stdout;
- idle disposal, on-demand reload, and tokenizer lifetime safety;
- exact asymmetric prompts;
- 768-dimensional finite L2 vectors and document order;
- fixed token IDs for prose, code, emoji, and multilingual text;
- query/document conformance timing through the shared harness;
- query-canary repeatability rejection;
- index-manifest and embedding-cache identity;
- embedded build and search, then HTTP search without a rebuild;
- lexical-only tool evidence exclusion from document embedding; and
- EmbeddingGemma-to-Octen incompatibility.

These are deterministic adapter and service measurements. They do not stand in for real-model or real-device acceptance.

After distribution approval and an explicit pinned-artifact download, run the unchanged quality corpus separately for CPU and one supported accelerator:

```bash
npm run evaluate:embeddinggemma -- --device cpu
npm run evaluate:embeddinggemma -- --device cuda # or metal/vulkan
```

The command never downloads a model, never reads production sessions, and never rewrites the Octen report. It records the complete model, prompt, tokenizer, canary, backend, adapter, selected device, candidate-policy, cache, quality, timing, throughput, and storage evidence in `docs/evaluation/embeddinggemma-quality-<device>.json`. See the [embedded profile acceptance ledger](../evaluation/embedded-profile-acceptance.md).

## External evidence still pending

This environment has no legal approval to accept the Gemma terms and no operator-approved 333,590,944-byte model download. No real CPU or accelerator model run was performed. The following evidence remains pending:

- Gemma distribution and notice review;
- verification of the real artifact's published byte size, SHA-256, and GGUF structure;
- frozen real-runtime token IDs for the committed prose, code, emoji, and multilingual inputs, compared with an independently accepted tokenizer;
- real repeated-query canary vectors;
- CPU model cold start, warm query and document inference latency, indexing throughput, index size, embedding-cache size, and device identity;
- Metal, CUDA, and Vulkan availability and selected device names on supported hardware;
- same-profile automatic GPU-to-CPU fallback against a real backend initialization failure;
- accelerated cold start, warm inference latency, throughput, storage, and stable parallelism by device class; and
- native runtime log capture under a real accelerated load.

Do not mark the profile release-ready until those checks run against the pinned artifact without changing the acceptance thresholds.

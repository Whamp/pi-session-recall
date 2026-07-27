# Embedded EmbeddingGemma execution

Conversation Recall can build and search a 768-dimensional index with the pinned `embeddinggemma-300m-q8-0-v1` profile on the CPU. The same profile also works through the built-in llama.cpp HTTP adapter. The profile, not the backend, controls prompts, dimensions, pooling, normalization, tokenizer identity, artifact identity, and canary behavior.

## Runtime behavior

`createEmbeddedEmbeddingGemmaProvider` loads `node-llama-cpp@3.18.1` only when embedding or tokenization first needs it. HTTP-only search does not import the native runtime. The package is an optional dependency, so an unavailable native package does not break installation or the HTTP adapter.

The embedded provider:

- verifies the approved model-cache artifact before importing the runtime;
- requests a CPU runtime with `gpu: false` and loads the model with `gpuLayers: 0`;
- applies `task: search result | query: ` to submitted queries;
- applies `title: none | text: ` to index documents;
- returns 768 finite, L2-normalized dimensions in input order;
- tokenizes chunk input through the tokenizer inside the checksum-pinned GGUF, without special tokens;
- shares one in-flight model load and serializes one embedding context;
- disables progress logs and requests error-only native logging;
- disposes the context, model, and runtime in dependency order.

The service runs the profile canary twice through the query operation. The vectors must have cosine similarity of at least `0.9995`. It stores the canonical vector and complete profile identity in the index manifest. Search repeats the canary check before opening zvec.

## Service wiring

The guided configuration flow tracked by #28 is not implemented yet. Current callers select embedded execution through the public service dependencies:

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
});

const service = createRecallConversationService(config, {
  embeddingProfile: profile,
  embeddingProvider: provider,
  tokenizerIdentity: createEmbeddingGemmaTokenizerManifestIdentity(profile),
  loadTokenizer: () => provider.loadConversationTokenizer(),
});

try {
  await service.index();
  const result = await service.search('source-backed decision', 5, {
    scope: RecallSearchScope.GLOBAL,
  });
} finally {
  await provider.dispose();
}
```

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
- CPU runtime and model-load options;
- exact asymmetric prompts;
- 768-dimensional finite L2 vectors and document order;
- fixed token IDs for prose, code, emoji, and multilingual text;
- query/document conformance timing through the shared harness;
- query-canary repeatability rejection;
- index-manifest and embedding-cache identity;
- embedded build and search, then HTTP search without a rebuild;
- lexical-only tool evidence exclusion from document embedding;
- EmbeddingGemma-to-Octen incompatibility.

These are deterministic adapter and service measurements. They do not stand in for real-model acceptance.

## External evidence still pending

This environment has no legal approval to accept the Gemma terms and no operator-approved 333,590,944-byte model download. The following evidence remains pending:

- Gemma distribution and notice review;
- verification of the real artifact's published byte size, SHA-256, and GGUF structure;
- frozen real-runtime token IDs for the committed prose, code, emoji, and multilingual inputs, compared with an independently accepted tokenizer;
- real repeated-query canary vectors;
- CPU model cold start, warm query and document inference latency, indexing throughput, index size, embedding-cache size, and device identity;
- accelerated-device evidence, which is outside this CPU ticket.

Do not mark the profile release-ready until those checks run against the pinned artifact without changing the acceptance thresholds.

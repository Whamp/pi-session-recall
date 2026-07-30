---
status: accepted
---

# Keep embedding profile semantics independent of execution

Conversation Recall adopts QMD 2.6.3 revision `e428df76bc0274d9e93eb7ca3e95673315c42e90` as the baseline for EmbeddingGemma prompts and local node-llama-cpp execution. Conversation Recall retains its own session import, document kinds, lexical-only tool evidence, project admission, provenance, rank fusion, and zvec storage.

One embedding profile owns query and document prompts, native and stored dimensions, pooling, normalization, tokenizer identity, artifact repository and revision, artifact SHA-256, and canary policy. Embedded and llama.cpp HTTP providers execute that profile without adding backend URL, device, or adapter location to vector compatibility.

`RecallConversationService` treats an injected profile as authoritative for manifest identity and dense-store dimensions. The legacy configuration fields continue to create the existing Octen profile when no profile is injected. EmbeddingGemma extends manifest version 5 with optional fields; the Octen projection remains unchanged, so existing Octen generations remain readable.

EmbeddingGemma chunking uses the tokenizer inside the checksum-pinned GGUF. The embedded provider loads node-llama-cpp dynamically, verifies the model cache first, and L2-normalizes every vector. Automatic execution probes supported accelerators, reports the selected compute backend and device, and retries the same profile on CPU with one warning when accelerator initialization fails. Explicit CPU, Metal, CUDA, and Vulkan overrides fail closed. The provider bounds context parallelism independently of reported aggregate GPU memory, routes native logs away from stdout, and disposes idle resources without invalidating a checked-out synchronous tokenizer. The service rejects non-repeatable query canaries before it writes a manifest or opens a new store.

This boundary lets an EmbeddingGemma generation move between conforming embedded and HTTP execution without rebuilding vectors. Moving to Octen changes semantic identity and requires another generation. A fresh HTTP-only build still needs an independently accepted exact tokenizer; the gated standalone Gemma tokenizer assets are not accepted by this decision.

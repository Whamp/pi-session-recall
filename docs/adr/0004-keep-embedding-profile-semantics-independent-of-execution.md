---
status: superseded by ADR-0005
---

# Keep embedding profile semantics independent of execution

This decision previously supported several embedding and reranking profiles across HTTP and embedded execution. ADR-0005 replaced that model-management architecture with one direct Octen HTTP profile, vendor-supported prefix storage, and inner-product Zvec search.

ADR-0015 later restores only the semantic principle for exactly two product profiles: one certified local Octen runtime and one direct Octen HTTP profile. It does not restore the generic provider registry, reranking, planning, cache, or worker architecture described by this superseded decision.

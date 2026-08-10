---
status: superseded
---

# Search dense conversations and compact Invocations together

> Superseded by ADR-0014. This document preserves the historical version 7 flat-Zvec-plus-SQLite decision.

Normal recall searches both parts of the Active recall database automatically. Flat Zvec search returns Dense recall documents. SQLite full-text search returns compact Invocation records. The service combines both candidate lists before it applies the caller's result limit; callers cannot select one fast store.

When both kinds match and the limit permits, mixed-result policy version 1 reserves at least one slot for each kind. It preserves up to four of the first five strong conversation results, then fills unused conversation capacity with Invocation results. Dense conversation ranking keeps duplicate suppression, active-branch preference, and Neighbor context. Invocation results retain SQLite rank and exact Source locators.

Complete tool results, bash output, and omitted payload arguments remain Source-backed evidence. Only explicit Source search reads them. Normal recall never scans canonical session JSONL.

A version 7 Index manifest requires the 1,024-dimension FP32 FLAT dense-store identity. A rebuild creates the dense store and Recall catalog in one Candidate recall database and activates them through the existing atomic generation pointer. Old manifests cannot serve the compact layout and require `psr index --rebuild`.

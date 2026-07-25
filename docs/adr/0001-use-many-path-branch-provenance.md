# Use many-path branch provenance

Pi session entries have parent links but no intrinsic branch IDs, and a shared ancestor can appear on several descendant paths. Recall documents therefore record every endpoint path that contains their source entry plus a separate active-branch flag; they never invent a singular branch ID. Path-specific document copies would duplicate shared evidence and make identity and deduplication depend on retrieval policy.

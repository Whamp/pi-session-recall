---
status: accepted
---

# Import historical Pi sessions through one canonical graph boundary

Conversation Recall frames physical session records only at byte `0x0A`. It removes one `0x0D` immediately before that byte and treats every other UTF-8 character, including U+2028 and U+2029, as JSON content. An unterminated final record still enters JSON parsing and cannot disappear silently.

The import boundary parses each physical record once, selects one exact format, and produces canonical logical sessions in memory. Every logical session then enters the same strict session graph parser used by current sessions. Converters do not build graphs, weaken validation, or write source files.

The boundary supports three disjoint formats:

1. Canonical JSONL has one complete leading session header at supported version 2 or 3. The strict graph parser validates its header count, entry IDs, parent links, cycles, effective leaf, branch and compaction relationships, and tool links.
2. Unversioned Pi v1 has one complete leading session header with no `version` field. Every later record is a supported v1 message, model change, thinking-level change, or compaction with the metadata Pi v1 wrote and with neither `id` nor `parentId`. Conversion assigns each entry a deterministic SHA-256 ID from the conversion-policy version, session ID, physical line, and unchanged source record. It chains entries in physical order, sets header version 2, and converts `firstKeptEntryIndex` to the corresponding deterministic `firstKeptEntryId` exactly as Pi's v1-to-v2 migration did.
3. Pi session-file reuse history has at least two complete version 2 or 3 session headers. The first nonblank record is a header, and each header begins a nonempty segment. Each segment becomes one independently validated logical session and retains its own ID, cwd, timestamp, parent-session metadata, physical path, and physical source lines. Import is all-or-nothing per physical file: if one logical session fails graph validation, no sibling segment emits searchable documents.

Pi's historical `/new` bug reused the active physical file. Upstream commit [`234f367d0ddbfb6e958e986bf962e02c21d06f4c`](https://github.com/earendil-works/pi/commit/234f367d0ddbfb6e958e986bf962e02c21d06f4c) changed `newSession()` to assign a fresh filename; [PR #649](https://github.com/earendil-works/pi/pull/649) records the reproduction and fix.

Detection never falls back to a converter because canonical graph parsing failed. Unsupported, ambiguous, malformed, truncated, cyclic, duplicate-ID, invalid-leaf, and missing-parent streams fail without documents. A changed physical file that becomes invalid also loses its previously indexed documents.

The index manifest records the import-policy version. Incremental state records the same policy under state schema version 2. Older generations and state files require an explicit rebuild. Import metadata does not enter embedding-cache identity, so unchanged searchable text can reuse cached vectors.

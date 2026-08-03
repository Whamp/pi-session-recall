---
status: superseded by ADR-0007
---

# Keep recall evidence complete and presentation collapsible

`pi-session-recall` keeps its complete formatted evidence in model-visible tool content. Pi presents that evidence through a custom Recall result presentation: successful matches show one summary line while collapsed, and Pi's native tool-output action reveals the execution text unchanged while expanded. The renderer follows Pi's global expanded state and resolves the `app.tools.expand` hint through Pi's keybinding helper.

The collapsed summary reports the returned result count, recall scope, indexed-document count, and output truncation when present. A zero-match search reports `No matching past conversations found` and the recall scope in either display state, without an expansion hint. Truncated execution retains its model-visible notice and stores Pi's full `TruncationResult` under result details. Untruncated execution omits that metadata. Validation and execution errors keep Pi's error content instead of using a successful summary.

The complete service-injected tool definition is the public construction and test seam. Production creates the Recall conversation service and registers that definition unchanged. This preserves trusted-cwd scope resolution, read-only maintenance, retrieval, result limits, evidence formatting, provenance, and source geometry. The tool defines no custom call renderer, toggle, state, or keybinding.

This decision excludes changes to:

- dense, lexical, or identifier retrieval; fusion, ranking, deduplication, project attribution, Neighbor context, or Source locators;
- indexing, manual maintenance, storage, embeddings, evaluation policy, or corpus scope;
- model-visible formatting, evidence ordering, excerpts, metadata, source geometry, result-count limits, output limits, or excerpt limits;
- tool-call or header rendering, query-argument display, validation messages, execution errors, or Pi's error framing;
- per-result toggles, recall-specific keybindings, a second expansion state, streaming, or partial-result updates;
- other Pi tools, Pi's global tool-output behavior, non-TUI consumers, persisted tool content, or the agent-facing contract beyond additive truncation details.

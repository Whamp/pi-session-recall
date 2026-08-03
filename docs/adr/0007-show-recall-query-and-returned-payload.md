---
status: accepted
---

# Show the recall query and returned payload

`pi-session-recall` shows a concise copy of its query in the tool call. The display copy trims outer whitespace, collapses internal whitespace to one line, and truncates at 60 columns with an ellipsis. This renderer never changes the stored tool arguments or the query sent through recall search. Error calls keep only Pi's existing tool title.

Successful nonzero results retain the UTF-8 byte count and line count reported by Pi's output-boundary operation. These metrics describe the bounded evidence body. A truncation notice is framing and does not contribute to either value. Zero-match results omit payload metrics, and older persisted results without metrics omit that summary segment.

The collapsed Recall result presentation reports the returned Hybrid recall result count, Recall scope, returned payload metrics, optional truncation disclosure, and Pi's configured global expansion action. It no longer reports the global indexed-document count. Expanded rendering remains the exact model-visible execution text. Zero-match and error rendering remain unchanged.

This decision changes presentation details only. It preserves complete model-visible evidence, Pi's native global expansion state, retrieval behavior, Recall scope resolution, result limits, provenance, Source locators, output bounds, Index maintenance, and read-only interactive search.

This ADR supersedes ADR-0006.

# Conversation Recall

Conversation Recall turns Pi session history into source-backed evidence that can be searched without losing the session structure that gave the text meaning.

## Language

**Session graph**:
The entries in one Pi session file, linked by their `id` and `parentId` values.
_Avoid_: Conversation list, transcript

**Effective leaf**:
The session graph position selected by the latest harness leaf record, or by the latest entry when no leaf record exists.
_Avoid_: Last message, branch ID

**Active branch**:
The path from a session graph root to the effective leaf.
_Avoid_: Current transcript

**Branch path membership**:
The set of endpoint paths that contain an entry. A shared ancestor can belong to several paths; Pi does not assign intrinsic branch IDs.
_Avoid_: Branch ID

**Active context**:
The entries visible from the effective leaf after Pi applies the latest compaction checkpoint.
_Avoid_: Active branch

**Visible text run**:
One or more adjacent, nonempty text blocks from one entry. Any non-text or empty block ends the run.
_Avoid_: Message text, joined content

**Atomic conversation chunk**:
A token-bounded part of one visible text run. It never crosses an entry, role, run, tool, thinking, image, result, or summary boundary.
_Avoid_: Character chunk, transcript chunk

**Tool evidence document**:
A lexical-only, verbatim tool name, argument object, result text, or direct bash command/output part bounded within one source block or message field. Tool calls and results are linked by call ID. Tool evidence is never sent to the embedding model.
_Avoid_: Tool transcript, conversation chunk

**Evidence part**:
The source component represented by one document: conversation or summary content, tool name, tool arguments, tool result, bash command, or bash output.
_Avoid_: Chunk type

**Dense candidate**:
An atomic conversation chunk surfaced because its meaning is close to the search query.
_Avoid_: Semantic result

**Lexical candidate**:
An atomic conversation chunk surfaced by case-insensitive ordinary-text retrieval.
_Avoid_: Keyword result

**Identifier candidate**:
An atomic conversation chunk surfaced by case-preserving retrieval of identifiers, filenames, hashes, or similar source tokens.
_Avoid_: Exact result

**Hybrid recall result**:
One conversation, summary, or tool evidence document deduplicated across retrieval channels, with its document kind and each component rank and score retained.
_Avoid_: Semantic match

**Index manifest**:
The versioned identity of the model, tokenizer, chunk policy, provenance schema, and zvec schema used by one index generation.
_Avoid_: Index state, configuration

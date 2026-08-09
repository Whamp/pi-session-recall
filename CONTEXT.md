# Conversation Recall

Conversation Recall turns Pi session history into source-backed evidence that can be searched without losing the session structure that gave the text meaning.

## Language

**Physical session file**:
One source JSONL file under Pi's session store. Historical file reuse can place several logical sessions in one physical session file.
_Avoid_: Session graph, logical session

**Ignored physical session path**:
One absolute, lexically normalized physical session path that PSR policy excludes from index maintenance. Identity is exact: no glob, realpath, containment, case, or Unicode normalization applies.
_Avoid_: Ignore pattern, parser exclusion

**Logical session**:
One complete session header and the records that follow it until the next complete header or physical end of file. Each logical session owns one independently validated session graph.
_Avoid_: Physical session file, segment

**Canonical session representation**:
The in-memory header and physical-line-backed records sent to the strict session graph parser after exact format detection and any deterministic virtual conversion.
_Avoid_: Repaired session, rewritten file

**Session import policy**:
The versioned rules for LF-byte framing, exact format detection, virtual conversion, and rejection before strict graph validation.
_Avoid_: Parser fallback, repair policy

**Session graph**:
The entries in one logical Pi session, linked by their `id` and `parentId` values.
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

**Turn-context document**:
A token-bounded secondary document that joins visible user text with visible assistant text on one parent-linked path until the next user entry. It may cross intervening tool activity but excludes thinking and raw tool output. It cites every entry whose text contributes.
_Avoid_: Flat turn, tool transcript

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

**Recall result presentation**:
The Pi TUI view of completed `pi-session-recall` output. It keeps model-visible recall evidence unchanged, shows a one-line summary while collapsed, and reveals the full output through Pi's configured tool-expansion action. It is a UI concept, not a Hybrid recall result or Tool evidence document.
_Avoid_: Recall result, tool evidence

**Evidence occurrence**:
One exact source location for recalled evidence, including its session, graph position, and source geometry. Copied evidence can have several occurrences.
_Avoid_: Duplicate result, source alias

**Duplicate evidence group**:
One representative recall candidate plus every overlapping-sibling or exact-copy occurrence suppressed from separate result slots. Raw evidence and synthetic summaries never share a group.
_Avoid_: Duplicate result list

**Ranked hybrid result**:
One duplicate evidence group ordered by fused retrieval score plus a small active-branch preference. Abandoned-branch evidence remains eligible and labeled.
_Avoid_: Reranked result, semantic match

**Neighbor context**:
Readable context formed from a winning atomic conversation chunk and its valid contiguous siblings in the same visible text run. The contributing chunks remain individually identified.
_Avoid_: Expanded transcript, joined messages

**Index manifest**:
The versioned identity of the Octen model, native and stored dimensions, prefix normalization, tokenizer, chunk policy, provenance schema, project identity, and zvec schema used by one explicitly maintained index.
_Avoid_: Index state, configuration

**Stored recall embedding**:
The first configured dimensions of one native Octen vector, L2-normalized and stored as FP32 for inner-product search.
_Avoid_: Raw embedding, independently verified MRL vector

**Index maintenance**:
One standalone `psr index` operation that scans physical session files and updates one zvec collection. An operator or opt-in user schedule may start it. `psr index --rebuild` replaces incompatible index state.
_Avoid_: Live ingestion, lifecycle reconciliation

**Maintenance workset**:
The eligible new or changed physical session files, missing previously indexed files, and ignored indexed files scheduled for removal during one index maintenance operation. It forecasts file-level work; the number of documents requiring embeddings emerges as changed files are processed.
_Avoid_: Sessions to embed, embedding total

**Index optimization**:
One explicit `psr optimize` operation that compacts the existing zvec collection without scanning Physical session files or changing searchable evidence. It runs manually or through an opt-in optimization schedule, never as a default part of Index maintenance. It uses the same writer lock as Index maintenance.
_Avoid_: Index maintenance, rebuild

**Index maintenance status**:
The durable completion record for the latest normally completed Index maintenance operation. It records when the operation completed and how many Physical session files it scanned or failed. Its absence means freshness is unavailable, not that a live backlog was measured.
_Avoid_: Index state, live backlog

**Source locator**:
The physical session path, source line range, and entry ID attached to one result so an agent can read the original JSONL records.
_Avoid_: Source neighborhood, expanded transcript

**Derived recall evidence**:
The `pi-session-recall` tool call and its result. The index excludes both because they restate search inputs or previously indexed evidence and would create a feedback loop.
_Avoid_: Tool evidence, primary evidence

**Session origin**:
The working directory recorded when a Pi session began. It applies to the whole session, not individual entries.
_Avoid_: Project path, current directory

**Project identity**:
The stable scalar that determines whether indexed evidence and an invocation belong to the same project boundary.
_Avoid_: Project path, affinity

**Repository identity**:
A Git project identity shared by equivalent clones and worktrees of one repository.
_Avoid_: Checkout path, worktree identity

**Project lineage**:
A personal, explicit assignment from historical session-origin roots to one canonical repository identity.
_Avoid_: Project graph, inferred migration

**Project lineage root**:
An absolute historical session origin whose exact path and descendants receive one configured project lineage assignment.
_Avoid_: Path alias, fuzzy project root

**Project identity source**:
The explicit basis that established a project identity: a canonical Git origin, shared local Git directory, or exact non-Git session origin.
_Avoid_: Project hint, inferred relation

**Recall scope**:
The candidate-eligibility boundary for one search: project scope admits one exact project identity, while global scope admits the whole corpus.
_Avoid_: Project ranking, affinity filter

**Evidence relation**:
The explicit relationship between one recall result and the invoking project: same repository, configured project lineage, same non-Git session origin, or unrestricted global evidence.
_Avoid_: Relevance relation, project score

**Evaluation identity**:
The versioned project-scope, repository, lineage, ranking, candidate-limit, and final-result policy measured by one bounded quality result.
_Avoid_: Benchmark settings, test configuration

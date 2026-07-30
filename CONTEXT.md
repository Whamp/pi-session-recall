# Conversation Recall

Conversation Recall turns Pi session history into source-backed evidence that can be searched without losing the session structure that gave the text meaning.

## Language

**Physical session file**:
One source JSONL file under Pi's session store. Historical file reuse can place several logical sessions in one physical session file.
_Avoid_: Session graph, logical session

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

**Recall eligibility**:
The monotonic state of evidence after compaction, branch navigation, session departure, session quiescence, or creation as a context-exit summary makes it available to recall. Evidence remains eligible if another runtime still has it active or a runtime later resumes it.
_Avoid_: Global inactivity, current-session status

**Context-exit summary**:
A compaction or branch summary created while source evidence leaves direct active context. It becomes recall-eligible when created even if Pi retains the summary itself in active context.
_Avoid_: Active-tail message, mutable summary

**Session quiescence**:
A sustained period with no growth in a dirty physical session file. Quiescence makes its remaining active tail recall-eligible without proving whether a Pi process is still alive.
_Avoid_: Process death, closed session, inactive session

**Recall horizon**:
The deliberate boundary behind active work where evidence becomes worth indexing for future recall. It favors forgotten and older context over immediate freshness so recall maintenance does not compete with interactive Pi use.
_Avoid_: Real-time freshness, active-session mirror

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

**Immutable recall evidence**:
Source-backed content finalized once when it becomes recall-eligible. Later leaf, branch, compaction, label, or session-name changes do not rewrite it.
_Avoid_: Session snapshot, mutable chunk

**Session projection**:
The small mutable account of physical source ingestion and logical session state kept separately from immutable recall evidence.
_Avoid_: Conversation index, evidence document, global index state

**Physical session projection**:
The sole durable per-source account of one physical session file's processed position, logical-session membership, source availability, marker coverage, expected store contents, and repair status.
_Avoid_: Logical session projection, process lease, file snapshot, global progress record

**Logical session projection**:
The session projection for one logical session's effective leaf, active context, branches, compaction boundary, labels, eligible spans, and repair status.
_Avoid_: Physical session projection, session graph, evidence document

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

**Submitted recall query**:
The exact search text that an invoking agent sends to Conversation Recall.
_Avoid_: User query, original query

**Recall intent**:
Optional context supplied by the invoking agent to disambiguate query planning and reranking without becoming a retrieval query itself.
_Avoid_: Search query, filter, agent prompt

**Query plan**:
One ordered, validated collection of planned retrieval queries produced by an invoking agent or a configured query planning model. It does not itself execute retrieval or ranking.
_Avoid_: Expanded query string, generated search

**Planned retrieval query**:
One typed `lex`, `vec`, or `hyde` query inside a query plan. Lexical queries target ordinary lexical evidence; semantic and hypothetical-answer queries target dense evidence.
_Avoid_: Search result, identifier query

**Query planning capability verification**:
One independent conformance operation that checks planner profile, adapter, prompt, grammar, typed bounds, recall intent, cancellation, timeout, and cache identity before query-planned search can use it.
_Avoid_: Query-planned search, model health check

**Query-planned recall**:
A recall search that keeps every submitted-query retrieval channel and adds ranked lists from a query plan. An invoking agent may supply the plan; otherwise a configured query-planning model produces it.
_Avoid_: Planned search, query rewrite, agentic search

**Model profile**:
The immutable semantic identity of one inference model's accepted inputs and produced outputs, independent of where or how the model executes.
_Avoid_: Backend, endpoint configuration, model name

**Inference adapter**:
A verified translation between one inference service contract and one Conversation Recall capability contract, including the service's capability-specific semantics.
_Avoid_: Endpoint mapping, generic model adapter, inference configuration

**Inference configuration**:
The atomic local selection of independently verified embedding, reranking, and query-planning model profiles plus their execution backends. Embeddings are required; the other capabilities are optional.
_Avoid_: Model stack, index manifest

**Inference configuration candidate**:
One exact capability profile, backend, adapter, artifact/device description, and conformance operation offered to setup. A candidate is never an automatic fallback.
_Avoid_: Provider default, generic endpoint

**Execution identity**:
The canonical profile, adapter revision and settings, evaluation policy, software, resolved compute backend, and physical device identity that can affect one measured inference result.
_Avoid_: Cache key, profile label

**Physical device identity**:
The normalized names and stable processor-count metadata for the resolved device used by one embedded inference capability.
_Avoid_: Caller device label, requested device policy, hardware identity

**Capability conformance record**:
The accepted profile, backend, adapter, cache identity, verification time, and bounded measurement persisted after one capability-specific conformance operation passes. Embedding records also carry the semantic identity that determines index-generation compatibility.
_Avoid_: Health check, model availability

**Pending embedding replacement**:
One verified embedding selection waiting for a matching replacement recall generation to activate. It does not replace the active embedding selection while replacement work is incomplete.
_Avoid_: Configured embedding, active embedding

**Evidence occurrence**:
One exact source location for recalled evidence, including its session, graph position, and source geometry. Copied evidence can have several occurrences.
_Avoid_: Duplicate result, source alias

**Duplicate evidence group**:
One representative recall candidate plus every overlapping-sibling or exact-copy occurrence suppressed from separate result slots. Raw evidence and synthetic summaries never share a group.
_Avoid_: Duplicate result list

**Reranked recall result**:
One duplicate evidence group ordered by query relevance plus a small active-branch preference. Abandoned-branch evidence remains eligible and labeled.
_Avoid_: Final match, semantic match

**Neighbor context**:
Readable context formed from a winning atomic conversation chunk and its valid contiguous siblings in the same visible text run. The contributing chunks remain individually identified.
_Avoid_: Expanded transcript, joined messages

**Replacement recall generation**:
One resumable recall generation built and validated beside the active generation. Its registry entry records building, failed, or ready state until atomic activation.
_Avoid_: Staging index generation, temporary index, partial active index

**Background index build**:
One detached child process that owns the crash-released rebuild lock while building a replacement recall generation until activation, failure, or an explicit stop.
_Avoid_: Daemon, background job framework

**Background index status record**:
The one bounded local record containing a background index build's generation, process state, progress, latest durable checkpoint, and latest actionable error.
_Avoid_: Event log, job history

**Corpus metadata estimate**:
A model-free count of physical session files and their total source bytes before first-index approval.
_Avoid_: Index estimate, corpus scan

**Measured indexing sample**:
Bounded first-index work that measures model cold start and throughput while retaining compatible embeddings for the approved full build.
_Avoid_: Benchmark, throwaway sample

**Recall readiness**:
Whether one complete active recall generation is available for search; selected inference configuration alone is not ready.
_Avoid_: Setup complete, model ready

**Index manifest**:
The fixed versioned identity and compatibility contract for one recall generation's three store schemas and indexes, text and provenance policies, embedding profile, dimensions, and canaries. It never records mutable progress, backlog, or counts.
_Avoid_: Index state, configuration, progress record

**Generation validation receipt**:
The immutable proof that a closed replacement generation matched its starting source snapshot and passed complete cross-store validation. It records validation evidence, not later incremental progress.
_Avoid_: Index manifest, background status, mutable certification

**Recall generation**:
One self-contained lexical/source store, dense store, session projection store, and fixed index manifest, plus a successful validation receipt once it becomes valid. A replacement generation is built beside the searchable generation rather than overwriting it.
_Avoid_: Database, index directory, mutable release, shared vector store

**Active generation pointer**:
The checksummed atomic selection of the sole recall generation served by search and targeted by incremental commits.
_Avoid_: Latest directory, generation scan, fallback generation

**Generation replay snapshot**:
The immutable set of pending and quarantined recall work marker IDs captured for one activation or rollback transition. Only those marker IDs gate that transition's replay; newer markers remain ordinary backlog.
_Avoid_: Live marker queue, freshness barrier, rebuild source snapshot

**Replay-pending generation**:
A newly active generation whose current generation replay snapshot has not yet been covered by durable physical session projections or still names unresolved quarantined markers. Search may serve it while the ordinary incremental worker completes that fixed snapshot.
_Avoid_: Building generation, dual-write generation, failed generation, stale generation

**Rollback generation**:
The one validated former active generation retained for a bounded period after cutover. Restoring it is an explicit quick switch followed by replay of one fixed generation replay snapshot.
_Avoid_: Backup copy, automatic fallback, second write target, rebuild

**Recall work marker**:
One immutable, atomically published event file telling an external worker that a physical session may contain newly eligible evidence. The worker orders and coalesces markers; Pi processes never overwrite them.
_Avoid_: Index job, mutable session marker, session lock

**Recall marker quarantine**:
The retained holding area for a corrupt or unsupported recall work marker removed from ordinary processing. Its diagnostics expose only a failure category, count, and age; an unresolved quarantined marker blocks any generation replay snapshot that names it.
_Avoid_: Failed marker deletion, retry queue, marker contents diagnostic

**Metadata recovery sweep**:
One bounded, resumable inspection of physical session file names and metadata used to observe crash-missed source arrivals or absences without reading session content.
_Avoid_: Full session scan, session parsing, deletion confirmation

**Incremental recall worker**:
The sole writer for deferred incremental transfer. This short-lived process runs outside Pi, drains recall work markers, processes eligible append deltas, and exits when no work remains.
_Avoid_: Daemon, Pi lifecycle handler, full indexer

**Recall maintenance class**:
The cost and scheduling category of recall work: immediate bookkeeping, deferred incremental transfer, or explicit reconciliation. Deferred work may be delayed and combined without a freshness deadline.
_Avoid_: Worker priority, ingestion mode, adaptive scheduler, freshness class

**Recall write window**:
The bounded period when one writer owns the active generation exclusively to commit prepared evidence and session projection changes. Search may wait for the current window but never for pending ingestion.
_Avoid_: Maintenance outage, freshness barrier, indexing run

**Recall recovery record**:
The durable generation-local description of one bounded store mutation that must be verified or replayed before read-only use. Its presence means the affected generation is not safe to search.
_Avoid_: Writer lock, batch backlog, progress record

**Recall diagnostic operation**:
One bounded, local account of recall work and its costs, identified independently from the session or query that caused it.
_Avoid_: Trace, span, telemetry

**Diagnostics mode**:
The persistence policy for recall diagnostic operations: `slow`, `all`, or `off`.
_Avoid_: Verbosity level, tracing mode

**Recall operator CLI**:
The standalone shell interface for recall setup, status, explicit catch-up, rebuild control, recovery, rollback, legacy adoption, and cleanup. Pi's TUI is not an operator control or maintenance-progress surface.
_Avoid_: Slash command, Pi maintenance command, lifecycle trigger

**Manual maintenance trigger**:
The explicit operation intent distinguishing a user-requested incremental catch-up from a rebuild. It originates from the recall operator CLI, never from Pi lifecycle activity.
_Avoid_: Lifecycle trigger, slash-command flag

**Physical session check**:
One determination of whether a physical session file changed and therefore needs reconciliation work.
_Avoid_: Session scan, logical session check

**Diagnostic start record**:
Evidence that a recall diagnostic operation began, retained when diagnostics mode is `all`.
_Avoid_: Progress record, trace start

**Diagnostic completion record**:
The bounded outcome, counts, and costs of one completed or failed recall diagnostic operation.
_Avoid_: Performance dump, trace end

**Phase timing**:
The exclusive elapsed time attributed to one named part of a recall diagnostic operation.
_Avoid_: Nested timing, overlapping timing

**Unattributed time**:
The nonnegative part of a recall diagnostic operation's elapsed time not assigned to a phase timing.
_Avoid_: Other phase, overhead bucket

**Active diagnostic log**:
The current bounded local sequence of recall diagnostic records.
_Avoid_: Telemetry stream, trace file

**Retained predecessor**:
The one previous active diagnostic log kept after rotation.
_Avoid_: Log archive, diagnostic history

**Read-only recall search**:
A query against the last durable recall generation. It neither waits for source ingestion nor writes recall state.
_Avoid_: Active-session freshness barrier, search-triggered reconciliation, read-your-writes search

**Confirmed session deletion**:
A source absence observed again on a later metadata sweep while the session root is healthy and no suspicious mass disappearance is present. Confirmation removes the physical projection, its logical projections, and every recall evidence occurrence backed by that source.
_Avoid_: Missing source, one-sweep deletion, retained historical memory

**Material recall backlog**:
Eligible source work not yet reflected in the active generation, including failed attempts and work that arrived during a rebuild. Its size or age measures staleness, not generation validity, and has no freshness deadline.
_Avoid_: Active-tail delay, freshness barrier, recovery-required state

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

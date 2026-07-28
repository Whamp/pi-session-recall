/** Kind of bounded local recall operation diagnostic. */
export enum RecallDiagnosticOperationKind {
  LIVE_SESSION_RECONCILIATION = 'live_session_reconciliation',
  SEARCH = 'search',
  FULL_INDEX = 'full_index',
  REBUILD = 'rebuild',
  PHYSICAL_SESSION_CHECK = 'physical_session_check',
  OPTIMIZATION = 'optimization',
}

/** Lifecycle state recorded for one recall diagnostic operation. */
export enum RecallDiagnosticStatus {
  STARTED = 'started',
  SUCCEEDED = 'succeeded',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

/** Stable privacy-safe category for a failed recall diagnostic operation. */
export enum RecallDiagnosticErrorCategory {
  OPERATION_FAILED = 'operation_failed',
  OPERATION_CANCELLED = 'operation_cancelled',
}

/** Local persistence policy for bounded recall diagnostic records. */
export enum RecallDiagnosticsMode {
  SLOW = 'slow',
  ALL = 'all',
  OFF = 'off',
}

/** Pi lifecycle event that requested one live session reconciliation. */
export enum RecallLifecycleTrigger {
  AGENT_SETTLED = 'agent_settled',
  SESSION_SHUTDOWN = 'session_shutdown',
  ACTIVE_SESSION_FRESHNESS = 'active_session_freshness',
}

/** Manual maintenance trigger distinguishing explicit incremental indexing from rebuilding. */
export enum RecallManualMaintenanceTrigger {
  MANUAL_INCREMENTAL_INDEX = 'manual_incremental_index',
  MANUAL_REBUILD = 'manual_rebuild',
}

/** Explicit source used to assign one stable project identity to recall evidence. */
export enum RecallProjectIdentitySource {
  GIT_ORIGIN = 'git_origin',
  GIT_COMMON_DIRECTORY = 'git_common_directory',
  NON_GIT_SESSION_ORIGIN = 'non_git_session_origin',
  CONFIGURED_PROJECT_LINEAGE = 'configured_project_lineage',
}

/** Exact physical Pi session format selected before strict graph validation. */
export enum SessionImportFormat {
  CANONICAL_JSONL = 'canonical_jsonl',
  PI_V1_LINEAR = 'pi_v1_linear',
  PI_SESSION_REUSE_HISTORY = 'pi_session_reuse_history',
}

/** Accepted or rejected outcome for one physical file in a read-only import replay. */
export enum SessionImportReplayOutcome {
  ACCEPTED = 'accepted',
  REJECTED = 'rejected',
}

/** Trigger kind persisted in one immutable recall work marker. */
export enum RecallWorkMarkerTrigger {
  ACTIVITY = 'activity',
  COMPACTION = 'compaction',
  BRANCH_EXIT = 'branch_exit',
  DEPARTURE = 'departure',
  ARRIVAL = 'arrival',
}

/** Privacy-safe reason an immutable recall work marker left the pending spool. */
export enum RecallMarkerQuarantineCategory {
  CORRUPT = 'corrupt',
  UNSUPPORTED = 'unsupported',
}

/** Bounded outcome of one metadata-only session recovery sweep slice. */
export enum RecallMetadataSweepStatus {
  COMPLETE = 'complete',
  CONTINUATION_REQUIRED = 'continuation_required',
  ROOT_UNAVAILABLE = 'root_unavailable',
  PERMISSION_DENIED = 'permission_denied',
  SUSPICIOUS_MASS_LOSS = 'suspicious_mass_loss',
}

/** Record kind persisted in the scalar-only session projection collection. */
export enum RecallSessionProjectionKind {
  PHYSICAL_SESSION = 'physical_session',
  LOGICAL_SESSION = 'logical_session',
}

/** Whether a session projection may continue incremental ingestion. */
export enum RecallProjectionRepairState {
  READY = 'ready',
  REQUIRES_RECONCILIATION = 'requires_reconciliation',
}

/** Stable reason that a session projection requires explicit reconciliation. */
export enum RecallProjectionRepairReason {
  APPEND_CURSOR_MISSING = 'append_cursor_missing',
  SOURCE_SHRANK = 'source_shrank',
  BOUNDARY_MISMATCH = 'boundary_mismatch',
  UNSUPPORTED_LAYOUT = 'unsupported_layout',
  MALFORMED_GRAPH = 'malformed_graph',
  PROJECTION_OVERFLOW = 'projection_overflow',
}

/** Durable observation state for one physical session source. */
export enum RecallSourceAvailability {
  PRESENT = 'present',
  SOURCE_MISSING = 'source_missing',
  DELETION_CONFIRMED = 'deletion_confirmed',
}

/** Outcome of encoding one bounded session projection candidate. */
export enum RecallProjectionEncodingStatus {
  ENCODED = 'encoded',
  REQUIRES_RECONCILIATION = 'requires_reconciliation',
}

/** Durable generation registry state used by explicit side-by-side cutover. */
export enum RecallGenerationCutoverState {
  BUILDING = 'building',
  READY = 'ready',
  ACTIVE = 'active',
  ROLLBACK = 'rollback',
  FAILED = 'failed',
}

/** Privacy-safe failure category exposed by the scalar material backlog summary. */
export enum RecallBacklogFailureCategory {
  MARKER_DECODE_FAILED = 'marker_decode_failed',
  PROJECTION_RECONCILIATION_REQUIRED = 'projection_reconciliation_required',
  WRITE_FAILED = 'write_failed',
  RECOVERY_REQUIRED = 'recovery_required',
  REBUILD_FAILED = 'rebuild_failed',
}

/** Version of trusted-invocation exact project filtering before every retrieval-channel limit. */
export const PROJECT_SCOPE_POLICY_VERSION = 1;

/** Corpus boundary selected for one recall search. */
export enum RecallSearchScope {
  PROJECT = 'project',
  GLOBAL = 'global',
}

/** Explicit relationship between recalled evidence and the invoking project. */
export enum RecallEvidenceRelation {
  SAME_REPOSITORY = 'same_repository',
  CONFIGURED_PROJECT_LINEAGE = 'configured_project_lineage',
  SAME_SESSION_ORIGIN = 'same_session_origin',
  UNRESTRICTED_GLOBAL = 'unrestricted_global_evidence',
}

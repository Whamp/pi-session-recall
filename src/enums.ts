/** Kind of bounded local recall operation diagnostic. */
export enum RecallDiagnosticOperationKind {
  LIVE_SESSION_RECONCILIATION = 'live_session_reconciliation',
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

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

/** Durable local Octen artifact state reported without changing the model cache. */
export enum LocalOctenModelStatusKind {
  MISSING = 'missing',
  PARTIAL = 'partial',
  CORRUPT = 'corrupt',
  READY = 'ready',
}

/** Observable stage of one explicitly approved local model download. */
export enum LocalOctenModelDownloadProgressKind {
  PREPARING = 'preparing',
  DOWNLOADING_FILE = 'downloading-file',
  FILE_VERIFIED = 'file-verified',
  ACTIVATED = 'activated',
}

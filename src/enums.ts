/** Explicit source used to assign one stable project identity to recall evidence. */
export enum RecallProjectIdentitySource {
  GIT_ORIGIN = 'git_origin',
  GIT_COMMON_DIRECTORY = 'git_common_directory',
  NON_GIT_SESSION_ORIGIN = 'non_git_session_origin',
}

/** Corpus boundary selected for one recall search. */
export enum RecallSearchScope {
  PROJECT = 'project',
  GLOBAL = 'global',
}

/** Explicit relationship between recalled evidence and the invoking project. */
export enum RecallEvidenceRelation {
  SAME_REPOSITORY = 'same_repository',
  SAME_SESSION_ORIGIN = 'same_session_origin',
  UNRESTRICTED_GLOBAL = 'unrestricted_global_evidence',
}

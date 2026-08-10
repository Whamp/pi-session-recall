/** Observable phases and cumulative facts from one standalone index maintenance operation. */
export type RecallIndexProgressEvent =
  | { kind: 'preparing' }
  | { kind: 'waiting-for-write-lock' }
  | { kind: 'discovering-physical-session-files' }
  | { kind: 'planning-maintenance-workset' }
  | { kind: 'preparing-rebuild-candidate'; staleCandidatesRemoved: number }
  | { kind: 'rebuild-candidate-failed' }
  | { kind: 'rebuild-candidate-activated'; previousAvailable: boolean }
  | {
      kind: 'maintenance-workset-planned';
      discoveredFiles: number;
      newFiles: number;
      changedFiles: number;
      missingFiles: number;
      ignoredRemovals: number;
      rebuild: boolean;
    }
  | { kind: 'indexing-changed-physical-session-files' }
  | {
      kind: 'indexing-maintenance-workset';
      completedFiles: number;
      totalFiles: number;
      sessionPath: string;
      indexedSessions: number;
      newlyEmbeddedDocuments: number;
      reusedVectors: number;
      deletedDocuments: number;
      failedSessions: number;
    }
  | { kind: 'physical-session-file-failed'; sessionPath: string }
  | { kind: 'optimizing-collection' }
  | { kind: 'completed' };

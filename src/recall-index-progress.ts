/** Content-free elapsed milliseconds for the expensive phases of one changed Physical session file. */
export interface PhysicalSessionIndexPhaseElapsedMilliseconds {
  readParse: number;
  graphValidation: number;
  documentConstructionTokenization: number;
  vectorLookup: number;
  embedding: number;
  sqliteReplacement: number;
}

/** Observable phases and cumulative facts from one standalone index maintenance operation. */
export type RecallIndexProgressEvent =
  | { kind: 'preparing' }
  | { kind: 'waiting-for-write-lock' }
  | { kind: 'discovering-physical-session-files' }
  | { kind: 'planning-maintenance-workset' }
  | { kind: 'preparing-rebuild-candidate'; staleCandidatesRemoved: number }
  | { kind: 'resuming-rebuild-candidate' }
  | { kind: 'rebuild-candidate-failed' }
  | { kind: 'rebuild-candidate-staged'; databaseTarget: string }
  | { kind: 'rebuild-candidate-activated' }
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
  | {
      kind: 'physical-session-file-profiled';
      sessionPath: string;
      change: 'new' | 'changed';
      sourceBytesAtPlanning: number;
      indexedSourceBytesBefore: number | null;
      denseDocuments: number;
      invocations: number;
      newlyEmbeddedDocuments: number;
      reusedVectors: number;
      totalElapsedMilliseconds: number;
      phaseElapsedMilliseconds: PhysicalSessionIndexPhaseElapsedMilliseconds;
    }
  | { kind: 'physical-session-file-failed'; sessionPath: string }
  | { kind: 'optimizing-collection' }
  | { kind: 'completed' };

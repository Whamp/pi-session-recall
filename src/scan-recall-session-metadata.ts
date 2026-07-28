import { randomUUID } from 'node:crypto';
import { lstat, mkdir, open, readFile, readdir, rename, rm } from 'node:fs/promises';
import { extname, join } from 'node:path';

import { Type } from 'typebox';
import { Value } from 'typebox/value';

import { RecallMetadataSweepStatus } from './enums.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Maximum files examined by one metadata recovery sweep slice. */
export const RECALL_METADATA_SWEEP_MAX_FILES = 10_000;

/** Maximum monotonic elapsed time allowed for one metadata recovery sweep slice. */
export const RECALL_METADATA_SWEEP_MAX_ELAPSED_MILLISECONDS = 500;

/** Current strict scalar continuation version for metadata recovery sweeps. */
export const RECALL_METADATA_SWEEP_CONTINUATION_VERSION = 1;

/** Filename of the strict scalar metadata sweep continuation under the marker control directory. */
export const RECALL_METADATA_SWEEP_CONTINUATION_FILENAME = 'metadata-sweep-continuation.json';

const DEFAULT_SUSPICIOUS_MASS_LOSS_MINIMUM_MISSING_SOURCES = 2;

/** Scalar metadata returned for one observed physical session file without reading its body. */
export interface ObservedRecallSessionMetadata {
  relativePath: string;
  sizeBytes: number;
  modifiedAtEpochMilliseconds: number;
}

/** Known physical-session identity used only to classify source-missing observations. */
export interface KnownRecallSessionMetadataSource {
  physicalSessionId: string;
  relativePath: string;
}

/** File kind and scalar stat fields needed by metadata-only traversal. */
export interface RecallSessionMetadataStat {
  isDirectory: boolean;
  isFile: boolean;
  sizeBytes: number;
  modifiedAtEpochMilliseconds: number;
}

/** Injectable directory-entry and stat-only boundary for session metadata recovery. */
export interface RecallSessionMetadataFilesystem {
  readDirectory: (path: string) => Promise<string[]>;
  statPath: (path: string) => Promise<RecallSessionMetadataStat>;
}

/** Strict scalar state that resumes one bounded metadata sweep without session content. */
export interface RecallMetadataSweepContinuation {
  version: 1;
  currentRelativeDirectory: string;
  afterEntryName: string | null;
  pendingRelativeDirectories: string[];
  observedPhysicalSessionIds: string[];
  observedSessionFileCount: number;
}

/** Durable continuation capability separated from session-directory traversal. */
export interface RecallMetadataSweepContinuationStore {
  readContinuation(): Promise<RecallMetadataSweepContinuation | null>;
  writeContinuation(continuation: RecallMetadataSweepContinuation): Promise<void>;
  clearContinuation(): Promise<void>;
}

/** Scalar and metadata-only result of one bounded recovery sweep slice. */
export interface RecallSessionMetadataSweepResult {
  status: RecallMetadataSweepStatus;
  rootHealthy: boolean;
  deletionConfirmationSuppressed: boolean;
  scannedFileCount: number;
  observedSessionFileCount: number;
  observedSessionMetadata: ObservedRecallSessionMetadata[];
  missingPhysicalSessionIds: string[];
  continuationPersisted: boolean;
  elapsedMilliseconds: number;
}

/** Bounds, policies, and injectable capabilities for metadata-only session recovery. */
export interface ScanRecallSessionMetadataOptions {
  sessionRootDirectory: string;
  controlDirectory: string;
  knownSources?: readonly KnownRecallSessionMetadataSource[];
  suspiciousMassLossMinimumMissingSources?: number;
  maxFiles?: number;
  maxElapsedMilliseconds?: number;
  monotonicNowMilliseconds?: () => number;
  filesystem?: RecallSessionMetadataFilesystem;
  continuationStore?: RecallMetadataSweepContinuationStore;
}

const relativeDirectorySchema = Type.String({
  pattern: '^(?!/)(?!.*(?:^|/)\\.\\.(?:/|$)).*$',
});
const entryNameSchema = Type.String({ minLength: 1, pattern: '^[^/]+$' });
const recallMetadataSweepContinuationSchema = Type.Object(
  {
    version: Type.Literal(RECALL_METADATA_SWEEP_CONTINUATION_VERSION),
    currentRelativeDirectory: relativeDirectorySchema,
    afterEntryName: Type.Union([entryNameSchema, Type.Null()]),
    pendingRelativeDirectories: Type.Array(relativeDirectorySchema),
    observedPhysicalSessionIds: Type.Array(Type.String({ minLength: 1 })),
    observedSessionFileCount: Type.Integer({ minimum: 0 }),
  },
  { additionalProperties: false },
);

async function statRecallSessionMetadata(path: string): Promise<RecallSessionMetadataStat> {
  const metadata = await lstat(path);
  return {
    isDirectory: metadata.isDirectory(),
    isFile: metadata.isFile(),
    sizeBytes: metadata.size,
    modifiedAtEpochMilliseconds: metadata.mtimeMs,
  };
}

function createRecallMetadataSweepContinuationStore(
  controlDirectory: string,
): RecallMetadataSweepContinuationStore {
  const continuationPath = join(controlDirectory, RECALL_METADATA_SWEEP_CONTINUATION_FILENAME);
  return {
    async readContinuation() {
      let source: string;
      try {
        source = await readFile(continuationPath, 'utf8');
      } catch (error) {
        if (readNodeErrorCode(error) === 'ENOENT') {
          return null;
        }
        throw error;
      }
      let value: unknown;
      try {
        value = JSON.parse(source);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall metadata sweep continuation unreadable: ${message}`, {
          cause: error,
        });
      }
      try {
        return Value.Parse(recallMetadataSweepContinuationSchema, value);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Recall metadata sweep continuation invalid: ${message}`, {
          cause: error,
        });
      }
    },
    async writeContinuation(continuation) {
      const parsed = Value.Parse(recallMetadataSweepContinuationSchema, continuation);
      await mkdir(controlDirectory, { recursive: true });
      const temporaryPath = join(
        controlDirectory,
        `.${RECALL_METADATA_SWEEP_CONTINUATION_FILENAME}.${randomUUID()}.tmp`,
      );
      const temporaryFile = await open(temporaryPath, 'wx', 0o600);
      try {
        await temporaryFile.writeFile(`${JSON.stringify(parsed)}\n`);
        await temporaryFile.sync();
      } finally {
        await temporaryFile.close();
      }
      try {
        await rename(temporaryPath, continuationPath);
        await syncRecallDirectory(controlDirectory);
      } catch (error) {
        await rm(temporaryPath, { force: true });
        throw error;
      }
    },
    async clearContinuation() {
      try {
        await rm(continuationPath);
      } catch (error) {
        if (readNodeErrorCode(error) === 'ENOENT') {
          return;
        }
        throw error;
      }
      await syncRecallDirectory(controlDirectory);
    },
  };
}

function createInitialRecallMetadataSweepContinuation(): RecallMetadataSweepContinuation {
  return {
    version: RECALL_METADATA_SWEEP_CONTINUATION_VERSION,
    currentRelativeDirectory: '',
    afterEntryName: null,
    pendingRelativeDirectories: [],
    observedPhysicalSessionIds: [],
    observedSessionFileCount: 0,
  };
}

function validatePositiveRecallMetadataBound(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Recall metadata sweep ${name} must be positive`);
  }
}

function classifyRecallMetadataTraversalFailure(error: unknown): RecallMetadataSweepStatus | null {
  switch (readNodeErrorCode(error)) {
    case 'EACCES':
    case 'EPERM':
      return RecallMetadataSweepStatus.PERMISSION_DENIED;
    case 'ENOENT':
    case 'ENOTDIR':
      return RecallMetadataSweepStatus.ROOT_UNAVAILABLE;
    default:
      return null;
  }
}

function createRecallMetadataSweepResult(
  status: RecallMetadataSweepStatus,
  startMilliseconds: number,
  monotonicNowMilliseconds: () => number,
  scannedFileCount: number,
  observedSessionFileCount: number,
  observedSessionMetadata: ObservedRecallSessionMetadata[],
  missingPhysicalSessionIds: string[],
  continuationPersisted: boolean,
): RecallSessionMetadataSweepResult {
  const rootHealthy =
    status !== RecallMetadataSweepStatus.ROOT_UNAVAILABLE &&
    status !== RecallMetadataSweepStatus.PERMISSION_DENIED;
  return {
    status,
    rootHealthy,
    deletionConfirmationSuppressed: status !== RecallMetadataSweepStatus.COMPLETE,
    scannedFileCount,
    observedSessionFileCount,
    observedSessionMetadata,
    missingPhysicalSessionIds,
    continuationPersisted,
    elapsedMilliseconds: Math.max(0, monotonicNowMilliseconds() - startMilliseconds),
  };
}

function hasRemainingRecallMetadataWork(
  entryIndex: number,
  entryCount: number,
  continuation: RecallMetadataSweepContinuation,
): boolean {
  return entryIndex + 1 < entryCount || continuation.pendingRelativeDirectories.length > 0;
}

/** Scans only directory entries and file metadata, persisting continuation at either hard bound. */
export async function scanRecallSessionMetadata(
  options: ScanRecallSessionMetadataOptions,
): Promise<RecallSessionMetadataSweepResult> {
  const maxFiles = options.maxFiles ?? RECALL_METADATA_SWEEP_MAX_FILES;
  const maxElapsedMilliseconds =
    options.maxElapsedMilliseconds ?? RECALL_METADATA_SWEEP_MAX_ELAPSED_MILLISECONDS;
  validatePositiveRecallMetadataBound(maxFiles, 'file bound');
  validatePositiveRecallMetadataBound(maxElapsedMilliseconds, 'elapsed-time bound');
  const monotonicNowMilliseconds =
    options.monotonicNowMilliseconds ?? performance.now.bind(performance);
  const startMilliseconds = monotonicNowMilliseconds();
  const readDirectory = options.filesystem?.readDirectory ?? readdir;
  const statPath = options.filesystem?.statPath ?? statRecallSessionMetadata;
  const continuationStore =
    options.continuationStore ??
    createRecallMetadataSweepContinuationStore(options.controlDirectory);
  const continuation =
    (await continuationStore.readContinuation()) ?? createInitialRecallMetadataSweepContinuation();
  const observedPhysicalSessionIds = new Set(continuation.observedPhysicalSessionIds);
  const knownSourceIdByRelativePath = new Map(
    (options.knownSources ?? []).map(({ physicalSessionId, relativePath }) => [
      relativePath,
      physicalSessionId,
    ]),
  );
  const observedSessionMetadata: ObservedRecallSessionMetadata[] = [];
  let scannedFileCount = 0;

  async function persistRecallMetadataSweepResult(
    status: RecallMetadataSweepStatus,
  ): Promise<RecallSessionMetadataSweepResult> {
    continuation.observedPhysicalSessionIds = [...observedPhysicalSessionIds].toSorted();
    await continuationStore.writeContinuation(continuation);
    return createRecallMetadataSweepResult(
      status,
      startMilliseconds,
      monotonicNowMilliseconds,
      scannedFileCount,
      continuation.observedSessionFileCount,
      observedSessionMetadata,
      [],
      true,
    );
  }

  while (true) {
    const directoryPath = join(options.sessionRootDirectory, continuation.currentRelativeDirectory);
    let entryNames: string[];
    try {
      entryNames = (await readDirectory(directoryPath)).toSorted();
    } catch (error) {
      const failureStatus = classifyRecallMetadataTraversalFailure(error);
      if (failureStatus === null) {
        throw error;
      }
      return persistRecallMetadataSweepResult(failureStatus);
    }

    const remainingEntryNames = entryNames.filter(
      (name) => continuation.afterEntryName === null || name > continuation.afterEntryName,
    );
    for (const [entryIndex, entryName] of remainingEntryNames.entries()) {
      if (!Value.Check(entryNameSchema, entryName)) {
        throw new Error('Recall metadata sweep directory returned an invalid entry name');
      }
      const elapsedBeforeStat = monotonicNowMilliseconds() - startMilliseconds;
      if (elapsedBeforeStat >= maxElapsedMilliseconds) {
        return persistRecallMetadataSweepResult(RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
      }
      const relativePath =
        continuation.currentRelativeDirectory === ''
          ? entryName
          : join(continuation.currentRelativeDirectory, entryName);
      let metadata: RecallSessionMetadataStat;
      try {
        metadata = await statPath(join(options.sessionRootDirectory, relativePath));
      } catch (error) {
        const failureStatus = classifyRecallMetadataTraversalFailure(error);
        if (failureStatus === null) {
          throw error;
        }
        return persistRecallMetadataSweepResult(failureStatus);
      }
      continuation.afterEntryName = entryName;
      if (metadata.isDirectory) {
        continuation.pendingRelativeDirectories.push(relativePath);
        continuation.pendingRelativeDirectories.sort();
      } else if (metadata.isFile) {
        scannedFileCount += 1;
        if (extname(entryName) === '.jsonl') {
          continuation.observedSessionFileCount += 1;
          observedSessionMetadata.push({
            relativePath,
            sizeBytes: metadata.sizeBytes,
            modifiedAtEpochMilliseconds: metadata.modifiedAtEpochMilliseconds,
          });
          const physicalSessionId = knownSourceIdByRelativePath.get(relativePath);
          if (physicalSessionId !== undefined) {
            observedPhysicalSessionIds.add(physicalSessionId);
          }
        }
      }
      const moreWork = hasRemainingRecallMetadataWork(
        entryIndex,
        remainingEntryNames.length,
        continuation,
      );
      const elapsedAfterStat = monotonicNowMilliseconds() - startMilliseconds;
      if (
        moreWork &&
        (scannedFileCount >= maxFiles || elapsedAfterStat >= maxElapsedMilliseconds)
      ) {
        return persistRecallMetadataSweepResult(RecallMetadataSweepStatus.CONTINUATION_REQUIRED);
      }
    }

    const nextRelativeDirectory = continuation.pendingRelativeDirectories.shift();
    if (nextRelativeDirectory === undefined) {
      break;
    }
    continuation.currentRelativeDirectory = nextRelativeDirectory;
    continuation.afterEntryName = null;
  }

  await continuationStore.clearContinuation();
  const missingPhysicalSessionIds = (options.knownSources ?? [])
    .filter(({ physicalSessionId }) => !observedPhysicalSessionIds.has(physicalSessionId))
    .map(({ physicalSessionId }) => physicalSessionId)
    .toSorted();
  const suspiciousMassLossMinimumMissingSources =
    options.suspiciousMassLossMinimumMissingSources ??
    DEFAULT_SUSPICIOUS_MASS_LOSS_MINIMUM_MISSING_SOURCES;
  validatePositiveRecallMetadataBound(
    suspiciousMassLossMinimumMissingSources,
    'suspicious mass-loss minimum',
  );
  const status =
    missingPhysicalSessionIds.length >= suspiciousMassLossMinimumMissingSources
      ? RecallMetadataSweepStatus.SUSPICIOUS_MASS_LOSS
      : RecallMetadataSweepStatus.COMPLETE;
  return createRecallMetadataSweepResult(
    status,
    startMilliseconds,
    monotonicNowMilliseconds,
    scannedFileCount,
    continuation.observedSessionFileCount,
    observedSessionMetadata,
    missingPhysicalSessionIds,
    false,
  );
}

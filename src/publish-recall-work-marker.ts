import { randomUUID } from 'node:crypto';
import { mkdir, open, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';

import {
  createRecallDetachedWorkerSignal,
  type RecallDetachedWorkerSignal,
} from './create-recall-detached-worker-signal.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import {
  encodeRecallWorkMarker,
  type RecallWorkMarker,
  type RecallWorkMarkerCodecOptions,
} from './recall-work-marker.js';

/** Minimal durable file capabilities used to publish one recall work marker atomically. */
export interface RecallMarkerPublicationFile {
  writeFile(content: string): Promise<void>;
  sync(): Promise<void>;
  close(): Promise<void>;
}

/** Injectable filesystem boundary for recall marker durability and fault tests. */
export interface RecallMarkerPublicationFilesystem {
  createDirectory(path: string): Promise<void>;
  openExclusiveFile(path: string): Promise<RecallMarkerPublicationFile>;
  renameFile(temporaryPath: string, markerPath: string): Promise<void>;
  openDirectory(path: string): Promise<RecallMarkerPublicationFile>;
  removeFile(path: string): Promise<void>;
}

/** Explicit spool, trust, and test boundaries for durable marker publication. */
export interface PublishRecallWorkMarkerOptions extends RecallWorkMarkerCodecOptions {
  markerSpoolDirectory: string;
  workerOwnershipLockPath?: string;
  filesystem?: RecallMarkerPublicationFilesystem;
  workerSignal?: RecallDetachedWorkerSignal;
}

const nodeMarkerPublicationFilesystem: RecallMarkerPublicationFilesystem = {
  async createDirectory(path) {
    await mkdir(path, { recursive: true });
  },
  async openExclusiveFile(path) {
    return open(path, 'wx', 0o600);
  },
  renameFile: rename,
  async openDirectory(path) {
    return open(path, 'r');
  },
  async removeFile(path) {
    await rm(path, { force: true });
  },
};

type RecallMarkerPublicationFailureStage =
  | 'directory close'
  | 'directory open'
  | 'directory sync'
  | 'encoding'
  | 'file sync'
  | 'rename'
  | 'spool create'
  | 'temporary file close'
  | 'temporary file open'
  | 'temporary file write'
  | 'worker signal';

function formatRecallMarkerPublicationError(
  stage: RecallMarkerPublicationFailureStage,
  error: unknown,
): Error {
  const code = readNodeErrorCode(error) ?? 'UNKNOWN';
  switch (stage) {
    case 'directory close':
      return new Error(`Recall marker publication directory close failed [${code}]`);
    case 'directory open':
      return new Error(`Recall marker publication directory open failed [${code}]`);
    case 'directory sync':
      return new Error(`Recall marker publication directory sync failed [${code}]`);
    case 'encoding':
      return new Error(`Recall marker publication encoding failed [${code}]`);
    case 'file sync':
      return new Error(`Recall marker publication file sync failed [${code}]`);
    case 'rename':
      return new Error(`Recall marker publication rename failed [${code}]`);
    case 'spool create':
      return new Error(`Recall marker publication spool create failed [${code}]`);
    case 'temporary file close':
      return new Error(`Recall marker publication temporary file close failed [${code}]`);
    case 'temporary file open':
      return new Error(`Recall marker publication temporary file open failed [${code}]`);
    case 'temporary file write':
      return new Error(`Recall marker publication temporary file write failed [${code}]`);
    case 'worker signal':
      return new Error(`Recall marker publication worker signal failed [${code}]`);
    default:
      return new Error(`Recall marker publication failed [${code}]`);
  }
}

async function closeRecallMarkerPublicationFile(
  file: RecallMarkerPublicationFile,
  stage: RecallMarkerPublicationFailureStage,
): Promise<void> {
  try {
    await file.close();
  } catch (error) {
    throw formatRecallMarkerPublicationError(stage, error);
  }
}

async function writeDurableRecallMarkerTemporaryFile(
  filesystem: RecallMarkerPublicationFilesystem,
  temporaryPath: string,
  content: string,
): Promise<void> {
  let file: RecallMarkerPublicationFile;
  try {
    file = await filesystem.openExclusiveFile(temporaryPath);
  } catch (error) {
    throw formatRecallMarkerPublicationError('temporary file open', error);
  }
  try {
    try {
      await file.writeFile(content);
    } catch (error) {
      throw formatRecallMarkerPublicationError('temporary file write', error);
    }
    try {
      await file.sync();
    } catch (error) {
      throw formatRecallMarkerPublicationError('file sync', error);
    }
  } finally {
    await closeRecallMarkerPublicationFile(file, 'temporary file close');
  }
}

async function syncRecallMarkerSpoolDirectory(
  filesystem: RecallMarkerPublicationFilesystem,
  markerSpoolDirectory: string,
): Promise<void> {
  let directory: RecallMarkerPublicationFile;
  try {
    directory = await filesystem.openDirectory(markerSpoolDirectory);
  } catch (error) {
    throw formatRecallMarkerPublicationError('directory open', error);
  }
  try {
    try {
      await directory.sync();
    } catch (error) {
      throw formatRecallMarkerPublicationError('directory sync', error);
    }
  } finally {
    await closeRecallMarkerPublicationFile(directory, 'directory close');
  }
}

function resolveRecallDetachedWorkerSignal(
  options: PublishRecallWorkMarkerOptions,
): RecallDetachedWorkerSignal {
  if (options.workerSignal !== undefined) {
    return options.workerSignal;
  }
  if (options.workerOwnershipLockPath === undefined) {
    throw new Error('Recall marker publication requires a worker ownership lock path');
  }
  return createRecallDetachedWorkerSignal(options.workerOwnershipLockPath);
}

/** Durably publishes one complete immutable marker before signaling a detached worker. */
export async function publishRecallWorkMarker(
  marker: RecallWorkMarker,
  options: PublishRecallWorkMarkerOptions,
): Promise<void> {
  const filesystem = options.filesystem ?? nodeMarkerPublicationFilesystem;
  const workerSignal = resolveRecallDetachedWorkerSignal(options);
  let content: string;
  try {
    content = await encodeRecallWorkMarker(marker, options);
  } catch (error) {
    throw formatRecallMarkerPublicationError('encoding', error);
  }
  try {
    await filesystem.createDirectory(options.markerSpoolDirectory);
  } catch (error) {
    throw formatRecallMarkerPublicationError('spool create', error);
  }
  const markerPath = join(options.markerSpoolDirectory, `${marker.markerId}.json`);
  const temporaryPath = join(
    options.markerSpoolDirectory,
    `.${marker.markerId}.${randomUUID()}.tmp`,
  );
  try {
    await writeDurableRecallMarkerTemporaryFile(filesystem, temporaryPath, content);
    try {
      await filesystem.renameFile(temporaryPath, markerPath);
    } catch (error) {
      throw formatRecallMarkerPublicationError('rename', error);
    }
    await syncRecallMarkerSpoolDirectory(filesystem, options.markerSpoolDirectory);
    try {
      workerSignal.signalDetachedWorker();
    } catch (error) {
      throw formatRecallMarkerPublicationError('worker signal', error);
    }
  } catch (error) {
    try {
      await filesystem.removeFile(temporaryPath);
    } catch (cleanupError) {
      process.emitWarning(
        `Recall marker temporary cleanup failed [${readNodeErrorCode(cleanupError) ?? 'UNKNOWN'}]`,
      );
    }
    throw error;
  }
}

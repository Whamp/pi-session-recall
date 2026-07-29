import { randomUUID } from 'node:crypto';
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { access, mkdir, open, rename, rm } from 'node:fs/promises';
import { dirname } from 'node:path';

import { RecallRecoveryRequiredError, RecallSearchBusyError } from './errors.js';
import { readNodeErrorCode } from './read-node-error-code.js';
import { resolveRecallFlockExecutable } from './resolve-recall-flock-executable.js';
import { syncRecallDirectory } from './sync-recall-directory.js';

/** Durable files that distinguish a current recall write window from required crash recovery. */
export interface RecallWriteWindowStatePaths {
  currentWindowPath: string;
  recoveryRequiredPath: string;
}

/** Exclusive write-window capability granted only after the kernel lock is held. */
export interface RecallWriteWindow {
  recovering: boolean;
  /** Attests that write-capable recovery and normal close completed successfully. */
  attestRecoveryCompleted(): void;
  /** Forces recovery state to remain after the operation returns. */
  retainRecoveryRequired(): void;
}

/** Kernel lock, recovery authority, and cancellation inputs for one bounded write window. */
export interface CoordinateRecallWriteWindowOptions {
  lockPath: string;
  allowRecovery: boolean;
  signal?: AbortSignal;
}

/** Kernel lock and bounded wait inputs for one read-only recall operation. */
export interface CoordinateRecallReadWindowOptions {
  lockPath: string;
  waitMilliseconds: number;
  signal?: AbortSignal;
}

/** Scalar state visible to read-only callers without inspecting process liveness. */
export interface RecallWriteWindowState {
  currentWindow: boolean;
  recoveryRequired: boolean;
}

interface HeldRecallKernelLock {
  release(): Promise<void>;
}

/** Returns the explicit current-window and recovery-required state paths for one operation lock. */
export function recallWriteWindowStatePaths(lockPath: string): RecallWriteWindowStatePaths {
  return {
    currentWindowPath: `${lockPath}.current-window.json`,
    recoveryRequiredPath: `${lockPath}.recovery-required.json`,
  };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (readNodeErrorCode(error) === 'ENOENT') {
      return false;
    }
    throw error;
  }
}

/** Combines caller cancellation with one bounded operation-lock wait. */
export function createRecallWriteWindowAcquisitionSignal(
  signal?: AbortSignal,
  lockWaitMilliseconds?: number,
): AbortSignal | undefined {
  if (lockWaitMilliseconds === undefined) {
    return signal;
  }
  const timeoutSignal = AbortSignal.timeout(lockWaitMilliseconds);
  return signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal;
}

/** Reads only scalar coordination state; it never removes or repairs an interrupted writer. */
export async function inspectRecallWriteWindow(lockPath: string): Promise<RecallWriteWindowState> {
  const paths = recallWriteWindowStatePaths(lockPath);
  const [currentWindow, recoveryRequired] = await Promise.all([
    pathExists(paths.currentWindowPath),
    pathExists(paths.recoveryRequiredPath),
  ]);
  return { currentWindow, recoveryRequired };
}

/** Rejects read-only access while a current or interrupted write window owns recovery state. */
export async function assertRecallWriteWindowAvailableForRead(lockPath: string): Promise<void> {
  const state = await inspectRecallWriteWindow(lockPath);
  if (state.recoveryRequired) {
    throw new RecallRecoveryRequiredError();
  }
  if (state.currentWindow) {
    throw new RecallSearchBusyError();
  }
}

async function writeRecallWindowState(path: string, value: object): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${randomUUID()}.tmp`;
  const handle = await open(temporaryPath, 'wx');
  try {
    await handle.writeFile(`${JSON.stringify(value)}\n`, 'utf8');
    await handle.sync();
  } finally {
    await handle.close();
  }
  await rename(temporaryPath, path);
  await syncRecallDirectory(dirname(path));
}

function waitForRecallKernelLock(
  child: ChildProcessWithoutNullStreams,
  token: string,
  signal?: AbortSignal,
  createAbortError: () => Error = () =>
    new Error('Recall conversation operation cancelled', { cause: signal?.reason }),
): Promise<void> {
  return new Promise((resolve, reject) => {
    let output = '';
    let settled = false;
    const cleanup = (): void => {
      child.stdout.off('data', onData);
      child.off('error', onError);
      child.off('exit', onExit);
      signal?.removeEventListener('abort', onAbort);
    };
    const finish = (error?: Error): void => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      if (error) {
        reject(error);
      } else {
        resolve();
      }
    };
    const onData = (chunk: Buffer): void => {
      output += chunk.toString('utf8');
      if (output.includes(token)) {
        finish();
      }
    };
    const onError = (error: Error): void => {
      finish(new Error('Recall write window kernel lock failed', { cause: error }));
    };
    const onExit = (code: number | null): void => {
      finish(new Error(`Recall write window kernel lock exited before acquisition: ${code}`));
    };
    const onAbort = (): void => {
      child.kill('SIGTERM');
      finish(createAbortError());
    };
    child.stdout.on('data', onData);
    child.once('error', onError);
    child.once('exit', onExit);
    signal?.addEventListener('abort', onAbort, { once: true });
    if (signal?.aborted) {
      onAbort();
    }
  });
}

async function acquireRecallKernelLock(
  lockPath: string,
  mode: 'exclusive' | 'shared',
  signal?: AbortSignal,
  createAbortError?: () => Error,
): Promise<HeldRecallKernelLock> {
  await mkdir(dirname(lockPath), { recursive: true });
  const token = `recall-${mode}-window-${randomUUID()}`;
  const child = spawn(resolveRecallFlockExecutable(), [`--${mode}`, lockPath, '/bin/cat'], {
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  child.stdin.write(`${token}\n`);
  await waitForRecallKernelLock(child, token, signal, createAbortError);
  return {
    async release() {
      if (child.exitCode !== null || child.signalCode !== null) {
        return;
      }
      const exited = new Promise<void>((resolve, reject) => {
        child.once('error', reject);
        child.once('exit', () => resolve());
      });
      child.stdin.end();
      await exited;
    },
  };
}

async function markRecallWriteWindow(
  paths: RecallWriteWindowStatePaths,
  recovering: boolean,
): Promise<void> {
  const windowId = randomUUID();
  const startedAtEpochMilliseconds = Date.now();
  await writeRecallWindowState(paths.recoveryRequiredPath, {
    version: 1,
    state: 'recovery_required',
    windowId,
    startedAtEpochMilliseconds,
  });
  await writeRecallWindowState(paths.currentWindowPath, {
    version: 1,
    state: 'current_window',
    windowId,
    startedAtEpochMilliseconds,
    recovering,
  });
}

async function clearRecallWriteWindow(paths: RecallWriteWindowStatePaths): Promise<void> {
  await rm(paths.currentWindowPath, { force: true });
  await syncRecallDirectory(dirname(paths.currentWindowPath));
  await rm(paths.recoveryRequiredPath, { force: true });
  await syncRecallDirectory(dirname(paths.recoveryRequiredPath));
}

/**
 * Runs one bounded write window under a crash-released kernel lock.
 * Recovery state clears only after normal completion and explicit attestation when recovering.
 */
export async function coordinateRecallWriteWindow<T>(
  options: CoordinateRecallWriteWindowOptions,
  operation: (window: RecallWriteWindow) => Promise<T>,
): Promise<T> {
  const kernelLock = await acquireRecallKernelLock(options.lockPath, 'exclusive', options.signal);
  const paths = recallWriteWindowStatePaths(options.lockPath);
  try {
    const priorState = await inspectRecallWriteWindow(options.lockPath);
    const recovering = priorState.currentWindow || priorState.recoveryRequired;
    if (recovering && !options.allowRecovery) {
      throw new Error('Recall write recovery required before another write window can open');
    }
    await markRecallWriteWindow(paths, recovering);
    let operationCompleted = false;
    let recoveryCompleted = !recovering;
    let retainRecoveryRequired = false;
    try {
      const result = await operation({
        recovering,
        attestRecoveryCompleted() {
          recoveryCompleted = true;
        },
        retainRecoveryRequired() {
          retainRecoveryRequired = true;
        },
      });
      operationCompleted = true;
      return result;
    } finally {
      const recoveryMayClear = !recovering || (operationCompleted && recoveryCompleted);
      if (recoveryMayClear && !retainRecoveryRequired) {
        await clearRecallWriteWindow(paths);
      }
    }
  } finally {
    await kernelLock.release();
  }
}

/**
 * Holds a shared kernel lock around one read-only recall operation.
 * It waits only for the current writer and never creates or removes coordination markers.
 */
export async function coordinateRecallReadWindow<T>(
  options: CoordinateRecallReadWindowOptions,
  operation: () => Promise<T>,
): Promise<T> {
  if (!Number.isFinite(options.waitMilliseconds) || options.waitMilliseconds < 0) {
    throw new Error('Recall search write-window wait must be a nonnegative number');
  }
  const timeoutSignal = AbortSignal.timeout(options.waitMilliseconds);
  const acquisitionSignal = options.signal
    ? AbortSignal.any([options.signal, timeoutSignal])
    : timeoutSignal;
  const kernelLock = await acquireRecallKernelLock(
    options.lockPath,
    'shared',
    acquisitionSignal,
    () =>
      options.signal?.aborted
        ? new Error('Recall conversation operation cancelled', { cause: options.signal.reason })
        : new RecallSearchBusyError(),
  );
  try {
    const state = await inspectRecallWriteWindow(options.lockPath);
    if (state.recoveryRequired || state.currentWindow) {
      throw new RecallRecoveryRequiredError();
    }
    return await operation();
  } finally {
    await kernelLock.release();
  }
}
